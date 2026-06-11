//! WHIR-ZK side of the Jolt PCS commitment benchmark.
//!
//! Reads a field-agnostic polynomial dump produced by `jolt-pcs-bench`,
//! encodes each vector into BN254 `Field256`, groups by `num_vars` (size
//! class), and commits each group via `whir_zk::Config::commit`. Times only
//! the commit calls and reports min/median/max in milliseconds.

// Allocation profiling: install dhat's global allocator wrapper when the
// `profile-alloc` feature is on. Recording is gated on the `--profile-alloc`
// CLI flag inside `main`; without it the wrapper exists but doesn't record.
#[cfg(feature = "profile-alloc")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ark_ff::Field as ArkField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::distributions::{Distribution, Standard};
use clap::Parser;
use rayon::prelude::*;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use whir::algebra::fields::Field256;
use whir::hash;
use whir::parameters::ProtocolParameters;
use whir::protocols::whir_zk::Config;
use whir::transcript::codecs::Empty;
use whir::transcript::{Codec, DomainSeparator, ProverState};

mod gkr;
mod gkr_bn254;
mod gruen;

const DUMP_MAGIC: &[u8; 8] = b"JOLTPCSB";
const DUMP_VERSION: u32 = 3;

/// `log_m` (per-family pushforward num_vars) is a deterministic protocol
/// parameter — matches `WHIR_MIN_NUM_VARS = 15` on the writer side. Dump v3
/// no longer carries per-chunk pushforward entries, so the reader baked it
/// in here.
const LOG_M_PER_FAMILY: usize = 15;

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser, Debug)]
struct Args {
    /// Path to the polynomial dump produced by jolt-pcs-bench.
    #[arg(long, default_value = "/tmp/jolt-pcs-bench/polys.bin")]
    dump: PathBuf,

    /// Warm-up runs (not timed).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Measured runs.
    #[arg(
        long,
        default_value_t = 3,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    runs: usize,

    /// Optional JSON output path.
    #[arg(long)]
    json: Option<PathBuf>,

    /// Write a Chrome / Perfetto trace of all tracing spans to this path.
    /// View at https://ui.perfetto.dev/ . Captures WHIR's internal spans
    /// (irs_commit, sumcheck, matrix_commit, …) alongside bench-side spans
    /// (read_dump_raw, encode_to_field, extract_families, commit_size_class).
    #[arg(long)]
    trace_chrome: Option<PathBuf>,

    /// Capture a dhat heap-allocation profile. Writes the JSON to the path
    /// given by `--dhat-output` (default: `./dhat-heap.json`), viewable at
    /// https://nnethercote.github.io/dh_view/dh_view.html . Requires building
    /// with `--features profile-alloc`; otherwise a no-op (warning printed).
    #[arg(long)]
    profile_alloc: bool,

    /// Output path for the dhat profile (only meaningful with --profile-alloc).
    #[arg(long, default_value = "dhat-heap.json")]
    dhat_output: PathBuf,

    /// Skip the GKR pushforward proving phase. Useful for regression-checking
    /// the commit-only numbers against the prior milestone.
    #[arg(long, default_value_t = false)]
    no_gkr: bool,
}

/// Install a tracing subscriber. Returns the chrome flush guard (must be held
/// until program exit). Returns `None` if `--trace-chrome` wasn't given.
fn install_tracing(trace_chrome: Option<&PathBuf>) -> Option<tracing_chrome::FlushGuard> {
    if let Some(path) = trace_chrome {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .file(path)
            .build();
        tracing_subscriber::registry().with(chrome_layer).init();
        println!(
            "[trace] writing Chrome/Perfetto trace to {} (open at https://ui.perfetto.dev/)",
            path.display()
        );
        Some(guard)
    } else {
        None
    }
}

#[cfg(feature = "profile-alloc")]
fn install_dhat(enabled: bool, output: &std::path::Path) -> Option<dhat::Profiler> {
    enabled.then(|| {
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        println!(
            "[dhat] heap profiling enabled — `{}` will be written at exit",
            output.display()
        );
        dhat::Profiler::builder().file_name(output).build()
    })
}

#[cfg(not(feature = "profile-alloc"))]
fn install_dhat(enabled: bool, _output: &std::path::Path) -> Option<()> {
    if enabled {
        eprintln!(
            "[dhat] WARNING: --profile-alloc given but binary was built without \
             `--features profile-alloc`. No allocation profile will be captured."
        );
    }
    None
}

// =============================================================================
// Field-agnostic dump loader
// =============================================================================

/// One raw vector, kept in its source-integer form. Encoded to F lazily.
#[derive(Debug)]
struct RawVector {
    label: String,
    num_vars: usize,
    kind: Kind,
    /// Packed source bytes. For `Kind::U8`, length = num_elements.
    /// For `Kind::U32`, length = 4 * num_elements. For `Kind::I128`, 16 * num_elements.
    payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    U8,
    U32,
    I128,
}

impl Kind {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => Kind::U8,
            1 => Kind::U32,
            2 => Kind::I128,
            other => panic!("unknown dump kind byte {other}"),
        }
    }
    fn bytes_per_element(self) -> usize {
        match self {
            Kind::U8 => 1,
            Kind::U32 => 4,
            Kind::I128 => 16,
        }
    }
}

#[tracing::instrument(skip_all, name = "bench.read_dump_raw")]
fn read_dump_raw(path: &PathBuf) -> Vec<RawVector> {
    let file =
        File::open(path).unwrap_or_else(|e| panic!("could not open dump {}: {e}", path.display()));
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).expect("read magic");
    assert_eq!(&magic, DUMP_MAGIC, "bad dump magic");

    let version = read_u32(&mut reader);
    assert_eq!(
        version, DUMP_VERSION,
        "dump version {version} not supported by this binary (expected {DUMP_VERSION})"
    );

    let num_vectors = read_u32(&mut reader) as usize;
    let mut vectors = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let mut kind_byte = [0u8; 1];
        reader.read_exact(&mut kind_byte).expect("read kind");
        let kind = Kind::from_byte(kind_byte[0]);
        let label_len = read_u32(&mut reader) as usize;
        let mut label_bytes = vec![0u8; label_len];
        reader.read_exact(&mut label_bytes).expect("read label");
        let label = String::from_utf8(label_bytes).expect("utf8 label");
        let len = read_u32(&mut reader) as usize;
        let num_vars = (len as u32).trailing_zeros() as usize;
        assert_eq!(
            1usize << num_vars,
            len,
            "vector {label} has non-power-of-2 length {len}"
        );
        let payload_bytes = len * kind.bytes_per_element();
        let mut payload = vec![0u8; payload_bytes];
        reader.read_exact(&mut payload).expect("read payload");
        vectors.push(RawVector {
            label,
            num_vars,
            kind,
            payload,
        });
    }
    vectors
}

fn read_u32(reader: &mut impl Read) -> u32 {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).expect("read u32");
    u32::from_le_bytes(buf)
}

// =============================================================================
// Integer → field encoding (field-agnostic)
// =============================================================================

/// i128 → F via signed mod-p reduction. Asserts |v| fits comfortably in F.
fn i128_to_f<F: ArkField>(v: i128) -> F {
    // For BN254 Fr (≈2^254), |v| ≤ 2^65 ≪ modulus.
    // The debug_assert catches a future small-field addition where this stops being true.
    debug_assert!(
        v.unsigned_abs() < (1u128 << 65),
        "i128 dump value {v} exceeds the i65 range expected for RdInc/RamInc"
    );
    if v >= 0 {
        F::from(v as u128)
    } else {
        -F::from((-v) as u128)
    }
}

#[tracing::instrument(skip_all, name = "bench.encode_to_field", fields(label = %raw.label, num_vars = raw.num_vars))]
fn encode_to_field<F: ArkField + Send>(raw: &RawVector) -> Vec<F> {
    match raw.kind {
        Kind::U8 => raw
            .payload
            .par_iter()
            .map(|&b| F::from(u64::from(b)))
            .collect(),
        Kind::U32 => raw
            .payload
            .par_chunks_exact(4)
            .map(|b| F::from(u64::from(u32::from_le_bytes(b.try_into().unwrap()))))
            .collect(),
        Kind::I128 => raw
            .payload
            .par_chunks_exact(16)
            .map(|b| i128_to_f::<F>(i128::from_le_bytes(b.try_into().unwrap())))
            .collect(),
    }
}

/// A typed-up vector ready for WHIR commit.
struct TypedVector<F> {
    label: String,
    num_vars: usize,
    values: Vec<F>,
    /// Populated only for `Kind::U8` vectors (ra_dense chunks). Carries the
    /// raw integer indices alongside the F-encoded `values` so the GKR module
    /// can scatter into the eq-weighted pushforward without doing F→usize
    /// inversions.
    raw_u8: Option<Vec<u8>>,
}

#[tracing::instrument(skip_all, name = "bench.convert_all")]
fn convert_all<F: ArkField + Send + Sync>(raw: &[RawVector]) -> Vec<TypedVector<F>> {
    raw.par_iter()
        .map(|r| TypedVector {
            label: r.label.clone(),
            num_vars: r.num_vars,
            values: encode_to_field::<F>(r),
            raw_u8: match r.kind {
                Kind::U8 => Some(r.payload.clone()),
                _ => None,
            },
        })
        .collect()
}

// =============================================================================
// Per-field run + dispatch
// =============================================================================

fn whir_params() -> ProtocolParameters {
    ProtocolParameters {
        security_level: 128,
        pow_bits: 20,
        initial_folding_factor: 4,
        folding_factor: 4,
        unique_decoding: false,
        starting_log_inv_rate: 1,
        batch_size: 1,
        hash_id: hash::BLAKE3,
    }
}

#[tracing::instrument(skip_all, name = "bench.commit_size_class", fields(num_vars = num_vars, num_polys = vectors.len()))]
fn commit_size_class<F>(num_vars: usize, vectors: &[&[F]], params: &ProtocolParameters) -> Duration
where
    F: ArkField + Codec<[u8]>,
    Standard: Distribution<F>,
{
    let size = 1usize << num_vars;
    let num_polys = vectors.len();
    let config = Config::<F>::new(size, params, num_polys);
    let ds = DomainSeparator::protocol(&config)
        .session(&"jolt-pcs-bench/whir-zk")
        .instance(&Empty);
    let mut prover_state = ProverState::new_std(&ds);

    let t0 = Instant::now();
    // `vectors` already has the right shape (`&[&[F]]`) — no per-call clone
    // or temporary `Vec<&[F]>` build needed.
    let _witness = config.commit(&mut prover_state, vectors);
    t0.elapsed()
}

fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

struct BenchStats {
    field_label: &'static str,
    field_bytes_per_elem: usize,
    total_elements: usize,
    size_classes: BTreeMap<usize, usize>,
    runs_ms: Vec<f64>,
    commit_runs_ms: Vec<f64>,
    per_class_ms: BTreeMap<usize, Vec<f64>>,
    encode_seconds: f64,
    /// Populated only when --no-gkr is NOT set.
    gkr: Option<GkrStats>,
}

struct GkrStats {
    /// Total per-run wall-clock = claim_eval_ms + gkr_ms (`commit_ms` is
    /// reported separately; `claim_eval_ms` and `gkr_ms` together cover the
    /// LogUp\* commitment-scheme prover overhead beyond commit).
    runs_ms: Vec<f64>,
    /// Per-run claim_eval phase (d MLE evaluations per family).
    claim_eval_ms: Vec<f64>,
    /// Per-run gkr phase (§4.5.2 RLC + P^F build + Figure-1 sumchecks).
    gkr_only_ms: Vec<f64>,
    /// Per-run per-family GKR cost (3 entries per run).
    per_family_ms: Vec<Vec<f64>>,
    /// Per-run per-layer GKR cost (max A-depth entries per run).
    per_layer_ms: Vec<Vec<f64>>,
}

impl BenchStats {
    fn min_med_max(&self) -> (f64, f64, f64) {
        min_med_max(&self.runs_ms)
    }
}

fn min_med_max(values: &[f64]) -> (f64, f64, f64) {
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    (s[0], s[s.len() / 2], s[s.len() - 1])
}

/// Per-family slice view over the typed-vector list. Holding borrowed
/// `&[F]` / `&[u8]` slices avoids the ~2 GB clone that an owned-Vec
/// representation cost when families were collected up-front.
struct FamilyData<'a, F> {
    name: &'static str,
    d: usize,
    log_t: usize,
    log_d: usize,
    log_m: usize,
    /// d ra_dense slices, F-encoded (used for commit + GKR leaf-denoms).
    ra_dense_chunks_f: Vec<&'a [F]>,
    /// d ra_dense slices as raw u8 indices (used for the eq-weighted
    /// pushforward scatter and the leaf-denom F::from in build_circuit_a).
    ra_dense_chunks_u8: Vec<&'a [u8]>,
}

/// Walk `typed` in dump order and group `ra_dense::FAMILY_chunk` entries into
/// 3 families (InstructionRa, BytecodeRa, RamRa). Dump v3 contains no
/// pushforward entries — the §4.1 eq-weighted pushforward is built at runtime
/// per family inside `gkr::prepare_pushforwards`. Returned `FamilyData<'a>`
/// borrows from `typed` rather than cloning the encoded values.
#[tracing::instrument(skip_all, name = "bench.extract_families")]
fn extract_families<'a, F: ArkField>(typed: &'a [TypedVector<F>]) -> Vec<FamilyData<'a, F>> {
    let mut families: Vec<FamilyData<'a, F>> = Vec::new();
    let mut current_family: Option<(&'static str, usize)> = None;
    let mut log_t_seen = 0usize;

    for v in typed {
        let Some(rest) = v.label.strip_prefix("ra_dense::") else {
            // dense::RdInc / dense::RamInc — committed verbatim, not grouped.
            continue;
        };
        let (fam_name_raw, _chunk_idx_str) = rest
            .rsplit_once('_')
            .unwrap_or_else(|| panic!("malformed ra_dense label `{}`", v.label));
        let fam_name: &'static str = match fam_name_raw {
            "InstructionRa" => "InstructionRa",
            "BytecodeRa" => "BytecodeRa",
            "RamRa" => "RamRa",
            other => panic!("unknown ra_dense family `{other}` in dump"),
        };
        let log_t = v.num_vars;
        if log_t_seen == 0 {
            log_t_seen = log_t;
        } else {
            assert_eq!(log_t_seen, log_t, "ra_dense vectors must share log_t");
        }
        let raw_u8 = v
            .raw_u8
            .as_deref()
            .unwrap_or_else(|| panic!("ra_dense `{}` missing raw_u8 backing", v.label));

        match current_family {
            Some((name, fam_idx)) if name == fam_name => {
                families[fam_idx]
                    .ra_dense_chunks_f
                    .push(v.values.as_slice());
                families[fam_idx].ra_dense_chunks_u8.push(raw_u8);
            }
            _ => {
                families.push(FamilyData {
                    name: fam_name,
                    d: 0,
                    log_t,
                    log_d: 0,
                    log_m: LOG_M_PER_FAMILY,
                    ra_dense_chunks_f: vec![v.values.as_slice()],
                    ra_dense_chunks_u8: vec![raw_u8],
                });
                current_family = Some((fam_name, families.len() - 1));
            }
        }
    }

    for family in &mut families {
        family.d = family.ra_dense_chunks_f.len();
        family.log_d = family.d.next_power_of_two().trailing_zeros() as usize;
    }

    families
}

type RunGkrFn<F> = fn(&mut ProverState, Vec<gkr::FamilyState<F>>, Duration) -> gkr::Timing;

fn run_bench<F>(
    args: &Args,
    raw: &[RawVector],
    field_label: &'static str,
    run_gkr_fn: RunGkrFn<F>,
) -> BenchStats
where
    F: ArkField + Codec<[u8]> + Send + Sync + 'static,
    Standard: Distribution<F>,
{
    let encode_start = Instant::now();
    let typed: Vec<TypedVector<F>> = convert_all::<F>(raw);

    // Group ra_dense entries by family; dump v3 contains no pushforward.
    let families: Vec<FamilyData<'_, F>> = extract_families::<F>(&typed);
    let encode_seconds = encode_start.elapsed().as_secs_f64();

    // We will RUN-LOCALLY add per-family eq-weighted pushforwards to the
    // typed list before each measured commit pass. Outside of that, we
    // don't know what to do with them — they depend on Fiat-Shamir squeezes
    // taken inside the GKR prover state.
    let total_elements_ra_only: usize = typed.iter().map(|t| t.values.len()).sum();
    let field_bytes_per_elem = F::default().compressed_size();

    println!(
        "[whir] load+encode: {:.2}s ({} vectors after family aggregation, {} field elements ra+dense, {} bytes/elem)",
        encode_seconds, typed.len(), total_elements_ra_only, field_bytes_per_elem
    );
    if !families.is_empty() {
        for f in &families {
            println!(
                "[whir] family `{}`: d={} log_t={} log_d={} log_m={}",
                f.name, f.d, f.log_t, f.log_d, f.log_m
            );
        }
    }

    let params = whir_params();
    println!(
        "[whir] params: security={} pow_bits={} init_fold={} fold={} rate=2^-{} hash=Blake3",
        params.security_level,
        params.pow_bits,
        params.initial_folding_factor,
        params.folding_factor,
        params.starting_log_inv_rate
    );

    // --- Warmup runs ---
    for w in 0..args.warmup {
        let t0 = Instant::now();
        let _ = one_run::<F>(&families, &typed, &params, args.no_gkr, run_gkr_fn);
        println!(
            "[whir] warmup {}/{}: {:.1}ms",
            w + 1,
            args.warmup,
            dur_ms(t0.elapsed())
        );
    }

    // --- Measured runs ---
    let mut runs_ms = Vec::with_capacity(args.runs);
    let mut commit_runs_ms = Vec::with_capacity(args.runs);
    let mut per_class_ms: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut gkr_runs_ms: Vec<f64> = Vec::new();
    let mut claim_eval_ms: Vec<f64> = Vec::new();
    let mut gkr_only_ms: Vec<f64> = Vec::new();
    let mut per_family_ms: Vec<Vec<f64>> = Vec::new();
    let mut per_layer_ms: Vec<Vec<f64>> = Vec::new();
    let mut last_size_classes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut last_total_elements: usize = total_elements_ra_only;

    for r in 0..args.runs {
        let t0 = Instant::now();
        let run = one_run::<F>(&families, &typed, &params, args.no_gkr, run_gkr_fn);
        let total = dur_ms(t0.elapsed());
        runs_ms.push(total);
        commit_runs_ms.push(run.commit_total_ms);

        // Aggregate per-class commit timings.
        for (num_vars, t) in &run.per_class_commit_ms {
            per_class_ms.entry(*num_vars).or_default().push(*t);
        }

        // Aggregate GKR timings if present.
        if let Some(gkr) = run.gkr {
            claim_eval_ms.push(gkr.claim_eval_ms);
            gkr_only_ms.push(gkr.gkr_only_ms);
            gkr_runs_ms.push(gkr.claim_eval_ms + gkr.gkr_only_ms);
            per_family_ms.push(gkr.per_family_ms);
            per_layer_ms.push(gkr.per_layer_ms);
            println!(
                "[whir] run {}/{}: commit={:.1}ms claim_eval={:.1}ms gkr={:.1}ms (total {:.1}ms)",
                r + 1,
                args.runs,
                run.commit_total_ms,
                gkr.claim_eval_ms,
                gkr.gkr_only_ms,
                total
            );
        } else {
            println!(
                "[whir] run {}/{}: commit={:.1}ms (total {:.1}ms, GKR skipped)",
                r + 1,
                args.runs,
                run.commit_total_ms,
                total
            );
        }

        last_size_classes = run.size_classes;
        last_total_elements = run.total_elements;
    }

    let gkr = if args.no_gkr || gkr_runs_ms.is_empty() {
        None
    } else {
        Some(GkrStats {
            runs_ms: gkr_runs_ms,
            claim_eval_ms,
            gkr_only_ms,
            per_family_ms,
            per_layer_ms,
        })
    };

    BenchStats {
        field_label,
        field_bytes_per_elem,
        total_elements: last_total_elements,
        size_classes: last_size_classes,
        runs_ms,
        commit_runs_ms,
        per_class_ms,
        encode_seconds,
        gkr,
    }
}

/// One measured run: (a) run the LogUp\* GKR prep to build eq-weighted P^F
/// vectors (if GKR enabled), (b) commit the full set ra_dense + dense + P^F
/// via WHIR, (c) run the GKR sumchecks.
struct OneRun {
    commit_total_ms: f64,
    per_class_commit_ms: Vec<(usize, f64)>,
    size_classes: BTreeMap<usize, usize>,
    total_elements: usize,
    gkr: Option<OneRunGkr>,
}

struct OneRunGkr {
    claim_eval_ms: f64,
    gkr_only_ms: f64,
    per_family_ms: Vec<f64>,
    per_layer_ms: Vec<f64>,
}

fn one_run<F>(
    families: &[FamilyData<'_, F>],
    typed_ra_and_dense: &[TypedVector<F>],
    params: &ProtocolParameters,
    no_gkr: bool,
    run_gkr_fn: RunGkrFn<F>,
) -> OneRun
where
    F: ArkField + Codec<[u8]> + Send + Sync + 'static,
    Standard: Distribution<F>,
{
    // === Phase 1: claim_eval + §4.5.2 + eq-weighted P^F build ===
    let (pushforwards, family_states, claim_eval_dur, gkr_pre_commit_dur, log_m_for_pf) =
        if no_gkr || families.is_empty() {
            (
                Vec::<Vec<F>>::new(),
                Vec::new(),
                Duration::ZERO,
                Duration::ZERO,
                0usize,
            )
        } else {
            let mut ps = new_logup_prover_state();
            // Build borrowed gkr::Family views from FamilyData. The
            // `Vec<&[F]>` / `Vec<&[u8]>` clones are shallow fat-ptr copies
            // since FamilyData already borrows into `typed`.
            let gkr_families: Vec<gkr::Family<'_, F>> = families
                .iter()
                .map(|f| gkr::Family {
                    name: f.name,
                    d: f.d,
                    log_t: f.log_t,
                    log_d: f.log_d,
                    log_m: f.log_m,
                    ra_dense_chunks_f: f.ra_dense_chunks_f.clone(),
                    ra_dense_chunks_u8: f.ra_dense_chunks_u8.clone(),
                })
                .collect();
            let prep = gkr::prepare_pushforwards::<F>(&mut ps, &gkr_families);
            let log_m = families[0].log_m;
            (
                prep.pushforwards,
                prep.family_states,
                prep.claim_eval_time,
                prep.gkr_pre_commit_time,
                log_m,
            )
        };

    // Owned `ProverState` needed by the sumcheck phase later; we stashed all
    // FS squeezes in `ps` above and now hand it off via `family_states`. The
    // sumcheck will reinitialize its own ProverState since the bench is
    // wall-clock-only and the prover-state hand-off is internal.
    // (See `gkr::run_gkr`: it threads its own ProverState through.)
    let mut gkr_ps = new_logup_prover_state();

    // === Phase 2: commit ===
    // Build (num_vars → slice list) groups directly. No intermediate
    // TypedVector clones — `typed_ra_and_dense` outlives one_run, and the
    // per-family eq-weighted pushforwards live in `pushforwards` already.
    let mut groups: BTreeMap<usize, Vec<&[F]>> = BTreeMap::new();
    for v in typed_ra_and_dense {
        groups
            .entry(v.num_vars)
            .or_default()
            .push(v.values.as_slice());
    }
    for p in &pushforwards {
        groups.entry(log_m_for_pf).or_default().push(p.as_slice());
    }

    let commit_t0 = Instant::now();
    let mut per_class_commit_ms: Vec<(usize, f64)> = Vec::new();
    for (num_vars, vs) in &groups {
        let elapsed = commit_size_class::<F>(*num_vars, vs, params);
        per_class_commit_ms.push((*num_vars, dur_ms(elapsed)));
    }
    let commit_total_ms = dur_ms(commit_t0.elapsed());

    let size_classes: BTreeMap<usize, usize> = groups.iter().map(|(k, v)| (*k, v.len())).collect();
    let total_elements: usize = groups
        .values()
        .flat_map(|vs| vs.iter().map(|s| s.len()))
        .sum();

    // === Phase 3: GKR sumcheck ===
    let gkr_payload = if no_gkr || family_states.is_empty() {
        None
    } else {
        let timing = run_gkr_fn(&mut gkr_ps, family_states, gkr_pre_commit_dur);
        let per_family_ms: Vec<f64> = timing.per_family_gkr.iter().map(|d| dur_ms(*d)).collect();
        let per_layer_ms: Vec<f64> = timing.per_layer_gkr.iter().map(|d| dur_ms(*d)).collect();
        Some(OneRunGkr {
            claim_eval_ms: dur_ms(claim_eval_dur),
            gkr_only_ms: dur_ms(timing.gkr),
            per_family_ms,
            per_layer_ms,
        })
    };

    OneRun {
        commit_total_ms,
        per_class_commit_ms,
        size_classes,
        total_elements,
        gkr: gkr_payload,
    }
}

fn new_logup_prover_state() -> ProverState {
    let ds = DomainSeparator::protocol(&"jolt-pcs-bench/logup-star")
        .session(&"figure-1")
        .instance(&Empty);
    ProverState::new_std(&ds)
}

// =============================================================================
// main
// =============================================================================

fn check_round_trip_bn254() {
    let cases = [0u64, 1, 2, 1_234_567, u64::MAX];
    for v in cases {
        let f = Field256::from(v);
        let mut buf = [0u8; 32];
        f.serialize_compressed(&mut buf[..]).expect("serialize");
        let recovered = Field256::deserialize_compressed(&buf[..]).expect("deserialize");
        assert_eq!(f, recovered, "Fp256 round-trip failed for {v}");
    }
    println!("[whir] Fp256 self-check: serialize/deserialize round-trip OK");
}

fn main() {
    let args = Args::parse();
    let _chrome_guard = install_tracing(args.trace_chrome.as_ref());
    let _dhat_guard = install_dhat(args.profile_alloc, &args.dhat_output);

    println!("[whir] reading dump from {}", args.dump.display());
    let load_start = Instant::now();
    let raw = read_dump_raw(&args.dump);
    println!(
        "[whir] read {} raw vectors in {:.2}s",
        raw.len(),
        load_start.elapsed().as_secs_f64()
    );

    // BN254-only benchmark. WHIR's `Field256` is `ark_bn254::Fr` (same Rust type
    // as `jolt_field::Fr`'s inner field), so the Jolt-optimized GKR hot path
    // (`gkr_bn254::run_gkr`) applies directly.
    let stats = {
        check_round_trip_bn254();
        println!("[whir] BN254 GKR backend: Jolt-optimized (gkr_bn254::run_gkr)");
        let gkr_fn: RunGkrFn<Field256> = gkr_bn254::run_gkr;
        run_bench::<Field256>(&args, &raw, "BN254 Fr (Field256)", gkr_fn)
    };

    let (min, median, max) = stats.min_med_max();
    println!(
        "[whir] summary [{}]: min={min:.1}ms median={median:.1}ms max={max:.1}ms  encode={:.2}s",
        stats.field_label, stats.encode_seconds,
    );
    let (commit_min, commit_median, commit_max) = min_med_max(&stats.commit_runs_ms);
    println!(
        "[whir] commit summary [{}]: min={commit_min:.1}ms median={commit_median:.1}ms max={commit_max:.1}ms",
        stats.field_label,
    );

    let gkr_summary: Option<serde_json::Value> = stats.gkr.as_ref().map(|g| {
        let (gmin, gmedian, gmax) = min_med_max(&g.runs_ms);
        let (ce_min, ce_median, ce_max) = min_med_max(&g.claim_eval_ms);
        let (gk_min, gk_median, gk_max) = min_med_max(&g.gkr_only_ms);
        println!(
            "[logup*] summary [{}]: claim_eval median={ce_median:.1}ms  gkr median={gk_median:.1}ms  total median={gmedian:.1}ms",
            stats.field_label,
        );
        println!(
            "[logup*] complete WHIR run median (commit + LogUp*) = {median:.1}ms ({} runs)",
            g.runs_ms.len(),
        );

        // Median-across-runs per-family (3 entries) and per-layer (24 entries).
        let runs = g.per_family_ms.len();
        let family_medians: Vec<f64> = if runs == 0 || g.per_family_ms[0].is_empty() {
            Vec::new()
        } else {
            let n_fam = g.per_family_ms[0].len();
            (0..n_fam)
                .map(|fam_idx| {
                    let mut v: Vec<f64> = (0..runs).map(|r| g.per_family_ms[r][fam_idx]).collect();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[v.len() / 2]
                })
                .collect()
        };
        let layer_medians: Vec<f64> = if runs == 0 || g.per_layer_ms[0].is_empty() {
            Vec::new()
        } else {
            let n_layers = g.per_layer_ms[0].len();
            (0..n_layers)
                .map(|l| {
                    let mut v: Vec<f64> = (0..runs).map(|r| g.per_layer_ms[r][l]).collect();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[v.len() / 2]
                })
                .collect()
        };

        serde_json::json!({
            "runs_ms": g.runs_ms,
            "min_ms": gmin,
            "median_ms": gmedian,
            "max_ms": gmax,
            "claim_eval_ms": {
                "runs_ms": g.claim_eval_ms,
                "min_ms": ce_min,
                "median_ms": ce_median,
                "max_ms": ce_max,
            },
            "gkr_only_ms": {
                "runs_ms": g.gkr_only_ms,
                "min_ms": gk_min,
                "median_ms": gk_median,
                "max_ms": gk_max,
            },
            "per_family_ms": family_medians,
            "per_layer_ms": layer_medians,
        })
    });

    if let Some(path) = &args.json {
        let mut report = serde_json::json!({
            "scheme": "whir-zk",
            "field": stats.field_label,
            "field_bytes_per_elem": stats.field_bytes_per_elem,
            "total_field_elements": stats.total_elements,
            "encode_seconds": stats.encode_seconds,
            "params": {
                "security_level": 128,
                "pow_bits": 20,
                "initial_folding_factor": 4,
                "folding_factor": 4,
                "starting_log_inv_rate": 1,
                "hash": "blake3",
            },
            "runs_ms": stats.runs_ms,
            "min_ms": min,
            "median_ms": median,
            "max_ms": max,
            "commit": {
                "runs_ms": stats.commit_runs_ms,
                "min_ms": commit_min,
                "median_ms": commit_median,
                "max_ms": commit_max,
            },
            "per_class_ms": stats.per_class_ms,
            "size_classes": stats.size_classes.iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect::<BTreeMap<_, _>>(),
        });
        if let Some(gkr_json) = gkr_summary {
            report["gkr"] = gkr_json;
        }
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).expect("write json");
        println!("[whir] wrote JSON report to {}", path.display());
    }
}
