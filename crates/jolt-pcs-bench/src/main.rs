// Force-link the p256 inlines so their `inventory::submit!`
// registrations are visible to the tracer at runtime.
use jolt_inlines_p256 as _;

// Allocation profiling: install dhat's global allocator wrapper when the
// `profile-alloc` feature is on. Recording is gated on the `--profile-alloc`
// CLI flag inside `main`; without it the wrapper exists but doesn't record.
#[cfg(feature = "profile-alloc")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod dory_bench;
mod dump;
mod jolt_polys;
mod logup_star;
mod sources;
mod verify;
mod workload;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use dory_bench::{bench_dory, DoryBenchSummary};
use dump::dump_for_whir;
use jolt_polys::build_polynomial_set;
use logup_star::transform;
use verify::verify_transformation;
use workload::build_ecdsa_workload;

#[derive(Parser, Debug)]
#[command(name = "jolt-pcs-bench")]
struct Args {
    /// Path the polynomial dump for whir-pcs-bench is written to.
    #[arg(long, default_value = "/tmp/jolt-pcs-bench/polys.bin")]
    dump: PathBuf,

    /// Skip the Dory commit benchmark (e.g. when only dumping for WHIR).
    #[arg(long)]
    no_dory: bool,

    /// Skip writing the WHIR polynomial dump.
    #[arg(long)]
    no_dump: bool,

    /// Only run the verify-transformation invariants, no timing.
    #[arg(long)]
    verify_only: bool,

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
    /// View at https://ui.perfetto.dev/ . Requires the WHIR `tracing` feature
    /// (already enabled) and adds bench-side spans for `build_polynomial_set`,
    /// `transform`, `verify_transformation`, `dump_for_whir`, `bench_dory`.
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
}

/// Install a tracing subscriber. Returns the chrome flush guard (must be held
/// until program exit). Returns `None` if `--trace-chrome` wasn't given —
/// the global subscriber is still installed so spans are inert no-ops.
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

fn main() {
    let args = Args::parse();
    let _chrome_guard = install_tracing(args.trace_chrome.as_ref());
    let _dhat_guard = install_dhat(args.profile_alloc, &args.dhat_output);
    let start = Instant::now();

    let workload = build_ecdsa_workload();
    println!(
        "[main] workload built: T={}, instruction_d={}, bytecode_d={}, ram_d={}",
        workload.trace_len,
        workload.one_hot_params.instruction_d,
        workload.one_hot_params.bytecode_d,
        workload.one_hot_params.ram_d,
    );

    let polys = build_polynomial_set(
        &workload.sources,
        &workload.one_hot_params,
        workload.trace_len,
    );
    let total = polys.total_field_elements();
    let one_hot_count: usize = polys
        .one_hot_families
        .iter()
        .map(|fam| fam.chunks.len())
        .sum();
    println!(
        "[main] Dory polynomial set: {} one-hot chunks across {} families, {} dense, total {} field elements ({:.1}M)",
        one_hot_count,
        polys.one_hot_families.len(),
        polys.dense.len(),
        total,
        (total as f64) / 1.0e6,
    );

    let logup = transform(&polys);
    let ra_dense_total: usize = logup.ra_dense.iter().map(|r| r.values.len()).sum();
    let dense_field: usize = polys.dense.iter().map(|d| d.values.len()).sum();

    // The WHIR side commits one eq-weighted P^F ∈ F^{2^WHIR_MIN_NUM_VARS}
    // per family (LogUp* §4.1).
    let n_families = polys.one_hot_families.len();
    let eq_weighted_pf_len = 1usize << logup_star::WHIR_MIN_NUM_VARS;
    let whir_total = ra_dense_total + n_families * eq_weighted_pf_len + dense_field;

    println!(
        "[main] WHIR-side committed set (after §4.1 family aggregation): \
         {} ra_dense (each 2^{}) + {} eq-weighted P^F (each 2^{}) + {} dense (each 2^{}) \
         = {} elements ({:.1}M)",
        logup.ra_dense.len(),
        workload.log_t,
        n_families,
        logup_star::WHIR_MIN_NUM_VARS,
        polys.dense.len(),
        workload.log_t,
        whir_total,
        (whir_total as f64) / 1.0e6,
    );
    println!(
        "[main] Dory/WHIR field-element ratio: {:.2}x (WHIR is {:.1}% of Dory)",
        (total as f64) / (whir_total as f64),
        100.0 * (whir_total as f64) / (total as f64),
    );

    if cfg!(debug_assertions) || args.verify_only {
        verify_transformation(&workload, &polys, &logup);
    }

    if args.verify_only {
        println!(
            "[main] --verify-only: exiting after invariants ({:.2}s)",
            start.elapsed().as_secs_f64()
        );
        return;
    }

    if !args.no_dump {
        let dump_start = Instant::now();
        let n = dump_for_whir(&polys, &workload.sources, workload.trace_len, &args.dump)
            .expect("dump polys");
        println!(
            "[dump] wrote {n} field elements to {} in {:.1}s",
            args.dump.display(),
            dump_start.elapsed().as_secs_f64()
        );
    }

    let mut dory_summary: Option<DoryBenchSummary> = None;
    if !args.no_dory {
        let summary = bench_dory(&polys, args.warmup, args.runs);
        let times = summary.total_times_ms();
        let (min, median, max) = min_median_max(&times);
        println!(
            "[dory] summary: min={min:.1}ms median={median:.1}ms max={max:.1}ms (setup={:.1}ms)",
            summary.setup_ms
        );
        dory_summary = Some(summary);
    }

    if let Some(path) = &args.json {
        write_json_report(path, total, whir_total, &workload, dory_summary.as_ref());
        println!("[main] wrote JSON report to {}", path.display());
    }

    println!(
        "[main] total wall time: {:.2}s",
        start.elapsed().as_secs_f64()
    );
}

fn min_median_max(times: &[f64]) -> (f64, f64, f64) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let min = sorted[0];
    let median = sorted[sorted.len() / 2];
    let max = sorted[sorted.len() - 1];
    (min, median, max)
}

fn write_json_report(
    path: &std::path::Path,
    dory_total_elements: usize,
    whir_total_elements: usize,
    workload: &workload::EcdsaWorkload,
    dory: Option<&DoryBenchSummary>,
) {
    let dory_json = dory.map(|summary| {
        let times = summary.total_times_ms();
        let (min, median, max) = min_median_max(&times);
        let per_oracle_serialized: Vec<Vec<serde_json::Value>> = summary
            .runs
            .iter()
            .map(|run| {
                run.per_oracle
                    .iter()
                    .map(|t| {
                        let name = match t.chunk_idx {
                            Some(idx) => format!("{}_{idx}", t.family_name),
                            None => t.family_name.to_string(),
                        };
                        serde_json::json!({
                            "name": name,
                            "num_vars": t.num_vars,
                            "elapsed_ms": t.elapsed_ms,
                        })
                    })
                    .collect()
            })
            .collect();
        serde_json::json!({
            "setup_ms": summary.setup_ms,
            "setup_num_vars": summary.setup_num_vars,
            "runs_ms": times,
            "min_ms": min,
            "median_ms": median,
            "max_ms": max,
            "per_oracle_runs": per_oracle_serialized,
        })
    });

    let report = serde_json::json!({
        "scheme": "dory",
        "field": "BN254 Fr",
        "workload": {
            "name": "p256-ecdsa-verify",
            "log_t": workload.log_t,
            "trace_len": workload.trace_len,
            "bytecode_k": workload.bytecode_k,
            "ram_k": workload.ram_k,
            "log_k_chunk": workload.one_hot_params.log_k_chunk,
            "instruction_d": workload.one_hot_params.instruction_d,
            "bytecode_d": workload.one_hot_params.bytecode_d,
            "ram_d": workload.one_hot_params.ram_d,
        },
        "dory_total_field_elements": dory_total_elements,
        "whir_total_field_elements": whir_total_elements,
        "ratio_dory_over_whir": (dory_total_elements as f64) / (whir_total_elements as f64),
        "dory": dory_json,
    });
    std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).expect("write json");
}
