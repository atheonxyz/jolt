//! Phase-1 live end-to-end: fibonacci guest → trace → base-Goldilocks limb
//! witness columns → WHIR base-commit, with validation and a commit-volume
//! report.
//!
//! This mirrors `workload.rs` (which drives the ECDSA guest) but for the
//! fibonacci guest, then runs it through the Phase-1 Goldilocks pipeline:
//!   `CommitmentTraceSources`  (jolt-witness, field-agnostic)
//!     → `GoldilocksWitnessColumns::build`  (base-limb columns: ra_dense + Inc)
//!     → `jolt_whir::commit_witness`  (WHIR commit over `Basefield<Field64_3>`)
//!
//! Validation performed by the e2e test:
//!   - the commit succeeds and the column count matches the trace geometry;
//!   - the `Inc` limbs **recompose** to the original signed increments;
//!   - a single-point WHIR **open/verify** round-trips on a non-degenerate column;
//!   - commits the SAME trace via the actual Jolt protocol (BN254 + Dory, the
//!     production `build_polynomial_set` + `bench_dory` path) and reports both
//!     schemes side by side: representation, committed volume, field-element
//!     width (32 B BN254 → 8 B Goldilocks), and measured commit wall-clock.
//!
//! The whir-`Field64` arithmetic cross-check lives in `jolt-whir/tests/crosscheck.rs`
//! (the field oracle from the commit side); this is the trace-driven e2e.
//!
//! Gated behind `--features goldilocks`. The e2e test is `#[ignore]` because it
//! compiles a RISC-V guest (needs the toolchain + `.bolt-dev-env`); run it with:
//!   `cargo nextest run -p jolt-pcs-bench --features goldilocks -- --ignored --no-capture`

#![allow(dead_code)] // workload + report are exercised only by the #[ignore] e2e test

use std::time::Instant;

use jolt_core::zkvm::config::OneHotParams;
use jolt_field::Fr;
use jolt_trace::bytecode::BytecodePreprocessing;
use jolt_trace::{extract_trace, Program};
use jolt_witness::goldilocks::{FamilyLayout, GoldilocksLayout};
use jolt_witness::{commitment_trace_sources, CommitmentTraceSources};

const GUEST_PACKAGE: &str = "fibonacci-guest";
const FUNC_NAME: &str = "fib";
const HEAP_SIZE: u64 = 32768; // matches #[jolt::provable(heap_size = 32768)]
const MAX_TRACE_LENGTH: usize = 65536; // 2^16, matches #[jolt::provable(max_trace_length = 65536)]
const NUM_VARS_PADDED: usize = 64;

/// Bytes per committed element: base Goldilocks commits over `Field64` (8 B).
const BASE_ELEM_BYTES: usize = 8;
/// Bytes per committed element in the actual Jolt protocol: BN254 `Fr` is a
/// 254-bit scalar serialized as 32 B. This is the per-element representation
/// the Goldilocks base field (8 B) replaces.
const BN254_ELEM_BYTES: usize = 32;

pub(crate) struct FibWorkload {
    pub log_t: usize,
    pub trace_len: usize,
    pub actual_cycles: usize,
    pub bytecode_k: usize,
    pub ram_k: usize,
    pub one_hot_params: OneHotParams,
    pub sources: CommitmentTraceSources,
}

/// Compile the fibonacci guest, trace `fib(n)`, and extract the dense
/// commitment-source index vectors via the canonical jolt-prover path.
pub(crate) fn build_fib_workload(n: u32) -> FibWorkload {
    let t_start = Instant::now();
    let mut program = Program::new(GUEST_PACKAGE);
    let _ = program.set_func(FUNC_NAME).set_heap_size(HEAP_SIZE);
    println!(
        "[fib-workload] compiling guest `{GUEST_PACKAGE}::{FUNC_NAME}` (heap={HEAP_SIZE}, max_trace={MAX_TRACE_LENGTH})"
    );

    let (instructions, memory_init, _program_size, entry_address) = program.decode();
    let bytecode = BytecodePreprocessing::preprocess(instructions, entry_address);
    let bytecode_k = bytecode.code_size;
    let ram_preprocessing = jolt_core::zkvm::ram::RAMPreprocessing::preprocess(memory_init);
    println!(
        "[fib-workload] bytecode preprocessing: code_size={bytecode_k}, entry={entry_address:#x}, elapsed={:.2}s",
        t_start.elapsed().as_secs_f64()
    );

    let inputs = postcard::to_stdvec(&n).expect("postcard n");
    let trace_start = Instant::now();
    let (_lazy, trace, _memory, jolt_device) = program.trace(&inputs, &[], &[]);
    let actual_cycles = trace.len();
    println!(
        "[fib-workload] guest trace fib({n}): cycles={actual_cycles}, elapsed={:.2}s",
        trace_start.elapsed().as_secs_f64()
    );

    let memory_layout = jolt_device.memory_layout.clone();

    // ram_K exactly as RV64IMACProver computes it (see workload.rs / prover.rs).
    let bytecode_end =
        jolt_core::zkvm::ram::remap_address(ram_preprocessing.min_bytecode_address, &memory_layout)
            .unwrap_or(0)
            + ram_preprocessing.bytecode_words.len() as u64
            + 1;
    let ram_k = trace
        .iter()
        .filter_map(|cycle| {
            jolt_core::zkvm::ram::remap_address(cycle.ram_access().address() as u64, &memory_layout)
        })
        .max()
        .unwrap_or(0)
        .max(bytecode_end)
        .next_power_of_two()
        .max(2) as usize;

    let trace_len = MAX_TRACE_LENGTH;
    let log_t = trace_len.trailing_zeros() as usize;
    assert!(trace.len() <= trace_len, "trace exceeds max length");

    let (cycle_inputs, _r1cs, _flags) = extract_trace::<_, Fr>(
        &trace,
        trace_len,
        &bytecode,
        &memory_layout,
        NUM_VARS_PADDED,
    );
    let sources = commitment_trace_sources(&cycle_inputs);

    let one_hot_params = OneHotParams::new(log_t, bytecode_k, ram_k);
    println!(
        "[fib-workload] log_T={log_t}, log_k_chunk={}, bytecode_k={bytecode_k}, ram_k={ram_k}, instruction_d={}, bytecode_d={}, ram_d={}",
        one_hot_params.log_k_chunk,
        one_hot_params.instruction_d,
        one_hot_params.bytecode_d,
        one_hot_params.ram_d,
    );

    FibWorkload {
        log_t,
        trace_len,
        actual_cycles,
        bytecode_k,
        ram_k,
        one_hot_params,
        sources,
    }
}

/// Map jolt-core's `OneHotParams` geometry onto the Goldilocks limb-column layout.
/// `chunk_bits = log_k_chunk` (4 or 8); instruction/bytecode pad with `Some(0)`,
/// RAM uses `None` (the committed-RA padding policy in jolt-core).
pub(crate) fn layout_from_params(p: &OneHotParams, trace_len: usize) -> GoldilocksLayout {
    let chunk_bits = p.log_k_chunk;
    GoldilocksLayout {
        trace_len,
        instruction: FamilyLayout {
            label: "InstructionRa",
            num_chunks: p.instruction_d,
            chunk_bits,
            padding: Some(0),
        },
        bytecode: FamilyLayout {
            label: "BytecodeRa",
            num_chunks: p.bytecode_d,
            chunk_bits,
            padding: Some(0),
        },
        ram: FamilyLayout {
            label: "RamRa",
            num_chunks: p.ram_d,
            chunk_bits,
            padding: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dory_bench::bench_dory;
    use crate::jolt_polys::build_polynomial_set;
    use jolt_field::goldilocks::decompose::sign_limbs_to_i128;
    use jolt_field::goldilocks::Goldilocks;
    use jolt_field::Field;
    use jolt_whir::{commit_witness, sanity_roundtrip};
    use jolt_witness::goldilocks::GoldilocksWitnessColumns;

    #[test]
    #[ignore = "compiles a RISC-V guest; run with --features goldilocks -- --ignored --no-capture"]
    fn fibonacci_goldilocks_e2e() {
        let wl = build_fib_workload(1000);
        let layout = layout_from_params(&wl.one_hot_params, wl.trace_len);

        let cols = GoldilocksWitnessColumns::build(&wl.sources, &layout);

        // Column count = ra_dense chunks (one per family-chunk) + 2 Inc × 3 limbs.
        let expected_ra = wl.one_hot_params.instruction_d
            + wl.one_hot_params.bytecode_d
            + wl.one_hot_params.ram_d;
        assert_eq!(
            cols.columns.len(),
            expected_ra + 6,
            "column count mismatch (ra={expected_ra} + 6 Inc limbs)"
        );
        assert_eq!(cols.log_t, wl.log_t);
        for c in &cols.columns {
            assert_eq!(c.values.len(), 1 << wl.log_t, "column {} length", c.label);
        }

        // Inc limbs must recompose to the original signed increments.
        let col = |name: &str| -> &[Goldilocks] {
            &cols
                .columns
                .iter()
                .find(|c| c.label == name)
                .unwrap_or_else(|| panic!("missing column {name}"))
                .values
        };
        for (label, inc) in [
            ("RdInc", &wl.sources.rd_inc),
            ("RamInc", &wl.sources.ram_inc),
        ] {
            let sign = col(&format!("{label}.sign"));
            let lo = col(&format!("{label}.lo"));
            let hi = col(&format!("{label}.hi"));
            for (i, &orig) in inc.iter().enumerate() {
                let recomposed = sign_limbs_to_i128(sign[i], [lo[i], hi[i]]);
                assert_eq!(
                    recomposed, orig,
                    "{label} limb recompose mismatch at cycle {i}"
                );
            }
        }

        // Commit every base-Goldilocks column via WHIR.
        let report = commit_witness(&cols);
        assert_eq!(report.log_t, wl.log_t);
        assert_eq!(report.num_columns, cols.columns.len());
        assert_eq!(report.total_base_elements, cols.total_elements());
        assert_eq!(
            report.committed_base_bytes,
            report.total_base_elements * BASE_ELEM_BYTES
        );

        // Single-point open/verify round-trip on a non-degenerate column. Real
        // witnesses contain all-zero columns (e.g. high instruction chunks fib
        // never reaches) — WHIR's open path divides by the polynomial evaluation,
        // which is 0 for the zero polynomial, so we must open a non-zero column.
        let zero = Goldilocks::from_u64(0);
        let nonzero = cols
            .columns
            .iter()
            .find(|c| c.values.iter().any(|&v| v != zero))
            .expect("at least one committed column must be non-zero");
        println!("[fib-e2e] sanity open/verify on column `{}`", nonzero.label);
        assert!(
            sanity_roundtrip(&nonzero.values),
            "WHIR open/verify round-trip failed for column `{}`",
            nonzero.label
        );

        // ---- Actual Jolt protocol commit: BN254 + Dory on the SAME trace ----
        // Build the exact polynomial set Jolt's prover commits (sparse one-hot RA
        // families + dense Inc polys), then commit it via Dory over BN254 — the
        // production path mirrored from `jolt-prover` stages/commitment.rs.
        let bn_polys = build_polynomial_set(&wl.sources, &wl.one_hot_params, wl.trace_len);
        let bn_chunks: usize = bn_polys
            .one_hot_families
            .iter()
            .map(|f| f.chunks.len())
            .sum();
        let bn_nonzeros: usize = bn_polys
            .one_hot_families
            .iter()
            .flat_map(|f| f.chunks.iter())
            .map(|c| c.indices.iter().filter(|x| x.is_some()).count())
            .sum();
        let bn_dense_elems: usize = bn_polys.dense.iter().map(|d| d.values.len()).sum();
        let bn_logical_elems = bn_polys.total_field_elements();
        let bn_logical_bytes = bn_logical_elems * BN254_ELEM_BYTES;

        // The Dory one-hot decomposition must match our Goldilocks ra_dense chunks.
        assert_eq!(
            bn_chunks, expected_ra,
            "Dory one-hot chunk count {bn_chunks} != Goldilocks ra_dense chunk count {expected_ra}"
        );

        // One measured run (warmup = 0): this is an e2e validation/report, not a
        // rigorous multi-run benchmark. Includes Dory's trusted-setup SRS
        // generation, which the transparent WHIR commit does not need.
        let dory = bench_dory(&bn_polys, 0, 1);
        let dory_commit_ms = dory.runs[0].total_ms;

        let gl_bytes = report.committed_base_bytes;
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);

        println!(
            "\n============ Phase-1 fibonacci e2e: Goldilocks/WHIR vs Jolt BN254/Dory ============"
        );
        println!(
            "  guest                 : {GUEST_PACKAGE}::{FUNC_NAME}(1000)  ({} cycles)",
            wl.actual_cycles
        );
        println!(
            "  committed length      : 2^{} = {}   (log_k_chunk={}, instruction_d={}, bytecode_d={}, ram_d={})",
            wl.log_t,
            1usize << wl.log_t,
            wl.one_hot_params.log_k_chunk,
            wl.one_hot_params.instruction_d,
            wl.one_hot_params.bytecode_d,
            wl.one_hot_params.ram_d,
        );
        println!("  bytecode_k / ram_k    : {} / {}", wl.bytecode_k, wl.ram_k);
        println!(
            "  ------------------------------------------------------------------------------"
        );
        println!(
            "  [Goldilocks base → WHIR]   (this implementation; transparent, no trusted setup)"
        );
        println!("    representation      : dense base-field columns (RA index/cycle + Inc sign/lo/hi limbs)");
        println!("    committed columns   : {}", report.num_columns);
        println!(
            "    committed elements  : {}  (dense, {} B each)",
            report.total_base_elements, BASE_ELEM_BYTES
        );
        println!(
            "    committed volume    : {gl_bytes} B ({:.2} MiB)",
            mib(gl_bytes)
        );
        println!("    commit time         : {:.2} ms", report.commit_ms);
        println!(
            "  ------------------------------------------------------------------------------"
        );
        println!("  [BN254 → Dory]             (actual Jolt protocol)");
        println!("    representation      : sparse one-hot RA polys + dense Inc polys");
        println!(
            "    one-hot chunks      : {} (layout 2^{} each; sparse: {} nonzeros total)",
            bn_chunks, dory.setup_num_vars, bn_nonzeros
        );
        println!("    dense Inc elements  : {bn_dense_elems}");
        println!(
            "    logical field elems : {}  ({} B each; one-hot is committed sparsely)",
            bn_logical_elems, BN254_ELEM_BYTES
        );
        println!(
            "    logical volume      : {} B ({:.2} MiB) dense-equivalent",
            bn_logical_bytes,
            mib(bn_logical_bytes)
        );
        println!(
            "    SRS setup (one-time): {:.2} ms  (num_vars={})",
            dory.setup_ms, dory.setup_num_vars
        );
        println!("    commit time         : {dory_commit_ms:.2} ms");
        println!(
            "  ------------------------------------------------------------------------------"
        );
        println!("  [Comparison — same fibonacci trace]");
        println!(
            "    field-element width : {} B (BN254 Fr) → {} B (Goldilocks) = {:.0}× narrower",
            BN254_ELEM_BYTES,
            BASE_ELEM_BYTES,
            BN254_ELEM_BYTES as f64 / BASE_ELEM_BYTES as f64
        );
        println!(
            "    commit wall-clock   : Dory {:.2} ms vs WHIR {:.2} ms = {:.2}× faster with WHIR",
            dory_commit_ms,
            report.commit_ms,
            dory_commit_ms / report.commit_ms
        );
        println!(
            "    trusted setup       : Dory needs a 2^{} SRS ({:.2} ms); WHIR is transparent (none)",
            dory.setup_num_vars, dory.setup_ms
        );
        println!(
            "==================================================================================\n"
        );
    }
}
