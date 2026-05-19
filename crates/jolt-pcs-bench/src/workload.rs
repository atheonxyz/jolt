//! ECDSA guest compile + trace → CommitmentTraceSources.
//!
//! Drives the p256-ecdsa-verify guest binary through the public
//! `jolt-trace` pipeline (no prover instrumentation). Output is the dense
//! per-cycle index vectors that the rest of the bench consumes.

use std::time::Instant;

use common::jolt_device::MemoryLayout;
use jolt_core::zkvm::config::OneHotParams;
use jolt_field::Fr;
use jolt_trace::bytecode::BytecodePreprocessing;
use jolt_trace::{extract_trace, CycleRow, Program};
use jolt_witness::{commitment_trace_sources, CommitmentTraceSources, CycleInput};

const GUEST_PACKAGE: &str = "p256-ecdsa-verify-guest";
const FUNC_NAME: &str = "p256_ecdsa_verify";
const MAX_TRACE_LENGTH: usize = 524288; // 2^19, matches #[jolt::provable(max_trace_length=524288)]
const HEAP_SIZE: u64 = 100_000;
const NUM_VARS_PADDED: usize = 64;

/// Test vectors copied from `examples/p256-ecdsa-verify/src/main.rs`.
/// Derived from RFC 6979 private key (d = 0xC9AFA9D8...).
fn ecdsa_test_input() -> Vec<u8> {
    // message hash z = SHA-256("sample")
    let z: [u64; 4] = [
        0x219f7c40307c8edf,
        0x83f30a857ad8f656,
        0x06d6364bd78467c1,
        0x4847be4ac21fe68a,
    ];
    let r: [u64; 4] = [
        0x61ba8a2e970ae87c,
        0xf81746f8e6b05ab8,
        0x15ab9e9a0f4fc6c8,
        0x42ed5ba7de86be7d,
    ];
    let s: [u64; 4] = [
        0xde14a271eb1fb4d6,
        0xbb8079f1b5d7dfc7,
        0x86880d7edb977acd,
        0x81f8aa8845318fbf,
    ];
    // public key Q (uncompressed: Qx || Qy)
    let q: [u64; 8] = [
        0xe669622e60f29fb6,
        0xc049b8923b61fa6c,
        0xc961eb74c6356d68,
        0x60fed4ba255a9d31,
        0x77a3c294d4462299,
        0xf2f1b20c2d7e9f51,
        0xa41ae9e95628bc64,
        0x7903fe1008b8bc99,
    ];

    let mut input_bytes = Vec::new();
    input_bytes.extend(postcard::to_stdvec(&z).expect("postcard z"));
    input_bytes.extend(postcard::to_stdvec(&r).expect("postcard r"));
    input_bytes.extend(postcard::to_stdvec(&s).expect("postcard s"));
    input_bytes.extend(postcard::to_stdvec(&q).expect("postcard q"));
    input_bytes
}

pub struct EcdsaWorkload {
    pub log_t: usize,
    pub trace_len: usize,
    pub bytecode_k: usize,
    pub ram_k: usize,
    pub one_hot_params: OneHotParams,
    /// Built via the canonical `extract_trace` + `commitment_trace_sources`
    /// path. This is what downstream phases (Dory commit, dump, etc.) consume.
    pub sources: CommitmentTraceSources,
    /// Same five columns, built by walking `Vec<Cycle>` once via `CycleRow`
    /// accessors. Skips `extract_trace`'s R1CS witness vector
    /// (`trace_len * NUM_VARS_PADDED` field zeros) and `InstructionFlagData`
    /// construction. Asserted equal to `sources` at build time.
    #[expect(dead_code, reason = "captured for inspection; bench consumes `sources`")]
    pub direct_sources: CommitmentTraceSources,
}

/// Drives the ECDSA guest pipeline end-to-end and returns the dense indices
/// the commitment phase would otherwise consume.
///
/// The returned `sources` and `one_hot_params` exactly mirror what
/// `jolt-prover`'s `SparseCommitmentInputs` uses internally.
#[tracing::instrument(skip_all, name = "bench.build_ecdsa_workload")]
pub fn build_ecdsa_workload() -> EcdsaWorkload {
    let t_start = Instant::now();
    let mut program = Program::new(GUEST_PACKAGE);
    let _ = program.set_func(FUNC_NAME).set_heap_size(HEAP_SIZE);
    println!(
        "[workload] compiling guest `{GUEST_PACKAGE}::{FUNC_NAME}` (heap={HEAP_SIZE}, max_trace={MAX_TRACE_LENGTH})"
    );

    // decode() compiles, then decodes the ELF. Returns (instructions, mem_init, program_size, entry_address)
    let (instructions, memory_init, _program_size, entry_address) = program.decode();
    let bytecode = BytecodePreprocessing::preprocess(instructions, entry_address);
    let bytecode_k = bytecode.code_size;
    // `memory_init` is consumed below by `RAMPreprocessing::preprocess` to compute the
    // bytecode-end clamp jolt-core applies to `ram_K`. Keep both `memory_init` (for the
    // bytecode_end calc) and `bytecode` (for trace extraction).
    let ram_preprocessing = jolt_core::zkvm::ram::RAMPreprocessing::preprocess(memory_init);
    println!(
        "[workload] bytecode preprocessing: code_size={bytecode_k}, entry={entry_address:#x}, elapsed={:.2}s",
        t_start.elapsed().as_secs_f64()
    );

    let inputs = ecdsa_test_input();
    let trace_start = Instant::now();
    let (_lazy, trace, _memory, jolt_device) = program.trace(&inputs, &[], &[]);
    println!(
        "[workload] guest trace: cycles={}, elapsed={:.2}s",
        trace.len(),
        trace_start.elapsed().as_secs_f64()
    );

    let memory_layout = jolt_device.memory_layout.clone();

    // Compute ram_K the same way RV64IMACProver does (jolt-core/src/zkvm/prover.rs:414).
    // Note: ram_K is the larger of (a) the largest RAM address actually accessed during
    // execution and (b) the end of the bytecode region (min_bytecode + bytecode_words + 1).
    // For ECDSA both terms produce a value < 2^14, so the result coincides regardless of
    // which term wins — but the clamp matters for sparse-RAM workloads where execution
    // never reaches the bytecode end.
    let bytecode_end = jolt_core::zkvm::ram::remap_address(
        ram_preprocessing.min_bytecode_address,
        &memory_layout,
    )
    .unwrap_or(0)
        + ram_preprocessing.bytecode_words.len() as u64
        + 1;
    let ram_k = trace
        .iter()
        .filter_map(|cycle| {
            jolt_core::zkvm::ram::remap_address(
                cycle.ram_access().address() as u64,
                &memory_layout,
            )
        })
        .max()
        .unwrap_or(0)
        .max(bytecode_end)
        .next_power_of_two()
        .max(2) as usize;

    // Pad trace to a power-of-2 length the prover would use
    let trace_len = MAX_TRACE_LENGTH;
    let log_t = trace_len.trailing_zeros() as usize;
    assert!(trace.len() <= trace_len, "trace exceeds max length");

    // Canonical path: extract_trace returns (cycle_inputs, r1cs_witness,
    // instruction_flags) padded to `trace_len`. The bench discards _r1cs / _flags.
    let extract_start = Instant::now();
    let (cycle_inputs, _r1cs, _flags) =
        extract_trace::<_, Fr>(&trace, trace_len, &bytecode, &memory_layout, NUM_VARS_PADDED);
    let sources = commitment_trace_sources(&cycle_inputs);
    let extract_elapsed = extract_start.elapsed();

    // Direct path: mirrors `extract_trace` but produces only the commitment
    // inputs — skips the `trace_len * NUM_VARS_PADDED` Fr-zero r1cs allocation
    // and `InstructionFlagData` construction. Per-cycle conversion duplicates
    // the private `jolt_trace::extract::cycle_input`; `assert_eq!` below
    // guarantees the two paths stay byte-identical.
    let direct_start = Instant::now();
    let direct_inputs =
        extract_commitment_inputs(&trace, trace_len, &bytecode, &memory_layout);
    let direct_sources = commitment_trace_sources(&direct_inputs);
    let direct_elapsed = direct_start.elapsed();

    assert_eq!(
        sources, direct_sources,
        "direct trace walk diverges from extract_trace + commitment_trace_sources"
    );

    let extract_ms = extract_elapsed.as_secs_f64() * 1e3;
    let direct_ms = direct_elapsed.as_secs_f64() * 1e3;
    println!(
        "[workload] commitment-source extraction: extract_trace+commitment_trace_sources={extract_ms:.2}ms, direct-from-Cycle={direct_ms:.2}ms, speedup={:.2}x",
        extract_ms / direct_ms.max(f64::MIN_POSITIVE),
    );

    let one_hot_params = OneHotParams::new(log_t, bytecode_k, ram_k);

    println!(
        "[workload] log_T={log_t}, log_k_chunk={}, bytecode_k={bytecode_k} (log={}), ram_k={ram_k} (log={}), instruction_d={}, bytecode_d={}, ram_d={}",
        one_hot_params.log_k_chunk,
        bytecode_k.trailing_zeros(),
        ram_k.trailing_zeros(),
        one_hot_params.instruction_d,
        one_hot_params.bytecode_d,
        one_hot_params.ram_d,
    );
    println!(
        "[workload] total setup time: {:.2}s",
        t_start.elapsed().as_secs_f64()
    );

    let _ = cycle_inputs; // consumed via `sources`; kept here just to anchor extract_trace ordering
    EcdsaWorkload {
        log_t,
        trace_len,
        bytecode_k,
        ram_k,
        one_hot_params,
        sources,
        direct_sources,
    }
}

/// Duplicate of the private `jolt_trace::extract::cycle_input`
/// ([extract.rs:68-98](../../../crates/jolt-trace/src/extract.rs#L68-L98)).
///
/// Kept in lockstep with upstream via the `assert_eq!` in
/// `build_ecdsa_workload` — if upstream's per-cycle derivation grows a new
/// term, the assertion fires and we update this body.
fn cycle_input(
    cycle: &impl CycleRow,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
) -> CycleInput {
    let rd_inc = match cycle.rd_write() {
        Some((_, pre, post)) => post as i128 - pre as i128,
        None => 0,
    };
    let ram_inc = match (cycle.ram_read_value(), cycle.ram_write_value()) {
        (Some(pre), Some(post)) => post as i128 - pre as i128,
        _ => 0,
    };
    let lowest = memory_layout.get_lowest_address();
    let ram_address = cycle.ram_access_address().map(|addr| {
        debug_assert!(
            addr >= lowest,
            "RAM address {addr:#x} below lowest {lowest:#x}"
        );
        ((addr - lowest) / 8) as u128
    });

    CycleInput {
        dense: [rd_inc, ram_inc],
        one_hot: [
            Some(cycle.lookup_index()),
            Some(bytecode.get_cycle_pc(cycle) as u128),
            ram_address,
        ],
    }
}

/// Bench-side counterpart to [`extract_trace`](jolt_trace::extract_trace) that
/// produces *only* the commitment inputs.
///
/// Skips `extract_trace`'s R1CS witness vector
/// (`trace_len * NUM_VARS_PADDED` Fr-zeros) and `InstructionFlagData`
/// construction — both of which the bench discards. Padding rule
/// (noop / out-of-range → [`CycleInput::PADDING`]) matches upstream exactly,
/// so the returned vector is byte-identical to
/// `extract_trace(..).0` for the same inputs.
fn extract_commitment_inputs<C: CycleRow>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
) -> Vec<CycleInput> {
    let mut inputs = Vec::with_capacity(size);
    for t in 0..size {
        if let Some(cycle) = trace.get(t).filter(|cycle| !cycle.is_noop()) {
            inputs.push(cycle_input(cycle, bytecode, memory_layout));
        } else {
            inputs.push(CycleInput::PADDING);
        }
    }
    inputs
}
