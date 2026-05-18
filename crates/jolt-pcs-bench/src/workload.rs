//! ECDSA guest compile + trace → CommitmentTraceSources.
//!
//! Drives the p256-ecdsa-verify guest binary through the public
//! `jolt-trace` pipeline (no prover instrumentation). Output is the dense
//! per-cycle index vectors that the rest of the bench consumes.

use std::time::Instant;

use jolt_core::zkvm::config::OneHotParams;
use jolt_field::Fr;
use jolt_trace::bytecode::BytecodePreprocessing;
use jolt_trace::{extract_trace, Program};
use jolt_witness::{commitment_trace_sources, CommitmentTraceSources};

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
    pub sources: CommitmentTraceSources,
}

/// Drives the ECDSA guest pipeline end-to-end and returns the dense indices
/// the commitment phase would otherwise consume.
///
/// The returned `sources` and `one_hot_params` exactly mirror what
/// `jolt-prover`'s `SparseCommitmentInputs` uses internally.
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

    // extract_trace returns (cycle_inputs, r1cs_witness, instruction_flags) padded to `trace_len`
    let (cycle_inputs, _r1cs, _flags) =
        extract_trace::<_, Fr>(&trace, trace_len, &bytecode, &memory_layout, NUM_VARS_PADDED);

    let sources = commitment_trace_sources(&cycle_inputs);
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
    }
}
