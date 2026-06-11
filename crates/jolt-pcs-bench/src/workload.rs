//! ECDSA guest compile + trace → native commitment sources.
//!
//! Drives the p256-ecdsa-verify guest through jolt-main's native host pipeline
//! (`jolt_core::host::Program`) and derives the dense per-cycle index vectors
//! the rest of the bench consumes (`crate::sources`).

use std::time::Instant;

use jolt_core::host::Program;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::config::OneHotParams;
use jolt_core::zkvm::ram::{remap_address, RAMPreprocessing};
use jolt_riscv::RV64IMAC_JOLT;

use crate::sources::{build_sources, CommitmentSources};

const GUEST_PACKAGE: &str = "p256-ecdsa-verify-guest";
const FUNC_NAME: &str = "p256_ecdsa_verify";
const MAX_TRACE_LENGTH: usize = 524288; // 2^19, matches #[jolt::provable(max_trace_length=524288)]
const HEAP_SIZE: u64 = 100_000;

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

pub(crate) struct EcdsaWorkload {
    pub log_t: usize,
    pub trace_len: usize,
    pub bytecode_k: usize,
    pub ram_k: usize,
    pub one_hot_params: OneHotParams,
    /// Dense per-cycle index/transition columns derived natively from the
    /// trace (the same values `CommittedPolynomial::generate_witness` commits).
    pub sources: CommitmentSources,
}

/// Drives the ECDSA guest pipeline end-to-end and returns the dense indices
/// the commitment phase would otherwise consume.
#[tracing::instrument(skip_all, name = "bench.build_ecdsa_workload")]
pub(crate) fn build_ecdsa_workload() -> EcdsaWorkload {
    let t_start = Instant::now();
    let mut program = Program::new(GUEST_PACKAGE);
    // jolt-main's setters return (); separate statements (no builder chaining).
    program.set_func(FUNC_NAME);
    program.set_heap_size(HEAP_SIZE);
    println!(
        "[workload] compiling guest `{GUEST_PACKAGE}::{FUNC_NAME}` (heap={HEAP_SIZE}, max_trace={MAX_TRACE_LENGTH})"
    );

    // decode() compiles, then decodes the ELF.
    // Returns (instructions, mem_init, program_size, entry_address).
    let (instructions, memory_init, _program_size, entry_address) = program.decode();
    let bytecode = BytecodePreprocessing::preprocess(instructions, entry_address, RV64IMAC_JOLT)
        .expect("bytecode preprocessing");
    let bytecode_k = bytecode.code_size;
    // `memory_init` is consumed by `RAMPreprocessing::preprocess` to compute the
    // bytecode-end clamp jolt-core applies to `ram_K`.
    let ram_preprocessing = RAMPreprocessing::preprocess(memory_init);
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

    // Compute ram_K the same way the prover does (jolt-core/src/zkvm/prover.rs):
    // the larger of (a) the largest RAM address actually accessed and (b) the
    // bytecode-image end (min_bytecode + bytecode_words + 1), next-power-of-two.
    // For ECDSA both terms are < 2^14; the clamp matters for sparse-RAM workloads.
    let bytecode_end = remap_address(ram_preprocessing.min_bytecode_address, &memory_layout)
        .unwrap_or(0)
        + ram_preprocessing.bytecode_words.len() as u64
        + 1;
    let ram_k = trace
        .iter()
        .filter_map(|cycle| remap_address(cycle.ram_access().address() as u64, &memory_layout))
        .max()
        .unwrap_or(0)
        .max(bytecode_end)
        .next_power_of_two()
        .max(2) as usize;

    // Bench commits at the padded power-of-2 trace length the prover would use.
    let trace_len = MAX_TRACE_LENGTH;
    let log_t = trace_len.trailing_zeros() as usize;
    assert!(trace.len() <= trace_len, "trace exceeds max length");

    // Native commitment-source extraction (replaces the Bolt extract_trace +
    // commitment_trace_sources). Produces unpadded per-cycle columns; downstream
    // chunking pads to `trace_len`.
    let extract_start = Instant::now();
    let sources = build_sources(&trace, &bytecode, &memory_layout);
    println!(
        "[workload] commitment-source extraction: {:.2}ms",
        extract_start.elapsed().as_secs_f64() * 1e3,
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

    EcdsaWorkload {
        log_t,
        trace_len,
        bytecode_k,
        ram_k,
        one_hot_params,
        sources,
    }
}
