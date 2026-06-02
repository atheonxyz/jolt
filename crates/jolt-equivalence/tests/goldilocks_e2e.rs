//! Full Goldilocks+WHIR prover e2e on a REAL muldiv trace (P10), gated against jolt-core.
//!
//! This compiles a RISC-V guest (the muldiv fixture) via the `jolt` CLI, so it is gated behind the
//! `goldilocks` feature and run explicitly (build ONLY this target — the full `--features goldilocks`
//! test set pulls the WHIR graph and fills the disk):
//!   `cargo test -p jolt-equivalence --features goldilocks --test goldilocks_e2e`
//!
//! Milestones land incrementally:
//!   M0 — assemble all binary-driver witnesses from the real trace; assert the limbed R1CS is
//!        satisfied (flushes out `cycle_to_z` op-coverage on real MUL/virtual-sequence cycles).
//!   M1 — the binary driver (Spartan -> memory -> booleanity) round-trips on the real trace.
//!   M2 — the full prove_e2e/verify_e2e adds the stage-8 WHIR open of the R1csAux columns.
#![cfg(feature = "goldilocks")]

use common::constants::REGISTER_COUNT;
use jolt_core::host;
use jolt_core::zkvm::ram::remap_address;
use jolt_prover_goldilocks::field::{ProverTranscript, VerifierTranscript};
use jolt_prover_goldilocks::zkvm::driver::{prove_binary, verify_binary};
use jolt_prover_goldilocks::zkvm::e2e::{prove_e2e, verify_e2e, VerifierParams};
use jolt_prover_goldilocks::zkvm::real_trace::{assemble_real_witness, RealWitness};
use jolt_prover_goldilocks::F;
use jolt_trace::{BytecodePreprocessing, CycleRow};

/// Trace the muldiv guest with the same inputs jolt-core's fixture uses and assemble the binary
/// driver's witnesses over Goldilocks. Returns the witnesses + the (unpadded) real trace length.
fn build_muldiv_real_witness() -> (RealWitness<F>, usize) {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("postcard inputs");

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode_instrs, _init_memory, _program_size, entry_address) = program.decode();
    let (_lazy, trace, _memory, io_device) = program.trace(&inputs, &[], &[]);
    let memory_layout = io_device.memory_layout.clone();
    let bytecode = BytecodePreprocessing::preprocess(bytecode_instrs, entry_address);

    let ram_lowest = memory_layout.get_lowest_address();
    // Goldilocks RAM witness is zero-initialised, so the address space need only cover the indices
    // the trace actually touches (remapped). `+1`, power-of-two, ≥ 2.
    let max_ram_index = trace
        .iter()
        .filter_map(CycleRow::ram_access_address)
        .filter_map(|addr| remap_address(addr, &memory_layout))
        .max()
        .unwrap_or(0);
    let ram_k = (max_ram_index + 1).next_power_of_two().max(2) as usize;

    let witness = assemble_real_witness::<F>(
        &trace,
        &bytecode,
        ram_lowest,
        ram_k,
        REGISTER_COUNT as usize,
    );
    (witness, trace.len())
}

#[test]
fn goldilocks_real_trace_r1cs_is_satisfied() {
    let (w, trace_len) = build_muldiv_real_witness();

    assert!(
        w.r1cs.is_satisfied(),
        "limbed RV64 R1CS must be satisfied by the real muldiv witness"
    );

    assert_eq!(
        w.ram.log_t, w.registers.log_t,
        "RAM and register stages must agree on log_t"
    );
    assert_eq!(
        w.r1cs.log_num_cycles, w.ram.log_t,
        "R1CS and memory stages must agree on the cycle count"
    );

    eprintln!(
        "[goldilocks-e2e/M0] muldiv real-trace witness OK: trace_len={trace_len}, \
         log_num_cycles={}, num_vars_padded={}, num_cons_padded={}, ram_log_k={}, reg_log_k={}",
        w.r1cs.log_num_cycles,
        w.r1cs.num_vars_padded,
        w.r1cs.num_cons_padded,
        w.ram.log_k,
        w.registers.log_k,
    );
}

#[test]
fn goldilocks_real_trace_binary_driver_round_trip() {
    let (w, _trace_len) = build_muldiv_real_witness();

    let mut prover_t = ProverTranscript::new("muldiv-binary-e2e");
    let proof = prove_binary(
        &w.r1cs,
        &w.ram,
        &w.registers,
        &w.ram_public,
        &w.key,
        &mut prover_t,
    );
    let narg = prover_t.into_proof();

    let mut verifier_t = VerifierTranscript::new("muldiv-binary-e2e", &narg);
    verify_binary(
        &proof,
        &w.key,
        w.r1cs.num_row_vars(),
        w.r1cs.log_num_cycles,
        w.ram.log_k,
        w.registers.log_k,
        &w.ram_public,
        &mut verifier_t,
    )
    .expect("binary driver (Spartan -> memory -> booleanity) must verify on the real muldiv trace");
}

#[test]
fn goldilocks_real_trace_e2e_with_stage8_r1cs_aux() {
    let (w, _trace_len) = build_muldiv_real_witness();

    let mut prover_t = ProverTranscript::new("muldiv-e2e");
    let proof = prove_e2e(&w, &mut prover_t)
        .expect("prove_e2e (binary driver + stage-8 WHIR open of R1csAux) must succeed");
    let narg = prover_t.into_proof();

    let params = VerifierParams::from_witness(&w);
    let mut verifier_t = VerifierTranscript::new("muldiv-e2e", &narg);
    verify_e2e(&proof, &params, &mut verifier_t)
        .expect("verify_e2e must accept the real muldiv proof (incl. the stage-8 R1csAux open)");
}
