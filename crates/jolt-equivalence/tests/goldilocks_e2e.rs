//! Full Goldilocks+WHIR prover e2e on a REAL muldiv trace (P10), gated against jolt-core.
//!
//! This compiles a RISC-V guest (the muldiv fixture) via the `jolt` CLI, so it is gated behind the
//! `goldilocks` feature and run explicitly (build ONLY this target — the full `--features goldilocks`
//! test set pulls the WHIR graph and fills the disk):
//!   `cargo test -p jolt-equivalence --features goldilocks --test goldilocks_e2e`
//!
//! Milestones land incrementally:
//!   M0  — assemble all binary-driver witnesses from the real trace; assert the limbed R1CS is
//!         satisfied (flushes out `cycle_to_z` op-coverage on real MUL/virtual-sequence cycles).
//!   M1  — the binary driver (Spartan -> memory -> booleanity) round-trips on the real trace.
//!   M2  — prove_e2e/verify_e2e adds the stage-8 WHIR open of the R1csAux + Inc columns.
//!   M3b — prove_e2e/verify_e2e adds the bytecode read-raf stage.
#![cfg(feature = "goldilocks")]

use common::constants::REGISTER_COUNT;
use jolt_core::host;
use jolt_core::zkvm::ram::remap_address;
use jolt_equivalence::core_oracle::core_muldiv_commitment_fixture;
use jolt_prover_goldilocks::field::{ProverTranscript, VerifierTranscript};
use jolt_prover_goldilocks::zkvm::driver::{prove_binary, verify_binary};
use jolt_prover_goldilocks::zkvm::e2e::{
    prove_e2e, verify_e2e, BytecodeProverInputs, BytecodeVerifierInputs, VerifierParams,
};
use jolt_prover_goldilocks::zkvm::real_trace::{assemble_real_witness, RealWitness};
use jolt_prover_goldilocks::zkvm::witness::CommittedWitness;
use jolt_prover_goldilocks::F;
use jolt_trace::{extract_trace, BytecodePreprocessing, Cycle, CycleRow, Instruction};
use jolt_witness::commitment_trace_sources;
use jolt_witness::goldilocks::{FamilyLayout, GoldilocksLayout};

/// muldiv has bytecode_d = 4 chunks at log_k_chunk = 4 (confirmed by the witness gate); the read-raf
/// is const-generic over D, so the e2e test pins D = 4, NE = D + 2 = 6.
const BYTECODE_D: usize = 4;
const BYTECODE_NE: usize = 6;

/// Everything the e2e needs from one real muldiv trace: the binary-driver witnesses, the committed
/// witness (for the bytecode RA chunk-index columns), the padded bytecode table, and the geometry.
struct Fixture {
    real: RealWitness<F>,
    committed: CommittedWitness<F>,
    bytecode_rows: Vec<Instruction>,
    /// The raw padded-or-unpadded execution trace, retained for the instruction-lookup family
    /// (per-cycle lookup index / table / interleaved flag) and the P3b-0 dispatch parity gate.
    trace: Vec<Cycle>,
    log_k_chunk: usize,
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
    log_register: usize,
    trace_len: usize,
}

/// Trace the muldiv guest with the same inputs jolt-core's fixture uses and assemble the full e2e
/// witness set over Goldilocks.
fn build_muldiv_fixture() -> Fixture {
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

    let real = assemble_real_witness::<F>(
        &trace,
        &bytecode,
        ram_lowest,
        ram_k,
        REGISTER_COUNT as usize,
    );

    // Committed witness from the projected sources (same path as the witness gate). `size` is the
    // padded committed length so the committed columns align with the binary witnesses at 2^log_t.
    let log_t = real.r1cs.log_num_cycles;
    let padded_len = 1usize << log_t;
    let (cycle_inputs, _r1cs, _flags) = extract_trace::<_, F>(
        &trace,
        padded_len,
        &bytecode,
        &memory_layout,
        real.r1cs.num_vars_padded,
    );
    let sources = commitment_trace_sources(&cycle_inputs);

    let log_k_chunk = if log_t < 25 { 4 } else { 8 };
    let log_k_bytecode = bytecode.code_size.trailing_zeros() as usize;
    let bytecode_d = log_k_bytecode.div_ceil(log_k_chunk);
    let instruction_d = 128 / log_k_chunk;
    let log_k_ram = ram_k.trailing_zeros() as usize;
    let ram_d = log_k_ram.div_ceil(log_k_chunk);
    let layout = GoldilocksLayout {
        trace_len: padded_len,
        instruction: FamilyLayout {
            label: "InstructionRa",
            num_chunks: instruction_d,
            chunk_bits: log_k_chunk,
            padding: Some(0),
        },
        bytecode: FamilyLayout {
            label: "BytecodeRa",
            num_chunks: bytecode_d,
            chunk_bits: log_k_chunk,
            padding: Some(0),
        },
        ram: FamilyLayout {
            label: "RamRa",
            num_chunks: ram_d,
            chunk_bits: log_k_chunk,
            padding: None,
        },
    };
    let committed = CommittedWitness::<F>::build(&sources, &layout);

    Fixture {
        real,
        committed,
        bytecode_rows: bytecode.bytecode.clone(),
        log_k_chunk,
        instruction_d,
        bytecode_d,
        ram_d,
        log_register: (REGISTER_COUNT as usize).trailing_zeros() as usize,
        trace_len: trace.len(),
        trace,
    }
}

/// The bytecode RA chunk-index columns (`ra_dense[bytecode_range]`) as the read-raf's `D` indices.
fn bytecode_indices(fx: &Fixture) -> [Vec<u32>; BYTECODE_D] {
    std::array::from_fn(|i| {
        fx.committed.ra_dense[fx.committed.bytecode_range.start + i]
            .indices
            .clone()
    })
}

#[test]
fn goldilocks_real_trace_r1cs_is_satisfied() {
    let fx = build_muldiv_fixture();
    let w = &fx.real;

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
        "[goldilocks-e2e/M0] muldiv real-trace witness OK: trace_len={}, log_num_cycles={}, \
         num_vars_padded={}, num_cons_padded={}, ram_log_k={}, reg_log_k={}, log_k_chunk={}, \
         bytecode_d={}",
        fx.trace_len,
        w.r1cs.log_num_cycles,
        w.r1cs.num_vars_padded,
        w.r1cs.num_cons_padded,
        w.ram.log_k,
        w.registers.log_k,
        fx.log_k_chunk,
        fx.bytecode_d,
    );
}

#[test]
fn goldilocks_real_trace_binary_driver_round_trip() {
    let fx = build_muldiv_fixture();
    let w = &fx.real;

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
fn goldilocks_real_trace_e2e_with_bytecode_read_raf() {
    let fx = build_muldiv_fixture();
    assert_eq!(
        fx.bytecode_d, BYTECODE_D,
        "muldiv bytecode_d must match the const D this test pins"
    );
    let log_k_chunks = [fx.log_k_chunk; BYTECODE_D];

    let mut prover_t = ProverTranscript::new("muldiv-e2e");
    let bc_prover = BytecodeProverInputs::<BYTECODE_D> {
        bytecode: &fx.bytecode_rows,
        indices: bytecode_indices(&fx),
        log_k_chunks,
        log_register: fx.log_register,
        base_index: fx.committed.bytecode_range.start,
    };
    let proof = prove_e2e::<BYTECODE_D, BYTECODE_NE>(&fx.real, &bc_prover, &mut prover_t)
        .expect("prove_e2e (binary + bytecode read-raf + stage-8 R1csAux/Inc opens) must succeed");
    let narg = prover_t.into_proof();

    let params = VerifierParams::from_witness(&fx.real);
    let bc_verifier = BytecodeVerifierInputs::<BYTECODE_D> {
        bytecode: &fx.bytecode_rows,
        log_k_chunks,
        log_register: fx.log_register,
        base_index: fx.committed.bytecode_range.start,
    };
    let mut verifier_t = VerifierTranscript::new("muldiv-e2e", &narg);
    verify_e2e::<BYTECODE_D, BYTECODE_NE>(&proof, &params, &bc_verifier, &mut verifier_t)
        .expect("verify_e2e must accept the real muldiv proof");
}

/// M4 — proof-level parity gate: the geometry the Goldilocks e2e proves matches jolt-core's
/// `JoltProtocolParams` for the same muldiv program. The ISA/bytecode-determined fields (`log_t`,
/// `log_k_chunk`, `instruction_d`, `bytecode_d`) must agree exactly; RAM diverges because the
/// Goldilocks memory stage uses a zero-init, max-accessed RAM model (so `ram_d ≤ jolt-core's`), an
/// interim gap that lands with real RAM initial-state loading. (Witness-integer parity is the
/// separate `goldilocks_witness_gate`.)
#[test]
fn goldilocks_e2e_geometry_matches_core_muldiv() {
    let core = core_muldiv_commitment_fixture();
    let p = &core.params;
    let fx = build_muldiv_fixture();

    assert_eq!(fx.real.r1cs.log_num_cycles, p.log_t, "log_t");
    assert_eq!(fx.log_k_chunk, p.log_k_chunk, "log_k_chunk");
    assert_eq!(fx.instruction_d, p.instruction_d, "instruction_d");
    assert_eq!(fx.bytecode_d, p.bytecode_d, "bytecode_d");
    assert!(
        fx.ram_d <= p.ram_d,
        "Goldilocks zero-init RAM is a subset of jolt-core's (ram_d {} <= {})",
        fx.ram_d,
        p.ram_d
    );

    eprintln!(
        "[goldilocks-e2e/M4] geometry parity vs jolt-core OK: log_t={}, log_k_chunk={}, \
         instruction_d={}, bytecode_d={} (ram_d goldilocks={} <= core={})",
        p.log_t, p.log_k_chunk, p.instruction_d, p.bytecode_d, fx.ram_d, p.ram_d,
    );
}

/// P3b-0 — the jolt-core-free instruction lookup-table dispatch
/// ([`jolt_lookup_tables::instruction_lookup_table_index`]) must agree with jolt-core's
/// `InstructionLookup::lookup_table(cycle).map(enum_index)` on every cycle of the real muldiv trace.
/// This gates both the per-opcode table choice AND that `LookupTableKind::index()` (the `#[repr(u8)]`
/// discriminant) is the same ordering as jolt-core's `LookupTables::enum_index`.
#[test]
fn goldilocks_instruction_lookup_dispatch_matches_core() {
    use common::constants::XLEN;
    use jolt_core::zkvm::instruction::InstructionLookup;
    use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
    use jolt_lookup_tables::instruction_lookup_table_index;

    let fx = build_muldiv_fixture();
    let mut tables_seen = std::collections::BTreeSet::new();
    for cycle in &fx.trace {
        let mine = instruction_lookup_table_index::<XLEN>(&cycle.instruction());
        let core = InstructionLookup::<XLEN>::lookup_table(cycle)
            .map(|table| CoreLookupTables::<XLEN>::enum_index(&table));
        assert_eq!(
            mine, core,
            "instruction lookup-table dispatch must match jolt-core (cycle: {cycle:?})"
        );
        if let Some(t) = mine {
            tables_seen.insert(t);
        }
    }
    assert!(!fx.trace.is_empty(), "muldiv trace must be non-empty");

    eprintln!(
        "[goldilocks-e2e/P3b-0] dispatch parity vs jolt-core OK over {} cycles: {} distinct tables {:?}",
        fx.trace.len(),
        tables_seen.len(),
        tables_seen,
    );
}
