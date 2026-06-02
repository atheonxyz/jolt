//! Witness-level structural parity gate (P11 / Fork 5): build the Goldilocks `CommittedWitness`
//! from the SAME real muldiv `CommitmentTraceSources` jolt-core produces, and assert geometry +
//! witness-integer parity against jolt-core's `JoltProtocolParams` geometry.
//!
//! This compiles a RISC-V guest (the muldiv fixture) via the `jolt` CLI, so it is gated behind the
//! `goldilocks` feature and run explicitly:
//!   `cargo nextest run -p jolt-equivalence --features goldilocks goldilocks_witness`
//!
//! It is the first cross-system check of the Goldilocks committed-witness layer on a REAL program:
//! the `ra_dense` chunk decomposition, family geometry, and Inc limbs are validated against the
//! jolt-core-derived sources + `OneHotParams`. The read-raf / pushforward sumchecks are NOT run here
//! (that is the full e2e, P10); this gate is transcript-free and witness-only.
#![cfg(feature = "goldilocks")]

use jolt_equivalence::core_oracle::core_muldiv_commitment_fixture;
use jolt_field::Field;
use jolt_prover_goldilocks::zkvm::witness::CommittedWitness;
use jolt_prover_goldilocks::F;
use jolt_witness::commitment_trace_sources;
use jolt_witness::goldilocks::{FamilyLayout, GoldilocksLayout};

#[test]
fn goldilocks_committed_witness_matches_core_muldiv() {
    let fixture = core_muldiv_commitment_fixture();
    let p = &fixture.params;
    let trace_len = p.trace_length;
    let chunk_bits = p.log_k_chunk;

    // Goldilocks committed-witness layout from jolt-core's OneHotParams geometry (mirrors
    // jolt-pcs-bench's `layout_from_params`): chunk_bits = log_k_chunk, instruction/bytecode pad with
    // `Some(0)`, RAM with `None` (the committed-RA padding policy).
    let layout = GoldilocksLayout {
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
    };

    let sources = commitment_trace_sources(&fixture.cycle_inputs);
    let w = CommittedWitness::<F>::build(&sources, &layout);

    // 1. Geometry parity vs jolt-core's OneHotParams.
    assert_eq!(
        w.instruction_range.len(),
        p.instruction_d,
        "instruction chunk count"
    );
    assert_eq!(w.bytecode_range.len(), p.bytecode_d, "bytecode chunk count");
    assert_eq!(w.ram_range.len(), p.ram_d, "ram chunk count");
    assert_eq!(
        w.num_ra_chunks(),
        p.instruction_d + p.bytecode_d + p.ram_d,
        "total committed RA chunks"
    );
    let committed_len = 1usize << w.log_t;
    assert!(
        committed_len >= trace_len.max(1) && committed_len < 2 * trace_len.max(1),
        "log_t is the tight power-of-two cover of the trace"
    );

    // 2. Index validity: every ra_dense index < 2^log_k_chunk; columns are length 2^log_t.
    for col in &w.ra_dense {
        assert_eq!(col.log_m, chunk_bits, "chunk width = log_k_chunk");
        assert_eq!(col.indices.len(), committed_len, "column length 2^log_t");
        assert!(
            col.indices.iter().all(|&k| k < (1u32 << chunk_bits)),
            "chunk {} index out of range 2^{chunk_bits}",
            col.global_index
        );
    }

    // 3. Bytecode chunk recomposition: the `bytecode_d` chunks (chunk 0 most significant) recompose
    //    to the source PC index, for every cycle with a real bytecode read.
    let bc = &w.ra_dense[w.bytecode_range.clone()];
    for (j, key) in sources.bytecode_indices.iter().enumerate() {
        if let Some(key) = key {
            let recomposed = bc
                .iter()
                .fold(0u128, |acc, col| (acc << chunk_bits) | u128::from(col.indices[j]));
            assert_eq!(recomposed, *key, "bytecode chunk recomposition at cycle {j}");
        }
    }

    // 4. Inc limbs recompose to the source signed increments (the value the Inc sumchecks consume).
    for (j, &inc) in sources.rd_inc.iter().enumerate() {
        assert_eq!(
            w.rd_inc.get_bound_coeff(j),
            F::from_i128(inc),
            "RdInc cycle {j}"
        );
    }
    for (j, &inc) in sources.ram_inc.iter().enumerate() {
        assert_eq!(
            w.ram_inc.get_bound_coeff(j),
            F::from_i128(inc),
            "RamInc cycle {j}"
        );
    }

    eprintln!(
        "[goldilocks-witness-gate] muldiv parity OK: log_t={}, instruction_d={}, bytecode_d={}, \
         ram_d={}, log_k_chunk={chunk_bits}, ra_chunks={}",
        w.log_t,
        p.instruction_d,
        p.bytecode_d,
        p.ram_d,
        w.num_ra_chunks()
    );
}
