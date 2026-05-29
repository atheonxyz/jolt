//! Functional test for the Phase-1 WHIR base-field commit pipeline:
//! synthetic trace → base-Goldilocks limb columns → WHIR commit + sanity
//! open/verify + limb recompose. (The live fibonacci trace is the e2e in
//! `tests/e2e.rs`; this validates the field/limb/commit path on synthetic data.)

#![cfg(feature = "goldilocks")]
#![expect(clippy::unwrap_used)]

use jolt_field::goldilocks::decompose::sign_limbs_to_i128;
use jolt_field::goldilocks::Goldilocks;
use jolt_whir::{commit_witness, sanity_roundtrip};
use jolt_witness::goldilocks::{FamilyLayout, GoldilocksLayout, GoldilocksWitnessColumns};
use jolt_witness::{CommitmentTraceSources, CycleInput};

const TRACE_LEN: usize = 64;

fn synthetic_sources() -> CommitmentTraceSources {
    // Values are spread across all bits so no committed column is all-zero
    // (WHIR's verify divides by the polynomial's evaluation, which is 0 for the
    // zero polynomial — a real concern for Phase-2 opening of arbitrary witness
    // columns; here we keep the synthetic data non-degenerate). Increments are
    // scaled past 2^32 so the `.hi` limb is exercised too.
    let cycles: Vec<CycleInput> = (0..TRACE_LEN)
        .map(|i| {
            let i = i as i128;
            CycleInput {
                dense: [
                    (i - 30) * 0x1_2345_6789, // rd_inc: mixed sign, |v| > 2^32 → hi nonzero
                    (i - 20) * 0x9_8765_4321, // ram_inc
                ],
                one_hot: [
                    Some(((i * 1031 + 17) % 65_536) as u128), // instruction: full 16-bit spread (4×4 bits)
                    Some(((i * 131 + 9) % 256) as u128), // bytecode: full 8-bit spread (2×4 bits)
                    if i % 3 == 0 {
                        None
                    } else {
                        Some(((i * 97 + 5) % 256) as u128) // ram (2×4 bits, padding None)
                    },
                ],
            }
        })
        .collect();
    CommitmentTraceSources::from_cycle_inputs(&cycles)
}

fn layout() -> GoldilocksLayout {
    GoldilocksLayout {
        trace_len: TRACE_LEN,
        instruction: FamilyLayout {
            label: "InstructionRa",
            num_chunks: 4,
            chunk_bits: 4,
            padding: Some(0),
        },
        bytecode: FamilyLayout {
            label: "BytecodeRa",
            num_chunks: 2,
            chunk_bits: 4,
            padding: Some(0),
        },
        ram: FamilyLayout {
            label: "RamRa",
            num_chunks: 2,
            chunk_bits: 4,
            padding: None,
        },
    }
}

#[test]
fn build_commit_and_open_synthetic_trace() {
    let sources = synthetic_sources();
    let layout = layout();
    let cols = GoldilocksWitnessColumns::build(&sources, &layout);

    // 4 + 2 + 2 = 8 ra_dense columns, plus 2 Inc × 3 limb columns = 6.
    assert_eq!(cols.columns.len(), 8 + 6);
    assert_eq!(cols.log_t, 6); // TRACE_LEN = 2^6
    for c in &cols.columns {
        assert_eq!(c.values.len(), 1 << 6);
    }

    let report = commit_witness(&cols);
    assert_eq!(report.log_t, 6);
    assert_eq!(report.num_columns, 14);
    assert_eq!(report.total_base_elements, 14 * 64);
    assert_eq!(report.committed_base_bytes, 14 * 64 * 8);
    assert!(report.commit_ms >= 0.0);

    // Sanity open/verify round-trips on non-degenerate columns (an ra_dense
    // chunk and an Inc limb). Columns are non-zero by construction (see
    // `synthetic_sources`), which WHIR's open path requires.
    assert!(sanity_roundtrip(&cols.columns[0].values));
    let lo = cols.columns.iter().find(|c| c.label == "RdInc.hi").unwrap();
    assert!(sanity_roundtrip(&lo.values));
}

#[test]
fn inc_limbs_recompose_to_originals() {
    let sources = synthetic_sources();
    let cols = GoldilocksWitnessColumns::build(&sources, &layout());

    let col = |name: &str| -> &[Goldilocks] {
        &cols
            .columns
            .iter()
            .find(|c| c.label == name)
            .unwrap()
            .values
    };
    let sign = col("RdInc.sign");
    let lo = col("RdInc.lo");
    let hi = col("RdInc.hi");

    for (i, &orig) in sources.rd_inc.iter().enumerate() {
        let recomposed = sign_limbs_to_i128(sign[i], [lo[i], hi[i]]);
        assert_eq!(
            recomposed, orig,
            "RdInc limb recompose mismatch at cycle {i}"
        );
    }
}
