//! Goldilocks base-field-limb witness columns (feature `goldilocks`).
//!
//! Transforms the field-agnostic [`CommitmentTraceSources`] into the committed
//! columns of the base-field-limb representation:
//! - `ra_dense` per one-hot chunk (`Some(k) → k`, `None → 0`), one column per
//!   chunk of each RA family;
//! - `RdInc`/`RamInc` decomposed into **signed two limbs** `lo + hi·2^32` (the
//!   high limb carries the sign), so recomposition is linear in the committed
//!   columns and the `Val = Σ inc·wa·LT` sumcheck stays degree-3;
//!
//! all in the base field [`Goldilocks`], padded to the committed length
//! `2^log_t`. There is **no** pushforward `P^F` here — that is the Phase-2
//! LogUp\* GKR. The recomposition / range-check *constraints* over these limbs
//! live in the Goldilocks prover crate; this module only produces the columns.

use jolt_field::goldilocks::decompose::i128_to_signed_limbs;
use jolt_field::goldilocks::Goldilocks;
use jolt_field::Field;

use crate::{one_hot_chunk_indices, CommitmentTraceSources};

/// One committed base-field column, length `2^log_t`.
#[derive(Clone, Debug)]
pub struct GoldilocksColumn {
    pub label: String,
    pub values: Vec<Goldilocks>,
}

/// One-hot RA family decomposition geometry (matches jolt-core's committed RA).
#[derive(Clone, Copy, Debug)]
pub struct FamilyLayout {
    /// Family name, e.g. `"InstructionRa"`.
    pub label: &'static str,
    /// Number of `d`-decomposition chunks.
    pub num_chunks: usize,
    /// Bits per chunk (the chunk-index domain is `2^chunk_bits`, `≤ 8`).
    pub chunk_bits: usize,
    /// Per-cycle padding policy (`Some(0)` for instruction/bytecode, `None` for RAM).
    pub padding: Option<u128>,
}

/// Geometry needed to build the base-field-limb committed columns.
#[derive(Clone, Debug)]
pub struct GoldilocksLayout {
    pub trace_len: usize,
    pub instruction: FamilyLayout,
    pub bytecode: FamilyLayout,
    pub ram: FamilyLayout,
}

/// The base-field-limb committed witness. All columns have length `2^log_t`.
#[derive(Clone, Debug)]
pub struct GoldilocksWitnessColumns {
    /// `log2` of the committed (padded, power-of-two) column length.
    pub log_t: usize,
    pub columns: Vec<GoldilocksColumn>,
}

/// `ceil(log2(n))`.
#[inline]
fn next_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

fn pad_to(mut v: Vec<Goldilocks>, len: usize) -> Vec<Goldilocks> {
    assert!(v.len() <= len, "column longer than committed length");
    v.resize(len, Goldilocks::from_u64(0));
    v
}

impl GoldilocksWitnessColumns {
    /// Build the committed base-field-limb columns from the trace sources.
    pub fn build(sources: &CommitmentTraceSources, layout: &GoldilocksLayout) -> Self {
        let trace_len = layout.trace_len;
        let log_t = next_log2(trace_len.max(1));
        let committed_len = 1usize << log_t;
        let mut columns = Vec::new();

        // ra_dense: one dense column per (family, chunk).
        for (src, fam) in [
            (&sources.instruction_keys, layout.instruction),
            (&sources.bytecode_indices, layout.bytecode),
            (&sources.ram_addresses, layout.ram),
        ] {
            for chunk in 0..fam.num_chunks {
                let indices = one_hot_chunk_indices(
                    src,
                    chunk,
                    fam.num_chunks,
                    fam.chunk_bits,
                    trace_len,
                    fam.padding,
                );
                let values = indices
                    .iter()
                    .map(|entry| match entry {
                        Some(k) => Goldilocks::from_u64(u64::from(*k)),
                        None => Goldilocks::from_u64(0),
                    })
                    .collect::<Vec<_>>();
                columns.push(GoldilocksColumn {
                    label: format!("ra_dense::{}_{chunk}", fam.label),
                    values: pad_to(values, committed_len),
                });
            }
        }

        // Inc columns: signed increment → (lo, hi) base-field limbs, hi signed.
        for (label, inc) in [("RdInc", &sources.rd_inc), ("RamInc", &sources.ram_inc)] {
            let mut lo = Vec::with_capacity(trace_len);
            let mut hi = Vec::with_capacity(trace_len);
            for &v in inc {
                let [l, h] = i128_to_signed_limbs(v);
                lo.push(l);
                hi.push(h);
            }
            columns.push(GoldilocksColumn {
                label: format!("{label}.lo"),
                values: pad_to(lo, committed_len),
            });
            columns.push(GoldilocksColumn {
                label: format!("{label}.hi"),
                values: pad_to(hi, committed_len),
            });
        }

        Self { log_t, columns }
    }

    /// Total committed field elements across all columns.
    pub fn total_elements(&self) -> usize {
        self.columns.iter().map(|c| c.values.len()).sum()
    }
}
