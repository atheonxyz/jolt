//! Committed base-field witness columns → framework MLEs, keyed by [`CommittedPolynomial`].
//!
//! The committed half of jolt-core's `zkvm/witness.rs`, over the Goldilocks base-limb
//! representation. Reuses the field-agnostic [`CommitmentTraceSources`] (jolt-witness) — the trace →
//! sources extraction (`extract_trace` + `commitment_trace_sources`) is the M8 e2e path; this module
//! is **decoupled** from the trace (takes the sources), matching the crate convention.
//!
//! Produces the committed columns the surviving subprotocols + the M7 Option C per-chunk pushforward
//! consume:
//! - **`ra_dense`** per `(family, chunk)`: the dense address-chunk index column (chunk 0 = the most
//!   significant chunk, matching jolt-core's committed RA decomposition). Indexed by a **global chunk
//!   index** across families (instruction chunks, then bytecode, then ram), which is exactly the
//!   `RaDense(idx)` / `Pushforward(idx)` accumulator key and the
//!   [`prove_family_per_chunk`](crate::zkvm::logup::driver::prove_family_per_chunk) base index. The
//!   raw `Vec<u32>` indices feed both the read-raf one-hot lift and the Option C pushforward.
//! - **`RdInc` / `RamInc`**: the **recomposed** signed increment as an `F` MLE — the value the Inc
//!   claim-reduction / val-evaluation / read-write-checking sumchecks consume (`Inc = lo + 2³²·hi`,
//!   kept as a single recomposed column so `Val = Σ inc·wa·LT` stays degree-3, per M4).
//!
//! **Deferred (documented, not silent):**
//! - The committed objects for `Inc` are actually the **two signed base limbs** (`lo`, `hi`); how
//!   stage-8 commits/opens them (per-limb columns vs the recomposed virtual poly) is the M8 piece-4
//!   stage-8-batching layout decision, so the limb columns are materialized there, not here.
//! - The **R1CS witness** `z` + `Az`/`Bz`/`Cz` and the virtual flag/value polynomials are
//!   materialized by the stage driver (M8 piece 3) from `extract_trace`'s R1CS/flag outputs.
//! - The **compact base-field MLE** variants (`base × ext` hot path) are a perf pivot layered on once
//!   the dense-`F` e2e is correct (review-guide §6/§8); these MLEs are dense `F` for now.

use jolt_field::Field;
use jolt_witness::goldilocks::GoldilocksLayout;
use jolt_witness::{one_hot_chunk_indices, CommitmentTraceSources};

use crate::framework::accumulator::CommittedPolynomial;
use crate::framework::poly::MultilinearPolynomial;

/// `ceil(log2(n))` (matches `jolt_witness::goldilocks`'s committed-length convention).
#[inline]
fn next_log2(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

/// One `ra_dense` address-chunk index column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaDenseColumn {
    /// Family label, e.g. `"InstructionRa"`.
    pub family: &'static str,
    /// Global chunk index across families — the `RaDense(idx)`/`Pushforward(idx)` accumulator key and
    /// the Option C per-chunk pushforward base index.
    pub global_index: usize,
    /// Chunk width (bits): the pushforward column space is `2^log_m`, so indices are `< 2^log_m`.
    pub log_m: usize,
    /// The dense chunk-index column (one-hot `None` padding → `0`), length `2^log_t`.
    pub indices: Vec<u32>,
}

impl RaDenseColumn {
    /// The committed-polynomial accumulator key for this chunk's dense column.
    #[inline]
    pub fn committed_key(&self) -> CommittedPolynomial {
        CommittedPolynomial::RaDense(self.global_index)
    }
}

/// Committed base-field witness, materialized from [`CommitmentTraceSources`] into framework MLEs and
/// raw `ra_dense` index columns, keyed for the accumulator + the M7 pushforward + the stage-8 open.
#[derive(Clone, Debug)]
pub struct CommittedWitness<F: Field> {
    /// `log2` of the committed (padded, power-of-two) column length.
    pub log_t: usize,
    /// `ra_dense` columns in global-index order: instruction chunks, then bytecode, then ram.
    pub ra_dense: Vec<RaDenseColumn>,
    /// Global-index range of the instruction family's chunks within [`Self::ra_dense`].
    pub instruction_range: std::ops::Range<usize>,
    /// Global-index range of the bytecode family's chunks.
    pub bytecode_range: std::ops::Range<usize>,
    /// Global-index range of the ram family's chunks.
    pub ram_range: std::ops::Range<usize>,
    /// Recomposed `RdInc` as an `F` MLE (the value the register Inc sumchecks consume).
    pub rd_inc: MultilinearPolynomial<F>,
    /// Recomposed `RamInc` as an `F` MLE (the value the RAM Inc sumchecks consume).
    pub ram_inc: MultilinearPolynomial<F>,
}

impl<F: Field> CommittedWitness<F> {
    /// Materialize the committed witness from the trace sources + the one-hot family geometry.
    pub fn build(sources: &CommitmentTraceSources, layout: &GoldilocksLayout) -> Self {
        let trace_len = layout.trace_len;
        let log_t = next_log2(trace_len.max(1));
        let committed_len = 1usize << log_t;

        let mut ra_dense = Vec::new();
        let mut global = 0usize;
        let mut ranges = Vec::with_capacity(3);
        for (src, fam) in [
            (&sources.instruction_keys, layout.instruction),
            (&sources.bytecode_indices, layout.bytecode),
            (&sources.ram_addresses, layout.ram),
        ] {
            let start = global;
            for chunk in 0..fam.num_chunks {
                let chunk_idx = one_hot_chunk_indices(
                    src,
                    chunk,
                    fam.num_chunks,
                    fam.chunk_bits,
                    trace_len,
                    fam.padding,
                );
                let mut indices: Vec<u32> = chunk_idx
                    .iter()
                    .map(|e| e.map_or(0u32, u32::from))
                    .collect();
                indices.resize(committed_len, 0);
                ra_dense.push(RaDenseColumn {
                    family: fam.label,
                    global_index: global,
                    log_m: fam.chunk_bits,
                    indices,
                });
                global += 1;
            }
            ranges.push(start..global);
        }

        let inc_mle = |inc: &[i128]| -> MultilinearPolynomial<F> {
            let mut values: Vec<F> = inc.iter().map(|&v| F::from_i128(v)).collect();
            assert!(
                values.len() <= committed_len,
                "Inc column longer than 2^log_t"
            );
            values.resize(committed_len, F::zero());
            MultilinearPolynomial::from(values)
        };

        Self {
            log_t,
            ra_dense,
            instruction_range: ranges[0].clone(),
            bytecode_range: ranges[1].clone(),
            ram_range: ranges[2].clone(),
            rd_inc: inc_mle(&sources.rd_inc),
            ram_inc: inc_mle(&sources.ram_inc),
        }
    }

    /// Total number of committed `ra_dense` chunks across all families.
    #[inline]
    pub fn num_ra_chunks(&self) -> usize {
        self.ra_dense.len()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_witness::goldilocks::FamilyLayout;

    /// A small synthetic layout: instruction d=2 / bytecode d=2 / ram d=1, all 2-bit chunks.
    fn layout(trace_len: usize) -> GoldilocksLayout {
        GoldilocksLayout {
            trace_len,
            instruction: FamilyLayout {
                label: "InstructionRa",
                num_chunks: 2,
                chunk_bits: 2,
                padding: Some(0),
            },
            bytecode: FamilyLayout {
                label: "BytecodeRa",
                num_chunks: 2,
                chunk_bits: 2,
                padding: Some(0),
            },
            ram: FamilyLayout {
                label: "RamRa",
                num_chunks: 1,
                chunk_bits: 2,
                padding: None,
            },
        }
    }

    /// Synthetic sources: `actual` cycles of data, the rest one-hot-padded to `trace_len`.
    fn synth_sources(actual: usize) -> CommitmentTraceSources {
        // 4-bit logical addresses split into two 2-bit chunks (chunk 0 = high 2 bits).
        let keys: Vec<Option<u128>> = (0..actual)
            .map(|j| Some(((j * 7 + 1) % 16) as u128))
            .collect();
        let bc: Vec<Option<u128>> = (0..actual)
            .map(|j| Some(((j * 5 + 2) % 16) as u128))
            .collect();
        let ram: Vec<Option<u128>> = (0..actual)
            .map(|j| {
                if j % 3 == 0 {
                    None
                } else {
                    Some((j % 4) as u128)
                }
            })
            .collect();
        let rd_inc: Vec<i128> = (0..actual as i128).map(|j| j - 2).collect();
        let ram_inc: Vec<i128> = (0..actual as i128).map(|j| (j - 1) * 1_000).collect();
        CommitmentTraceSources {
            rd_inc,
            ram_inc,
            instruction_keys: keys,
            ram_addresses: ram,
            bytecode_indices: bc,
        }
    }

    #[test]
    fn ra_dense_global_indexing_and_ranges() {
        let trace_len = 6; // pads to 2^3 = 8
        let sources = synth_sources(4);
        let w = CommittedWitness::<F>::build(&sources, &layout(trace_len));

        assert_eq!(w.log_t, 3);
        assert_eq!(w.num_ra_chunks(), 5); // 2 + 2 + 1
        assert_eq!(w.instruction_range, 0..2);
        assert_eq!(w.bytecode_range, 2..4);
        assert_eq!(w.ram_range, 4..5);

        // Global indices are contiguous and match the column's stored key.
        for (i, col) in w.ra_dense.iter().enumerate() {
            assert_eq!(col.global_index, i);
            assert_eq!(col.committed_key(), CommittedPolynomial::RaDense(i));
            assert_eq!(col.indices.len(), 1 << w.log_t);
            assert!(
                col.indices.iter().all(|&k| k < (1u32 << col.log_m)),
                "chunk {i} index out of range 2^{}",
                col.log_m
            );
        }
        // Families land at the expected labels.
        assert_eq!(w.ra_dense[0].family, "InstructionRa");
        assert_eq!(w.ra_dense[2].family, "BytecodeRa");
        assert_eq!(w.ra_dense[4].family, "RamRa");
    }

    #[test]
    fn ra_dense_indices_match_one_hot_decomposition() {
        let trace_len = 8;
        let sources = synth_sources(8);
        let lay = layout(trace_len);
        let w = CommittedWitness::<F>::build(&sources, &lay);

        // Chunk 0 is the high 2 bits, chunk 1 the low 2 bits, of the 4-bit instruction key.
        for (cycle, key) in sources.instruction_keys.iter().enumerate() {
            let k = key.unwrap() as u32;
            assert_eq!(w.ra_dense[0].indices[cycle], (k >> 2) & 0b11, "hi chunk");
            assert_eq!(w.ra_dense[1].indices[cycle], k & 0b11, "lo chunk");
        }
    }

    #[test]
    fn inc_recomposes_and_pads() {
        let trace_len = 6; // pads to 8
        let actual = 4;
        let sources = synth_sources(actual);
        let w = CommittedWitness::<F>::build(&sources, &layout(trace_len));

        assert_eq!(w.rd_inc.len(), 1 << w.log_t);
        for (j, &orig) in sources.rd_inc.iter().enumerate() {
            assert_eq!(
                w.rd_inc.get_bound_coeff(j),
                F::from_i128(orig),
                "RdInc cycle {j}"
            );
        }
        for j in actual..w.rd_inc.len() {
            assert_eq!(
                w.rd_inc.get_bound_coeff(j),
                F::from_u64(0),
                "RdInc padding must be zero"
            );
        }

        for (j, &orig) in sources.ram_inc.iter().enumerate() {
            assert_eq!(
                w.ram_inc.get_bound_coeff(j),
                F::from_i128(orig),
                "RamInc cycle {j}"
            );
        }
    }
}
