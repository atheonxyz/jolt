//! Materialize the stage-8 committed base-Goldilocks columns from the real witness (P9-S3b) — the
//! bridge from the zkvm witness ([`CommittedWitness`] + [`CommitmentTraceSources`] + the `R1csAux`
//! boolean columns) to the framework WHIR-open inputs
//! ([`Stage8Columns`](crate::framework::stage8_open::Stage8Columns) +
//! [`IncLimbColumns`](crate::framework::stage8_open::IncLimbColumns)).
//!
//! Each committed object is lowered to its base-Goldilocks commit form:
//! - **`RaDense(idx)`** — the dense chunk-index column (`u32 → Goldilocks`), length `2^log_t`.
//! - **`R1csAux(i)`** — the boolean aux columns (`Fp3 0/1 → Goldilocks` coeff-0 lift; the other two
//!   Fp3 coefficients are zero for a boolean), length `2^log_t`.
//! - **`RdInc`/`RamInc`** — each split into its two signed base limbs `lo`/`hi`
//!   (`i128 → i128_to_signed_limbs`, Fork 3), padded to `2^log_t`.
//!
//! The `Pushforward` `P^F` base-limb columns are produced by the M7 GKR (Fp3 → 3 limbs via
//! [`Fp3LimbColumns::from_fp3`](crate::framework::stage8_open::Fp3LimbColumns)); they are materialized
//! where the pushforward runs (the full-driver integration), not here.

use jolt_field::goldilocks::decompose::i128_to_signed_limbs;
use jolt_field::Field;
use jolt_witness::CommitmentTraceSources;

use crate::field::{Base, F};
use crate::framework::accumulator::CommittedPolynomial;
use crate::framework::stage8_open::{IncLimbColumns, Stage8Columns};
use crate::zkvm::witness::CommittedWitness;

/// Lower the witness-derived committed objects to their base-Goldilocks commit columns: the
/// `RaDense` chunk-index columns + the `R1csAux` boolean columns into [`Stage8Columns`] (the generic
/// inventory open), and the `RdInc`/`RamInc` signed limbs into [`IncLimbColumns`] (the limb
/// reconstruct). All columns have length `2^log_t`.
pub fn build_committed_columns(
    committed: &CommittedWitness<F>,
    sources: &CommitmentTraceSources,
    aux_columns: &[Vec<F>],
) -> (Stage8Columns, IncLimbColumns) {
    let committed_len = 1usize << committed.log_t;
    let mut columns = Stage8Columns::new();

    // RaDense: the dense chunk-index column, u32 → base-Goldilocks.
    for col in &committed.ra_dense {
        let base: Vec<Base> = col
            .indices
            .iter()
            .map(|&k| Base::from_u64(u64::from(k)))
            .collect();
        columns.insert(CommittedPolynomial::RaDense(col.global_index), base);
    }

    // R1csAux: boolean Fp3 columns → base (coeff-0 lift; the other coeffs are zero for a boolean).
    for (i, col) in aux_columns.iter().enumerate() {
        let base: Vec<Base> = col
            .iter()
            .map(|x| {
                let c = x.coeffs();
                debug_assert!(
                    c[1] == Base::from_u64(0) && c[2] == Base::from_u64(0),
                    "R1csAux column must be base-representable (boolean)"
                );
                c[0]
            })
            .collect();
        columns.insert(CommittedPolynomial::R1csAux(i), base);
    }

    // Inc: signed two-limb decomposition, padded to the committed length.
    let limbs = |inc: &[i128]| -> (Vec<Base>, Vec<Base>) {
        let mut lo = Vec::with_capacity(committed_len);
        let mut hi = Vec::with_capacity(committed_len);
        for &v in inc {
            let [l, h] = i128_to_signed_limbs(v);
            lo.push(l);
            hi.push(h);
        }
        lo.resize(committed_len, Base::from_u64(0));
        hi.resize(committed_len, Base::from_u64(0));
        (lo, hi)
    };
    let (rd_inc_lo, rd_inc_hi) = limbs(&sources.rd_inc);
    let (ram_inc_lo, ram_inc_hi) = limbs(&sources.ram_inc);

    (
        columns,
        IncLimbColumns {
            rd_inc_lo,
            rd_inc_hi,
            ram_inc_lo,
            ram_inc_hi,
        },
    )
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript, WhirScheme};
    use crate::framework::accumulator::{OpeningPoint, BIG_ENDIAN};
    use crate::framework::stage8::Stage8Inventory;
    use crate::framework::stage8_open::{
        prove_inc_open, prove_stage8, verify_inc_open, verify_stage8,
    };
    use jolt_field::goldilocks::decompose::signed_limbs_recompose;
    use jolt_witness::goldilocks::{FamilyLayout, GoldilocksLayout};

    fn layout(trace_len: usize) -> GoldilocksLayout {
        let fam = |label| FamilyLayout {
            label,
            num_chunks: 2,
            chunk_bits: 2,
            padding: Some(0),
        };
        GoldilocksLayout {
            trace_len,
            instruction: fam("InstructionRa"),
            bytecode: fam("BytecodeRa"),
            ram: FamilyLayout {
                label: "RamRa",
                num_chunks: 1,
                chunk_bits: 2,
                padding: None,
            },
        }
    }

    fn synth_sources(actual: usize) -> CommitmentTraceSources {
        CommitmentTraceSources {
            rd_inc: (0..actual as i128)
                .map(|j| (j - 2) * 0x1_0000_0001)
                .collect(),
            ram_inc: (0..actual as i128)
                .map(|j| (j - 1) * 0x2_0000_0003)
                .collect(),
            instruction_keys: (0..actual)
                .map(|j| Some(((j * 7 + 1) % 16) as u128))
                .collect(),
            ram_addresses: (0..actual).map(|j| Some((j % 4) as u128)).collect(),
            bytecode_indices: (0..actual)
                .map(|j| Some(((j * 5 + 2) % 16) as u128))
                .collect(),
        }
    }

    /// Materialized columns match the witness integers (RaDense indices, Inc limbs recompose) and are
    /// the committed length; the `R1csAux` columns lift the boolean values.
    #[test]
    fn materialization_matches_witness() {
        let trace_len = 8;
        let sources = synth_sources(6);
        let committed = CommittedWitness::<F>::build(&sources, &layout(trace_len));
        let aux = vec![
            (0..trace_len)
                .map(|j| F::from_u64((j % 2) as u64))
                .collect::<Vec<_>>(),
            (0..trace_len)
                .map(|j| F::from_u64(((j + 1) % 2) as u64))
                .collect::<Vec<_>>(),
        ];
        let (columns, inc) = build_committed_columns(&committed, &sources, &aux);
        let n = 1usize << committed.log_t;

        for col in &committed.ra_dense {
            let materialized = &columns.columns[&CommittedPolynomial::RaDense(col.global_index)];
            assert_eq!(materialized.len(), n);
            for (j, &k) in col.indices.iter().enumerate() {
                assert_eq!(materialized[j], Base::from_u64(u64::from(k)));
            }
        }
        // Inc limbs recompose to from_i128(source increment).
        for (j, &v) in sources.rd_inc.iter().enumerate() {
            assert_eq!(
                signed_limbs_recompose([inc.rd_inc_lo[j], inc.rd_inc_hi[j]]),
                Base::from_i128(v)
            );
        }
        assert_eq!(inc.rd_inc_lo.len(), n);
        assert_eq!(columns.columns[&CommittedPolynomial::R1csAux(0)].len(), n);
    }

    /// The materialized columns commit + open + verify through the WHIR stage-8 open: build the
    /// inventory (RaDense + R1csAux at random native points, claims = the honest WHIR evals) and the
    /// Inc-limb opens (recomposed claims = the recomposed base column's eval), and round-trip.
    #[test]
    fn materialized_columns_whir_round_trip() {
        // WHIR's RS interleaving needs a column length above its minimum (2^3 is too small).
        let trace_len = 64;
        let sources = synth_sources(trace_len);
        let committed = CommittedWitness::<F>::build(&sources, &layout(trace_len));
        let aux: Vec<Vec<F>> = vec![(0..trace_len)
            .map(|j| F::from_u64((j % 2) as u64))
            .collect()];
        let (columns, inc) = build_committed_columns(&committed, &sources, &aux);
        let n = 1usize << committed.log_t;
        let cfg = WhirScheme::config(n);
        let point = |seed: u64| {
            (0..committed.log_t)
                .map(|i| F::from_u64(seed.wrapping_mul(i as u64 + 7) | 1))
                .collect::<Vec<F>>()
        };

        // Inventory over RaDense + R1csAux, each at its own point, claims = honest WHIR evals.
        let mut inventory = Stage8Inventory::<F>::new();
        for (i, col) in committed.ra_dense.iter().enumerate() {
            let key = CommittedPolynomial::RaDense(col.global_index);
            let pt = point(0x100 + i as u64);
            let claim = WhirScheme::evaluate(&cfg, &columns.columns[&key], &pt);
            let _ = inventory.insert_or_alias(
                key,
                OpeningPoint::<BIG_ENDIAN, F>::new(pt),
                claim,
                committed.log_t,
            );
        }
        let aux_key = CommittedPolynomial::R1csAux(0);
        let aux_pt = point(0x200);
        let aux_claim = WhirScheme::evaluate(&cfg, &columns.columns[&aux_key], &aux_pt);
        let _ = inventory.insert_or_alias(
            aux_key,
            OpeningPoint::<BIG_ENDIAN, F>::new(aux_pt),
            aux_claim,
            committed.log_t,
        );

        // Inc recomposed claims = the recomposed base column's WHIR eval at each family's point.
        let recompose = |lo: &[Base], hi: &[Base]| -> Vec<Base> {
            lo.iter()
                .zip(hi.iter())
                .map(|(&l, &h)| signed_limbs_recompose([l, h]))
                .collect()
        };
        let rd_point = point(0x300);
        let ram_point = point(0x301);
        let rd_claim =
            WhirScheme::evaluate(&cfg, &recompose(&inc.rd_inc_lo, &inc.rd_inc_hi), &rd_point);
        let ram_claim = WhirScheme::evaluate(
            &cfg,
            &recompose(&inc.ram_inc_lo, &inc.ram_inc_hi),
            &ram_point,
        );

        let mut prover_t = ProverTranscript::new("s3b");
        prove_stage8(&mut prover_t, &columns, &inventory).expect("prove inventory");
        let inc_proof = prove_inc_open(&mut prover_t, &inc, &rd_point, &ram_point);
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("s3b", &narg);
        verify_stage8(&mut verifier_t, &inventory).expect("verify inventory");
        verify_inc_open(
            &mut verifier_t,
            &rd_point,
            &ram_point,
            &inc_proof,
            rd_claim,
            ram_claim,
        )
        .expect("verify inc");
    }
}
