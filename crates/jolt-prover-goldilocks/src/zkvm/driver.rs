//! Top-level binary-Spartan prove/verify driver (M8, growing toward the full 8-stage e2e).
//!
//! Wires the per-stage instances onto one shared transcript + opening accumulator, the way the full
//! `prove()`/`verify()` will. Currently composes the **Spartan stage** (outer + inner reduction, see
//! [`crate::zkvm::spartan::stage`]) and the **booleanity stage** (the M6 carry/sign residual over
//! `R1csAux`). The remaining stages (read-raf, RAM/registers read-write-checking + val-evaluation +
//! output-check, claim-reductions, the M7 per-chunk pushforward) and the stage-8 WHIR batched open
//! follow the same template — each needs its per-stage witness columns materialized from the trace
//! (the `K×T` one-hot matrices), which is the remaining witness-gen work (see task #3).
//!
//! The committed openings the verifier checks against (`Az/Bz/Cz(r_x)`, `z(r_y)`, `R1csAux(i)(ρ)`)
//! are carried in the proof and will be discharged by the stage-8 WHIR open.

use jolt_field::Field;
use jolt_r1cs::R1csKey;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, Openings, SumcheckId,
};
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
use crate::framework::transcript::{ProverFs, VerifierFs};
use crate::zkvm::booleanity::{Booleanity, BooleanityParams};
use crate::zkvm::r1cs_witness::R1csWitness;
use crate::zkvm::spartan::stage::{prove_spartan, verify_spartan, SpartanProof, SpartanStageError};

const BOOLEANITY_DEGREE: usize = 3;

/// The binary-Spartan proof (Spartan stage + booleanity stage; grows as stages are wired).
#[derive(Clone, Debug)]
pub struct BinaryProof<F: Field> {
    pub spartan: SpartanProof<F>,
    /// `R1csAux(i)(ρ)` openings the booleanity reduction discharges against (PCS-opened at stage 8).
    pub aux_evals: Vec<F>,
}

/// Top-level prover: Fiat-Shamir-threaded Spartan stage → booleanity stage.
pub fn prove_binary<F, T>(
    witness: &R1csWitness<F>,
    key: &R1csKey<F>,
    transcript: &mut T,
) -> BinaryProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let mut accumulator = Openings::<F>::new(witness.log_num_cycles);

    let spartan = prove_spartan(witness, key, &mut accumulator, transcript);

    let aux_cols = witness.boolean_aux_columns();
    let n_aux = aux_cols.len();
    let r_bool = transcript.challenge_vector(witness.log_num_cycles);
    let bparams = BooleanityParams::new(r_bool, n_aux, transcript);
    let mut booleanity = Booleanity::new_prover(bparams, aux_cols);
    let _ = prove(&mut booleanity, &mut accumulator, transcript);

    let aux_evals = (0..n_aux)
        .map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::R1csAux(i),
                    SumcheckId::Booleanity,
                )
                .1
        })
        .collect();

    BinaryProof { spartan, aux_evals }
}

/// Top-level verifier (mirror of [`prove_binary`]).
pub fn verify_binary<F, T>(
    proof: &BinaryProof<F>,
    key: &R1csKey<F>,
    num_row_vars: usize,
    log_num_cycles: usize,
    transcript: &mut T,
) -> Result<(), SpartanStageError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let mut accumulator = Openings::<F>::new(log_num_cycles);

    verify_spartan(
        &proof.spartan,
        key,
        num_row_vars,
        &mut accumulator,
        transcript,
    )?;

    let n_aux = proof.aux_evals.len();
    let r_bool = transcript.challenge_vector(log_num_cycles);
    let bparams = BooleanityParams::new(r_bool, n_aux, transcript);
    let booleanity = Booleanity::new_verifier(bparams);
    let bclaim = SumcheckClaim {
        num_vars: log_num_cycles,
        degree: BOOLEANITY_DEGREE,
        claimed_sum: F::zero(),
    };
    let EvaluationClaim { point, value } =
        verify(&bclaim, transcript).map_err(|_| SpartanStageError::Sumcheck)?;

    let rho = booleanity.normalize_opening_point(&point);
    for (i, &eval) in proof.aux_evals.iter().enumerate() {
        accumulator.append_dense(
            CommittedPolynomial::R1csAux(i),
            SumcheckId::Booleanity,
            rho.clone(),
            eval,
        );
    }
    if value != booleanity.expected_output_claim(&accumulator, &point) {
        return Err(SpartanStageError::InnerClaim);
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::r1cs::rv64_limbed_constraints;
    use crate::zkvm::r1cs_witness::{build_limbed_z, tests_support::MockCycle};
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    fn witness_and_key(trace: &[MockCycle]) -> (R1csWitness<F>, R1csKey<F>) {
        let pcs = vec![0u64; trace.len()];
        let per_cycle = build_limbed_z::<MockCycle, F>(trace, &pcs);
        let witness = R1csWitness::<F>::materialize(&per_cycle);
        let cycles_pad = 1usize << witness.log_num_cycles;
        let key = R1csKey::new(rv64_limbed_constraints::<F>(), cycles_pad);
        (witness, key)
    }

    /// The multi-stage binary driver round-trips on a real (mock) `CycleRow` trace: Spartan stage +
    /// booleanity stage, one shared transcript + accumulator, prove → verify.
    #[test]
    fn binary_driver_round_trip() {
        let trace = [
            MockCycle::add(0, 7, 11),
            MockCycle::add(4, 0xFFFF_FFFF, 1),
            MockCycle::add(8, 100, 40),
            MockCycle::noop_at(12),
        ];
        let (witness, key) = witness_and_key(&trace);
        assert!(witness.is_satisfied(), "witness must satisfy the R1CS");

        let mut prover_t = ProverTranscript::new("binary-driver");
        let proof = prove_binary(&witness, &key, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("binary-driver", &narg);
        verify_binary(
            &proof,
            &key,
            witness.num_row_vars(),
            witness.log_num_cycles,
            &mut verifier_t,
        )
        .expect("binary driver must verify");
    }

    /// Tampering a booleanity opening (a corrupted `R1csAux` eval) is rejected.
    #[test]
    fn tampered_aux_rejected() {
        let trace = [MockCycle::add(0, 3, 5), MockCycle::noop_at(4)];
        let (witness, key) = witness_and_key(&trace);
        let mut prover_t = ProverTranscript::new("binary-driver");
        let mut proof = prove_binary(&witness, &key, &mut prover_t);
        let narg = prover_t.into_proof();
        // Tamper to a NON-boolean value (2) so `b² − b ≠ 0` — flipping to 1 would stay boolean.
        proof.aux_evals[0] += F::from_u64(2);

        let mut verifier_t = VerifierTranscript::new("binary-driver", &narg);
        assert!(
            verify_binary(
                &proof,
                &key,
                witness.num_row_vars(),
                witness.log_num_cycles,
                &mut verifier_t,
            )
            .is_err(),
            "tampered R1csAux eval must be rejected",
        );
    }
}
