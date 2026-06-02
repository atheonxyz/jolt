//! Binary Spartan **stage**: the outer R1CS-satisfaction zero-check + the inner reduction wired into
//! one prove/verify pair. Given the materialized [`R1csWitness`] (`z` + `Az/Bz/Cz`) and the
//! preprocessed [`R1csKey`], this is the Spartan portion of the M8 stage driver — the template the
//! other stages follow (construct instances → `framework::sumcheck` prove/verify → thread the
//! accumulator + transcript).
//!
//! Flow: draw `τ`; prove the outer zero-check `0 = Σ_x eq(τ,x)(Az·Bz−Cz)` → `Az/Bz/Cz(r_x)`; draw
//! `ρ`; prove the inner reduction `ρ·(Az,Bz,Cz)(r_x) = Σ_y M(r_x,y)·z(y)` → `z(r_y)`. The verifier
//! mirrors, discharging the outer claim against `eq·(Az·Bz−Cz)` (from the sent `Az/Bz/Cz(r_x)`) and
//! the inner against `M(r_x,r_y)·z(r_y)` (via [`R1csKey::evaluate_matrix_mles`]). `z(r_y)` is the
//! single committed-witness opening the stage-8 WHIR open discharges.

use jolt_field::Field;
use jolt_r1cs::R1csKey;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

use crate::framework::accumulator::{OpeningAccumulator, Openings, SumcheckId, VirtualPolynomial};
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
use crate::framework::transcript::{ProverFs, VerifierFs};
use crate::zkvm::r1cs_witness::R1csWitness;
use crate::zkvm::spartan::inner::{SpartanInner, SpartanInnerParams};
use crate::zkvm::spartan::outer::{SpartanOuter, SpartanOuterParams};

const OUTER_DEGREE: usize = 3;
const INNER_DEGREE: usize = 2;

/// Spartan-stage verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpartanStageError {
    /// A sub-sumcheck (outer or inner) was rejected by the workspace verifier.
    Sumcheck,
    /// The outer reduced claim did not equal `eq(τ, r_x)·(Az·Bz − Cz)(r_x)`.
    OuterClaim,
    /// The inner reduced claim did not equal `M(r_x, r_y)·z(r_y)`.
    InnerClaim,
}

/// The Spartan-stage opening claims the verifier discharges (`Az/Bz/Cz(r_x)` and the witness opening
/// `z(r_y)`, the latter PCS-opened at stage 8). The outer/inner sumcheck round polynomials live in
/// the shared NARG, not here.
#[derive(Clone, Debug)]
pub struct SpartanProof<F: Field> {
    pub az_rx: F,
    pub bz_rx: F,
    pub cz_rx: F,
    pub z_ry: F,
}

/// Prove the Spartan stage from the materialized witness + key, threading `accumulator`/`transcript`.
pub fn prove_spartan<F, T>(
    witness: &R1csWitness<F>,
    key: &R1csKey<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> SpartanProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let num_row_vars = witness.num_row_vars();
    let tau = transcript.challenge_vector(num_row_vars);

    let mut outer = SpartanOuter::new_prover(
        SpartanOuterParams::new(tau),
        witness.az.clone(),
        witness.bz.clone(),
        witness.cz.clone(),
    );
    let _ = prove(&mut outer, accumulator, transcript);

    let (r_x, az_rx) = accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
    let (_, bz_rx) = accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
    let (_, cz_rx) = accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

    let rho = transcript.challenge_vector(3);
    let params = SpartanInnerParams {
        key: key.clone(),
        r_x: r_x.r.clone(),
        rho: [rho[0], rho[1], rho[2]],
    };
    let mut inner = SpartanInner::new_prover(params, witness.z.clone());
    let _ = prove(&mut inner, accumulator, transcript);

    let (_, z_ry) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::SpartanWitnessZ,
        SumcheckId::SpartanInner,
    );

    SpartanProof {
        az_rx,
        bz_rx,
        cz_rx,
        z_ry,
    }
}

/// Verify the Spartan stage (mirror of [`prove_spartan`]).
pub fn verify_spartan<F, T>(
    proof: &SpartanProof<F>,
    key: &R1csKey<F>,
    num_row_vars: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), SpartanStageError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let tau = transcript.challenge_vector(num_row_vars);
    let outer = SpartanOuter::new_verifier(SpartanOuterParams::new(tau));
    let outer_claim = SumcheckClaim {
        num_vars: num_row_vars,
        degree: OUTER_DEGREE,
        claimed_sum: F::zero(),
    };
    let EvaluationClaim {
        point: outer_ch,
        value: outer_value,
    } = verify(&outer_claim, transcript).map_err(|_| SpartanStageError::Sumcheck)?;

    // Seed the sent Az/Bz/Cz(r_x) so the outer claim discharges.
    let r_x = outer.normalize_opening_point(&outer_ch);
    accumulator.append_virtual(
        VirtualPolynomial::SpartanAz,
        SumcheckId::SpartanOuter,
        r_x.clone(),
        proof.az_rx,
    );
    accumulator.append_virtual(
        VirtualPolynomial::SpartanBz,
        SumcheckId::SpartanOuter,
        r_x.clone(),
        proof.bz_rx,
    );
    accumulator.append_virtual(
        VirtualPolynomial::SpartanCz,
        SumcheckId::SpartanOuter,
        r_x.clone(),
        proof.cz_rx,
    );
    if outer_value != outer.expected_output_claim(accumulator, &outer_ch) {
        return Err(SpartanStageError::OuterClaim);
    }

    let rho = transcript.challenge_vector(3);
    let params = SpartanInnerParams {
        key: key.clone(),
        r_x: r_x.r.clone(),
        rho: [rho[0], rho[1], rho[2]],
    };
    let inner = SpartanInner::new_verifier(params);
    let inner_claim = SumcheckClaim {
        num_vars: key.num_col_vars(),
        degree: INNER_DEGREE,
        claimed_sum: rho[0] * proof.az_rx + rho[1] * proof.bz_rx + rho[2] * proof.cz_rx,
    };
    let EvaluationClaim {
        point: inner_ch,
        value: inner_value,
    } = verify(&inner_claim, transcript).map_err(|_| SpartanStageError::Sumcheck)?;

    let r_y = inner.normalize_opening_point(&inner_ch);
    accumulator.append_virtual(
        VirtualPolynomial::SpartanWitnessZ,
        SumcheckId::SpartanInner,
        r_y,
        proof.z_ry,
    );
    if inner_value != inner.expected_output_claim(accumulator, &inner_ch) {
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

    /// Build a satisfying witness from a small `MockCycle` trace + the R1csKey for it.
    fn witness_and_key(trace: &[MockCycle]) -> (R1csWitness<F>, R1csKey<F>) {
        let pcs = vec![0u64; trace.len()];
        let per_cycle = build_limbed_z::<MockCycle, F>(trace, &pcs);
        let witness = R1csWitness::<F>::materialize(&per_cycle);
        let cycles_pad = 1usize << witness.log_num_cycles;
        let key = R1csKey::new(rv64_limbed_constraints::<F>(), cycles_pad);
        (witness, key)
    }

    /// The Spartan stage round-trips on a real (mock) `CycleRow` trace: outer zero-check + inner
    /// reduction prove → verify, over a satisfying ADD/no-op witness.
    #[test]
    fn spartan_stage_round_trip() {
        let trace = [
            MockCycle::add(0, 7, 11),
            MockCycle::add(4, 1_000_000, 2_000_000),
            MockCycle::add(8, 0xFFFF_FFFF, 1),
            MockCycle::noop_at(12),
        ];
        let (witness, key) = witness_and_key(&trace);
        assert!(witness.is_satisfied(), "witness must satisfy the R1CS");

        let mut prover_acc = Openings::<F>::new(witness.log_num_cycles);
        let mut prover_t = ProverTranscript::new("spartan-stage");
        let proof = prove_spartan(&witness, &key, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(witness.log_num_cycles);
        let mut verifier_t = VerifierTranscript::new("spartan-stage", &narg);
        verify_spartan(
            &proof,
            &key,
            witness.num_row_vars(),
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("Spartan stage must verify");
    }

    /// Tampering the outer reduced claim (a corrupted `Az(r_x)`) is rejected.
    #[test]
    fn tampered_az_rejected() {
        let trace = [MockCycle::add(0, 3, 5), MockCycle::noop_at(4)];
        let (witness, key) = witness_and_key(&trace);
        let mut prover_acc = Openings::<F>::new(witness.log_num_cycles);
        let mut prover_t = ProverTranscript::new("spartan-stage");
        let mut proof = prove_spartan(&witness, &key, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();
        proof.az_rx += F::from_u64(1);

        let mut verifier_acc = Openings::<F>::new(witness.log_num_cycles);
        let mut verifier_t = VerifierTranscript::new("spartan-stage", &narg);
        assert!(
            verify_spartan(
                &proof,
                &key,
                witness.num_row_vars(),
                &mut verifier_acc,
                &mut verifier_t,
            )
            .is_err(),
            "tampered Az(r_x) must be rejected",
        );
    }
}
