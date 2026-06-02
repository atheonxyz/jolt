//! Registers **stage**: the three register sumchecks wired into one prove/verify pipeline on a
//! shared transcript + opening accumulator —
//! `RegistersClaimReduction → RegistersReadWriteChecking → RegistersValEvaluation`. Given the
//! materialized [`RegisterWitness`], this is the registers portion of the M8 stage driver.
//!
//! ## Interim binary-Spartan seeding (documented soundness gap — see `PHASE3_REVIEW_GUIDE.md` §7)
//!
//! `RegistersClaimReduction` reads its input register-value claims (`RdWriteValue`/`Rs1Value`/
//! `Rs2Value`) from [`SumcheckId::SpartanOuter`] — openings the **uni-skip** Spartan outer would emit
//! but the **binary** Spartan does not. In this interim path the driver **seeds them directly from
//! the materialized witness** (their MLE at a fresh cycle point `r_spartan`) and **carries the seed
//! values in the proof** (the verifier has no witness). This is **not yet bound** — the binding
//! arrives with the deferred uni-skip Spartan (task #6). For the witness-level equivalence gate the
//! seeded claims are the correct values; full soundness is the uni-skip pass.
//!
//! Flow: draw `r_spartan`; seed the four SpartanOuter register openings; reduce them to one point
//! (`RegistersClaimReduction`); prove read/write consistency (`RegistersReadWriteChecking`) →
//! `RegistersVal(r_address ‖ r_cycle')` + `Rs1Ra`/`Rs2Ra`/`RdWa`/`RdInc`; materialize `wa(r_address,·)`
//! and prove the value evolution (`RegistersValEvaluation`) → `RdInc`/`RdWa` at the reduced cycle.

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim, SumcheckProof};
use jolt_transcript::Transcript;

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
};
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
use crate::zkvm::claim_reductions::{RegistersClaimReduction, RegistersClaimReductionParams};
use crate::zkvm::registers::read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteCheckingParams,
};
use crate::zkvm::registers::val_evaluation::{
    RegistersValEvaluation, RegistersValEvaluationParams,
};
use crate::zkvm::registers::witness::RegisterWitness;

const CLAIM_REDUCTION_DEGREE: usize = 2;
const RW_DEGREE: usize = 3;
const VAL_EVAL_DEGREE: usize = 3;

/// Registers-stage verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistersStageError {
    Sumcheck,
    ClaimReductionClaim,
    ReadWriteClaim,
    ValEvaluationClaim,
}

/// The registers-stage proof: three sumcheck transcripts + the opening claims the verifier
/// discharges (the interim SpartanOuter seeds + each stage's cached openings — the committed `RdInc`
/// openings are PCS-opened at stage 8).
#[derive(Clone, Debug)]
pub struct RegistersStageProof<F: Field> {
    /// Interim SpartanOuter seeds `(RdWriteValue, Rs1Value, Rs2Value)(r_spartan)` (fork 2).
    pub spartan_seeds: [F; 3],
    pub claim_reduction: SumcheckProof<F>,
    /// `(RdWriteValue, Rs1Value, Rs2Value)` @ RegistersClaimReduction.
    pub cr_openings: [F; 3],
    pub rw_checking: SumcheckProof<F>,
    /// `(RegistersVal, Rs1Ra, Rs2Ra, RdWa, RdInc)` @ RegistersReadWriteChecking.
    pub rw_openings: [F; 5],
    pub val_evaluation: SumcheckProof<F>,
    /// `(RdInc, RdWa)` @ RegistersValEvaluation.
    pub ve_openings: [F; 2],
}

fn mle<F: Field>(col: &[F], eq: &[F]) -> F {
    col.iter()
        .zip(eq.iter())
        .fold(F::zero(), |a, (x, e)| a + *x * *e)
}

/// Seed the four SpartanOuter register openings at `r_spartan` from the witness value columns.
/// `LookupOutput`'s value is unused (only its point is read for `r_spartan`); the three register
/// values are the interim seeds.
fn seed_spartan_outer<F: Field>(accumulator: &mut Openings<F>, r_spartan: &[F], seeds: [F; 3]) {
    let point = OpeningPoint::new(r_spartan.to_vec());
    accumulator.append_virtual(
        VirtualPolynomial::LookupOutput,
        SumcheckId::SpartanOuter,
        point.clone(),
        F::zero(),
    );
    for (poly, value) in [
        VirtualPolynomial::RdWriteValue,
        VirtualPolynomial::Rs1Value,
        VirtualPolynomial::Rs2Value,
    ]
    .into_iter()
    .zip(seeds)
    {
        accumulator.append_virtual(poly, SumcheckId::SpartanOuter, point.clone(), value);
    }
}

/// Prove the registers stage from the materialized witness, threading `accumulator`/`transcript`.
pub fn prove_registers<F, T>(
    reg: &RegisterWitness<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> RegistersStageProof<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let r_spartan = transcript.challenge_vector(reg.log_t);
    let eq_spartan = EqPolynomial::<F>::evals(&r_spartan, None);
    let spartan_seeds = [
        mle(&reg.rd_write_value, &eq_spartan),
        mle(&reg.rs1_value, &eq_spartan),
        mle(&reg.rs2_value, &eq_spartan),
    ];
    seed_spartan_outer(accumulator, &r_spartan, spartan_seeds);

    // Stage 1: reduce the three SpartanOuter claims to one point.
    let cr_params = RegistersClaimReductionParams::new(reg.log_t, accumulator, transcript);
    let mut cr = RegistersClaimReduction::new_prover(
        cr_params,
        reg.rd_write_value.clone(),
        reg.rs1_value.clone(),
        reg.rs2_value.clone(),
    );
    let (claim_reduction, _) = prove(&mut cr, accumulator, transcript);
    let cr_openings = read_virtual(
        accumulator,
        SumcheckId::RegistersClaimReduction,
        &[
            VirtualPolynomial::RdWriteValue,
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
        ],
    );

    // Stage 2: read/write consistency over the full K·T matrices.
    let rw_params = RegistersReadWriteCheckingParams::new(accumulator, reg.log_k, transcript);
    let mut rw = RegistersReadWriteChecking::new_prover(
        rw_params,
        reg.ra1.clone(),
        reg.ra2.clone(),
        reg.wa.clone(),
        reg.val.clone(),
        reg.inc.clone(),
    );
    let (rw_checking, _) = prove(&mut rw, accumulator, transcript);
    let rw_openings = [
        get_virtual(
            accumulator,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        ),
        get_virtual(
            accumulator,
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        get_virtual(
            accumulator,
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        get_virtual(
            accumulator,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        ),
        get_committed(
            accumulator,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        ),
    ];

    // Stage 3: value evolution. `wa(r_address, ·)` is materialized now that r_address is known.
    let (val_point, _) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RegistersVal,
        SumcheckId::RegistersReadWriteChecking,
    );
    let (r_address, _) = val_point.split_at(reg.log_k);
    let wa_col = wa_at_address(reg, &r_address.r);
    let ve_params = RegistersValEvaluationParams::new(accumulator, reg.log_k);
    let mut ve = RegistersValEvaluation::new_prover(ve_params, reg.inc.clone(), wa_col);
    let (val_evaluation, _) = prove(&mut ve, accumulator, transcript);
    let ve_openings = [
        get_committed(
            accumulator,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        ),
        get_virtual(
            accumulator,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        ),
    ];

    RegistersStageProof {
        spartan_seeds,
        claim_reduction,
        cr_openings,
        rw_checking,
        rw_openings,
        val_evaluation,
        ve_openings,
    }
}

/// Verify the registers stage (mirror of [`prove_registers`]).
pub fn verify_registers<F, T>(
    proof: &RegistersStageProof<F>,
    log_t: usize,
    log_k: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), RegistersStageError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let r_spartan = transcript.challenge_vector(log_t);
    seed_spartan_outer(accumulator, &r_spartan, proof.spartan_seeds);

    // Stage 1.
    let cr_params = RegistersClaimReductionParams::new(log_t, accumulator, transcript);
    let cr = RegistersClaimReduction::new_verifier(cr_params.clone());
    let [s_rd, s_rs1, s_rs2] = proof.spartan_seeds;
    let cr_input = s_rd + cr_params.gamma * s_rs1 + cr_params.gamma_sqr * s_rs2;
    let cr_eval = verify(
        &SumcheckClaim {
            num_vars: log_t,
            degree: CLAIM_REDUCTION_DEGREE,
            claimed_sum: cr_input,
        },
        &proof.claim_reduction,
        transcript,
    )
    .map_err(|_| RegistersStageError::Sumcheck)?;
    seed_virtual(
        accumulator,
        &cr_eval,
        SumcheckId::RegistersClaimReduction,
        &[
            VirtualPolynomial::RdWriteValue,
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
        ],
        &proof.cr_openings,
    );
    if cr_eval.value != cr.expected_output_claim(accumulator, &cr_eval.point) {
        return Err(RegistersStageError::ClaimReductionClaim);
    }

    // Stage 2.
    let rw_params = RegistersReadWriteCheckingParams::new(accumulator, log_k, transcript);
    let rw = RegistersReadWriteChecking::new_verifier(rw_params.clone());
    let [cr_rd, cr_rs1, cr_rs2] = proof.cr_openings;
    let rw_input = cr_rd + rw_params.gamma * (cr_rs1 + rw_params.gamma * cr_rs2);
    let rw_eval = verify(
        &SumcheckClaim {
            num_vars: log_k + log_t,
            degree: RW_DEGREE,
            claimed_sum: rw_input,
        },
        &proof.rw_checking,
        transcript,
    )
    .map_err(|_| RegistersStageError::Sumcheck)?;
    let rw_point = rw.normalize_opening_point(&rw_eval.point);
    let (_, rw_cycle) = rw_point.split_at(log_k);
    accumulator.append_virtual(
        VirtualPolynomial::RegistersVal,
        SumcheckId::RegistersReadWriteChecking,
        rw_point.clone(),
        proof.rw_openings[0],
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs1Ra,
        SumcheckId::RegistersReadWriteChecking,
        rw_point.clone(),
        proof.rw_openings[1],
    );
    accumulator.append_virtual(
        VirtualPolynomial::Rs2Ra,
        SumcheckId::RegistersReadWriteChecking,
        rw_point.clone(),
        proof.rw_openings[2],
    );
    accumulator.append_virtual(
        VirtualPolynomial::RdWa,
        SumcheckId::RegistersReadWriteChecking,
        rw_point,
        proof.rw_openings[3],
    );
    accumulator.append_dense(
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersReadWriteChecking,
        rw_cycle,
        proof.rw_openings[4],
    );
    if rw_eval.value != rw.expected_output_claim(accumulator, &rw_eval.point) {
        return Err(RegistersStageError::ReadWriteClaim);
    }

    // Stage 3.
    let ve_params = RegistersValEvaluationParams::new(accumulator, log_k);
    let ve = RegistersValEvaluation::new_verifier(ve_params);
    let ve_eval = verify(
        &SumcheckClaim {
            num_vars: log_t,
            degree: VAL_EVAL_DEGREE,
            claimed_sum: proof.rw_openings[0],
        },
        &proof.val_evaluation,
        transcript,
    )
    .map_err(|_| RegistersStageError::Sumcheck)?;
    let ve_cycle = ve.normalize_opening_point(&ve_eval.point);
    accumulator.append_dense(
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersValEvaluation,
        ve_cycle.clone(),
        proof.ve_openings[0],
    );
    let (r_addr_ve, _) = {
        let (rval_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        rval_point.split_at(log_k)
    };
    let rd_wa_point = OpeningPoint::new([r_addr_ve.r.as_slice(), ve_cycle.r.as_slice()].concat());
    accumulator.append_virtual(
        VirtualPolynomial::RdWa,
        SumcheckId::RegistersValEvaluation,
        rd_wa_point,
        proof.ve_openings[1],
    );
    if ve_eval.value != ve.expected_output_claim(accumulator, &ve_eval.point) {
        return Err(RegistersStageError::ValEvaluationClaim);
    }
    Ok(())
}

/// `wa(r_address, j) = Σ_k eq(r_address, k)·wa[k·T + j]` — the write-address indicator MLE per cycle,
/// materialized once `r_address` is known (from the read-write-checking opening point).
fn wa_at_address<F: Field>(reg: &RegisterWitness<F>, r_address: &[F]) -> Vec<F> {
    let t = 1usize << reg.log_t;
    let k = 1usize << reg.log_k;
    let eq_addr = EqPolynomial::<F>::evals(r_address, None);
    (0..t)
        .map(|j| (0..k).fold(F::zero(), |acc, kk| acc + eq_addr[kk] * reg.wa[kk * t + j]))
        .collect()
}

fn read_virtual<F: Field>(
    acc: &Openings<F>,
    sumcheck: SumcheckId,
    polys: &[VirtualPolynomial],
) -> [F; 3] {
    std::array::from_fn(|i| get_virtual(acc, polys[i], sumcheck))
}

fn get_virtual<F: Field>(acc: &Openings<F>, poly: VirtualPolynomial, sumcheck: SumcheckId) -> F {
    acc.get_virtual_polynomial_opening(poly, sumcheck).1
}

fn get_committed<F: Field>(
    acc: &Openings<F>,
    poly: CommittedPolynomial,
    sumcheck: SumcheckId,
) -> F {
    acc.get_committed_polynomial_opening(poly, sumcheck).1
}

fn seed_virtual<F: Field>(
    acc: &mut Openings<F>,
    eval: &EvaluationClaim<F>,
    sumcheck: SumcheckId,
    polys: &[VirtualPolynomial],
    values: &[F],
) {
    let point = OpeningPoint::new(eval.point.iter().rev().copied().collect());
    for (poly, &value) in polys.iter().zip(values) {
        acc.append_virtual(*poly, sumcheck, point.clone(), value);
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::zkvm::r1cs_witness::tests_support::MockCycle;
    use crate::zkvm::registers::witness::register_witness;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_transcript::Blake2bTranscript;

    fn sample_trace() -> Vec<MockCycle> {
        vec![
            MockCycle::add(0, 0, 0).with_rd(3, 0, 5),
            MockCycle::add(4, 0, 0)
                .with_reads(Some(3), Some(0))
                .with_rd(4, 0, 9),
            MockCycle::add(8, 0, 0)
                .with_reads(Some(3), Some(4))
                .with_rd(3, 5, 14),
            MockCycle::noop_at(12),
        ]
    }

    #[test]
    fn registers_stage_round_trip() {
        let trace = sample_trace();
        let reg = register_witness::<MockCycle, F>(&trace, 8);

        let mut prover_acc = Openings::<F>::new(reg.log_t);
        let mut prover_t = Blake2bTranscript::<F>::new(b"registers-stage");
        let proof = prove_registers(&reg, &mut prover_acc, &mut prover_t);

        let mut verifier_acc = Openings::<F>::new(reg.log_t);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"registers-stage");
        verify_registers(
            &proof,
            reg.log_t,
            reg.log_k,
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("registers stage must verify");
    }

    #[test]
    fn tampered_rw_opening_rejected() {
        let trace = sample_trace();
        let reg = register_witness::<MockCycle, F>(&trace, 8);
        let mut prover_acc = Openings::<F>::new(reg.log_t);
        let mut prover_t = Blake2bTranscript::<F>::new(b"registers-stage");
        let mut proof = prove_registers(&reg, &mut prover_acc, &mut prover_t);
        // Corrupt the cached RegistersVal opening (rw_openings[0]) → the read-write claim check fails.
        proof.rw_openings[0] += F::from_u64(1);

        let mut verifier_acc = Openings::<F>::new(reg.log_t);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"registers-stage");
        assert!(
            verify_registers(
                &proof,
                reg.log_t,
                reg.log_k,
                &mut verifier_acc,
                &mut verifier_t
            )
            .is_err(),
            "tampered RegistersVal opening must be rejected"
        );
    }
}
