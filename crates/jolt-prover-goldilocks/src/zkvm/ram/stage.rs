//! RAM **stage**: the four RAM sumchecks wired into one prove/verify pipeline. The first three —
//! `RamReadWriteChecking`, `RamRafEvaluation`, `RamOutputCheck` — are run as a **front-loaded
//! batched** sumcheck so their address rounds bind in lockstep and share one `r_address`; then
//! `RamValCheck` runs sequentially, consuming `RamVal`(from RW) and `RamValFinal`(from output-check)
//! at that aligned `r_address`.
//!
//! ## Why batched (the alignment, see `PHASE3_REVIEW_GUIDE.md` §7)
//!
//! `RamValCheck` batches two value identities and reads `RamVal@RamReadWriteChecking` **and**
//! `RamValFinal@RamOutputCheck` — both must be at the **same** `r_address`. `RamReadWriteChecking`
//! binds `(cycle ‖ address)` low→high (length `log_K+log_T`); `RamRafEvaluation`/`RamOutputCheck`
//! (length `log_K`, `round_offset = log_T`) bind in rounds `[log_T, log_T+log_K)` — exactly RW's
//! address rounds. So `reverse(challenges[log_T..])` is the shared `r_address` across all three, and
//! `RamValCheck` consumes both openings consistently. The front-loaded
//! [`prove_batched`](crate::framework::sumcheck::prove_batched) provides this lockstep.
//!
//! ## Interim binary-Spartan seeding (fork 2)
//!
//! `RamReadWriteChecking`/`RamRafEvaluation` read `RamReadValue`/`RamWriteValue`/`RamAddress` from
//! [`SumcheckId::SpartanOuter`] — openings binary Spartan does not emit. As with the registers stage
//! they are **interim-seeded** from the witness at a fresh `r_spartan` and **carried in the proof**;
//! the binding is the deferred uni-skip Spartan (task #6).

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
};
use crate::framework::sumcheck::{prove, prove_batched, verify, verify_batched, SumcheckInstance};
use crate::framework::transcript::{ProverFs, VerifierFs};
use crate::zkvm::ram::output_check::{RamOutputCheck, RamOutputCheckParams};
use crate::zkvm::ram::raf_evaluation::{RamRafEvaluation, RamRafEvaluationParams};
use crate::zkvm::ram::read_write_checking::{RamReadWriteChecking, RamReadWriteCheckingParams};
use crate::zkvm::ram::val_check::{RamValCheck, RamValCheckParams};
use crate::zkvm::ram::witness::RamWitness;

const RW_DEGREE: usize = 3;
const RAF_DEGREE: usize = 2;
const OC_DEGREE: usize = 3;
const VAL_CHECK_DEGREE: usize = 3;

/// RAM-stage verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RamStageError {
    Sumcheck,
    BatchedClaim,
    ValCheckClaim,
}

/// The RAM-stage opening claims the verifier discharges (the committed `RamInc` openings are
/// PCS-opened at stage 8). The batched (RW+RAF+OutputCheck) and val-check round polynomials live in
/// the shared NARG, not here.
#[derive(Clone, Debug)]
pub struct RamStageProof<F: Field> {
    /// Interim SpartanOuter seeds `(RamReadValue, RamWriteValue, RamAddress)(r_spartan)`.
    pub spartan_seeds: [F; 3],
    /// `(RamVal, RamRa, RamInc)` @ RamReadWriteChecking.
    pub rw_openings: [F; 3],
    /// `RamRa` @ RamRafEvaluation.
    pub raf_opening: F,
    /// `RamValFinal` @ RamOutputCheck.
    pub oc_opening: F,
    /// `(RamRa, RamInc)` @ RamValCheck.
    pub vc_openings: [F; 2],
}

fn mle<F: Field>(col: &[F], eq: &[F]) -> F {
    col.iter()
        .zip(eq.iter())
        .fold(F::zero(), |a, (x, e)| a + *x * *e)
}

/// `Σ_k eq(r_address, k) · M[k·T + j]` per cycle `j` — collapse an address-major `K·T` matrix to a
/// length-`T` column at the address point. Used for both `wa(r_address, ·)` (val-check) and seeds.
fn at_address<F: Field>(matrix: &[F], r_address: &[F], log_t: usize, log_k: usize) -> Vec<F> {
    let t = 1usize << log_t;
    let k = 1usize << log_k;
    let eq = EqPolynomial::<F>::evals(r_address, None);
    (0..t)
        .map(|j| (0..k).fold(F::zero(), |acc, kk| acc + eq[kk] * matrix[kk * t + j]))
        .collect()
}

/// `ra_count[k] = Σ_j eq(r_cycle, j) · ra[k·T + j]` — the per-address access count weighted by the
/// cycle eq (the RAF instance's `ra` column).
fn ra_count<F: Field>(ra: &[F], eq_cycle: &[F], log_t: usize, log_k: usize) -> Vec<F> {
    let t = 1usize << log_t;
    let k = 1usize << log_k;
    (0..k)
        .map(|kk| (0..t).fold(F::zero(), |acc, j| acc + eq_cycle[j] * ra[kk * t + j]))
        .collect()
}

/// Per-cycle original RAM address `Σ_k ra[k·T+j]·unmap[k]` (0 for non-access cycles).
fn orig_address<F: Field>(ra: &[F], unmap: &[F], log_t: usize, log_k: usize) -> Vec<F> {
    let t = 1usize << log_t;
    let k = 1usize << log_k;
    (0..t)
        .map(|j| (0..k).fold(F::zero(), |acc, kk| acc + ra[kk * t + j] * unmap[kk]))
        .collect()
}

/// Prove the RAM stage. `unmap`/`val_io`/`io_mask` are the public columns (length `K`): `unmap` is
/// the remap inverse (affine), `io_mask` the {0,1} I/O-region indicator, `val_io` the public output.
pub fn prove_ram<F, T>(
    reg: &RamWitness<F>,
    unmap: &[F],
    val_io: &[F],
    io_mask: &[F],
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> RamStageProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let (log_t, log_k) = (reg.log_t, reg.log_k);
    let r_spartan = transcript.challenge_vector(log_t);
    let eq_spartan = EqPolynomial::<F>::evals(&r_spartan, None);
    let orig_addr = orig_address(&reg.ra, unmap, log_t, log_k);
    let spartan_seeds = [
        mle(&reg.ram_read_value, &eq_spartan),
        mle(&reg.ram_write_value, &eq_spartan),
        mle(&orig_addr, &eq_spartan),
    ];
    seed_spartan_outer(accumulator, &r_spartan, spartan_seeds);

    // Fresh eq-weight point for the output-check zero-check (Schwartz-Zippel).
    let r_address_weight = transcript.challenge_vector(log_k);

    let rw_params = RamReadWriteCheckingParams::new(accumulator, log_k, transcript);
    let mut rw = RamReadWriteChecking::new_prover(
        rw_params,
        reg.ra.clone(),
        reg.val.clone(),
        reg.inc.clone(),
    );
    let raf_params = RamRafEvaluationParams::new(accumulator, log_k);
    let mut raf = RamRafEvaluation::new_prover(
        raf_params,
        ra_count(&reg.ra, &eq_spartan, log_t, log_k),
        unmap.to_vec(),
    );
    let oc_params = RamOutputCheckParams::new(r_address_weight);
    let mut oc = RamOutputCheck::new_prover(
        oc_params,
        reg.val_final.clone(),
        val_io.to_vec(),
        io_mask.to_vec(),
    );

    let instances: Vec<&mut dyn SumcheckInstance<F>> = vec![&mut rw, &mut raf, &mut oc];
    let _ = prove_batched(instances, accumulator, transcript);

    let rw_openings = [
        get_virtual(
            accumulator,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        ),
        get_virtual(
            accumulator,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        ),
        get_committed(
            accumulator,
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        ),
    ];
    let raf_opening = get_virtual(
        accumulator,
        VirtualPolynomial::RamRa,
        SumcheckId::RamRafEvaluation,
    );
    let oc_opening = get_virtual(
        accumulator,
        VirtualPolynomial::RamValFinal,
        SumcheckId::RamOutputCheck,
    );

    // Sequential val-check at the aligned r_address.
    let (val_point, _) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RamVal,
        SumcheckId::RamReadWriteChecking,
    );
    let (r_address, _) = val_point.split_at(log_k);
    let wa_col = at_address(&reg.ra, &r_address.r, log_t, log_k);
    let initial_ram = vec![F::zero(); 1 << log_k];
    let vc_params = RamValCheckParams::new(accumulator, log_k, &initial_ram, transcript);
    let mut vc = RamValCheck::new_prover(vc_params, reg.inc.clone(), wa_col);
    let _ = prove(&mut vc, accumulator, transcript);
    let vc_openings = [
        get_virtual(
            accumulator,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
        ),
        get_committed(
            accumulator,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
        ),
    ];

    RamStageProof {
        spartan_seeds,
        rw_openings,
        raf_opening,
        oc_opening,
        vc_openings,
    }
}

/// Verify the RAM stage (mirror of [`prove_ram`]). The public columns must match the prover's.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors prove_ram: proof + geometry (log_t/log_k) + the three public columns + acc/transcript"
)]
pub fn verify_ram<F, T>(
    proof: &RamStageProof<F>,
    log_t: usize,
    log_k: usize,
    unmap: &[F],
    val_io: &[F],
    io_mask: &[F],
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), RamStageError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let r_spartan = transcript.challenge_vector(log_t);
    seed_spartan_outer(accumulator, &r_spartan, proof.spartan_seeds);
    let r_address_weight = transcript.challenge_vector(log_k);

    let rw_params = RamReadWriteCheckingParams::new(accumulator, log_k, transcript);
    let rw = RamReadWriteChecking::new_verifier(rw_params.clone());
    let raf_params = RamRafEvaluationParams::new(accumulator, log_k);
    let raf = RamRafEvaluation::new_verifier(raf_params.clone(), unmap.to_vec());
    let oc_params = RamOutputCheckParams::new(r_address_weight);
    let oc = RamOutputCheck::new_verifier(oc_params, val_io.to_vec(), io_mask.to_vec());

    let [seed_rv, seed_wv, seed_raf] = proof.spartan_seeds;
    let claims = [
        SumcheckClaim {
            num_vars: log_k + log_t,
            degree: RW_DEGREE,
            claimed_sum: seed_rv + rw_params.gamma * seed_wv,
        },
        SumcheckClaim {
            num_vars: log_k,
            degree: RAF_DEGREE,
            claimed_sum: seed_raf,
        },
        SumcheckClaim {
            num_vars: log_k,
            degree: OC_DEGREE,
            claimed_sum: F::zero(),
        },
    ];
    let (
        EvaluationClaim {
            point: challenges,
            value,
        },
        coeffs,
    ) = verify_batched(&claims, transcript).map_err(|_| RamStageError::Sumcheck)?;

    // Seed the batched stages' cached openings at their (aligned) points.
    let rw_point = rw.normalize_opening_point(&challenges);
    let (r_address, rw_cycle) = rw_point.split_at(log_k);
    accumulator.append_virtual(
        VirtualPolynomial::RamVal,
        SumcheckId::RamReadWriteChecking,
        rw_point.clone(),
        proof.rw_openings[0],
    );
    accumulator.append_virtual(
        VirtualPolynomial::RamRa,
        SumcheckId::RamReadWriteChecking,
        rw_point,
        proof.rw_openings[1],
    );
    accumulator.append_dense(
        CommittedPolynomial::RamInc,
        SumcheckId::RamReadWriteChecking,
        rw_cycle,
        proof.rw_openings[2],
    );
    // RAF caches RamRa at (r_address ‖ r_spartan) (its r_cycle is the RamAddress seed point).
    let raf_point = OpeningPoint::new([r_address.r.as_slice(), r_spartan.as_slice()].concat());
    accumulator.append_virtual(
        VirtualPolynomial::RamRa,
        SumcheckId::RamRafEvaluation,
        raf_point,
        proof.raf_opening,
    );
    accumulator.append_virtual(
        VirtualPolynomial::RamValFinal,
        SumcheckId::RamOutputCheck,
        r_address.clone(),
        proof.oc_opening,
    );

    let rw_slice = &challenges[..];
    let raf_slice = &challenges[log_t..];
    let oc_slice = &challenges[log_t..];
    let combined = coeffs[0] * rw.expected_output_claim(accumulator, rw_slice)
        + coeffs[1] * raf.expected_output_claim(accumulator, raf_slice)
        + coeffs[2] * oc.expected_output_claim(accumulator, oc_slice);
    if value != combined {
        return Err(RamStageError::BatchedClaim);
    }

    // Sequential val-check at the aligned r_address.
    let initial_ram = vec![F::zero(); 1 << log_k];
    let vc_params = RamValCheckParams::new(accumulator, log_k, &initial_ram, transcript);
    let vc = RamValCheck::new_verifier(vc_params.clone());
    let vc_input = (proof.rw_openings[0] - vc_params.init_eval)
        + vc_params.gamma * (proof.oc_opening - vc_params.init_eval);
    let vc_eval = verify(
        &SumcheckClaim {
            num_vars: log_t,
            degree: VAL_CHECK_DEGREE,
            claimed_sum: vc_input,
        },
        transcript,
    )
    .map_err(|_| RamStageError::Sumcheck)?;
    let vc_cycle = vc.normalize_opening_point(&vc_eval.point);
    let vc_ra_point = OpeningPoint::new([r_address.r.as_slice(), vc_cycle.r.as_slice()].concat());
    accumulator.append_virtual(
        VirtualPolynomial::RamRa,
        SumcheckId::RamValCheck,
        vc_ra_point,
        proof.vc_openings[0],
    );
    accumulator.append_dense(
        CommittedPolynomial::RamInc,
        SumcheckId::RamValCheck,
        vc_cycle,
        proof.vc_openings[1],
    );
    if vc_eval.value != vc.expected_output_claim(accumulator, &vc_eval.point) {
        return Err(RamStageError::ValCheckClaim);
    }
    Ok(())
}

/// Seed the three SpartanOuter RAM openings at `r_spartan` from the witness columns.
fn seed_spartan_outer<F: Field>(accumulator: &mut Openings<F>, r_spartan: &[F], seeds: [F; 3]) {
    let point = OpeningPoint::new(r_spartan.to_vec());
    for (poly, value) in [
        VirtualPolynomial::RamReadValue,
        VirtualPolynomial::RamWriteValue,
        VirtualPolynomial::RamAddress,
    ]
    .into_iter()
    .zip(seeds)
    {
        accumulator.append_virtual(poly, SumcheckId::SpartanOuter, point.clone(), value);
    }
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

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::zkvm::r1cs_witness::tests_support::MockCycle;
    use crate::zkvm::ram::witness::ram_witness;
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    fn sample_trace() -> Vec<MockCycle> {
        vec![
            MockCycle::add(0, 0, 0).with_ram(2, 0, 5),
            MockCycle::add(4, 0, 0).with_ram(2, 5, 5),
            MockCycle::add(8, 0, 0).with_ram(5, 0, 9),
            MockCycle::add(12, 0, 0).with_ram(2, 5, 14),
            MockCycle::noop_at(16),
        ]
    }

    /// Public columns: affine unmap, empty I/O region (io_mask = val_io = 0 → output-check is an
    /// honest trivial zero-check that still opens RamValFinal for the val-check).
    fn public_columns(log_k: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
        let k = 1usize << log_k;
        let unmap: Vec<F> = (0..k)
            .map(|i| F::from_u64(0x8000_0000 + i as u64))
            .collect();
        let zero = vec![F::from_u64(0); k];
        (unmap, zero.clone(), zero)
    }

    #[test]
    fn ram_stage_round_trip() {
        let trace = sample_trace();
        let w = ram_witness::<MockCycle, F>(&trace, 8);
        let (unmap, val_io, io_mask) = public_columns(w.log_k);

        let mut prover_acc = Openings::<F>::new(w.log_t);
        let mut prover_t = ProverTranscript::new("ram-stage");
        let proof = prove_ram(
            &w,
            &unmap,
            &val_io,
            &io_mask,
            &mut prover_acc,
            &mut prover_t,
        );
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(w.log_t);
        let mut verifier_t = VerifierTranscript::new("ram-stage", &narg);
        verify_ram(
            &proof,
            w.log_t,
            w.log_k,
            &unmap,
            &val_io,
            &io_mask,
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("RAM stage must verify");
    }

    #[test]
    fn tampered_rw_opening_rejected() {
        let trace = sample_trace();
        let w = ram_witness::<MockCycle, F>(&trace, 8);
        let (unmap, val_io, io_mask) = public_columns(w.log_k);
        let mut prover_acc = Openings::<F>::new(w.log_t);
        let mut prover_t = ProverTranscript::new("ram-stage");
        let mut proof = prove_ram(
            &w,
            &unmap,
            &val_io,
            &io_mask,
            &mut prover_acc,
            &mut prover_t,
        );
        let narg = prover_t.into_proof();
        // Corrupt RamVal@RW → both the batched RW claim and the val-check input break.
        proof.rw_openings[0] += F::from_u64(1);

        let mut verifier_acc = Openings::<F>::new(w.log_t);
        let mut verifier_t = VerifierTranscript::new("ram-stage", &narg);
        assert!(
            verify_ram(
                &proof,
                w.log_t,
                w.log_k,
                &unmap,
                &val_io,
                &io_mask,
                &mut verifier_acc,
                &mut verifier_t,
            )
            .is_err(),
            "tampered RamVal opening must be rejected"
        );
    }
}
