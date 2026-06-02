//! Inc opening-reduction sumcheck — ported from jolt-core's
//! `zkvm/claim_reductions/increments.rs` to the framework ([`crate::framework`]) over the lean
//! `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Reduces the four `RamInc`/`RdInc` openings (emitted by RAM/register read-write checking +
//! val-evaluation at distinct cycle points) to a **single** opening of each at a shared point `ρ`:
//!
//! ```text
//! input:  v₁ = RamInc(r2), v₂ = RamInc(r4), w₁ = RdInc(s4), w₂ = RdInc(s5)
//! claim:  v₁ + γ·v₂ + γ²·w₁ + γ³·w₂
//! sumcheck (log T rounds, degree 2):
//!   Σ_j RamInc(j)·[eq(r2,j) + γ·eq(r4,j)]  +  γ²·Σ_j RdInc(j)·[eq(s4,j) + γ·eq(s5,j)]
//! output: RamInc(ρ), RdInc(ρ)
//! ```
//!
//! This is the **single-phase** form (the jolt-core prefix/suffix two-phase materialization is a
//! perf optimization deferred with the trace witness-gen). `RamInc`/`RdInc` are taken as
//! pre-materialized recomposed values (`Fp3`), decoupling the sumcheck from the trace →
//! signed-limb materialization (M8). The committed-limb opening of `ρ` is a stage-8 concern.

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// Fiat-Shamir + opening-point parameters, fetched from the accumulator (matches jolt-core
/// `IncClaimReductionSumcheckParams`).
#[derive(Clone, Debug)]
pub struct IncClaimReductionParams<F: Field> {
    /// `[γ, γ², γ³]`.
    pub gamma_powers: [F; 3],
    pub n_cycle_vars: usize,
    pub r_cycle_stage2: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>,
    pub s_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>,
    pub s_cycle_stage5: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: Field> IncClaimReductionParams<F> {
    pub fn new(
        n_cycle_vars: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Challenge<F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let gamma_sqr = gamma * gamma;
        let gamma_cub = gamma_sqr * gamma;

        let (r_cycle_stage2, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_cycle_stage4, _) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
        let (s_cycle_stage4, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (s_cycle_stage5, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );

        Self {
            gamma_powers: [gamma, gamma_sqr, gamma_cub],
            n_cycle_vars,
            r_cycle_stage2,
            r_cycle_stage4,
            s_cycle_stage4,
            s_cycle_stage5,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let [gamma, gamma_sqr, gamma_cub] = self.gamma_powers;
        let (_, v1) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, v2) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
        let (_, w1) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, w2) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );
        v1 + gamma * v2 + gamma_sqr * w1 + gamma_cub * w2
    }
}

/// Prover/verifier instance. Holds the materialized polynomials; the verifier instance carries the
/// same `params` and ignores the (empty) polynomials.
pub struct IncClaimReduction<F: Field> {
    pub params: IncClaimReductionParams<F>,
    ram_inc: MultilinearPolynomial<F>,
    rd_inc: MultilinearPolynomial<F>,
    /// `eq(r2,·) + γ·eq(r4,·)`.
    eq_ram: MultilinearPolynomial<F>,
    /// `eq(s4,·) + γ·eq(s5,·)`.
    eq_rd: MultilinearPolynomial<F>,
}

impl<F: Field> IncClaimReduction<F> {
    /// Build the prover instance from materialized `RamInc`/`RdInc` value columns.
    pub fn new_prover(params: IncClaimReductionParams<F>, ram_inc: Vec<F>, rd_inc: Vec<F>) -> Self {
        let gamma = params.gamma_powers[0];
        let eq_ram = combine_eq(&params.r_cycle_stage2.r, &params.r_cycle_stage4.r, gamma);
        let eq_rd = combine_eq(&params.s_cycle_stage4.r, &params.s_cycle_stage5.r, gamma);
        Self {
            params,
            ram_inc: MultilinearPolynomial::from(ram_inc),
            rd_inc: MultilinearPolynomial::from(rd_inc),
            eq_ram: MultilinearPolynomial::from(eq_ram),
            eq_rd: MultilinearPolynomial::from(eq_rd),
        }
    }

    /// Build a verifier instance (no polynomials needed; `expected_output_claim` reads cached
    /// openings + recomputes the eq factors).
    pub fn new_verifier(params: IncClaimReductionParams<F>) -> Self {
        Self {
            params,
            ram_inc: MultilinearPolynomial::from(vec![F::zero()]),
            rd_inc: MultilinearPolynomial::from(vec![F::zero()]),
            eq_ram: MultilinearPolynomial::from(vec![F::zero()]),
            eq_rd: MultilinearPolynomial::from(vec![F::zero()]),
        }
    }
}

/// `eq(a,·) + γ·eq(b,·)` as a dense column.
fn combine_eq<F: Field>(a: &[F], b: &[F], gamma: F) -> Vec<F> {
    let eq_a = EqPolynomial::<F>::evals(a, None);
    let eq_b = EqPolynomial::<F>::evals(b, None);
    eq_a.iter()
        .zip(eq_b.iter())
        .map(|(x, y)| *x + gamma * *y)
        .collect()
}

impl<F: Field> SumcheckInstance<F> for IncClaimReduction<F> {
    fn num_rounds(&self) -> usize {
        self.params.n_cycle_vars
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let gamma_sqr = self.params.gamma_powers[1];
        let half = self.ram_inc.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for j in 0..half {
            let ri = self
                .ram_inc
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let er = self
                .eq_ram
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let di = self
                .rd_inc
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let ed = self
                .eq_rd
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            for k in 0..3 {
                acc[k].fmadd(ri[k], er[k]);
                acc[k].fmadd(gamma_sqr * di[k], ed[k]);
            }
        }
        let evals: [F; 3] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.ram_inc.bind_parallel(r, BindingOrder::LowToHigh);
        self.rd_inc.bind_parallel(r, BindingOrder::LowToHigh);
        self.eq_ram.bind_parallel(r, BindingOrder::LowToHigh);
        self.eq_rd.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
            point.clone(),
            self.ram_inc.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
            point,
            self.rd_inc.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let [gamma, gamma_sqr, _] = self.params.gamma_powers;
        let point = self.normalize_opening_point(challenges);

        let eq_r2 = EqPolynomial::<F>::mle(&point.r, &self.params.r_cycle_stage2.r);
        let eq_r4 = EqPolynomial::<F>::mle(&point.r, &self.params.r_cycle_stage4.r);
        let eq_s4 = EqPolynomial::<F>::mle(&point.r, &self.params.s_cycle_stage4.r);
        let eq_s5 = EqPolynomial::<F>::mle(&point.r, &self.params.s_cycle_stage5.r);

        let (_, ram_inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        ram_inc_claim * (eq_r2 + gamma * eq_r4) + gamma_sqr * rd_inc_claim * (eq_s4 + gamma * eq_s5)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::sumcheck::{prove, verify};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<F> {
        (0..n).map(|_| F::from_u64(rng.next())).collect()
    }

    fn dot(poly: &[F], eq: &[F]) -> F {
        poly.iter()
            .zip(eq.iter())
            .fold(F::from_u64(0), |a, (p, e)| a + *p * *e)
    }

    fn round_trip(seed: u64, log_t: usize) {
        let mut rng = Rng(seed);
        let t = 1usize << log_t;
        let ram_inc = rand_vec(&mut rng, t);
        let rd_inc = rand_vec(&mut rng, t);
        let r2 = rand_vec(&mut rng, log_t);
        let r4 = rand_vec(&mut rng, log_t);
        let s4 = rand_vec(&mut rng, log_t);
        let s5 = rand_vec(&mut rng, log_t);

        // Pre-seed the four input openings (v1,v2,w1,w2) — RamInc(r2) = Σ_j RamInc[j]·eq(r2,j), etc.
        let v1 = dot(&ram_inc, &EqPolynomial::<F>::evals(&r2, None));
        let v2 = dot(&ram_inc, &EqPolynomial::<F>::evals(&r4, None));
        let w1 = dot(&rd_inc, &EqPolynomial::<F>::evals(&s4, None));
        let w2 = dot(&rd_inc, &EqPolynomial::<F>::evals(&s5, None));

        let seed_acc = |acc: &mut Openings<F>| {
            acc.append_dense(
                CommittedPolynomial::RamInc,
                SumcheckId::RamReadWriteChecking,
                OpeningPoint::new(r2.clone()),
                v1,
            );
            acc.append_dense(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValCheck,
                OpeningPoint::new(r4.clone()),
                v2,
            );
            acc.append_dense(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersReadWriteChecking,
                OpeningPoint::new(s4.clone()),
                w1,
            );
            acc.append_dense(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersValEvaluation,
                OpeningPoint::new(s5.clone()),
                w2,
            );
        };

        // Prover
        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut prover_t = ProverTranscript::new("inc-claim-reduction");
        let params = IncClaimReductionParams::new(log_t, &prover_acc, &mut prover_t);
        let input_claim = params.input_claim(&prover_acc);
        let mut prover =
            IncClaimReduction::new_prover(params.clone(), ram_inc.clone(), rd_inc.clone());
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        // Verifier
        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut verifier_t = VerifierTranscript::new("inc-claim-reduction", &narg);
        let vparams = IncClaimReductionParams::new(log_t, &verifier_acc, &mut verifier_t);
        let verifier = IncClaimReduction::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("inc reduction must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        // The verifier caches the reduced openings (carried in the real proof); here from the
        // prover's accumulator, then checks the reduced claim against the eq-weighted formula.
        let (_, ram_rho) = prover_acc.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_rho) = prover_acc.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );
        verifier_acc.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
            OpeningPoint::new(point.clone()),
            ram_rho,
        );
        verifier_acc.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
            OpeningPoint::new(point.clone()),
            rd_rho,
        );

        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match the eq-weighted output formula"
        );

        // The cached reduced openings equal the polynomials' MLEs at ρ = normalize(challenges)
        // (big-endian = reversed challenge order, matching how v1..w2 were seeded via evals).
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        assert_eq!(
            ram_rho,
            dot(&ram_inc, &eq_rho),
            "RamInc(ρ) matches direct MLE"
        );
        assert_eq!(rd_rho, dot(&rd_inc, &eq_rho), "RdInc(ρ) matches direct MLE");
    }

    #[test]
    fn inc_claim_reduction_round_trip() {
        for log_t in 1..=8 {
            round_trip(0xF000 + log_t as u64, log_t);
        }
    }
}
