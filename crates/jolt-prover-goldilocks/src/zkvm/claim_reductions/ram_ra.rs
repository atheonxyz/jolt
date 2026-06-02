//! RAM RA claim-reduction sumcheck — ported from jolt-core's `zkvm/claim_reductions/ram_ra.rs`
//! onto the framework ([`crate::framework`]) over the lean `Field` (`C = F = Fp3`). jolt-core is the
//! parity oracle.
//!
//! Consolidates the three `RamRa` openings (from `RamRafEvaluation`, `RamReadWriteChecking`,
//! `RamValCheck`) — which by Stage-2 alignment share the **same** RAM address point `r_address` but
//! differ in their cycle points — into a single `RamRa(r_address ‖ ρ)` opening:
//!
//! ```text
//! input:  claim_raf + γ·claim_rw + γ²·claim_val
//! sumcheck (log T cycle rounds, degree 2; address fixed to the aligned r_address):
//!   Σ_c ( eq(r_cycle_raf, c) + γ·eq(r_cycle_rw, c) + γ²·eq(r_cycle_val, c) ) · ra(r_address, c)
//! output: RamRa(r_address ‖ ρ)
//! ```
//!
//! where `ra(r_address, c) = Σ_k eq(r_address, k)·ra(k, c)` is the cycle-indexed RAM-access column.
//! This is the **single-phase** form (jolt-core's prefix/suffix two-phase materialization is a perf
//! optimization deferred with the trace witness-gen, matching [`super::increments`]). The
//! `ra(r_address, ·)` column is taken pre-materialized (`Fp3`), decoupling from the trace → RAM
//! address-remap extraction (M8).

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;

use crate::framework::accumulator::{
    OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial, BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// Fiat-Shamir + opening-point parameters, fetched from the accumulator (matches jolt-core
/// `RaReductionParams`).
#[derive(Clone, Debug)]
pub struct RamRaReductionParams<F: Field> {
    pub gamma: F,
    pub gamma_squared: F,
    pub log_t: usize,
    pub log_k: usize,
    /// The aligned RAM address point (BIG_ENDIAN), shared by all three input openings.
    pub r_address: Vec<F>,
    pub r_cycle_raf: Vec<F>,
    pub r_cycle_rw: Vec<F>,
    pub r_cycle_val: Vec<F>,
    pub claim_raf: F,
    pub claim_rw: F,
    pub claim_val: F,
}

impl<F: Field> RamRaReductionParams<F> {
    pub fn new(
        log_t: usize,
        log_k: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript<Challenge = F>,
    ) -> Self {
        let (r_raf, claim_raf) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (r_rw, claim_rw) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_val, claim_val) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValCheck);

        let (r_address_raf, r_cycle_raf) = r_raf.split_at(log_k);
        let (_, r_cycle_rw) = r_rw.split_at(log_k);
        let (_, r_cycle_val) = r_val.split_at(log_k);

        let gamma = transcript.challenge();
        let gamma_squared = gamma * gamma;

        Self {
            gamma,
            gamma_squared,
            log_t,
            log_k,
            r_address: r_address_raf.r,
            r_cycle_raf: r_cycle_raf.r,
            r_cycle_rw: r_cycle_rw.r,
            r_cycle_val: r_cycle_val.r,
            claim_raf,
            claim_rw,
            claim_val,
        }
    }

    fn input_claim(&self) -> F {
        self.claim_raf + self.gamma * self.claim_rw + self.gamma_squared * self.claim_val
    }

    /// `[r_address ‖ reverse(challenges)]` — the full `(address, cycle)` opening point.
    fn opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_cycle_be: Vec<F> = challenges.iter().rev().copied().collect();
        OpeningPoint::new([self.r_address.clone(), r_cycle_be].concat())
    }
}

/// Prover/verifier instance. The prover holds the `ra(r_address,·)` cycle column + the three
/// `eq(r_cycle_*,·)` columns; the verifier carries the same `params` and ignores the polynomials.
pub struct RamRaClaimReduction<F: Field> {
    pub params: RamRaReductionParams<F>,
    ra: MultilinearPolynomial<F>,
    eq_raf: MultilinearPolynomial<F>,
    eq_rw: MultilinearPolynomial<F>,
    eq_val: MultilinearPolynomial<F>,
}

impl<F: Field> RamRaClaimReduction<F> {
    /// Build the prover instance from the materialized `ra(r_address, ·)` cycle column (length `2^log_t`).
    pub fn new_prover(params: RamRaReductionParams<F>, ra: Vec<F>) -> Self {
        let eq_raf = EqPolynomial::<F>::evals(&params.r_cycle_raf, None);
        let eq_rw = EqPolynomial::<F>::evals(&params.r_cycle_rw, None);
        let eq_val = EqPolynomial::<F>::evals(&params.r_cycle_val, None);
        Self {
            params,
            ra: MultilinearPolynomial::from(ra),
            eq_raf: MultilinearPolynomial::from(eq_raf),
            eq_rw: MultilinearPolynomial::from(eq_rw),
            eq_val: MultilinearPolynomial::from(eq_val),
        }
    }

    /// Build a verifier instance (no polynomials; `expected_output_claim` reads the cached reduced
    /// opening + recomputes the combined eq factor).
    pub fn new_verifier(params: RamRaReductionParams<F>) -> Self {
        Self {
            params,
            ra: MultilinearPolynomial::from(vec![F::zero()]),
            eq_raf: MultilinearPolynomial::from(vec![F::zero()]),
            eq_rw: MultilinearPolynomial::from(vec![F::zero()]),
            eq_val: MultilinearPolynomial::from(vec![F::zero()]),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for RamRaClaimReduction<F> {
    fn num_rounds(&self) -> usize {
        self.params.log_t
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim()
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let gamma = self.params.gamma;
        let gamma_sqr = self.params.gamma_squared;
        let half = self.ra.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for j in 0..half {
            let ra = self
                .ra
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let e_raf = self
                .eq_raf
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let e_rw = self
                .eq_rw
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let e_val = self
                .eq_val
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            for k in 0..3 {
                let eq_combined = e_raf[k] + gamma * e_rw[k] + gamma_sqr * e_val[k];
                acc[k].fmadd(ra[k], eq_combined);
            }
        }
        let evals: [F; 3] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.ra.bind_parallel(r, BindingOrder::LowToHigh);
        self.eq_raf.bind_parallel(r, BindingOrder::LowToHigh);
        self.eq_rw.bind_parallel(r, BindingOrder::LowToHigh);
        self.eq_val.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        accumulator.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            self.params.opening_point(challenges),
            self.ra.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let r_cycle_reduced: Vec<F> = challenges.iter().rev().copied().collect();
        let eq_combined = EqPolynomial::<F>::mle(&self.params.r_cycle_raf, &r_cycle_reduced)
            + self.params.gamma * EqPolynomial::<F>::mle(&self.params.r_cycle_rw, &r_cycle_reduced)
            + self.params.gamma_squared
                * EqPolynomial::<F>::mle(&self.params.r_cycle_val, &r_cycle_reduced);

        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
        );
        eq_combined * ra_claim
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::sumcheck::{prove, verify};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};
    use jolt_transcript::Blake2bTranscript;

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

    /// Seed the three RamRa input openings, all at `[r_address ‖ r_cycle_x]` with value = `ra`'s MLE
    /// at that cycle point.
    fn seed_acc(
        acc: &mut Openings<F>,
        r_address: &[F],
        ra: &[F],
        cycles: [(&SumcheckId, &[F]); 3],
    ) {
        for (sid, r_cycle) in cycles {
            let eq = EqPolynomial::<F>::evals(r_cycle, None);
            let point = OpeningPoint::new([r_address.to_vec(), r_cycle.to_vec()].concat());
            acc.append_virtual(VirtualPolynomial::RamRa, *sid, point, dot(ra, &eq));
        }
    }

    fn round_trip(seed: u64, log_t: usize, log_k: usize) {
        let mut rng = Rng(seed);
        let t = 1usize << log_t;
        let ra = rand_vec(&mut rng, t);
        let r_address = rand_vec(&mut rng, log_k);
        let r_cycle_raf = rand_vec(&mut rng, log_t);
        let r_cycle_rw = rand_vec(&mut rng, log_t);
        let r_cycle_val = rand_vec(&mut rng, log_t);

        let seed_both = |acc: &mut Openings<F>| {
            seed_acc(
                acc,
                &r_address,
                &ra,
                [
                    (&SumcheckId::RamRafEvaluation, &r_cycle_raf),
                    (&SumcheckId::RamReadWriteChecking, &r_cycle_rw),
                    (&SumcheckId::RamValCheck, &r_cycle_val),
                ],
            );
        };

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_both(&mut prover_acc);
        let mut prover_t = Blake2bTranscript::<F>::new(b"ram-ra-claim-reduction");
        let params = RamRaReductionParams::new(log_t, log_k, &prover_acc, &mut prover_t);
        let input_claim = params.input_claim();
        let mut prover = RamRaClaimReduction::new_prover(params.clone(), ra.clone());
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_both(&mut verifier_acc);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"ram-ra-claim-reduction");
        let vparams = RamRaReductionParams::new(log_t, log_k, &verifier_acc, &mut verifier_t);
        let verifier = RamRaClaimReduction::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("ram ra reduction must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        let (_, ra_rho) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
        );
        verifier_acc.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            OpeningPoint::new(point.clone()),
            ra_rho,
        );

        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match eq_combined(ρ)·ra(r_address,ρ)"
        );

        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        assert_eq!(
            ra_rho,
            dot(&ra, &eq_rho),
            "ra(r_address,ρ) matches direct MLE"
        );
    }

    #[test]
    fn ram_ra_claim_reduction_round_trip() {
        for log_t in 1..=7 {
            round_trip(0xB000u64.wrapping_add(log_t as u64), log_t, 3);
        }
    }

    /// A tampered reduced opening (corrupted `RamRa(r_address‖ρ)`) breaks the output-claim check.
    #[test]
    fn tampered_reduced_opening_rejected() {
        let (log_t, log_k) = (5, 3);
        let mut rng = Rng(0x6262);
        let t = 1usize << log_t;
        let ra = rand_vec(&mut rng, t);
        let r_address = rand_vec(&mut rng, log_k);
        let r_cycle_raf = rand_vec(&mut rng, log_t);
        let r_cycle_rw = rand_vec(&mut rng, log_t);
        let r_cycle_val = rand_vec(&mut rng, log_t);

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(
            &mut prover_acc,
            &r_address,
            &ra,
            [
                (&SumcheckId::RamRafEvaluation, &r_cycle_raf),
                (&SumcheckId::RamReadWriteChecking, &r_cycle_rw),
                (&SumcheckId::RamValCheck, &r_cycle_val),
            ],
        );
        let mut prover_t = Blake2bTranscript::<F>::new(b"ram-ra-claim-reduction");
        let params = RamRaReductionParams::new(log_t, log_k, &prover_acc, &mut prover_t);
        let mut prover = RamRaClaimReduction::new_prover(params.clone(), ra);
        let (_, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let verifier = RamRaClaimReduction::new_verifier(params);
        let (_, ra_rho) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
        );
        let point = OpeningPoint::new(challenges.clone());

        let mut honest_acc = Openings::<F>::new(log_t);
        honest_acc.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            point.clone(),
            ra_rho,
        );
        let honest = verifier.expected_output_claim(&honest_acc, &challenges);

        let mut tampered_acc = Openings::<F>::new(log_t);
        tampered_acc.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            point,
            ra_rho + F::from_u64(1),
        );
        let tampered = verifier.expected_output_claim(&tampered_acc, &challenges);
        assert_ne!(
            honest, tampered,
            "tampered RamRa(ρ) must change the output claim"
        );
    }
}
