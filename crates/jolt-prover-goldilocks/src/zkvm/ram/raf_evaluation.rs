//! RAM RAF-evaluation sumcheck — ported from jolt-core's `zkvm/ram/raf_evaluation.rs` onto
//! [`crate::framework`] over the lean `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves the read-address-fingerprint relation over the `log_K` RAM-address variables:
//!
//! ```text
//! Σ_{k=0}^{K-1} ra(k) · unmap(k) = raf_claim,
//! ```
//!
//! where `ra(k) = Σ_j eq(r_cycle, j)·1[address(j) = k]` aggregates per-address access counts, and
//! `unmap(k)` maps the remapped index `k` back to its original RAM address. The input claim is the
//! `RamAddress` opening from [`SumcheckId::SpartanOuter`]. Degree-2 (`ra · unmap`).
//!
//! Caches the `RamRa` opening at `r_address ‖ r_cycle` under [`SumcheckId::RamRafEvaluation`].
//!
//! **Decoupled from the trace** (the M5 convention): takes the materialized `ra` and (public)
//! `unmap` columns; the verifier recomputes `unmap(ρ)` as the MLE of the public column. jolt-core's
//! split-eq `ra` materialization, the `UnmapRamAddressPolynomial` structural evaluator, and the
//! phase-2/3 gap-round alignment with `2^phase3_cycle_rounds` pre-scaling are deferred here
//! (single-phase, all `log_K` rounds are address rounds, no gap scaling).

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial, BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// Opening parameters (matches jolt-core `RafEvaluationSumcheckParams`, minus the phase/gap fields).
#[derive(Clone, Debug)]
pub struct RamRafEvaluationParams<F: Field> {
    pub log_k: usize,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: Field> RamRafEvaluationParams<F> {
    /// Reads `r_cycle` from the `RamAddress` Spartan-outer opening.
    pub fn new(accumulator: &dyn OpeningAccumulator<F>, log_k: usize) -> Self {
        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );
        Self { log_k, r_cycle }
    }

    /// The input claim is the `RamAddress` Spartan-outer opening verbatim (self-free, so it doesn't
    /// trip `clippy::unused_self`).
    fn input_claim(accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, raf) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );
        raf
    }
}

/// Prover/verifier instance. Both sides hold the public `unmap` column (for the MLE recompute); the
/// prover additionally holds `ra`.
pub struct RamRafEvaluation<F: Field> {
    pub params: RamRafEvaluationParams<F>,
    ra: MultilinearPolynomial<F>,
    unmap: MultilinearPolynomial<F>,
    unmap_col: Vec<F>,
}

impl<F: Field> RamRafEvaluation<F> {
    /// Build the prover instance from the materialized `ra` access-count column and the public
    /// `unmap` column (both length `K`).
    pub fn new_prover(params: RamRafEvaluationParams<F>, ra: Vec<F>, unmap: Vec<F>) -> Self {
        Self {
            params,
            ra: MultilinearPolynomial::from(ra),
            unmap: MultilinearPolynomial::from(unmap.clone()),
            unmap_col: unmap,
        }
    }

    pub fn new_verifier(params: RamRafEvaluationParams<F>, unmap: Vec<F>) -> Self {
        Self {
            params,
            ra: MultilinearPolynomial::from(vec![F::zero()]),
            unmap: MultilinearPolynomial::from(vec![F::zero()]),
            unmap_col: unmap,
        }
    }
}

impl<F: Field> SumcheckInstance<F> for RamRafEvaluation<F> {
    fn num_rounds(&self) -> usize {
        self.params.log_k
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        RamRafEvaluationParams::input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-2 product `ra · unmap` ⇒ 3 evaluation points (0,1,2).
        let half = self.ra.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for k in 0..half {
            let ra_e = self
                .ra
                .sumcheck_evals_array::<3>(k, BindingOrder::LowToHigh);
            let um_e = self
                .unmap
                .sumcheck_evals_array::<3>(k, BindingOrder::LowToHigh);
            for i in 0..3 {
                acc[i].fmadd(ra_e[i], um_e[i]);
            }
        }
        let evals: [F; 3] = std::array::from_fn(|i| acc[i].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.ra.bind_parallel(r, BindingOrder::LowToHigh);
        self.unmap.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let r_address = self.normalize_opening_point(challenges);
        let ra_point =
            OpeningPoint::new([r_address.r.as_slice(), self.params.r_cycle.r.as_slice()].concat());
        accumulator.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_point,
            self.ra.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let r = self.normalize_opening_point(challenges);
        let eq_rho = EqPolynomial::<F>::evals(&r.r, None);
        let unmap_eval = self
            .unmap_col
            .iter()
            .zip(eq_rho.iter())
            .fold(F::zero(), |acc, (v, e)| acc + *v * *e);
        let (_, ra_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        unmap_eval * ra_claim
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

    fn round_trip(seed: u64, log_k: usize, log_t: usize) {
        let mut rng = Rng(seed);
        let k = 1usize << log_k;
        let ra = rand_vec(&mut rng, k);
        // Public unmap column: start_address + index (a concrete affine remap inverse).
        let start_address = 0x8000_0000u64;
        let unmap: Vec<F> = (0..k)
            .map(|idx| F::from_u64(start_address + idx as u64))
            .collect();
        let r_cycle = rand_vec(&mut rng, log_t);

        let raf_claim: F = ra
            .iter()
            .zip(unmap.iter())
            .fold(F::from_u64(0), |a, (x, u)| a + *x * *u);

        let seed_acc = |acc: &mut Openings<F>| {
            acc.append_virtual(
                VirtualPolynomial::RamAddress,
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.clone()),
                raf_claim,
            );
        };

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let params = RamRafEvaluationParams::new(&prover_acc, log_k);
        let input_claim = RamRafEvaluationParams::<F>::input_claim(&prover_acc);
        let mut prover = RamRafEvaluation::new_prover(params, ra.clone(), unmap.clone());
        let mut prover_t = ProverTranscript::new("ram-raf-evaluation");
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let vparams = RamRafEvaluationParams::new(&verifier_acc, log_k);
        let verifier = RamRafEvaluation::new_verifier(vparams, unmap.clone());
        let claim = SumcheckClaim {
            num_vars: log_k,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("ram-raf-evaluation", &narg);
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("raf-evaluation must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        let (ra_pt, ra_rho) = prover_acc
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        verifier_acc.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_pt,
            ra_rho,
        );
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(value, expected, "reduced claim must match unmap(ρ)·ra(ρ)");

        // Cached RamRa(ρ) equals the direct MLE at ρ = reverse(challenges).
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let ra_mle = ra
            .iter()
            .zip(eq_rho.iter())
            .fold(F::from_u64(0), |a, (x, e)| a + *x * *e);
        assert_eq!(ra_rho, ra_mle, "RamRa(ρ) matches direct MLE");
    }

    #[test]
    fn ram_raf_evaluation_round_trip() {
        for log_k in 1..=8 {
            round_trip(0x3A00 + log_k as u64, log_k, 4);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_k = 5;
        let mut rng = Rng(0x3AFE);
        let k = 1usize << log_k;
        let ra = rand_vec(&mut rng, k);
        let unmap: Vec<F> = (0..k).map(|idx| F::from_u64(idx as u64)).collect();
        let r_cycle = rand_vec(&mut rng, 4);
        let raf_claim: F = ra
            .iter()
            .zip(unmap.iter())
            .fold(F::from_u64(0), |a, (x, u)| a + *x * *u);

        let mut acc = Openings::<F>::new(4);
        acc.append_virtual(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(r_cycle),
            raf_claim,
        );
        let params = RamRafEvaluationParams::new(&acc, log_k);
        let input_claim = RamRafEvaluationParams::<F>::input_claim(&acc);
        let mut prover = RamRafEvaluation::new_prover(params, ra, unmap);
        let mut prover_t = ProverTranscript::new("t");
        let _ = prove(&mut prover, &mut acc, &mut prover_t);
        let mut narg = prover_t.into_proof();

        narg.narg_string[0] ^= 0x01;
        let claim = SumcheckClaim {
            num_vars: log_k,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("t", &narg);
        assert!(
            verify(&claim, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }
}
