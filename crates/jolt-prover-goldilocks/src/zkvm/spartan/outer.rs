//! Spartan outer (R1CS satisfaction) sumcheck — ported from jolt-core's `zkvm/spartan/outer.rs`
//! onto [`crate::framework`] over the lean `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves R1CS satisfaction as the zero-check over the constraint/cycle hypercube (degree-3):
//!
//! ```text
//! 0 = Σ_x eq(τ, x) · ( Az(x)·Bz(x) − Cz(x) ),
//! ```
//!
//! reducing it to the three claims `Az(r)`, `Bz(r)`, `Cz(r)` (cached as `SpartanAz`/`Bz`/`Cz`
//! under [`SumcheckId::SpartanOuter`]). The verifier re-evals `eq(τ, ρ)` and reads the three.
//!
//! **Decoupled / correctness-first** (the M5 convention): takes the materialized `Az`/`Bz`/`Cz`
//! matrix-vector-product columns directly. Deferred (OPT-E): jolt-core's **univariate-skip** first
//! round (the `L(τ_high, Y)·Az·Bz` univariate over the constraint domain), the **streaming**
//! cycle rounds, and the `R1CSEval`/`ALL_R1CS_INPUTS` reduction of `Az(r)`/`Bz(r)`/`Cz(r)` to the
//! committed `z`-input openings via the (limbed) R1CS matrices — that matrix→`z` step is the inner
//! Spartan reduction, distinct from this outer satisfaction check.

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{OpeningAccumulator, Openings, SumcheckId, VirtualPolynomial};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// Tau (eq) parameters.
#[derive(Clone, Debug)]
pub struct SpartanOuterParams<F: Field> {
    pub tau: Vec<F>,
}

impl<F: Field> SpartanOuterParams<F> {
    pub fn new(tau: Vec<F>) -> Self {
        Self { tau }
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) polynomials.
pub struct SpartanOuter<F: Field> {
    pub params: SpartanOuterParams<F>,
    eq: MultilinearPolynomial<F>,
    az: MultilinearPolynomial<F>,
    bz: MultilinearPolynomial<F>,
    cz: MultilinearPolynomial<F>,
}

impl<F: Field> SpartanOuter<F> {
    /// Build the prover instance from the materialized `Az`/`Bz`/`Cz` columns (each length `2^|τ|`).
    pub fn new_prover(params: SpartanOuterParams<F>, az: Vec<F>, bz: Vec<F>, cz: Vec<F>) -> Self {
        let eq = EqPolynomial::<F>::evals(&params.tau, None);
        Self {
            params,
            eq: MultilinearPolynomial::from(eq),
            az: MultilinearPolynomial::from(az),
            bz: MultilinearPolynomial::from(bz),
            cz: MultilinearPolynomial::from(cz),
        }
    }

    pub fn new_verifier(params: SpartanOuterParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::zero()]);
        Self {
            params,
            eq: dummy(),
            az: dummy(),
            bz: dummy(),
            cz: dummy(),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for SpartanOuter<F> {
    fn num_rounds(&self) -> usize {
        self.params.tau.len()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-3: eq·(Az·Bz − Cz) ⇒ 4 evaluation points; unreduced accumulation.
        let half = self.eq.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 4];
        for x in 0..half {
            let eq = self
                .eq
                .sumcheck_evals_array::<4>(x, BindingOrder::LowToHigh);
            let az = self
                .az
                .sumcheck_evals_array::<4>(x, BindingOrder::LowToHigh);
            let bz = self
                .bz
                .sumcheck_evals_array::<4>(x, BindingOrder::LowToHigh);
            let cz = self
                .cz
                .sumcheck_evals_array::<4>(x, BindingOrder::LowToHigh);
            for k in 0..4 {
                acc[k].fmadd(eq[k], az[k] * bz[k] - cz[k]);
            }
        }
        let evals: [F; 4] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
        self.az.bind_parallel(r, BindingOrder::LowToHigh);
        self.bz.bind_parallel(r, BindingOrder::LowToHigh);
        self.cz.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        accumulator.append_virtual(
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            point.clone(),
            self.az.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            point.clone(),
            self.bz.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            point,
            self.cz.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let eq_eval = EqPolynomial::<F>::mle(&self.params.tau, &point.r);
        let (_, az) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, bz) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, cz) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
        eq_eval * (az * bz - cz)
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

    fn round_trip(seed: u64, log_n: usize) {
        let mut rng = Rng(seed);
        let n = 1usize << log_n;
        let tau = rand_vec(&mut rng, log_n);
        let az = rand_vec(&mut rng, n);
        let bz = rand_vec(&mut rng, n);
        // Honest R1CS: Cz = Az ∘ Bz on the hypercube, so the zero-check holds.
        let cz: Vec<F> = az.iter().zip(bz.iter()).map(|(a, b)| *a * *b).collect();

        let mut prover_acc = Openings::<F>::new(log_n);
        let params = SpartanOuterParams::new(tau.clone());
        let mut prover = SpartanOuter::new_prover(params, az.clone(), bz.clone(), cz.clone());
        let input_claim = prover.input_claim(&prover_acc);
        assert_eq!(input_claim, F::from_u64(0), "outer is a zero-check");
        let mut prover_t = ProverTranscript::new("spartan-outer");
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_n);
        let verifier = SpartanOuter::new_verifier(SpartanOuterParams::new(tau));
        let claim = SumcheckClaim {
            num_vars: log_n,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("spartan-outer", &narg);
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("spartan outer must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for key in [
            VirtualPolynomial::SpartanAz,
            VirtualPolynomial::SpartanBz,
            VirtualPolynomial::SpartanCz,
        ] {
            let (pt, c) = prover_acc.get_virtual_polynomial_opening(key, SumcheckId::SpartanOuter);
            verifier_acc.append_virtual(key, SumcheckId::SpartanOuter, pt, c);
        }
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(value, expected, "reduced claim must match eq·(Az·Bz − Cz)");

        // Cached Az(ρ)/Bz(ρ)/Cz(ρ) equal the direct MLEs at ρ = reverse(challenges).
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let dot = |p: &[F]| {
            p.iter()
                .zip(eq_rho.iter())
                .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
        };
        let (_, az_c) = prover_acc
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        assert_eq!(az_c, dot(&az), "Az(ρ) matches direct MLE");
        let _ = (&bz, &cz);
    }

    #[test]
    fn spartan_outer_round_trip() {
        for log_n in 1..=8 {
            round_trip(0x0017 + log_n as u64, log_n);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_n = 4;
        let mut rng = Rng(0x00FE);
        let n = 1usize << log_n;
        let tau = rand_vec(&mut rng, log_n);
        let az = rand_vec(&mut rng, n);
        let bz = rand_vec(&mut rng, n);
        let cz: Vec<F> = az.iter().zip(bz.iter()).map(|(a, b)| *a * *b).collect();
        let mut acc = Openings::<F>::new(log_n);
        let mut prover = SpartanOuter::new_prover(SpartanOuterParams::new(tau), az, bz, cz);
        let input_claim = prover.input_claim(&acc);
        let mut prover_t = ProverTranscript::new("t");
        let _ = prove(&mut prover, &mut acc, &mut prover_t);
        let mut narg = prover_t.into_proof();
        narg.narg_string[0] ^= 0x01;
        let claim = SumcheckClaim {
            num_vars: log_n,
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
