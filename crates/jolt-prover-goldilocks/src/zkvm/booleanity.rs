//! Booleanity sumcheck — ported from jolt-core's `subprotocols/booleanity.rs` onto
//! [`crate::framework`] over the lean `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves a batch of committed columns is Boolean (`x²−x = 0` everywhere) via the zero-check:
//!
//! ```text
//! 0 = Σ_x eq(r, x) · Σ_i γ^{2i} · ( b_i(x)² − b_i(x) ).
//! ```
//!
//! **M6 retarget:** jolt-core runs this over the one-hot `Ra` selectors of all three families; the
//! LogUp\*-GKR design (M7) subsumes that one-hot booleanity, so the *only* booleanity surviving is
//! this residual over the **limbed-RV64-R1CS carry/sign columns** (`CommittedPolynomial::R1csAux`).
//! The shape (degree-3, single `eq`, `γ^{2i}` batching) is unchanged. Soundness of the limbed R1CS
//! (`LIMBED_R1CS.md` §"Degree / soundness") depends on this check + the wide-limb range checks
//! (which fold into the stage-5 Shout `RangeCheck`/`LowerHalfWord`/`UpperWord` tables — M8 wiring).
//!
//! Uses the Gruen + Dao-Thaler split-eq round polynomial (`gruen_poly_deg_3`) with unreduced
//! accumulation. **Decoupled** (the M5 convention): takes the materialized Boolean columns; the
//! verifier reads the cached `R1csAux(i)` openings.

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, Openings, SumcheckId,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// Batching/opening parameters.
#[derive(Clone, Debug)]
pub struct BooleanityParams<F: Field> {
    pub r: Vec<F>,
    /// `γ^{2i}` for each of the `n` columns.
    pub gamma_sq_powers: Vec<F>,
}

impl<F: Field> BooleanityParams<F> {
    /// Draws `γ` and forms the `[γ^0, γ², γ⁴, …]` batching coefficients for `num_cols` columns.
    pub fn new(r: Vec<F>, num_cols: usize, transcript: &mut impl Challenge<F>) -> Self {
        let gamma = transcript.challenge();
        let gamma_sq = gamma * gamma;
        let mut powers = Vec::with_capacity(num_cols);
        let mut p = F::one();
        for _ in 0..num_cols {
            powers.push(p);
            p *= gamma_sq;
        }
        Self {
            r,
            gamma_sq_powers: powers,
        }
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) columns.
pub struct Booleanity<F: Field> {
    pub params: BooleanityParams<F>,
    eq: GruenSplitEqPolynomial<F>,
    cols: Vec<MultilinearPolynomial<F>>,
}

impl<F: Field> Booleanity<F> {
    /// Build the prover instance from the materialized Boolean columns (each length `2^|r|`).
    pub fn new_prover(params: BooleanityParams<F>, cols: Vec<Vec<F>>) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, BindingOrder::LowToHigh);
        Self {
            params,
            eq,
            cols: cols.into_iter().map(MultilinearPolynomial::from).collect(),
        }
    }

    pub fn new_verifier(params: BooleanityParams<F>) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, BindingOrder::LowToHigh);
        Self {
            params,
            eq,
            cols: vec![],
        }
    }
}

impl<F: Field> SumcheckInstance<F> for Booleanity<F> {
    fn num_rounds(&self) -> usize {
        self.params.r.len()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        // q(X) = Σ_i γ^{2i}·(b_i(X)² − b_i(X)); per pair: constant = b0²−b0, X² coeff = (b1−b0)².
        // Gruen handles the eq factor; E_out·E_in-weighted, unreduced.
        let cols = &self.cols;
        let gpow = &self.params.gamma_sq_powers;
        let [q_constant, q_quadratic] = self.eq.fold_out_in(
            || [<F as Field>::Accumulator::default(); 2],
            |inner: &mut [<F as Field>::Accumulator; 2], group, _x_in, e_in| {
                let mut qc = F::zero();
                let mut qq = F::zero();
                for (i, col) in cols.iter().enumerate() {
                    let b0 = col.get_bound_coeff(2 * group);
                    let b1 = col.get_bound_coeff(2 * group + 1);
                    let d = b1 - b0;
                    qc += gpow[i] * (b0 * b0 - b0);
                    qq += gpow[i] * (d * d);
                }
                inner[0].fmadd(e_in, qc);
                inner[1].fmadd(e_in, qq);
            },
            |_x_out, e_out, inner: [<F as Field>::Accumulator; 2]| {
                [e_out * inner[0].reduce(), e_out * inner[1].reduce()]
            },
            |a: [F; 2], b: [F; 2]| [a[0] + b[0], a[1] + b[1]],
        );
        self.eq
            .gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.eq.bind(r);
        for col in &mut self.cols {
            col.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        for (i, col) in self.cols.iter().enumerate() {
            accumulator.append_dense(
                CommittedPolynomial::R1csAux(i),
                SumcheckId::Booleanity,
                point.clone(),
                col.final_sumcheck_claim(),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let eq_eval = jolt_poly::EqPolynomial::<F>::mle(&self.params.r, &point.r);
        let mut q = F::zero();
        for (i, &g) in self.params.gamma_sq_powers.iter().enumerate() {
            let (_, b) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::R1csAux(i),
                SumcheckId::Booleanity,
            );
            q += g * (b * b - b);
        }
        eq_eval * q
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::sumcheck::{prove, verify};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_poly::EqPolynomial;
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

    /// Random Boolean columns (entries in {0,1}); the booleanity sum is then 0.
    fn bool_cols(rng: &mut Rng, n_cols: usize, len: usize) -> Vec<Vec<F>> {
        (0..n_cols)
            .map(|_| (0..len).map(|_| F::from_u64(rng.next() & 1)).collect())
            .collect()
    }

    fn round_trip(seed: u64, log_k: usize, n_cols: usize) {
        let mut rng = Rng(seed);
        let k = 1usize << log_k;
        let r: Vec<F> = (0..log_k).map(|_| F::from_u64(rng.next())).collect();
        let cols = bool_cols(&mut rng, n_cols, k);

        let mut prover_acc = Openings::<F>::new(log_k);
        let mut prover_t = ProverTranscript::new("booleanity");
        let params = BooleanityParams::new(r.clone(), n_cols, &mut prover_t);
        let mut prover = Booleanity::new_prover(params, cols.clone());
        let input_claim = prover.input_claim(&prover_acc);
        assert_eq!(input_claim, F::from_u64(0), "booleanity is a zero-check");
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_k);
        let mut verifier_t = VerifierTranscript::new("booleanity", &narg);
        let vparams = BooleanityParams::new(r, n_cols, &mut verifier_t);
        let verifier = Booleanity::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_k,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("booleanity must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for i in 0..n_cols {
            let (pt, c) = prover_acc.get_committed_polynomial_opening(
                CommittedPolynomial::R1csAux(i),
                SumcheckId::Booleanity,
            );
            verifier_acc.append_dense(
                CommittedPolynomial::R1csAux(i),
                SumcheckId::Booleanity,
                pt,
                c,
            );
        }
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match eq·Σ γ^{{2i}}·(b²−b)"
        );

        // Cached column claims equal the direct MLEs at ρ.
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let dot = |p: &[F]| {
            p.iter()
                .zip(eq_rho.iter())
                .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
        };
        let (_, b0) = prover_acc.get_committed_polynomial_opening(
            CommittedPolynomial::R1csAux(0),
            SumcheckId::Booleanity,
        );
        assert_eq!(b0, dot(&cols[0]), "R1csAux(0)(ρ) matches direct MLE");
    }

    #[test]
    fn booleanity_round_trip() {
        for log_k in 1..=8 {
            round_trip(0xB001 + log_k as u64, log_k, 3);
        }
        round_trip(0xB0FF, 6, 1);
        round_trip(0xB0FE, 5, 5);
    }

    #[test]
    fn non_boolean_column_mismatch() {
        // A non-Boolean entry makes the honest summand nonzero. `gruen_poly_deg_3` still forces
        // each round to sum to the claim, so `verify()` succeeds — but the reduced value then
        // disagrees with `expected_output_claim` computed from the (honest) cached openings, which
        // is how the booleanity failure is caught (the output-claim discharge, not the round check).
        let log_k = 4;
        let mut rng = Rng(0xBAD0);
        let k = 1usize << log_k;
        let r: Vec<F> = (0..log_k).map(|_| F::from_u64(rng.next())).collect();
        let mut cols = bool_cols(&mut rng, 1, k);
        cols[0][3] = F::from_u64(7); // not in {0,1}

        let mut acc = Openings::<F>::new(log_k);
        let mut prover_t = ProverTranscript::new("booleanity");
        let params = BooleanityParams::new(r.clone(), 1, &mut prover_t);
        let mut prover = Booleanity::new_prover(params, cols);
        let challenges = prove(&mut prover, &mut acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("booleanity", &narg);
        let vparams = BooleanityParams::new(r, 1, &mut verifier_t);
        let verifier = Booleanity::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_k,
            degree: DEGREE,
            claimed_sum: F::from_u64(0),
        };
        let EvaluationClaim { value, .. } =
            verify(&claim, &mut verifier_t).expect("rounds are internally consistent");
        let (pt, c) = acc.get_committed_polynomial_opening(
            CommittedPolynomial::R1csAux(0),
            SumcheckId::Booleanity,
        );
        acc.append_dense(
            CommittedPolynomial::R1csAux(0),
            SumcheckId::Booleanity,
            pt,
            c,
        );
        let expected = verifier.expected_output_claim(&acc, &challenges);
        assert_ne!(
            value, expected,
            "a non-Boolean column must fail the output-claim discharge"
        );
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_k = 4;
        let mut rng = Rng(0xB0AA);
        let k = 1usize << log_k;
        let r: Vec<F> = (0..log_k).map(|_| F::from_u64(rng.next())).collect();
        let cols = bool_cols(&mut rng, 2, k);
        let mut acc = Openings::<F>::new(log_k);
        let mut prover_t = ProverTranscript::new("t");
        let params = BooleanityParams::new(r.clone(), 2, &mut prover_t);
        let mut prover = Booleanity::new_prover(params, cols);
        let _ = prove(&mut prover, &mut acc, &mut prover_t);
        let mut narg = prover_t.into_proof();
        narg.narg_string[0] ^= 0x01;
        let claim = SumcheckClaim {
            num_vars: log_k,
            degree: DEGREE,
            claimed_sum: F::from_u64(0),
        };
        let mut verifier_t = VerifierTranscript::new("t", &narg);
        // Replay the prover's pre-round γ squeeze so the verifier transcript stays in lockstep.
        let _ = BooleanityParams::new(r, 2, &mut verifier_t);
        assert!(
            verify(&claim, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }
}
