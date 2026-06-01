//! One fan-in-2 fractional-add GKR layer, as a framework [`SumcheckInstance`].
//!
//! Ported from `crates/whir-pcs-bench/src/gkr.rs::prove_single_instance_sumcheck` (the per-layer
//! body). A layer reduces a claim about level `k` to a claim about level `k+1` of a [`Circuit`]. Its
//! claim is
//!
//! ```text
//! claim = α·Ñ_k(point) + β·D̃_k(point) = Σ_i eq(point, i) · F(i),
//! F(i) = α·(NL[i]·DR[i] + NR[i]·DL[i]) + β·(DL[i]·DR[i]),
//! ```
//!
//! where `NL/NR/DL/DR` are level `k+1` split on its low bit (`2i`/`2i+1`) and `point ∈ F^k` is the
//! layer's claim point (built up from the merge challenges of the layers above). `F` is degree-2 in
//! each variable, so with the `eq(point, ·)` factor the round polynomial is degree-3.
//!
//! The `eq(point, ·)` factor is handled by the **Gruen / Dao-Thaler** split-eq
//! ([`GruenSplitEqPolynomial`] + `gruen_poly_deg_3`), exactly as `ram/output_check.rs`. The LogUp\*
//! paper / prototype is **LSB-first** (`point[0]` ↔ the first-bound variable), so the split-eq is
//! built from the **reversed** point — its high-to-low coordinate consumption then matches the
//! `LowToHigh` binding of `NL/NR/DL/DR`, and `current_scalar()` after all `k` binds equals
//! `eq(point, r')`.

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{OpeningAccumulator, Openings};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

/// Degree of each layer's round polynomial: `eq` (1) × `F` (2).
pub const DEGREE: usize = 3;

/// The fan-in-2 fractional-add gate value `F(NL,NR,DL,DR) = α·(NL·DR + NR·DL) + β·(DL·DR)`.
#[inline]
pub fn f_combine<F: Field>(alpha: F, beta: F, nl: F, nr: F, dl: F, dr: F) -> F {
    alpha * (nl * dr + nr * dl) + beta * (dl * dr)
}

/// One GKR layer reducing a level-`k` claim to a level-`(k+1)` claim. `num_rounds() = k`.
pub struct GkrLayer<F: Field> {
    nl: MultilinearPolynomial<F>,
    nr: MultilinearPolynomial<F>,
    dl: MultilinearPolynomial<F>,
    dr: MultilinearPolynomial<F>,
    /// Gruen split-eq over the layer point (built from its reverse). `None` for the root layer
    /// (`k = 0`, zero rounds), where the bound eq scalar is `1`.
    eq: Option<GruenSplitEqPolynomial<F>>,
    alpha: F,
    beta: F,
    input_claim: F,
}

impl<F: Field> GkrLayer<F> {
    /// Build the layer from `level_kp1` (the `2^{k+1}` gates at level `k+1`), the layer claim point
    /// `point` (len `k`, LSB-first), the combine challenges `(α, β)`, and the carried `input_claim`.
    pub fn new(level_kp1: &[(F, F)], point: Vec<F>, alpha: F, beta: F, input_claim: F) -> Self {
        let half = level_kp1.len() / 2;
        debug_assert_eq!(half, 1usize << point.len());
        let mut nl = Vec::with_capacity(half);
        let mut nr = Vec::with_capacity(half);
        let mut dl = Vec::with_capacity(half);
        let mut dr = Vec::with_capacity(half);
        for pair in level_kp1.chunks_exact(2) {
            let (n0, d0) = pair[0];
            let (n1, d1) = pair[1];
            nl.push(n0);
            nr.push(n1);
            dl.push(d0);
            dr.push(d1);
        }

        let eq = if point.is_empty() {
            None
        } else {
            // LSB-first → reverse so the split-eq's high-to-low binding matches LowToHigh.
            let reversed: Vec<F> = point.iter().rev().copied().collect();
            Some(GruenSplitEqPolynomial::new(
                &reversed,
                BindingOrder::LowToHigh,
            ))
        };

        Self {
            nl: MultilinearPolynomial::from(nl),
            nr: MultilinearPolynomial::from(nr),
            dl: MultilinearPolynomial::from(dl),
            dr: MultilinearPolynomial::from(dr),
            eq,
            alpha,
            beta,
            input_claim,
        }
    }

    /// The bound `eq(point, r')` scalar after all rounds (`1` for the root layer).
    #[inline]
    pub fn bound_eq_scalar(&self) -> F {
        self.eq
            .as_ref()
            .map_or_else(F::one, GruenSplitEqPolynomial::current_scalar)
    }

    /// The four leaf values `(NL(r'), NR(r'), DL(r'), DR(r'))` after binding.
    #[inline]
    pub fn leaf_values(&self) -> (F, F, F, F) {
        (
            self.nl.final_sumcheck_claim(),
            self.nr.final_sumcheck_claim(),
            self.dl.final_sumcheck_claim(),
            self.dr.final_sumcheck_claim(),
        )
    }
}

impl<F: Field> SumcheckInstance<F> for GkrLayer<F> {
    fn num_rounds(&self) -> usize {
        self.eq.as_ref().map_or(0, GruenSplitEqPolynomial::num_vars)
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    #[expect(
        clippy::expect_used,
        reason = "the framework driver only calls compute_message when num_rounds > 0, so eq is Some"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let (alpha, beta) = (self.alpha, self.beta);
        let nl = &self.nl;
        let nr = &self.nr;
        let dl = &self.dl;
        let dr = &self.dr;
        let eq = self
            .eq
            .as_ref()
            .expect("compute_message is only called when num_rounds > 0");
        // Gruen: the per-pair quadratic q(X) = F(NL(X),NR(X),DL(X),DR(X)) condensed to its constant
        // F(X=0) and X² coefficient, E_in-weighted and accumulated unreduced (the OPT-A pattern).
        let [q_constant, q_quadratic] = eq.fold_out_in(
            || [<F as Field>::Accumulator::default(); 2],
            |inner: &mut [<F as Field>::Accumulator; 2], group, _x_in, e_in| {
                let nl0 = nl.get_bound_coeff(2 * group);
                let nl1 = nl.get_bound_coeff(2 * group + 1);
                let nr0 = nr.get_bound_coeff(2 * group);
                let nr1 = nr.get_bound_coeff(2 * group + 1);
                let dl0 = dl.get_bound_coeff(2 * group);
                let dl1 = dl.get_bound_coeff(2 * group + 1);
                let dr0 = dr.get_bound_coeff(2 * group);
                let dr1 = dr.get_bound_coeff(2 * group + 1);
                let f0 = f_combine(alpha, beta, nl0, nr0, dl0, dr0);
                let fx2 = f_combine(alpha, beta, nl1 - nl0, nr1 - nr0, dl1 - dl0, dr1 - dr0);
                inner[0].fmadd(e_in, f0);
                inner[1].fmadd(e_in, fx2);
            },
            |_x_out, e_out, inner: [<F as Field>::Accumulator; 2]| {
                [e_out * inner[0].reduce(), e_out * inner[1].reduce()]
            },
            |a: [F; 2], b: [F; 2]| [a[0] + b[0], a[1] + b[1]],
        );
        eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn bind(&mut self, r: F, _round: usize) {
        if let Some(eq) = self.eq.as_mut() {
            eq.bind(r);
        }
        self.nl.bind_parallel(r, BindingOrder::LowToHigh);
        self.nr.bind_parallel(r, BindingOrder::LowToHigh);
        self.dl.bind_parallel(r, BindingOrder::LowToHigh);
        self.dr.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, _accumulator: &mut Openings<F>, _challenges: &[F]) {}

    fn expected_output_claim(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
        _challenges: &[F],
    ) -> F {
        let (nl, nr, dl, dr) = self.leaf_values();
        self.bound_eq_scalar() * f_combine(self.alpha, self.beta, nl, nr, dl, dr)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::sumcheck::{prove, verify};
    use crate::zkvm::logup::{lsb_eq_table, mle_eval_lsb};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};
    use jolt_transcript::{Blake2bTranscript, Transcript};

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

    /// Round-trip a single layer: build random `NL/NR/DL/DR` of `2^k` and a point of `k`, seed the
    /// claim as `Σ_i eq(point,i)·F(i)`, prove, verify, and check the reduced claim equals
    /// `eq(point,r')·F(NL(r'),..)` (the per-layer consistency that becomes a fallible check in the
    /// GKR driver).
    fn layer_round_trip(seed: u64, k: usize) {
        let mut rng = Rng(seed);
        let half = 1usize << k;
        let alpha = F::from_u64(rng.next());
        let beta = F::from_u64(rng.next());
        let point = rand_vec(&mut rng, k);

        // Random level-(k+1): 2^{k+1} (num, denom) gates.
        let level: Vec<(F, F)> = (0..(2 * half))
            .map(|_| (F::from_u64(rng.next()), F::from_u64(rng.next())))
            .collect();

        // F(i) over the 2^k level-k gates, then the eq-weighted claim.
        let f_vals: Vec<F> = (0..half)
            .map(|i| {
                let (n0, d0) = level[2 * i];
                let (n1, d1) = level[2 * i + 1];
                f_combine(alpha, beta, n0, n1, d0, d1)
            })
            .collect();
        let input_claim = mle_eval_lsb(&f_vals, &point);
        // Sanity: input_claim == Σ_i eq(point,i)·F(i) (mle_eval_lsb is exactly that).
        let eq_pt = lsb_eq_table(&point);
        let direct: F = f_vals.iter().zip(eq_pt.iter()).map(|(&f, &e)| f * e).sum();
        assert_eq!(input_claim, direct);

        let mut instance = GkrLayer::new(&level, point.clone(), alpha, beta, input_claim);
        let mut acc = Openings::<F>::new(k.max(1));
        let mut prover_t = Blake2bTranscript::<F>::new(b"gkr-layer");
        let (proof, challenges) = prove(&mut instance, &mut acc, &mut prover_t);
        assert_eq!(challenges.len(), k);

        let claim = SumcheckClaim {
            num_vars: k,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"gkr-layer");
        let EvaluationClaim {
            point: r_prime,
            value,
        } = verify(&claim, &proof, &mut verifier_t).expect("layer must verify");
        assert_eq!(
            r_prime, challenges,
            "verifier point matches prover challenges"
        );

        // Per-layer consistency (the GKR driver's assert #3): value == eq(point,r')·F(leaves).
        let (nl, nr, dl, dr) = instance.leaf_values();
        let eq_bound = EqPolynomial::<F>::mle(&point, &r_prime);
        let expected = eq_bound * f_combine(alpha, beta, nl, nr, dl, dr);
        assert_eq!(value, expected, "reduced claim == eq·F(leaves)");
        // And it matches the instance's own expected_output_claim.
        assert_eq!(value, instance.expected_output_claim(&acc, &challenges));
        assert_eq!(eq_bound, instance.bound_eq_scalar());
    }

    #[test]
    fn gkr_layer_round_trip() {
        for k in 0..=8 {
            layer_round_trip(0x1A00 + k as u64, k);
        }
    }

    #[test]
    fn tampered_layer_rejected() {
        let mut rng = Rng(0x1AFE);
        let k = 4;
        let half = 1usize << k;
        let alpha = F::from_u64(rng.next());
        let beta = F::from_u64(rng.next());
        let point = rand_vec(&mut rng, k);
        let level: Vec<(F, F)> = (0..(2 * half))
            .map(|_| (F::from_u64(rng.next()), F::from_u64(rng.next())))
            .collect();
        let f_vals: Vec<F> = (0..half)
            .map(|i| {
                let (n0, d0) = level[2 * i];
                let (n1, d1) = level[2 * i + 1];
                f_combine(alpha, beta, n0, n1, d0, d1)
            })
            .collect();
        let input_claim = mle_eval_lsb(&f_vals, &point);

        let mut instance = GkrLayer::new(&level, point, alpha, beta, input_claim);
        let mut acc = Openings::<F>::new(k);
        let mut prover_t = Blake2bTranscript::<F>::new(b"t");
        let (mut proof, _) = prove(&mut instance, &mut acc, &mut prover_t);
        proof.round_polynomials[0] = UnivariatePoly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let claim = SumcheckClaim {
            num_vars: k,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"t");
        assert!(verify(&claim, &proof, &mut verifier_t).is_err());
    }
}
