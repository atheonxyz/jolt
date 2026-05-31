//! Prover-side multilinear polynomial — vendored from jolt-core's
//! `poly/multilinear_polynomial.rs` (commit `90d5926a0` era), retargeted to the lean
//! [`jolt_field::Field`]: challenges are plain `F` (the `C = F = Fp3` convention; jolt-core's
//! `F::Challenge` collapses to `F`).
//!
//! This is the **dense** variant only — enough for the framework and the first claim-reduction
//! ports. The compact base-field variants (the `base × ext` sumcheck hot path that motivates the
//! Goldilocks move, via the M0 `Fp3Accumulator::fmadd_base`) and the `OneHot`/`RLC` variants land
//! incrementally as the subprotocols that need them are ported. `jolt-core` is the parity oracle
//! for the bind / `sumcheck_evals` semantics (see `specs/jolt-prover-model-crate.md`).

use jolt_field::Field;
use jolt_poly::BindingOrder;
use rayon::prelude::*;

/// A multilinear polynomial over `F`, currently always dense. The enum shape is kept so the
/// committed-witness compact/one-hot variants can be added without touching call sites.
#[derive(Clone, Debug)]
pub enum MultilinearPolynomial<F: Field> {
    /// Dense coefficient vector (length a power of two), MLE over `log2(len)` variables.
    Dense(Vec<F>),
}

impl<F: Field> MultilinearPolynomial<F> {
    #[inline]
    fn coeffs(&self) -> &[F] {
        match self {
            Self::Dense(c) => c,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeffs().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of (remaining, unbound) variables.
    #[inline]
    pub fn num_vars(&self) -> usize {
        debug_assert!(self.len().is_power_of_two());
        self.len().trailing_zeros() as usize
    }

    /// The current coefficient at `index` (after any binds).
    #[inline]
    pub fn get_bound_coeff(&self, index: usize) -> F {
        self.coeffs()[index]
    }

    /// Round-polynomial evaluations of this factor at the points `0, 1, …, DEGREE-1`, for the
    /// pair selected by `index` under `order`. The factor is multilinear in the bound variable, so
    /// the evals lie on the line through `(0, e0)` and `(1, e1)` — `evals[k] = e0 + k·(e1 − e0)`.
    /// Mirrors jolt-core `MultilinearPolynomial::sumcheck_evals_array`.
    #[inline]
    pub fn sumcheck_evals_array<const DEGREE: usize>(
        &self,
        index: usize,
        order: BindingOrder,
    ) -> [F; DEGREE] {
        debug_assert!(DEGREE > 0);
        debug_assert!(index < self.len() / 2);

        let half = self.len() / 2;
        let (e0, e1) = match order {
            BindingOrder::HighToLow => (
                self.get_bound_coeff(index),
                self.get_bound_coeff(index + half),
            ),
            BindingOrder::LowToHigh => (
                self.get_bound_coeff(2 * index),
                self.get_bound_coeff(2 * index + 1),
            ),
        };

        let mut evals = [F::zero(); DEGREE];
        evals[0] = e0;
        if DEGREE == 1 {
            return evals;
        }
        let m = e1 - e0;
        let mut eval = e1;
        evals[1] = e1;
        for slot in evals.iter_mut().take(DEGREE).skip(2) {
            eval += m;
            *slot = eval;
        }
        evals
    }

    /// Bind the next variable to `r`, halving the polynomial: `new[i] = lo + r·(hi − lo)` where
    /// `(lo, hi)` is the pair selected by `order`. Mirrors jolt-core `bind_parallel`.
    pub fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        match self {
            Self::Dense(coeffs) => {
                let half = coeffs.len() / 2;
                let bound: Vec<F> = (0..half)
                    .into_par_iter()
                    .map(|i| {
                        let (lo, hi) = match order {
                            BindingOrder::HighToLow => (coeffs[i], coeffs[i + half]),
                            BindingOrder::LowToHigh => (coeffs[2 * i], coeffs[2 * i + 1]),
                        };
                        lo + r * (hi - lo)
                    })
                    .collect();
                *coeffs = bound;
            }
        }
    }

    /// The single remaining coefficient after every variable has been bound.
    pub fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(
            self.len(),
            1,
            "final_sumcheck_claim requires a fully-bound polynomial"
        );
        self.get_bound_coeff(0)
    }
}

impl<F: Field> From<Vec<F>> for MultilinearPolynomial<F> {
    fn from(coeffs: Vec<F>) -> Self {
        debug_assert!(
            coeffs.len().is_power_of_two(),
            "MultilinearPolynomial requires a power-of-two coefficient count"
        );
        Self::Dense(coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};

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

    /// `bind_parallel` matches the defining recurrence `lo + r·(hi − lo)` element-wise.
    fn bind_matches_recurrence<F: Field>(seed: u64) {
        let mut rng = Rng(seed);
        let coeffs: Vec<F> = (0..16).map(|_| F::from_u64(rng.next())).collect();
        let r = F::from_u64(rng.next());

        let mut p = MultilinearPolynomial::from(coeffs.clone());
        p.bind_parallel(r, BindingOrder::LowToHigh);
        for i in 0..8 {
            let expected = coeffs[2 * i] + r * (coeffs[2 * i + 1] - coeffs[2 * i]);
            assert_eq!(p.get_bound_coeff(i), expected);
        }

        let mut p_hi = MultilinearPolynomial::from(coeffs.clone());
        p_hi.bind_parallel(r, BindingOrder::HighToLow);
        for i in 0..8 {
            let expected = coeffs[i] + r * (coeffs[i + 8] - coeffs[i]);
            assert_eq!(p_hi.get_bound_coeff(i), expected);
        }
    }

    /// The sumcheck invariant: each round's message `g` satisfies `g(0) + g(1) = claim`, and after
    /// binding every variable the running claim equals `final_sumcheck_claim`. Exercises
    /// `sumcheck_evals_array`, `bind_parallel`, and `final_sumcheck_claim` together.
    fn mini_sumcheck_consistency<F: Field>(seed: u64, log_len: usize) {
        let mut rng = Rng(seed);
        let coeffs: Vec<F> = (0..(1usize << log_len))
            .map(|_| F::from_u64(rng.next()))
            .collect();
        let mut claim: F = coeffs.iter().fold(F::zero(), |a, b| a + *b);

        let mut p = MultilinearPolynomial::from(coeffs);
        while p.len() > 1 {
            let half = p.len() / 2;
            let (g0, g1) = (0..half).fold((F::zero(), F::zero()), |(s0, s1), i| {
                let e = p.sumcheck_evals_array::<2>(i, BindingOrder::LowToHigh);
                (s0 + e[0], s1 + e[1])
            });
            assert_eq!(
                g0 + g1,
                claim,
                "round message must sum to the running claim"
            );

            let r = F::from_u64(rng.next());
            claim = g0 + r * (g1 - g0);
            p.bind_parallel(r, BindingOrder::LowToHigh);
        }
        assert_eq!(
            p.final_sumcheck_claim(),
            claim,
            "final claim must equal the reduced sum"
        );
    }

    /// Degree-3 extrapolation: `sumcheck_evals_array::<3>` lies on the line through the pair.
    fn evals_extrapolate<F: Field>() {
        let coeffs: Vec<F> = vec![
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(10),
            F::from_u64(2),
        ];
        let p = MultilinearPolynomial::from(coeffs);
        let e = p.sumcheck_evals_array::<3>(0, BindingOrder::LowToHigh);
        // e0=3, e1=7, m=4 -> e2 = 11
        assert_eq!(e[0], F::from_u64(3));
        assert_eq!(e[1], F::from_u64(7));
        assert_eq!(e[2], F::from_u64(11));
    }

    #[test]
    fn dense_poly_layer_goldilocks() {
        bind_matches_recurrence::<Goldilocks>(0x9001);
        evals_extrapolate::<Goldilocks>();
        for log_len in 1..=8 {
            mini_sumcheck_consistency::<Goldilocks>(0xA000 + log_len as u64, log_len);
        }
    }

    #[test]
    fn dense_poly_layer_fp3() {
        bind_matches_recurrence::<GoldilocksFp3>(0xB001);
        evals_extrapolate::<GoldilocksFp3>();
        for log_len in 1..=8 {
            mini_sumcheck_consistency::<GoldilocksFp3>(0xC000 + log_len as u64, log_len);
        }
    }
}
