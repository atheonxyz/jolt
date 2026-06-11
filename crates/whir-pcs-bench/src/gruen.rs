//! Minimal Gruen / Dao-Thaler split-equality polynomial for the BN254 GKR
//! round-polynomial construction (`gkr_bn254.rs`).
//!
//! This is a thin, field-element-typed reimplementation of the subset of
//! `jolt_poly::GruenSplitEqPolynomial` that the fractional-GKR per-layer
//! sumcheck needs: `new`, `bind`, `current_scalar`, and `gruen_poly_deg_3`.
//! The arithmetic is copied verbatim from the upstream split-eq implementation,
//! dropping the `e_in`/`e_out` prefix tables (which the GKR does not use — it
//! maintains its own `eq_unbound` table).
//!
//! jolt-main's `crates/jolt-poly` does not yet export `GruenSplitEqPolynomial`
//! (only the legacy `jolt-core` carries one, and it is challenge-typed). The
//! in-development `jolt-prover` adds a field-typed `jolt_poly::GruenSplitEqPolynomial`
//! with this exact API, so once it lands on jolt-main's `jolt-poly` this module
//! can be deleted and replaced with `use jolt_poly::GruenSplitEqPolynomial;`.

use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};

#[derive(Clone, Debug)]
pub struct GruenSplitEqPolynomial<F: Field> {
    current_index: usize,
    current_scalar: F,
    w: Vec<F>,
    binding_order: BindingOrder,
}

impl<F: Field> GruenSplitEqPolynomial<F> {
    pub fn new(w: &[F], binding_order: BindingOrder) -> Self {
        assert!(!w.is_empty(), "split eq requires at least one variable");
        let current_index = match binding_order {
            BindingOrder::LowToHigh => w.len(),
            BindingOrder::HighToLow => 0,
        };
        Self {
            current_index,
            current_scalar: F::one(),
            w: w.to_vec(),
            binding_order,
        }
    }

    pub fn current_scalar(&self) -> F {
        self.current_scalar
    }

    fn current_w(&self) -> F {
        match self.binding_order {
            BindingOrder::LowToHigh => self.w[self.current_index - 1],
            BindingOrder::HighToLow => self.w[self.current_index],
        }
    }

    pub fn bind(&mut self, r: F) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                let w = self.w[self.current_index - 1];
                let prod = w * r;
                self.current_scalar *= F::one() - w - r + prod + prod;
                self.current_index -= 1;
            }
            BindingOrder::HighToLow => {
                let w = self.w[self.current_index];
                let prod = w * r;
                self.current_scalar *= F::one() - w - r + prod + prod;
                self.current_index += 1;
            }
        }
    }

    /// Degree-3 round polynomial from the eq factorization, matching upstream
    /// `GruenSplitEqPolynomial::gruen_poly_deg_3`.
    pub fn gruen_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> UnivariatePoly<F> {
        let eq_eval_1 = self.current_scalar * self.current_w();
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_slope = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_slope;
        let eq_eval_3 = eq_eval_2 + eq_slope;

        let quadratic_eval_0 = q_constant;
        let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        // `jolt_field::Field` provides `*` but not `/`; divide via the
        // `Invertible` supertrait (eq_eval_1 is nonzero on the Gruen path).
        let quadratic_eval_1 = cubic_eval_1 * eq_eval_1.inverse().expect("eq_eval_1 nonzero");
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

        UnivariatePoly::from_evals(&[
            cubic_eval_0,
            cubic_eval_1,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ])
    }
}
