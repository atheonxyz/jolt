//! Spartan shift (PC) sumcheck — ported from jolt-core's `zkvm/spartan/shift.rs` onto
//! [`crate::framework`] over the lean `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves the batched `eq+1`-shift identity over cycles `j` (degree-2):
//!
//! ```text
//! Σ_j [ EqPlusOne(r_outer, j)·(s0 + γ·s1 + γ²·s2 + γ³·s3)(j) + γ⁴·EqPlusOne(r_product, j)·(1 − s4(j)) ]
//!   = NextUnexpandedPC + γ·NextPC + γ²·NextIsVirtual + γ³·NextIsFirstInSequence + γ⁴·(1 − NextIsNoop),
//! ```
//!
//! where `s0..s4` are the `f(j+1)`-aligned shift MLEs (`UnexpandedPC`, `PC`, `IsVirtual`,
//! `IsFirstInSequence`, `IsNoop`), `EqPlusOne(r, ·)` is the successor MLE (1 iff index `= j+1`),
//! and the right-hand `Next*` claims come from the Spartan outer / product-virtualization
//! sumchecks. Two `eq+1` points are used: `r_outer` (terms 0–3) and `r_product` (term 4).
//!
//! **Decoupled from the trace** (the M5 convention): takes the materialized shift columns; the
//! `EqPlusOne` tables are built via [`jolt_poly::EqPlusOnePolynomial::evals`] and bound `LowToHigh`.
//! jolt-core's prefix-suffix two-phase `EqPlusOne` materialization is a perf opt deferred (OPT-E).
//! The five shift openings are cached under [`SumcheckId::SpartanShift`]; the flag-carrying
//! `VirtualPolynomial` variants jolt-core keys them with are mapped to distinct existing variants.

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPlusOnePolynomial, UnivariatePoly};

use crate::framework::accumulator::{OpeningAccumulator, Openings, SumcheckId, VirtualPolynomial};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// The five shift-column opening keys (cached under [`SumcheckId::SpartanShift`]). jolt-core uses
/// `UnexpandedPC`/`PC`/`OpFlags(VirtualInstruction)`/`OpFlags(IsFirstInSequence)`/
/// `InstructionFlags(IsNoop)`; the decoupled port maps them to these distinct existing variants.
const SHIFT_KEYS: [VirtualPolynomial; 5] = [
    VirtualPolynomial::UnexpandedPC,
    VirtualPolynomial::PC,
    VirtualPolynomial::NextIsVirtual,
    VirtualPolynomial::NextIsFirstInSequence,
    VirtualPolynomial::NextIsNoop,
];

/// Batching/opening parameters (matches jolt-core `ShiftSumcheckParams`).
#[derive(Clone, Debug)]
pub struct SpartanShiftParams<F: Field> {
    /// `[γ⁰, …, γ⁴]`.
    pub gamma_powers: [F; 5],
    pub log_t: usize,
    pub r_outer: Vec<F>,
    pub r_product: Vec<F>,
}

impl<F: Field> SpartanShiftParams<F> {
    /// Draws `[γ⁰..γ⁴]` and reads `r_outer` (from `NextPC`@SpartanOuter) and `r_product` (from
    /// `NextIsNoop`@SpartanProductVirtualization), each truncated to the `log_t` cycle variables.
    pub fn new(
        accumulator: &dyn OpeningAccumulator<F>,
        log_t: usize,
        transcript: &mut impl Challenge<F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let mut gp = [F::one(); 5];
        for i in 1..5 {
            gp[i] = gp[i - 1] * gamma;
        }
        let (r_outer, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (r_outer, _) = r_outer.split_at(log_t);
        let (r_product, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanProductVirtualization,
        );
        let (r_product, _) = r_product.split_at(log_t);
        Self {
            gamma_powers: gp,
            log_t,
            r_outer: r_outer.r,
            r_product: r_product.r,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let g = &self.gamma_powers;
        let (_, nuexp) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, npc) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, nvirt) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, nfirst) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsFirstInSequence,
            SumcheckId::SpartanOuter,
        );
        let (_, nnoop) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanProductVirtualization,
        );
        nuexp + g[1] * npc + g[2] * nvirt + g[3] * nfirst + g[4] * (F::one() - nnoop)
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) polynomials.
pub struct SpartanShift<F: Field> {
    pub params: SpartanShiftParams<F>,
    eqp_outer: MultilinearPolynomial<F>,
    eqp_product: MultilinearPolynomial<F>,
    /// `s0..s4`: the `f(j+1)`-aligned shift columns (length `T`).
    shift: [MultilinearPolynomial<F>; 5],
}

impl<F: Field> SpartanShift<F> {
    /// Build the prover instance from the five materialized shift columns (each length `T`).
    pub fn new_prover(params: SpartanShiftParams<F>, shift_cols: [Vec<F>; 5]) -> Self {
        let (_, eqp_o) = EqPlusOnePolynomial::<F>::evals(&params.r_outer, None);
        let (_, eqp_p) = EqPlusOnePolynomial::<F>::evals(&params.r_product, None);
        Self {
            params,
            eqp_outer: MultilinearPolynomial::from(eqp_o),
            eqp_product: MultilinearPolynomial::from(eqp_p),
            shift: shift_cols.map(MultilinearPolynomial::from),
        }
    }

    pub fn new_verifier(params: SpartanShiftParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::zero()]);
        Self {
            params,
            eqp_outer: dummy(),
            eqp_product: dummy(),
            shift: std::array::from_fn(|_| dummy()),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for SpartanShift<F> {
    fn num_rounds(&self) -> usize {
        self.params.log_t
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-2 ⇒ 3 evaluation points; unreduced accumulation.
        let g = &self.params.gamma_powers;
        let half = self.eqp_outer.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for j in 0..half {
            let eo = self
                .eqp_outer
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let ep = self
                .eqp_product
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let s: [[F; 3]; 5] = std::array::from_fn(|i| {
                self.shift[i].sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh)
            });
            for k in 0..3 {
                let outer_terms = s[0][k] + g[1] * s[1][k] + g[2] * s[2][k] + g[3] * s[3][k];
                acc[k].fmadd(eo[k], outer_terms);
                acc[k].fmadd(g[4] * ep[k], F::one() - s[4][k]);
            }
        }
        let evals: [F; 3] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.eqp_outer.bind_parallel(r, BindingOrder::LowToHigh);
        self.eqp_product.bind_parallel(r, BindingOrder::LowToHigh);
        for poly in &mut self.shift {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        for (i, key) in SHIFT_KEYS.iter().enumerate() {
            accumulator.append_virtual(
                *key,
                SumcheckId::SpartanShift,
                point.clone(),
                self.shift[i].final_sumcheck_claim(),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let g = &self.params.gamma_powers;
        let eqp_outer =
            EqPlusOnePolynomial::<F>::new(self.params.r_outer.clone()).evaluate(&point.r);
        let eqp_product =
            EqPlusOnePolynomial::<F>::new(self.params.r_product.clone()).evaluate(&point.r);

        let claim = |i: usize| {
            accumulator
                .get_virtual_polynomial_opening(SHIFT_KEYS[i], SumcheckId::SpartanShift)
                .1
        };
        let outer_terms = claim(0) + g[1] * claim(1) + g[2] * claim(2) + g[3] * claim(3);
        eqp_outer * outer_terms + g[4] * eqp_product * (F::one() - claim(4))
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::accumulator::OpeningPoint;
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

    fn round_trip(seed: u64, log_t: usize) {
        let mut rng = Rng(seed);
        let t = 1usize << log_t;
        let r_outer = rand_vec(&mut rng, log_t);
        let r_product = rand_vec(&mut rng, log_t);
        let shift_cols: [Vec<F>; 5] = std::array::from_fn(|_| rand_vec(&mut rng, t));

        // Seed the five Next* claims (γ-independent) so input_claim == the actual shift sum.
        let (_, eqp_o) = EqPlusOnePolynomial::<F>::evals(&r_outer, None);
        let (_, eqp_p) = EqPlusOnePolynomial::<F>::evals(&r_product, None);
        let dot = |a: &[F], b: &[F]| {
            a.iter()
                .zip(b.iter())
                .fold(F::from_u64(0), |s, (x, y)| s + *x * *y)
        };
        let nuexp = dot(&eqp_o, &shift_cols[0]);
        let npc = dot(&eqp_o, &shift_cols[1]);
        let nvirt = dot(&eqp_o, &shift_cols[2]);
        let nfirst = dot(&eqp_o, &shift_cols[3]);
        // term4 = Σ eqp_p·(1−s4); seed NextIsNoop = 1 − term4 so γ⁴·(1−NextIsNoop) = γ⁴·term4.
        let term4 = eqp_p
            .iter()
            .zip(shift_cols[4].iter())
            .fold(F::from_u64(0), |s, (e, v)| s + *e * (F::from_u64(1) - *v));
        let nnoop = F::from_u64(1) - term4;

        let seed_acc = |acc: &mut Openings<F>| {
            let pt_o = OpeningPoint::new(r_outer.clone());
            let pt_p = OpeningPoint::new(r_product.clone());
            acc.append_virtual(
                VirtualPolynomial::NextUnexpandedPC,
                SumcheckId::SpartanOuter,
                pt_o.clone(),
                nuexp,
            );
            acc.append_virtual(
                VirtualPolynomial::NextPC,
                SumcheckId::SpartanOuter,
                pt_o.clone(),
                npc,
            );
            acc.append_virtual(
                VirtualPolynomial::NextIsVirtual,
                SumcheckId::SpartanOuter,
                pt_o.clone(),
                nvirt,
            );
            acc.append_virtual(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
                pt_o,
                nfirst,
            );
            acc.append_virtual(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::SpartanProductVirtualization,
                pt_p,
                nnoop,
            );
        };

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut prover_t = ProverTranscript::new("spartan-shift");
        let params = SpartanShiftParams::new(&prover_acc, log_t, &mut prover_t);
        let input_claim = params.input_claim(&prover_acc);
        let mut prover = SpartanShift::new_prover(params, shift_cols.clone());
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut verifier_t = VerifierTranscript::new("spartan-shift", &narg);
        let vparams = SpartanShiftParams::new(&verifier_acc, log_t, &mut verifier_t);
        let verifier = SpartanShift::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("spartan shift must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for key in SHIFT_KEYS {
            let (pt, c) = prover_acc.get_virtual_polynomial_opening(key, SumcheckId::SpartanShift);
            verifier_acc.append_virtual(key, SumcheckId::SpartanShift, pt, c);
        }
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match the eq+1 shift formula"
        );
    }

    #[test]
    fn spartan_shift_round_trip() {
        for log_t in 1..=8 {
            round_trip(0x5417 + log_t as u64, log_t);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_t = 4;
        let mut rng = Rng(0x54FE);
        let t = 1usize << log_t;
        let r_outer = rand_vec(&mut rng, log_t);
        let r_product = rand_vec(&mut rng, log_t);
        let shift_cols: [Vec<F>; 5] = std::array::from_fn(|_| rand_vec(&mut rng, t));
        let (_, eqp_o) = EqPlusOnePolynomial::<F>::evals(&r_outer, None);
        let (_, eqp_p) = EqPlusOnePolynomial::<F>::evals(&r_product, None);
        let dot = |a: &[F], b: &[F]| {
            a.iter()
                .zip(b.iter())
                .fold(F::from_u64(0), |s, (x, y)| s + *x * *y)
        };
        let term4 = eqp_p
            .iter()
            .zip(shift_cols[4].iter())
            .fold(F::from_u64(0), |s, (e, v)| s + *e * (F::from_u64(1) - *v));
        let mut acc = Openings::<F>::new(log_t);
        let pt_o = OpeningPoint::new(r_outer.clone());
        let pt_p = OpeningPoint::new(r_product);
        acc.append_virtual(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
            pt_o.clone(),
            dot(&eqp_o, &shift_cols[0]),
        );
        acc.append_virtual(
            VirtualPolynomial::NextPC,
            SumcheckId::SpartanOuter,
            pt_o.clone(),
            dot(&eqp_o, &shift_cols[1]),
        );
        acc.append_virtual(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
            pt_o.clone(),
            dot(&eqp_o, &shift_cols[2]),
        );
        acc.append_virtual(
            VirtualPolynomial::NextIsFirstInSequence,
            SumcheckId::SpartanOuter,
            pt_o,
            dot(&eqp_o, &shift_cols[3]),
        );
        acc.append_virtual(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanProductVirtualization,
            pt_p,
            F::from_u64(1) - term4,
        );

        let mut prover_t = ProverTranscript::new("t");
        let params = SpartanShiftParams::new(&acc, log_t, &mut prover_t);
        let input_claim = params.input_claim(&acc);
        let mut prover = SpartanShift::new_prover(params, shift_cols);
        let _ = prove(&mut prover, &mut acc, &mut prover_t);
        let mut narg = prover_t.into_proof();
        narg.narg_string[0] ^= 0x01;
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("t", &narg);
        // Replay the prover's pre-round γ squeeze to keep the verifier transcript aligned.
        let _ = SpartanShiftParams::new(&acc, log_t, &mut verifier_t);
        assert!(
            verify(&claim, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }
}
