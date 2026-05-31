//! Spartan instruction-input virtualization sumcheck — ported from jolt-core's
//! `zkvm/spartan/instruction_input.rs` onto [`crate::framework`] over the lean `Field`
//! (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves (degree-3) that the lookup-operand inputs are the flag-selected register/PC/imm values:
//!
//! ```text
//! Σ_j eq(r_cycle, j)·( RightInput(j) + γ·LeftInput(j) ) = right_claim + γ·left_claim,
//!   LeftInput  = left_is_rs1·rs1 + left_is_pc·unexpanded_pc,
//!   RightInput = right_is_rs2·rs2 + right_is_imm·imm.
//! ```
//!
//! The input claim batches the `Left`/`RightInstructionInput` openings from
//! [`SumcheckId::SpartanProductVirtualization`]; the eight flag/value openings are cached under
//! [`SumcheckId::InstructionInputVirtualization`].
//!
//! **Decoupled from the trace** (the M5 convention): takes the eight materialized flag/value
//! columns; the verifier reconstructs `Left/RightInput` from the cached openings and re-evals
//! `eq(r_cycle, ρ)`. jolt-core keys the flag openings with `InstructionFlags(...)` variants; the
//! decoupled port maps them to distinct existing variants.

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;

use crate::framework::accumulator::{OpeningAccumulator, Openings, SumcheckId, VirtualPolynomial};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// The eight flag/value opening keys, in order
/// `[left_is_rs1, rs1, left_is_pc, upc, right_is_rs2, rs2, right_is_imm, imm]`. jolt-core keys the
/// flags with `InstructionFlags(...)`; the decoupled port maps them to distinct existing variants.
const KEYS: [VirtualPolynomial; 8] = [
    VirtualPolynomial::LeftLookupOperand,
    VirtualPolynomial::Rs1Value,
    VirtualPolynomial::RightLookupOperand,
    VirtualPolynomial::UnexpandedPC,
    VirtualPolynomial::InstructionRaf,
    VirtualPolynomial::Rs2Value,
    VirtualPolynomial::InstructionRafFlag,
    VirtualPolynomial::Imm,
];

/// Batching/opening parameters (matches jolt-core `InstructionInputParams`).
#[derive(Clone, Debug)]
pub struct InstructionInputParams<F: Field> {
    pub r_cycle: Vec<F>,
    pub gamma: F,
}

impl<F: Field> InstructionInputParams<F> {
    /// Reads `r_cycle` from the `LeftInstructionInput`@SpartanProductVirtualization opening and
    /// draws `γ`.
    pub fn new(
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript<Challenge = F>,
    ) -> Self {
        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanProductVirtualization,
        );
        let gamma = transcript.challenge();
        Self {
            r_cycle: r_cycle.r,
            gamma,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, left) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, right) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::SpartanProductVirtualization,
        );
        right + self.gamma * left
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) polynomials.
pub struct InstructionInput<F: Field> {
    pub params: InstructionInputParams<F>,
    eq: MultilinearPolynomial<F>,
    /// `[left_is_rs1, rs1, left_is_pc, upc, right_is_rs2, rs2, right_is_imm, imm]`.
    cols: [MultilinearPolynomial<F>; 8],
}

impl<F: Field> InstructionInput<F> {
    /// Build the prover instance from the eight materialized flag/value columns (each length `T`).
    pub fn new_prover(params: InstructionInputParams<F>, cols: [Vec<F>; 8]) -> Self {
        let eq = EqPolynomial::<F>::evals(&params.r_cycle, None);
        Self {
            params,
            eq: MultilinearPolynomial::from(eq),
            cols: cols.map(MultilinearPolynomial::from),
        }
    }

    pub fn new_verifier(params: InstructionInputParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::zero()]);
        Self {
            params,
            eq: dummy(),
            cols: std::array::from_fn(|_| dummy()),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for InstructionInput<F> {
    fn num_rounds(&self) -> usize {
        self.params.r_cycle.len()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-3: eq·(right_input + γ·left_input) ⇒ 4 evaluation points; unreduced accumulation.
        let gamma = self.params.gamma;
        let half = self.eq.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 4];
        for j in 0..half {
            let eq = self
                .eq
                .sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh);
            let c: [[F; 4]; 8] = std::array::from_fn(|i| {
                self.cols[i].sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh)
            });
            for k in 0..4 {
                let left = c[0][k] * c[1][k] + c[2][k] * c[3][k];
                let right = c[4][k] * c[5][k] + c[6][k] * c[7][k];
                acc[k].fmadd(eq[k], right + gamma * left);
            }
        }
        let evals: [F; 4] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
        for poly in &mut self.cols {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        for (i, key) in KEYS.iter().enumerate() {
            accumulator.append_virtual(
                *key,
                SumcheckId::InstructionInputVirtualization,
                point.clone(),
                self.cols[i].final_sumcheck_claim(),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let e2 = EqPolynomial::<F>::mle(&self.params.r_cycle, &point.r);
        let c = |i: usize| {
            accumulator
                .get_virtual_polynomial_opening(KEYS[i], SumcheckId::InstructionInputVirtualization)
                .1
        };
        let left = c(0) * c(1) + c(2) * c(3);
        let right = c(4) * c(5) + c(6) * c(7);
        e2 * (right + self.params.gamma * left)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::accumulator::OpeningPoint;
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

    fn round_trip(seed: u64, log_t: usize) {
        let mut rng = Rng(seed);
        let t = 1usize << log_t;
        let r_cycle = rand_vec(&mut rng, log_t);
        let cols: [Vec<F>; 8] = std::array::from_fn(|_| rand_vec(&mut rng, t));

        // Seed Left/RightInstructionInput@SpartanProductVirtualization (γ-independent) so
        // input_claim == Σ eq·(right_input + γ·left_input).
        let eq = EqPolynomial::<F>::evals(&r_cycle, None);
        let mut left_claim = F::from_u64(0);
        let mut right_claim = F::from_u64(0);
        for j in 0..t {
            left_claim += eq[j] * (cols[0][j] * cols[1][j] + cols[2][j] * cols[3][j]);
            right_claim += eq[j] * (cols[4][j] * cols[5][j] + cols[6][j] * cols[7][j]);
        }
        let seed_acc = |acc: &mut Openings<F>| {
            let pt = OpeningPoint::new(r_cycle.clone());
            acc.append_virtual(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanProductVirtualization,
                pt.clone(),
                left_claim,
            );
            acc.append_virtual(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanProductVirtualization,
                pt,
                right_claim,
            );
        };

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut prover_t = Blake2bTranscript::<F>::new(b"spartan-instruction-input");
        let params = InstructionInputParams::new(&prover_acc, &mut prover_t);
        let input_claim = params.input_claim(&prover_acc);
        let mut prover = InstructionInput::new_prover(params, cols.clone());
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"spartan-instruction-input");
        let vparams = InstructionInputParams::new(&verifier_acc, &mut verifier_t);
        let verifier = InstructionInput::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("instruction-input must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for key in KEYS {
            let (pt, cl) = prover_acc
                .get_virtual_polynomial_opening(key, SumcheckId::InstructionInputVirtualization);
            verifier_acc.append_virtual(key, SumcheckId::InstructionInputVirtualization, pt, cl);
        }
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match eq·(right + γ·left)"
        );
    }

    #[test]
    fn spartan_instruction_input_round_trip() {
        for log_t in 1..=8 {
            round_trip(0x1417 + log_t as u64, log_t);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_t = 4;
        let mut rng = Rng(0x14FE);
        let t = 1usize << log_t;
        let r_cycle = rand_vec(&mut rng, log_t);
        let cols: [Vec<F>; 8] = std::array::from_fn(|_| rand_vec(&mut rng, t));
        let eq = EqPolynomial::<F>::evals(&r_cycle, None);
        let mut left_claim = F::from_u64(0);
        let mut right_claim = F::from_u64(0);
        for j in 0..t {
            left_claim += eq[j] * (cols[0][j] * cols[1][j] + cols[2][j] * cols[3][j]);
            right_claim += eq[j] * (cols[4][j] * cols[5][j] + cols[6][j] * cols[7][j]);
        }
        let mut acc = Openings::<F>::new(log_t);
        let pt = OpeningPoint::new(r_cycle);
        acc.append_virtual(
            VirtualPolynomial::LeftInstructionInput,
            SumcheckId::SpartanProductVirtualization,
            pt.clone(),
            left_claim,
        );
        acc.append_virtual(
            VirtualPolynomial::RightInstructionInput,
            SumcheckId::SpartanProductVirtualization,
            pt,
            right_claim,
        );
        let mut prover_t = Blake2bTranscript::<F>::new(b"t");
        let params = InstructionInputParams::new(&acc, &mut prover_t);
        let input_claim = params.input_claim(&acc);
        let mut prover = InstructionInput::new_prover(params, cols);
        let (mut proof, _) = prove(&mut prover, &mut acc, &mut prover_t);
        proof.round_polynomials[0] = UnivariatePoly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"t");
        assert!(
            verify(&claim, &proof, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }
}
