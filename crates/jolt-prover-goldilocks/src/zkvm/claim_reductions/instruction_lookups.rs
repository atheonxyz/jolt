//! Instruction-lookups claim-reduction sumcheck — ported from jolt-core's
//! `zkvm/claim_reductions/instruction_lookups.rs` onto the framework ([`crate::framework`]) over the
//! lean `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Aggregates the five instruction-lookup claims emitted by the Spartan outer sumcheck (stage 1) at
//! the shared point `r_spartan` into a single opening point `ρ`:
//!
//! ```text
//! input:  LookupOutput(r_spartan) + γ·LeftLookupOperand + γ²·RightLookupOperand
//!         + γ³·LeftInstructionInput + γ⁴·RightInstructionInput   (all at r_spartan)
//! sumcheck (log T rounds, degree 2):
//!   Σ_j eq(r_spartan, j)·( LookupOutput(j) + γ·LeftLookupOperand(j) + γ²·RightLookupOperand(j)
//!                          + γ³·LeftInstructionInput(j) + γ⁴·RightInstructionInput(j) )
//! output: the five claims at ρ (SumcheckId::InstructionClaimReduction)
//! ```
//!
//! This is the **single-phase** form (jolt-core's prefix/suffix two-phase materialization is a perf
//! optimization deferred with the trace witness-gen, matching [`super::increments`]). The five value
//! columns are taken pre-materialized (`Fp3`), decoupling from the trace → lookup-operand extraction.

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;

use crate::framework::accumulator::{
    OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial, BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// The five instruction-lookup value polynomials, in `γ`-power order.
const POLYS: [VirtualPolynomial; 5] = [
    VirtualPolynomial::LookupOutput,
    VirtualPolynomial::LeftLookupOperand,
    VirtualPolynomial::RightLookupOperand,
    VirtualPolynomial::LeftInstructionInput,
    VirtualPolynomial::RightInstructionInput,
];

/// Fiat-Shamir + opening-point parameters, fetched from the accumulator (matches jolt-core
/// `InstructionLookupsClaimReductionSumcheckParams`).
#[derive(Clone, Debug)]
pub struct InstructionLookupsClaimReductionParams<F: Field> {
    /// `[1, γ, γ², γ³, γ⁴]`.
    pub coeffs: [F; 5],
    pub n_cycle_vars: usize,
    /// The shared stage-1 Spartan-outer challenge point (BIG_ENDIAN).
    pub r_spartan: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: Field> InstructionLookupsClaimReductionParams<F> {
    pub fn new(
        n_cycle_vars: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript<Challenge = F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let g2 = gamma * gamma;
        let coeffs = [F::from_u64(1), gamma, g2, g2 * gamma, g2 * g2];
        let (r_spartan, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        Self {
            coeffs,
            n_cycle_vars,
            r_spartan,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        POLYS
            .iter()
            .zip(self.coeffs.iter())
            .map(|(&poly, &coeff)| {
                let (_, claim) =
                    accumulator.get_virtual_polynomial_opening(poly, SumcheckId::SpartanOuter);
                coeff * claim
            })
            .fold(F::zero(), |a, b| a + b)
    }
}

/// Prover/verifier instance. The prover holds the five value columns + the `eq(r_spartan,·)` column;
/// the verifier carries the same `params` and ignores the polynomials.
pub struct InstructionLookupsClaimReduction<F: Field> {
    pub params: InstructionLookupsClaimReductionParams<F>,
    polys: [MultilinearPolynomial<F>; 5],
    eq: MultilinearPolynomial<F>,
}

impl<F: Field> InstructionLookupsClaimReduction<F> {
    /// Build the prover instance from the five value columns (each length `2^n_cycle_vars`), in the
    /// [`POLYS`] order.
    pub fn new_prover(
        params: InstructionLookupsClaimReductionParams<F>,
        columns: [Vec<F>; 5],
    ) -> Self {
        let eq = EqPolynomial::<F>::evals(&params.r_spartan.r, None);
        Self {
            params,
            polys: columns.map(MultilinearPolynomial::from),
            eq: MultilinearPolynomial::from(eq),
        }
    }

    /// Build a verifier instance (no polynomials; `expected_output_claim` reads cached reduced
    /// openings + recomputes `eq(ρ, r_spartan)`).
    pub fn new_verifier(params: InstructionLookupsClaimReductionParams<F>) -> Self {
        Self {
            params,
            polys: std::array::from_fn(|_| MultilinearPolynomial::from(vec![F::zero()])),
            eq: MultilinearPolynomial::from(vec![F::zero()]),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for InstructionLookupsClaimReduction<F> {
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
        let coeffs = self.params.coeffs;
        let half = self.eq.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for j in 0..half {
            let e = self
                .eq
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let p: [[F; 3]; 5] = std::array::from_fn(|i| {
                self.polys[i].sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh)
            });
            for k in 0..3 {
                let combo = (0..5).fold(F::zero(), |a, i| a + coeffs[i] * p[i][k]);
                acc[k].fmadd(combo, e[k]);
            }
        }
        let evals: [F; 3] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        for poly in &mut self.polys {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        for (i, &poly) in POLYS.iter().enumerate() {
            accumulator.append_virtual(
                poly,
                SumcheckId::InstructionClaimReduction,
                point.clone(),
                self.polys[i].final_sumcheck_claim(),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let eq_eval = EqPolynomial::<F>::mle(&point.r, &self.params.r_spartan.r);
        let combined = POLYS
            .iter()
            .zip(self.params.coeffs.iter())
            .map(|(&poly, &coeff)| {
                let (_, claim) = accumulator
                    .get_virtual_polynomial_opening(poly, SumcheckId::InstructionClaimReduction);
                coeff * claim
            })
            .fold(F::zero(), |a, b| a + b);
        eq_eval * combined
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

    fn seed_acc(acc: &mut Openings<F>, r_spartan: &[F], columns: &[Vec<F>; 5]) {
        let eq = EqPolynomial::<F>::evals(r_spartan, None);
        let point = OpeningPoint::new(r_spartan.to_vec());
        for (i, &poly) in POLYS.iter().enumerate() {
            acc.append_virtual(
                poly,
                SumcheckId::SpartanOuter,
                point.clone(),
                dot(&columns[i], &eq),
            );
        }
    }

    fn round_trip(seed: u64, log_t: usize) {
        let mut rng = Rng(seed);
        let t = 1usize << log_t;
        let columns: [Vec<F>; 5] = std::array::from_fn(|_| rand_vec(&mut rng, t));
        let r_spartan = rand_vec(&mut rng, log_t);

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc, &r_spartan, &columns);
        let mut prover_t = Blake2bTranscript::<F>::new(b"instr-lookups-claim-reduce");
        let params = InstructionLookupsClaimReductionParams::new(log_t, &prover_acc, &mut prover_t);
        let input_claim = params.input_claim(&prover_acc);
        let mut prover =
            InstructionLookupsClaimReduction::new_prover(params.clone(), columns.clone());
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc, &r_spartan, &columns);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"instr-lookups-claim-reduce");
        let vparams =
            InstructionLookupsClaimReductionParams::new(log_t, &verifier_acc, &mut verifier_t);
        let verifier = InstructionLookupsClaimReduction::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("instruction reduction must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for &poly in &POLYS {
            let (_, claim) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::InstructionClaimReduction);
            verifier_acc.append_virtual(
                poly,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::new(point.clone()),
                claim,
            );
        }

        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match eq(ρ,r_spartan)·Σ γⁱ·claimᵢ"
        );

        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        for (i, &poly) in POLYS.iter().enumerate() {
            let (_, claim) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::InstructionClaimReduction);
            assert_eq!(
                claim,
                dot(&columns[i], &eq_rho),
                "{poly:?}(ρ) matches direct MLE"
            );
        }
    }

    #[test]
    fn instruction_lookups_claim_reduction_round_trip() {
        for log_t in 1..=7 {
            round_trip(0xC000u64.wrapping_add(log_t as u64), log_t);
        }
    }

    /// A tampered reduced opening (corrupted `RightInstructionInput(ρ)`) breaks the output check.
    #[test]
    fn tampered_reduced_opening_rejected() {
        let log_t = 5;
        let mut rng = Rng(0x7373);
        let t = 1usize << log_t;
        let columns: [Vec<F>; 5] = std::array::from_fn(|_| rand_vec(&mut rng, t));
        let r_spartan = rand_vec(&mut rng, log_t);

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc, &r_spartan, &columns);
        let mut prover_t = Blake2bTranscript::<F>::new(b"instr-lookups-claim-reduce");
        let params = InstructionLookupsClaimReductionParams::new(log_t, &prover_acc, &mut prover_t);
        let mut prover = InstructionLookupsClaimReduction::new_prover(params.clone(), columns);
        let (_, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let verifier = InstructionLookupsClaimReduction::new_verifier(params);
        let point = OpeningPoint::new(challenges.clone());
        let build = |tamper: bool| {
            let mut acc = Openings::<F>::new(log_t);
            for &poly in &POLYS {
                let (_, mut claim) = prover_acc
                    .get_virtual_polynomial_opening(poly, SumcheckId::InstructionClaimReduction);
                if tamper && poly == VirtualPolynomial::RightInstructionInput {
                    claim += F::from_u64(1);
                }
                acc.append_virtual(
                    poly,
                    SumcheckId::InstructionClaimReduction,
                    point.clone(),
                    claim,
                );
            }
            verifier.expected_output_claim(&acc, &challenges)
        };
        assert_ne!(
            build(false),
            build(true),
            "tampered claim must change the output"
        );
    }
}
