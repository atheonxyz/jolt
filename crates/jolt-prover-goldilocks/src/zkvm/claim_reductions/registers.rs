//! Registers claim-reduction sumcheck — ported from jolt-core's
//! `zkvm/claim_reductions/registers.rs` onto the framework ([`crate::framework`]) over the lean
//! `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Aggregates the three register-value claims emitted by the Spartan outer sumcheck (stage 1) at the
//! shared point `r_spartan` into a single opening point `ρ`:
//!
//! ```text
//! input:  RdWriteValue(r_spartan) + γ·Rs1Value(r_spartan) + γ²·Rs2Value(r_spartan)
//! sumcheck (log T rounds, degree 2):
//!   Σ_j eq(r_spartan, j)·(RdWriteValue(j) + γ·Rs1Value(j) + γ²·Rs2Value(j))
//! output: RdWriteValue(ρ), Rs1Value(ρ), Rs2Value(ρ)
//! ```
//!
//! This is the **single-phase** form (the jolt-core prefix/suffix two-phase materialization is a
//! perf optimization deferred with the trace witness-gen, matching [`super::increments`]). The
//! `RdWriteValue`/`Rs1Value`/`Rs2Value` value columns are taken pre-materialized (`Fp3`), decoupling
//! the sumcheck from the trace → register-file extraction (M8).

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial, BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// Fiat-Shamir + opening-point parameters, fetched from the accumulator (matches jolt-core
/// `RegistersClaimReductionSumcheckParams`).
#[derive(Clone, Debug)]
pub struct RegistersClaimReductionParams<F: Field> {
    pub gamma: F,
    pub gamma_sqr: F,
    pub n_cycle_vars: usize,
    /// The shared stage-1 Spartan-outer challenge point (BIG_ENDIAN), where the three input claims
    /// were opened.
    pub r_spartan: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: Field> RegistersClaimReductionParams<F> {
    pub fn new(
        n_cycle_vars: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Challenge<F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let gamma_sqr = gamma * gamma;
        let (r_spartan, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        Self {
            gamma,
            gamma_sqr,
            n_cycle_vars,
            r_spartan,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rd_write_value) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );
        let (_, rs1_value) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rs2_value) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        rd_write_value + self.gamma * rs1_value + self.gamma_sqr * rs2_value
    }
}

/// Prover/verifier instance. The prover holds the materialized value columns + the `eq(r_spartan,·)`
/// column; the verifier instance carries the same `params` and ignores the (empty) polynomials.
pub struct RegistersClaimReduction<F: Field> {
    pub params: RegistersClaimReductionParams<F>,
    rd_write_value: MultilinearPolynomial<F>,
    rs1_value: MultilinearPolynomial<F>,
    rs2_value: MultilinearPolynomial<F>,
    eq: MultilinearPolynomial<F>,
}

impl<F: Field> RegistersClaimReduction<F> {
    /// Build the prover instance from the materialized register-value columns (length `2^n_cycle_vars`).
    pub fn new_prover(
        params: RegistersClaimReductionParams<F>,
        rd_write_value: Vec<F>,
        rs1_value: Vec<F>,
        rs2_value: Vec<F>,
    ) -> Self {
        let eq = EqPolynomial::<F>::evals(&params.r_spartan.r, None);
        Self {
            params,
            rd_write_value: MultilinearPolynomial::from(rd_write_value),
            rs1_value: MultilinearPolynomial::from(rs1_value),
            rs2_value: MultilinearPolynomial::from(rs2_value),
            eq: MultilinearPolynomial::from(eq),
        }
    }

    /// Build a verifier instance (no polynomials needed; `expected_output_claim` reads cached
    /// reduced openings + recomputes `eq(ρ, r_spartan)`).
    pub fn new_verifier(params: RegistersClaimReductionParams<F>) -> Self {
        Self {
            params,
            rd_write_value: MultilinearPolynomial::from(vec![F::zero()]),
            rs1_value: MultilinearPolynomial::from(vec![F::zero()]),
            rs2_value: MultilinearPolynomial::from(vec![F::zero()]),
            eq: MultilinearPolynomial::from(vec![F::zero()]),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for RegistersClaimReduction<F> {
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
        let gamma = self.params.gamma;
        let gamma_sqr = self.params.gamma_sqr;
        let half = self.eq.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for j in 0..half {
            let e = self
                .eq
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let rd = self
                .rd_write_value
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let rs1 = self
                .rs1_value
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            let rs2 = self
                .rs2_value
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            for k in 0..3 {
                let combo = rd[k] + gamma * rs1[k] + gamma_sqr * rs2[k];
                acc[k].fmadd(combo, e[k]);
            }
        }
        let evals: [F; 3] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.rd_write_value
            .bind_parallel(r, BindingOrder::LowToHigh);
        self.rs1_value.bind_parallel(r, BindingOrder::LowToHigh);
        self.rs2_value.bind_parallel(r, BindingOrder::LowToHigh);
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        accumulator.append_virtual(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
            point.clone(),
            self.rd_write_value.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
            point.clone(),
            self.rs1_value.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
            point,
            self.rs2_value.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let eq_eval = EqPolynomial::<F>::mle(&point.r, &self.params.r_spartan.r);

        let (_, rd_write_value) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs1_value) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs2_value) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
        );

        eq_eval
            * (rd_write_value + self.params.gamma * rs1_value + self.params.gamma_sqr * rs2_value)
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

    fn dot(poly: &[F], eq: &[F]) -> F {
        poly.iter()
            .zip(eq.iter())
            .fold(F::from_u64(0), |a, (p, e)| a + *p * *e)
    }

    /// Seed the three SpartanOuter register-value openings (plus the `LookupOutput` point the params
    /// reads `r_spartan` from), all at the shared `r_spartan` point.
    fn seed_acc(acc: &mut Openings<F>, r_spartan: &[F], rd: &[F], rs1: &[F], rs2: &[F]) {
        let eq = EqPolynomial::<F>::evals(r_spartan, None);
        let point = OpeningPoint::new(r_spartan.to_vec());
        acc.append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
            point.clone(),
            F::from_u64(0),
        );
        acc.append_virtual(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
            point.clone(),
            dot(rd, &eq),
        );
        acc.append_virtual(
            VirtualPolynomial::Rs1Value,
            SumcheckId::SpartanOuter,
            point.clone(),
            dot(rs1, &eq),
        );
        acc.append_virtual(
            VirtualPolynomial::Rs2Value,
            SumcheckId::SpartanOuter,
            point,
            dot(rs2, &eq),
        );
    }

    fn round_trip(seed: u64, log_t: usize) {
        let mut rng = Rng(seed);
        let t = 1usize << log_t;
        let rd = rand_vec(&mut rng, t);
        let rs1 = rand_vec(&mut rng, t);
        let rs2 = rand_vec(&mut rng, t);
        let r_spartan = rand_vec(&mut rng, log_t);

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc, &r_spartan, &rd, &rs1, &rs2);
        let mut prover_t = ProverTranscript::new("registers-claim-reduction");
        let params = RegistersClaimReductionParams::new(log_t, &prover_acc, &mut prover_t);
        let input_claim = params.input_claim(&prover_acc);
        let mut prover = RegistersClaimReduction::new_prover(
            params.clone(),
            rd.clone(),
            rs1.clone(),
            rs2.clone(),
        );
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc, &r_spartan, &rd, &rs1, &rs2);
        let mut verifier_t = VerifierTranscript::new("registers-claim-reduction", &narg);
        let vparams = RegistersClaimReductionParams::new(log_t, &verifier_acc, &mut verifier_t);
        let verifier = RegistersClaimReduction::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("registers reduction must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        // Carry the reduced openings (in the proof) from the prover's accumulator.
        for poly in [
            VirtualPolynomial::RdWriteValue,
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
        ] {
            let (_, claim) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::RegistersClaimReduction);
            verifier_acc.append_virtual(
                poly,
                SumcheckId::RegistersClaimReduction,
                OpeningPoint::new(point.clone()),
                claim,
            );
        }

        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match eq(ρ,r_spartan)·(rd + γ·rs1 + γ²·rs2)"
        );

        // The reduced openings equal the value MLEs at ρ = reversed challenges.
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        for (poly, col) in [
            (VirtualPolynomial::RdWriteValue, &rd),
            (VirtualPolynomial::Rs1Value, &rs1),
            (VirtualPolynomial::Rs2Value, &rs2),
        ] {
            let (_, claim) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::RegistersClaimReduction);
            assert_eq!(claim, dot(col, &eq_rho), "{poly:?}(ρ) matches direct MLE");
        }
    }

    #[test]
    fn registers_claim_reduction_round_trip() {
        for log_t in 1..=8 {
            round_trip(0xA000u64.wrapping_add(log_t as u64), log_t);
        }
    }

    /// A tampered reduced opening (corrupted `Rs1Value(ρ)`) breaks the output-claim check.
    #[test]
    fn tampered_reduced_opening_rejected() {
        let log_t = 5;
        let mut rng = Rng(0x5151);
        let t = 1usize << log_t;
        let rd = rand_vec(&mut rng, t);
        let rs1 = rand_vec(&mut rng, t);
        let rs2 = rand_vec(&mut rng, t);
        let r_spartan = rand_vec(&mut rng, log_t);

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc, &r_spartan, &rd, &rs1, &rs2);
        let mut prover_t = ProverTranscript::new("registers-claim-reduction");
        let params = RegistersClaimReductionParams::new(log_t, &prover_acc, &mut prover_t);
        let mut prover = RegistersClaimReduction::new_prover(params.clone(), rd.clone(), rs1, rs2);
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let verifier = RegistersClaimReduction::new_verifier(params);
        let point: Vec<F> = challenges.clone();
        let mut verifier_acc = Openings::<F>::new(log_t);
        for poly in [
            VirtualPolynomial::RdWriteValue,
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
        ] {
            let (_, mut claim) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::RegistersClaimReduction);
            if poly == VirtualPolynomial::Rs1Value {
                claim += F::from_u64(1);
            }
            verifier_acc.append_virtual(
                poly,
                SumcheckId::RegistersClaimReduction,
                OpeningPoint::new(point.clone()),
                claim,
            );
        }
        let tampered = verifier.expected_output_claim(&verifier_acc, &challenges);
        // The honest reduced claim is the prover's final sumcheck value; tampering must diverge.
        let mut honest_acc = Openings::<F>::new(log_t);
        for poly in [
            VirtualPolynomial::RdWriteValue,
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
        ] {
            let (_, claim) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::RegistersClaimReduction);
            honest_acc.append_virtual(
                poly,
                SumcheckId::RegistersClaimReduction,
                OpeningPoint::new(point.clone()),
                claim,
            );
        }
        let honest = verifier.expected_output_claim(&honest_acc, &challenges);
        assert_ne!(
            tampered, honest,
            "tampered Rs1Value(ρ) must change the output claim"
        );
    }
}
