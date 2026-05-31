//! Register value-evaluation sumcheck — ported from jolt-core's
//! `zkvm/registers/val_evaluation.rs` onto [`crate::framework`] over the lean `Field`
//! (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves the relation
//!
//! ```text
//! Val(r_address, r_cycle) = Σ_{j=0}^{T-1} inc(j) · wa(r_address, j) · LT(j, r_cycle)
//! ```
//!
//! where
//! - `inc(j)` is the register write-increment at cycle `j` (`RdInc`; 0 if no write),
//! - `wa(r_address, j)` is the write-address indicator MLE = `eq(r_address, rd_j)` when cycle `j`
//!   writes register `rd_j`, else 0,
//! - `LT(j, r_cycle)` is the strict less-than MLE (1 iff `j < r_cycle` as integers), accumulating
//!   the writes that occurred strictly before the queried cycle.
//!
//! The input claim `Val(r_address, r_cycle)` is the opening produced by
//! [`SumcheckId::RegistersReadWriteChecking`]; the opening point is `r_address ‖ r_cycle`, split at
//! `log_k` (the register-address bit width). Degree-3 (a product of three multilinear factors).
//!
//! **Decoupled from the trace** (the M5 convention): the instance takes the materialized `inc` and
//! `wa` value columns; the dense `LT(·, r_cycle)` table is materialized via
//! [`jolt_poly::LtPolynomial::evaluations`] and bound `LowToHigh` like every other factor. jolt-core
//! folds `eq(r_address)` into a `RaPolynomial` for `wa` and uses the split-LT representation + a
//! two-phase materialization; those are perf optimizations deferred here (correctness-first).

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, LtPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// Opening-point parameters, fetched from the accumulator (matches jolt-core
/// `RegistersValEvaluationSumcheckParams`). `log_k` is the register-address bit width; jolt-core
/// hard-codes it as `REGISTER_COUNT.ilog2()`, parameterized here to keep the instance decoupled.
#[derive(Clone, Debug)]
pub struct RegistersValEvaluationParams<F: Field> {
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: Field> RegistersValEvaluationParams<F> {
    pub fn new(accumulator: &dyn OpeningAccumulator<F>, log_k: usize) -> Self {
        let (r, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, r_cycle) = r.split_at(log_k);
        Self { r_address, r_cycle }
    }

    /// The input claim is the `RegistersVal` opening verbatim (no params needed) — an associated
    /// function so the (self-free) read doesn't trip `clippy::unused_self`.
    fn input_claim(accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, val) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        val
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) polynomials;
/// its `expected_output_claim` recomputes `LT(ρ, r_cycle)` and reads the cached `RdInc`/`RdWa`
/// openings.
pub struct RegistersValEvaluation<F: Field> {
    pub params: RegistersValEvaluationParams<F>,
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    lt: MultilinearPolynomial<F>,
}

impl<F: Field> RegistersValEvaluation<F> {
    /// Build the prover instance from the materialized `RdInc` and write-address columns (both
    /// length `T`). The dense `LT(·, r_cycle)` table is materialized internally.
    pub fn new_prover(params: RegistersValEvaluationParams<F>, inc: Vec<F>, wa: Vec<F>) -> Self {
        let lt = LtPolynomial::evaluations(&params.r_cycle.r);
        Self {
            params,
            inc: MultilinearPolynomial::from(inc),
            wa: MultilinearPolynomial::from(wa),
            lt: MultilinearPolynomial::from(lt),
        }
    }

    /// Build a verifier instance (no polynomials needed).
    pub fn new_verifier(params: RegistersValEvaluationParams<F>) -> Self {
        Self {
            params,
            inc: MultilinearPolynomial::from(vec![F::zero()]),
            wa: MultilinearPolynomial::from(vec![F::zero()]),
            lt: MultilinearPolynomial::from(vec![F::zero()]),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for RegistersValEvaluation<F> {
    fn num_rounds(&self) -> usize {
        self.params.r_cycle.len()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        RegistersValEvaluationParams::input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-3 product ⇒ 4 evaluation points (0,1,2,3); unreduced accumulation per point.
        let half = self.inc.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 4];
        for j in 0..half {
            let i = self
                .inc
                .sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh);
            let w = self
                .wa
                .sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh);
            let l = self
                .lt
                .sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh);
            for k in 0..4 {
                acc[k].fmadd(i[k] * w[k], l[k]);
            }
        }
        let evals: [F; 4] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.inc.bind_parallel(r, BindingOrder::LowToHigh);
        self.wa.bind_parallel(r, BindingOrder::LowToHigh);
        self.lt.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let r_cycle = self.normalize_opening_point(challenges);

        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            r_cycle.clone(),
            self.inc.final_sumcheck_claim(),
        );

        let r = [self.params.r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
            self.wa.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        // `LT(ρ, r_cycle)` where ρ = normalize(challenges) (the bound cycle point). Matches
        // jolt-core's hand-rolled `Σ_i (1−ρ_i)·r_cycle_i·eq_prefix` loop.
        let lt_eval = LtPolynomial::evaluate(&point.r, &self.params.r_cycle.r);

        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        inc_claim * wa_claim * lt_eval
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::sumcheck::{prove, verify};
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

    fn round_trip(seed: u64, log_k: usize, log_t: usize) {
        let mut rng = Rng(seed);
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        // Materialized `inc` column + a faithful write-address column `wa[j] = eq(r_address, rd_j)`.
        let inc = rand_vec(&mut rng, t);
        let r_address = rand_vec(&mut rng, log_k);
        let r_cycle = rand_vec(&mut rng, log_t);
        let eq_address = EqPolynomial::<F>::evals(&r_address, None);
        let rd: Vec<usize> = (0..t).map(|_| (rng.next() as usize) % k).collect();
        let wa: Vec<F> = rd.iter().map(|&idx| eq_address[idx]).collect();

        // Honest input claim: Val = Σ_j inc[j]·wa[j]·LT(j, r_cycle).
        let lt_table = LtPolynomial::<F>::evaluations(&r_cycle);
        let val: F = (0..t).fold(F::from_u64(0), |acc, j| acc + inc[j] * wa[j] * lt_table[j]);

        let r_combined: Vec<F> = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let seed_acc = |acc: &mut Openings<F>| {
            acc.append_virtual(
                VirtualPolynomial::RegistersVal,
                SumcheckId::RegistersReadWriteChecking,
                OpeningPoint::new(r_combined.clone()),
                val,
            );
        };

        // Prover
        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let params = RegistersValEvaluationParams::new(&prover_acc, log_k);
        let input_claim = RegistersValEvaluationParams::<F>::input_claim(&prover_acc);
        let mut prover = RegistersValEvaluation::new_prover(params, inc.clone(), wa.clone());
        let mut prover_t = Blake2bTranscript::<F>::new(b"registers-val-evaluation");
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        // Verifier
        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let vparams = RegistersValEvaluationParams::new(&verifier_acc, log_k);
        let verifier = RegistersValEvaluation::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"registers-val-evaluation");
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("val-evaluation must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        // Carry the reduced openings (cached by the prover) into the verifier accumulator, then
        // check the reduced claim against `inc·wa·LT`.
        let (_, inc_rho) = prover_acc.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );
        let (wa_pt, wa_rho) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        verifier_acc.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(point.clone()),
            inc_rho,
        );
        verifier_acc.append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            wa_pt,
            wa_rho,
        );
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(value, expected, "reduced claim must match inc·wa·LT");

        // Cached reduced openings equal the columns' MLEs at ρ = reverse(challenges).
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let dot = |p: &[F]| {
            p.iter()
                .zip(eq_rho.iter())
                .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
        };
        assert_eq!(inc_rho, dot(&inc), "RdInc(ρ) matches direct MLE");
        assert_eq!(wa_rho, dot(&wa), "RdWa(ρ) matches direct MLE");
    }

    #[test]
    fn registers_val_evaluation_round_trip() {
        for log_t in 1..=8 {
            round_trip(0x5A00 + log_t as u64, 3, log_t);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_k = 2;
        let log_t = 4;
        let mut rng = Rng(0x5AFE);
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        let inc = rand_vec(&mut rng, t);
        let r_address = rand_vec(&mut rng, log_k);
        let r_cycle = rand_vec(&mut rng, log_t);
        let eq_address = EqPolynomial::<F>::evals(&r_address, None);
        let wa: Vec<F> = (0..t)
            .map(|_| eq_address[(rng.next() as usize) % k])
            .collect();
        let lt_table = LtPolynomial::<F>::evaluations(&r_cycle);
        let val: F = (0..t).fold(F::from_u64(0), |acc, j| acc + inc[j] * wa[j] * lt_table[j]);
        let r_combined: Vec<F> = [r_address.as_slice(), r_cycle.as_slice()].concat();

        let mut acc = Openings::<F>::new(log_t);
        acc.append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            OpeningPoint::new(r_combined),
            val,
        );
        let params = RegistersValEvaluationParams::new(&acc, log_k);
        let input_claim = RegistersValEvaluationParams::<F>::input_claim(&acc);
        let mut prover = RegistersValEvaluation::new_prover(params, inc, wa);
        let mut prover_t = Blake2bTranscript::<F>::new(b"t");
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
