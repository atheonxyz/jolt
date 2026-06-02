//! RAM batched value-check sumcheck — ported from jolt-core's `zkvm/ram/val_check.rs` onto
//! [`crate::framework`] over the lean `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! A single `log_T`-round sumcheck batching the two RAM value identities at a unified address
//! point `r_address` via a transcript challenge `γ`:
//!
//! ```text
//! (1) Val(r_address, r_cycle) − Val_init(r_address) = Σ_j inc(j)·wa(r_address,j)·LT(j, r_cycle)
//! (2) Val_final(r_address)    − Val_init(r_address) = Σ_j inc(j)·wa(r_address,j)
//!
//! (1) + γ·(2):  Σ_j inc(j)·wa(r_address,j)·(LT(j, r_cycle) + γ)
//! ```
//!
//! so `input_claim = (val_rw − init_eval) + γ·(val_final − init_eval)`, where `val_rw` is the
//! `RamVal` opening from [`SumcheckId::RamReadWriteChecking`], `val_final` is the `RamValFinal`
//! opening from [`SumcheckId::RamOutputCheck`], and `init_eval = Val_init(r_address)`. Degree-3
//! (`inc · wa · (LT+γ)`, three multilinear factors — `(LT+γ)` is degree-1 in the round variable).
//!
//! Caches a single RAM `RamRa` opening (at `r_address ‖ r_cycle′`) and the `RamInc` opening (at
//! `r_cycle′`) under [`SumcheckId::RamValCheck`].
//!
//! **Decoupled from the trace** (the M5 convention): takes materialized `inc` + write-address (wa)
//! columns and the initial-RAM column (`init_eval` = its MLE at `r_address`); the LT table is
//! materialized via [`jolt_poly::LtPolynomial::evaluations`]. The ZK `init_eval_public`/advice
//! decomposition is dropped (non-ZK), and jolt-core's split-LT + two-phase materialization are
//! perf optimizations deferred here.

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, LtPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// Batching/opening parameters (matches jolt-core `RamValCheckSumcheckParams`, minus the ZK
/// `init_eval_public`/advice fields). `log_k` is the RAM-address bit width.
#[derive(Clone, Debug)]
pub struct RamValCheckParams<F: Field> {
    pub gamma: F,
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    pub init_eval: F,
}

impl<F: Field> RamValCheckParams<F> {
    /// Draws `γ`, reads `(r_address ‖ r_cycle)` from the `RamVal` opening, and computes
    /// `init_eval = Val_init(r_address)` from the materialized initial-RAM column.
    pub fn new(
        accumulator: &dyn OpeningAccumulator<F>,
        log_k: usize,
        initial_ram_state: &[F],
        transcript: &mut impl Challenge<F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let (r, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, r_cycle) = r.split_at(log_k);
        let eq_addr = EqPolynomial::<F>::evals(&r_address.r, None);
        let init_eval = initial_ram_state
            .iter()
            .zip(eq_addr.iter())
            .fold(F::zero(), |acc, (v, e)| acc + *v * *e);
        Self {
            gamma,
            r_address,
            r_cycle,
            init_eval,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, val_rw) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_final) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );
        (val_rw - self.init_eval) + self.gamma * (val_final - self.init_eval)
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) polynomials.
pub struct RamValCheck<F: Field> {
    pub params: RamValCheckParams<F>,
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    lt: MultilinearPolynomial<F>,
}

impl<F: Field> RamValCheck<F> {
    /// Build the prover instance from the materialized `RamInc` and write-address columns (length
    /// `T`). The dense `LT(·, r_cycle)` table is materialized internally.
    pub fn new_prover(params: RamValCheckParams<F>, inc: Vec<F>, wa: Vec<F>) -> Self {
        let lt = LtPolynomial::evaluations(&params.r_cycle.r);
        Self {
            params,
            inc: MultilinearPolynomial::from(inc),
            wa: MultilinearPolynomial::from(wa),
            lt: MultilinearPolynomial::from(lt),
        }
    }

    pub fn new_verifier(params: RamValCheckParams<F>) -> Self {
        Self {
            params,
            inc: MultilinearPolynomial::from(vec![F::zero()]),
            wa: MultilinearPolynomial::from(vec![F::zero()]),
            lt: MultilinearPolynomial::from(vec![F::zero()]),
        }
    }

    /// `RamRa` opening point `r_address ‖ r_cycle′`.
    fn wa_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_cycle_prime = self.normalize_opening_point(challenges);
        OpeningPoint::new(
            [
                self.params.r_address.r.as_slice(),
                r_cycle_prime.r.as_slice(),
            ]
            .concat(),
        )
    }
}

impl<F: Field> SumcheckInstance<F> for RamValCheck<F> {
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
        // Degree-3 product `inc · wa · (LT + γ)` ⇒ 4 points (0,1,2,3); unreduced accumulation.
        let gamma = self.params.gamma;
        let half = self.inc.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 4];
        for j in 0..half {
            let inc_e = self
                .inc
                .sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh);
            let wa_e = self
                .wa
                .sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh);
            let lt_e = self
                .lt
                .sumcheck_evals_array::<4>(j, BindingOrder::LowToHigh);
            for k in 0..4 {
                acc[k].fmadd(inc_e[k] * wa_e[k], lt_e[k] + gamma);
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
        let wa_point = self.wa_opening_point(challenges);
        accumulator.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
            wa_point,
            self.wa.final_sumcheck_claim(),
        );
        let r_cycle_prime = self.normalize_opening_point(challenges);
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
            r_cycle_prime,
            self.inc.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let r_cycle_prime = self.normalize_opening_point(challenges);
        let lt_eval = LtPolynomial::evaluate(&r_cycle_prime.r, &self.params.r_cycle.r);

        let (_, inc_claim) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
        let (_, wa_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValCheck);

        inc_claim * wa_claim * (lt_eval + self.params.gamma)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::sumcheck::{prove, verify};
    use crate::framework::transcript::Challenge;
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

    fn round_trip(seed: u64, log_k: usize, log_t: usize) {
        let mut rng = Rng(seed);
        let k = 1usize << log_k;
        let t = 1usize << log_t;

        let inc = rand_vec(&mut rng, t);
        let r_address = rand_vec(&mut rng, log_k);
        let r_cycle = rand_vec(&mut rng, log_t);
        let initial_ram_state = rand_vec(&mut rng, k);
        let eq_address = EqPolynomial::<F>::evals(&r_address, None);
        let wa: Vec<F> = (0..t)
            .map(|_| eq_address[(rng.next() as usize) % k])
            .collect();

        // gamma is drawn from the transcript by params::new; replicate its draw to seed the
        // accumulator's val_rw so input_claim == Σ_j inc·wa·(LT+γ).
        let lt_table = LtPolynomial::<F>::evaluations(&r_cycle);
        let init_eval = initial_ram_state
            .iter()
            .zip(eq_address.iter())
            .fold(F::from_u64(0), |a, (v, e)| a + *v * *e);
        let val_final = F::from_u64(rng.next());

        // r_address ‖ r_cycle is the RamVal opening point.
        let r_combined: Vec<F> = [r_address.as_slice(), r_cycle.as_slice()].concat();

        let build_acc = |gamma: F| -> (Openings<F>, F) {
            // S = Σ_j inc·wa·(LT+γ); choose val_rw so input_claim == S.
            let s: F = (0..t).fold(F::from_u64(0), |acc, j| {
                acc + inc[j] * wa[j] * (lt_table[j] + gamma)
            });
            let val_rw = s - gamma * (val_final - init_eval) + init_eval;
            let mut acc = Openings::<F>::new(log_t);
            acc.append_virtual(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
                OpeningPoint::new(r_combined.clone()),
                val_rw,
            );
            acc.append_virtual(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
                OpeningPoint::new(r_address.clone()),
                val_final,
            );
            (acc, s)
        };

        // Determine gamma by running params::new on a throwaway transcript primed identically.
        let mut probe_t = ProverTranscript::new("ram-val-check");
        let gamma = {
            // params::new draws challenge first; mirror by drawing here.
            probe_t.challenge()
        };
        let (mut prover_acc, s) = build_acc(gamma);

        let mut prover_t = ProverTranscript::new("ram-val-check");
        let params = RamValCheckParams::new(&prover_acc, log_k, &initial_ram_state, &mut prover_t);
        assert_eq!(params.gamma, gamma, "gamma draw matches the probe");
        let input_claim = params.input_claim(&prover_acc);
        assert_eq!(input_claim, s, "input claim equals Σ inc·wa·(LT+γ)");
        let mut prover = RamValCheck::new_prover(params, inc.clone(), wa.clone());
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        // Verifier
        let (mut verifier_acc, _) = build_acc(gamma);
        let mut verifier_t = VerifierTranscript::new("ram-val-check", &narg);
        let vparams =
            RamValCheckParams::new(&verifier_acc, log_k, &initial_ram_state, &mut verifier_t);
        let verifier = RamValCheck::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("val-check must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        let (inc_pt, inc_rho) = prover_acc
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
        let (wa_pt, wa_rho) = prover_acc
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValCheck);
        verifier_acc.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
            inc_pt,
            inc_rho,
        );
        verifier_acc.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
            wa_pt,
            wa_rho,
        );
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(value, expected, "reduced claim must match inc·wa·(LT+γ)");

        // Cached openings equal direct MLEs at ρ = reverse(challenges).
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let dot = |p: &[F]| {
            p.iter()
                .zip(eq_rho.iter())
                .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
        };
        assert_eq!(inc_rho, dot(&inc), "RamInc(ρ) matches direct MLE");
        assert_eq!(wa_rho, dot(&wa), "RamRa(ρ) matches direct MLE");
    }

    #[test]
    fn ram_val_check_round_trip() {
        for log_t in 1..=8 {
            round_trip(0x4A00 + log_t as u64, 3, log_t);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_k = 2;
        let log_t = 4;
        let mut rng = Rng(0x4AFE);
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        let inc = rand_vec(&mut rng, t);
        let r_address = rand_vec(&mut rng, log_k);
        let r_cycle = rand_vec(&mut rng, log_t);
        let initial_ram_state = rand_vec(&mut rng, k);
        let eq_address = EqPolynomial::<F>::evals(&r_address, None);
        let wa: Vec<F> = (0..t)
            .map(|_| eq_address[(rng.next() as usize) % k])
            .collect();
        let lt_table = LtPolynomial::<F>::evaluations(&r_cycle);
        let init_eval = initial_ram_state
            .iter()
            .zip(eq_address.iter())
            .fold(F::from_u64(0), |a, (v, e)| a + *v * *e);
        let val_final = F::from_u64(rng.next());
        let r_combined: Vec<F> = [r_address.as_slice(), r_cycle.as_slice()].concat();

        let mut probe_t = ProverTranscript::new("t");
        let gamma = { probe_t.challenge() };
        let s: F = (0..t).fold(F::from_u64(0), |acc, j| {
            acc + inc[j] * wa[j] * (lt_table[j] + gamma)
        });
        let val_rw = s - gamma * (val_final - init_eval) + init_eval;
        let mut acc = Openings::<F>::new(log_t);
        acc.append_virtual(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            OpeningPoint::new(r_combined),
            val_rw,
        );
        acc.append_virtual(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            OpeningPoint::new(r_address),
            val_final,
        );
        let mut prover_t = ProverTranscript::new("t");
        let params = RamValCheckParams::new(&acc, log_k, &initial_ram_state, &mut prover_t);
        let input_claim = params.input_claim(&acc);
        let mut prover = RamValCheck::new_prover(params, inc, wa);
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
        let _ = RamValCheckParams::new(&acc, log_k, &initial_ram_state, &mut verifier_t);
        assert!(
            verify(&claim, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }
}
