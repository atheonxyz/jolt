//! RAM read-write-checking sumcheck — ported from jolt-core's
//! `zkvm/ram/read_write_checking.rs` onto [`crate::framework`] over the lean `Field`
//! (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves the RAM read/write consistency relation over the `(address, cycle)` hypercube:
//!
//! ```text
//! Σ_{k,j} eq(r_cycle, j) · ra(k,j) · (Val(k,j) + γ·(inc(j) + Val(k,j))) = rv_claim + γ·wv_claim,
//! ```
//!
//! where `ra(k,j)` is the access indicator, `Val(k,j)` the RAM value just before cycle `j`, and
//! `inc(j)` the write increment. The input claim batches the `RamReadValue`/`RamWriteValue`
//! openings from [`SumcheckId::SpartanOuter`] with `γ`. Degree-3 over `log_K + log_T` variables.
//!
//! Caches `RamVal`/`RamRa` (virtual) at the full opening point and `RamInc` (committed) at the
//! cycle sub-point, all under [`SumcheckId::RamReadWriteChecking`].
//!
//! **Decoupled from the trace** (the M5 convention): takes the full dense `ra`/`val` matrices
//! (`K·T`, address-major index `k·T + j`) plus the cycle-only `inc` and `eq(r_cycle,·)` (broadcast
//! across the `K` address blocks). jolt-core's sparse `ReadWriteMatrix` two-phase materialization
//! and Gruen split-eq are perf optimizations deferred here (single-phase, uniform `LowToHigh`).

use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// Batching/opening parameters (matches jolt-core `RamReadWriteCheckingParams`, minus the
/// phase-round counts). `log_k` is the RAM-address bit width.
#[derive(Clone, Debug)]
pub struct RamReadWriteCheckingParams<F: Field> {
    pub gamma: F,
    pub log_k: usize,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: Field> RamReadWriteCheckingParams<F> {
    /// Draws `γ` and reads `r_cycle` from the `RamReadValue` Spartan-outer opening.
    pub fn new(
        accumulator: &dyn OpeningAccumulator<F>,
        log_k: usize,
        transcript: &mut impl Transcript<Challenge = F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        Self {
            gamma,
            log_k,
            r_cycle,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        let (_, wv) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamWriteValue,
            SumcheckId::SpartanOuter,
        );
        rv + self.gamma * wv
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) polynomials.
pub struct RamReadWriteChecking<F: Field> {
    pub params: RamReadWriteCheckingParams<F>,
    eq: MultilinearPolynomial<F>,
    ra: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
    inc: MultilinearPolynomial<F>,
}

impl<F: Field> RamReadWriteChecking<F> {
    /// Build the prover instance. `ra`/`val` are the full `K·T` address-major matrices (index
    /// `k·T + j`); `inc` is the cycle-only increment column (length `T`).
    pub fn new_prover(
        params: RamReadWriteCheckingParams<F>,
        ra: Vec<F>,
        val: Vec<F>,
        inc: Vec<F>,
    ) -> Self {
        let t = inc.len();
        let k = ra.len() / t;
        debug_assert_eq!(ra.len(), k * t);
        let eq_cycle = EqPolynomial::<F>::evals(&params.r_cycle.r, None);
        let eq_full: Vec<F> = (0..k * t).map(|idx| eq_cycle[idx % t]).collect();
        let inc_full: Vec<F> = (0..k * t).map(|idx| inc[idx % t]).collect();
        Self {
            params,
            eq: MultilinearPolynomial::from(eq_full),
            ra: MultilinearPolynomial::from(ra),
            val: MultilinearPolynomial::from(val),
            inc: MultilinearPolynomial::from(inc_full),
        }
    }

    pub fn new_verifier(params: RamReadWriteCheckingParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::zero()]);
        Self {
            params,
            eq: dummy(),
            ra: dummy(),
            val: dummy(),
            inc: dummy(),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for RamReadWriteChecking<F> {
    fn num_rounds(&self) -> usize {
        self.params.log_k + self.params.r_cycle.len()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-3: eq·ra·(val + γ·(val+inc)) ⇒ 4 evaluation points (0,1,2,3).
        let gamma = self.params.gamma;
        let half = self.eq.len() / 2;
        let mut evals = [F::zero(); 4];
        for idx in 0..half {
            let eq_e = self
                .eq
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let ra_e = self
                .ra
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let val_e = self
                .val
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let inc_e = self
                .inc
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            for k in 0..4 {
                evals[k] += eq_e[k] * ra_e[k] * (val_e[k] + gamma * (val_e[k] + inc_e[k]));
            }
        }
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
        self.ra.bind_parallel(r, BindingOrder::LowToHigh);
        self.val.bind_parallel(r, BindingOrder::LowToHigh);
        self.inc.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let opening_point = self.normalize_opening_point(challenges);
        let (_, r_cycle) = opening_point.split_at(self.params.log_k);

        accumulator.append_virtual(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            self.val.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point,
            self.ra.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle,
            self.inc.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let r = self.normalize_opening_point(challenges);
        let (_, r_cycle) = r.split_at(self.params.log_k);
        let eq_eval = EqPolynomial::<F>::mle(&self.params.r_cycle.r, &r_cycle.r);

        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );

        eq_eval * ra_claim * (val_claim + self.params.gamma * (val_claim + inc_claim))
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

    /// Seed the two Spartan-outer component claims (γ-independent): rv = Σ eq·ra·val,
    /// wv = Σ eq·ra·(inc+val), so input_claim = rv + γ·wv = Σ eq·ra·(val + γ(inc+val)).
    fn seed_components(
        acc: &mut Openings<F>,
        r_cycle: &[F],
        ra: &[F],
        val: &[F],
        inc: &[F],
        log_k: usize,
    ) {
        let t = inc.len();
        let k = 1usize << log_k;
        let eq_cycle = EqPolynomial::<F>::evals(r_cycle, None);
        let mut rv = F::from_u64(0);
        let mut wv = F::from_u64(0);
        for kk in 0..k {
            for j in 0..t {
                let idx = kk * t + j;
                rv += eq_cycle[j] * ra[idx] * val[idx];
                wv += eq_cycle[j] * ra[idx] * (inc[j] + val[idx]);
            }
        }
        let pt = OpeningPoint::new(r_cycle.to_vec());
        acc.append_virtual(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
            pt.clone(),
            rv,
        );
        acc.append_virtual(
            VirtualPolynomial::RamWriteValue,
            SumcheckId::SpartanOuter,
            pt,
            wv,
        );
    }

    fn round_trip(seed: u64, log_k: usize, log_t: usize) {
        let mut rng = Rng(seed);
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        let n = k * t;

        let ra = rand_vec(&mut rng, n);
        let val = rand_vec(&mut rng, n);
        let inc = rand_vec(&mut rng, t);
        let r_cycle = rand_vec(&mut rng, log_t);

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_components(&mut prover_acc, &r_cycle, &ra, &val, &inc, log_k);
        let mut prover_t = Blake2bTranscript::<F>::new(b"ram-rw-checking");
        let params = RamReadWriteCheckingParams::new(&prover_acc, log_k, &mut prover_t);
        let input_claim = params.input_claim(&prover_acc);
        let mut prover =
            RamReadWriteChecking::new_prover(params, ra.clone(), val.clone(), inc.clone());
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_components(&mut verifier_acc, &r_cycle, &ra, &val, &inc, log_k);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"ram-rw-checking");
        let vparams = RamReadWriteCheckingParams::new(&verifier_acc, log_k, &mut verifier_t);
        let verifier = RamReadWriteChecking::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_k + log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("ram rw-checking must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for poly in [VirtualPolynomial::RamVal, VirtualPolynomial::RamRa] {
            let (pt, c) =
                prover_acc.get_virtual_polynomial_opening(poly, SumcheckId::RamReadWriteChecking);
            verifier_acc.append_virtual(poly, SumcheckId::RamReadWriteChecking, pt, c);
        }
        let (inc_pt, inc_c) = prover_acc.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        verifier_acc.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            inc_pt,
            inc_c,
        );
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match eq·ra·(val+γ(val+inc))"
        );

        // Cached RamRa/RamVal equal direct MLEs at ρ.
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let dot = |p: &[F]| {
            p.iter()
                .zip(eq_rho.iter())
                .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
        };
        let (_, ra_rho) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_rho) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        assert_eq!(ra_rho, dot(&ra), "RamRa(ρ) matches direct MLE");
        assert_eq!(val_rho, dot(&val), "RamVal(ρ) matches direct MLE");
    }

    #[test]
    fn ram_rw_checking_round_trip() {
        round_trip(0x6600, 2, 2);
        round_trip(0x6601, 3, 3);
        round_trip(0x6602, 2, 5);
        round_trip(0x6603, 4, 3);
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_k = 3;
        let log_t = 3;
        let mut rng = Rng(0x66FE);
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        let n = k * t;
        let ra = rand_vec(&mut rng, n);
        let val = rand_vec(&mut rng, n);
        let inc = rand_vec(&mut rng, t);
        let r_cycle = rand_vec(&mut rng, log_t);

        let mut acc = Openings::<F>::new(log_t);
        seed_components(&mut acc, &r_cycle, &ra, &val, &inc, log_k);
        let mut prover_t = Blake2bTranscript::<F>::new(b"t");
        let params = RamReadWriteCheckingParams::new(&acc, log_k, &mut prover_t);
        let input_claim = params.input_claim(&acc);
        let mut prover = RamReadWriteChecking::new_prover(params, ra, val, inc);
        let (mut proof, _) = prove(&mut prover, &mut acc, &mut prover_t);

        proof.round_polynomials[0] = UnivariatePoly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let claim = SumcheckClaim {
            num_vars: log_k + log_t,
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
