//! Register read-write-checking sumcheck — ported from jolt-core's
//! `zkvm/registers/read_write_checking.rs` onto [`crate::framework`] over the lean `Field`
//! (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves the combined read/write consistency relation over the `(address, cycle)` hypercube:
//!
//! ```text
//! Σ_{k,j} eq(r_cycle, j) · [ ra_merged(k,j)·Val(k,j) + wa(k,j)·(Val(k,j) + inc(j)) ]
//!   = rd_wv_claim + γ·rs1_rv_claim + γ²·rs2_rv_claim,
//! ```
//!
//! where `ra_merged = γ·ra1 + γ²·ra2` (the merged read indicators), `wa` is the write indicator,
//! `Val(k,j)` is register `k`'s value just before cycle `j`, and `inc(j)` is the write increment.
//! The input claim batches the three [`SumcheckId::RegistersClaimReduction`] openings
//! (`RdWriteValue`, `Rs1Value`, `Rs2Value`) with `γ`. Degree-3 over `LOG_K + log_T` variables.
//!
//! Caches `RegistersVal`/`Rs1Ra`/`Rs2Ra`/`RdWa` (virtual) at the full opening point and `RdInc`
//! (committed) at the cycle sub-point, all under [`SumcheckId::RegistersReadWriteChecking`].
//!
//! **Decoupled from the trace** (the M5 convention): takes the full dense `ra1`/`ra2`/`wa`/`val`
//! matrices (`K·T`, address-major index `k·T + j`) plus the cycle-only `inc` and `eq(r_cycle,·)`.
//! jolt-core's sparse `ReadWriteMatrix` two-phase (cycle-major → address-major) materialization,
//! the Gruen split-eq, and the `compute_rs2_ra_claim` derivation trick (which avoids materializing
//! both read matrices) are perf optimizations deferred here — this single-phase form binds every
//! variable `LowToHigh` over uniformly-broadcast dense columns and reads `rs1_ra`/`rs2_ra` directly.

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// Batching/opening parameters (matches jolt-core `RegistersReadWriteCheckingParams`, minus the
/// phase-round counts the decoupled single-phase form does not need). `log_k` is the
/// register-address bit width.
#[derive(Clone, Debug)]
pub struct RegistersReadWriteCheckingParams<F: Field> {
    pub gamma: F,
    pub log_k: usize,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: Field> RegistersReadWriteCheckingParams<F> {
    /// Draws `γ` and reads `r_cycle` from the `RdWriteValue` claim-reduction opening.
    pub fn new(
        accumulator: &dyn OpeningAccumulator<F>,
        log_k: usize,
        transcript: &mut impl Challenge<F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        Self {
            gamma,
            log_k,
            r_cycle,
        }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rd_wv) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs1_rv) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs2_rv) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
        );
        rd_wv + self.gamma * (rs1_rv + self.gamma * rs2_rv)
    }
}

/// Prover/verifier instance. The verifier carries `params` and ignores the (empty) polynomials.
pub struct RegistersReadWriteChecking<F: Field> {
    pub params: RegistersReadWriteCheckingParams<F>,
    eq: MultilinearPolynomial<F>,
    ra1: MultilinearPolynomial<F>,
    ra2: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
    inc: MultilinearPolynomial<F>,
}

impl<F: Field> RegistersReadWriteChecking<F> {
    /// Build the prover instance. `ra1`/`ra2`/`wa`/`val` are the full `K·T` address-major matrices
    /// (index `k·T + j`); `inc` is the cycle-only increment column (length `T`). The cycle-only
    /// `eq`/`inc` are broadcast across the `K` address blocks so every column binds uniformly.
    pub fn new_prover(
        params: RegistersReadWriteCheckingParams<F>,
        ra1: Vec<F>,
        ra2: Vec<F>,
        wa: Vec<F>,
        val: Vec<F>,
        inc: Vec<F>,
    ) -> Self {
        let t = inc.len();
        let k = ra1.len() / t;
        debug_assert_eq!(ra1.len(), k * t);
        let eq_cycle = EqPolynomial::<F>::evals(&params.r_cycle.r, None);
        let eq_full: Vec<F> = (0..k * t).map(|idx| eq_cycle[idx % t]).collect();
        let inc_full: Vec<F> = (0..k * t).map(|idx| inc[idx % t]).collect();
        Self {
            params,
            eq: MultilinearPolynomial::from(eq_full),
            ra1: MultilinearPolynomial::from(ra1),
            ra2: MultilinearPolynomial::from(ra2),
            wa: MultilinearPolynomial::from(wa),
            val: MultilinearPolynomial::from(val),
            inc: MultilinearPolynomial::from(inc_full),
        }
    }

    pub fn new_verifier(params: RegistersReadWriteCheckingParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::zero()]);
        Self {
            params,
            eq: dummy(),
            ra1: dummy(),
            ra2: dummy(),
            wa: dummy(),
            val: dummy(),
            inc: dummy(),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for RegistersReadWriteChecking<F> {
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
        // Degree-3: eq·(ra_merged·val + wa·(val+inc)) ⇒ 4 evaluation points.
        let gamma = self.params.gamma;
        let gamma_sq = gamma * gamma;
        let half = self.eq.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 4];
        for idx in 0..half {
            let eq_e = self
                .eq
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let ra1_e = self
                .ra1
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let ra2_e = self
                .ra2
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let wa_e = self
                .wa
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let val_e = self
                .val
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let inc_e = self
                .inc
                .sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            for k in 0..4 {
                let ra_merged = gamma * ra1_e[k] + gamma_sq * ra2_e[k];
                acc[k].fmadd(
                    eq_e[k],
                    ra_merged * val_e[k] + wa_e[k] * (val_e[k] + inc_e[k]),
                );
            }
        }
        let evals: [F; 4] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
        self.ra1.bind_parallel(r, BindingOrder::LowToHigh);
        self.ra2.bind_parallel(r, BindingOrder::LowToHigh);
        self.wa.bind_parallel(r, BindingOrder::LowToHigh);
        self.val.bind_parallel(r, BindingOrder::LowToHigh);
        self.inc.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let opening_point = self.normalize_opening_point(challenges);
        let (_, r_cycle) = opening_point.split_at(self.params.log_k);

        accumulator.append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            self.val.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            self.ra1.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            self.ra2.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point,
            self.wa.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
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

        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rd_wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );

        let rd_write_value = rd_wa_claim * (inc_claim + val_claim);
        let rs1_value = rs1_ra_claim * val_claim;
        let rs2_value = rs2_ra_claim * val_claim;
        let gamma = self.params.gamma;

        EqPolynomial::<F>::mle(&r_cycle.r, &self.params.r_cycle.r)
            * (rd_write_value + gamma * (rs1_value + gamma * rs2_value))
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

    /// Seed the three [`SumcheckId::RegistersClaimReduction`] component claims (γ-independent) so
    /// that `input_claim = rd_wv + γ·rs1 + γ²·rs2 = Σ eq·(ra_merged·val + wa·(val+inc))` for any γ.
    #[expect(clippy::too_many_arguments)]
    fn seed_components(
        acc: &mut Openings<F>,
        r_cycle: &[F],
        ra1: &[F],
        ra2: &[F],
        wa: &[F],
        val: &[F],
        inc: &[F],
        log_k: usize,
    ) {
        let t = inc.len();
        let k = 1usize << log_k;
        let eq_cycle = EqPolynomial::<F>::evals(r_cycle, None);
        let mut rd_wv = F::from_u64(0);
        let mut rs1 = F::from_u64(0);
        let mut rs2 = F::from_u64(0);
        for kk in 0..k {
            for j in 0..t {
                let idx = kk * t + j;
                rd_wv += eq_cycle[j] * wa[idx] * (val[idx] + inc[j]);
                rs1 += eq_cycle[j] * ra1[idx] * val[idx];
                rs2 += eq_cycle[j] * ra2[idx] * val[idx];
            }
        }
        // Opening points are irrelevant to the value-only input claim; use r_cycle as a placeholder.
        let pt = OpeningPoint::new(r_cycle.to_vec());
        acc.append_virtual(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
            pt.clone(),
            rd_wv,
        );
        acc.append_virtual(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
            pt.clone(),
            rs1,
        );
        acc.append_virtual(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
            pt,
            rs2,
        );
    }

    fn round_trip(seed: u64, log_k: usize, log_t: usize) {
        let mut rng = Rng(seed);
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        let n = k * t;

        let ra1 = rand_vec(&mut rng, n);
        let ra2 = rand_vec(&mut rng, n);
        let wa = rand_vec(&mut rng, n);
        let val = rand_vec(&mut rng, n);
        let inc = rand_vec(&mut rng, t);
        let r_cycle = rand_vec(&mut rng, log_t);

        // Prover
        let mut prover_acc = Openings::<F>::new(log_t);
        seed_components(
            &mut prover_acc,
            &r_cycle,
            &ra1,
            &ra2,
            &wa,
            &val,
            &inc,
            log_k,
        );
        let mut prover_t = ProverTranscript::new("registers-rw-checking");
        let params = RegistersReadWriteCheckingParams::new(&prover_acc, log_k, &mut prover_t);
        let input_claim = params.input_claim(&prover_acc);
        let mut prover = RegistersReadWriteChecking::new_prover(
            params,
            ra1.clone(),
            ra2.clone(),
            wa.clone(),
            val.clone(),
            inc.clone(),
        );
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        // Verifier
        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_components(
            &mut verifier_acc,
            &r_cycle,
            &ra1,
            &ra2,
            &wa,
            &val,
            &inc,
            log_k,
        );
        let mut verifier_t = VerifierTranscript::new("registers-rw-checking", &narg);
        let vparams = RegistersReadWriteCheckingParams::new(&verifier_acc, log_k, &mut verifier_t);
        let verifier = RegistersReadWriteChecking::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_k + log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("rw-checking must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        // Carry the five reduced openings into the verifier accumulator, then discharge.
        for poly in [
            VirtualPolynomial::RegistersVal,
            VirtualPolynomial::Rs1Ra,
            VirtualPolynomial::Rs2Ra,
            VirtualPolynomial::RdWa,
        ] {
            let (pt, c) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::RegistersReadWriteChecking);
            verifier_acc.append_virtual(poly, SumcheckId::RegistersReadWriteChecking, pt, c);
        }
        let (inc_pt, inc_c) = prover_acc.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        verifier_acc.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            inc_pt,
            inc_c,
        );
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match the eq-weighted output formula"
        );

        // Cached rs1_ra/rs2_ra equal direct MLEs of the separate read matrices at ρ.
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let dot = |p: &[F]| {
            p.iter()
                .zip(eq_rho.iter())
                .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
        };
        let (_, rs1_ra) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        assert_eq!(rs1_ra, dot(&ra1), "Rs1Ra(ρ) matches direct MLE");
        assert_eq!(rs2_ra, dot(&ra2), "Rs2Ra(ρ) matches direct MLE");
    }

    #[test]
    fn registers_rw_checking_round_trip() {
        round_trip(0x7700, 2, 2);
        round_trip(0x7701, 3, 3);
        round_trip(0x7702, 2, 5);
        round_trip(0x7703, 4, 3);
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_k = 3;
        let log_t = 3;
        let mut rng = Rng(0x77FE);
        let k = 1usize << log_k;
        let t = 1usize << log_t;
        let n = k * t;
        let ra1 = rand_vec(&mut rng, n);
        let ra2 = rand_vec(&mut rng, n);
        let wa = rand_vec(&mut rng, n);
        let val = rand_vec(&mut rng, n);
        let inc = rand_vec(&mut rng, t);
        let r_cycle = rand_vec(&mut rng, log_t);

        let mut acc = Openings::<F>::new(log_t);
        seed_components(&mut acc, &r_cycle, &ra1, &ra2, &wa, &val, &inc, log_k);
        let mut prover_t = ProverTranscript::new("t");
        let params = RegistersReadWriteCheckingParams::new(&acc, log_k, &mut prover_t);
        let input_claim = params.input_claim(&acc);
        let mut prover = RegistersReadWriteChecking::new_prover(params, ra1, ra2, wa, val, inc);
        let _ = prove(&mut prover, &mut acc, &mut prover_t);
        let mut narg = prover_t.into_proof();

        narg.narg_string[0] ^= 0x01;
        let claim = SumcheckClaim {
            num_vars: log_k + log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("t", &narg);
        // Replay the prover's pre-round γ squeeze to keep the verifier transcript aligned.
        let _ = RegistersReadWriteCheckingParams::new(&acc, log_k, &mut verifier_t);
        assert!(
            verify(&claim, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }
}
