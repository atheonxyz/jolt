//! RAM output-check sumcheck — ported from jolt-core's `zkvm/ram/output_check.rs` onto
//! [`crate::framework`] over the lean `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Proves the zero-check
//!
//! ```text
//! 0 = Σ_k eq(r_address, k) · io_mask(k) · (Val_final(k) − Val_io(k))
//! ```
//!
//! over the `log_K` RAM-address variables, where
//! - `io_mask(k)` is the {0,1} indicator of the public I/O region of memory,
//! - `Val_final(k)` is the final RAM value at address `k` (committed/opened),
//! - `Val_io(k)` is the publicly-claimed output value (`= Val_final(k)` on the I/O region, else 0).
//!
//! For an honest prover the summand vanishes at every hypercube point (`Val_final = Val_io` where
//! `io_mask = 1`, and `io_mask = 0` elsewhere), so the input claim is `0`. The sumcheck binds the
//! result to the `Val_final` opening, and the verifier recomputes the public `io_mask`/`Val_io`
//! MLEs at the bound point. Degree-3 (`eq · io_mask · Val_diff`, three multilinear factors —
//! mirrors jolt-core `OUTPUT_SUMCHECK_DEGREE_BOUND = 3`).
//!
//! **Decoupled from the trace / program I/O** (the M5 convention): the instance takes the
//! materialized `val_final`, `val_io`, and `io_mask` columns directly, instead of jolt-core's
//! `JoltDevice`/`RangeMaskPolynomial`/`remap_address`/`eval_io_mle` machinery. jolt-core's Gruen
//! split-eq round polynomial and the phase-1/2/3 gap-round interleaving (with the
//! `2^phase3_cycle_rounds` pre-scaling) are perf optimizations deferred here (single-phase,
//! plain dense `eq` table, all `log_K` rounds are address rounds).

use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{OpeningAccumulator, Openings, SumcheckId, VirtualPolynomial};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 3;

/// Address-challenge parameters (matches jolt-core `OutputSumcheckParams`, minus the phase/gap and
/// program-I/O fields the decoupled instance does not need).
#[derive(Clone, Debug)]
pub struct RamOutputCheckParams<F: Field> {
    pub r_address: Vec<F>,
}

impl<F: Field> RamOutputCheckParams<F> {
    pub fn new(r_address: Vec<F>) -> Self {
        Self { r_address }
    }

    #[inline]
    fn log_k(&self) -> usize {
        self.r_address.len()
    }
}

/// Prover/verifier instance. The verifier carries `params` + the public `io_mask`/`val_io` columns
/// (to recompute their MLEs); only `Val_final` is opened.
pub struct RamOutputCheck<F: Field> {
    pub params: RamOutputCheckParams<F>,
    eq: MultilinearPolynomial<F>,
    io_mask: MultilinearPolynomial<F>,
    val_final: MultilinearPolynomial<F>,
    val_io: MultilinearPolynomial<F>,
    /// Retained for the verifier's `expected_output_claim` (public MLE recompute).
    io_mask_col: Vec<F>,
    val_io_col: Vec<F>,
}

impl<F: Field> RamOutputCheck<F> {
    /// Build the prover instance from the materialized columns (all length `K`).
    pub fn new_prover(
        params: RamOutputCheckParams<F>,
        val_final: Vec<F>,
        val_io: Vec<F>,
        io_mask: Vec<F>,
    ) -> Self {
        let eq = EqPolynomial::<F>::evals(&params.r_address, None);
        Self {
            params,
            eq: MultilinearPolynomial::from(eq),
            io_mask: MultilinearPolynomial::from(io_mask.clone()),
            val_final: MultilinearPolynomial::from(val_final),
            val_io: MultilinearPolynomial::from(val_io.clone()),
            io_mask_col: io_mask,
            val_io_col: val_io,
        }
    }

    /// Build a verifier instance holding only the public `io_mask`/`val_io` columns.
    pub fn new_verifier(params: RamOutputCheckParams<F>, val_io: Vec<F>, io_mask: Vec<F>) -> Self {
        Self {
            params,
            eq: MultilinearPolynomial::from(vec![F::zero()]),
            io_mask: MultilinearPolynomial::from(vec![F::zero()]),
            val_final: MultilinearPolynomial::from(vec![F::zero()]),
            val_io: MultilinearPolynomial::from(vec![F::zero()]),
            io_mask_col: io_mask,
            val_io_col: val_io,
        }
    }
}

impl<F: Field> SumcheckInstance<F> for RamOutputCheck<F> {
    fn num_rounds(&self) -> usize {
        self.params.log_k()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-3 product `eq · io_mask · (val_final − val_io)` ⇒ 4 evaluation points.
        let half = self.eq.len() / 2;
        let mut evals = [F::zero(); 4];
        for g in 0..half {
            let eq_e = self
                .eq
                .sumcheck_evals_array::<4>(g, BindingOrder::LowToHigh);
            let io_e = self
                .io_mask
                .sumcheck_evals_array::<4>(g, BindingOrder::LowToHigh);
            let vf_e = self
                .val_final
                .sumcheck_evals_array::<4>(g, BindingOrder::LowToHigh);
            let vio_e = self
                .val_io
                .sumcheck_evals_array::<4>(g, BindingOrder::LowToHigh);
            for k in 0..4 {
                evals[k] += eq_e[k] * io_e[k] * (vf_e[k] - vio_e[k]);
            }
        }
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
        self.io_mask.bind_parallel(r, BindingOrder::LowToHigh);
        self.val_final.bind_parallel(r, BindingOrder::LowToHigh);
        self.val_io.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        accumulator.append_virtual(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            point,
            self.val_final.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let eq_rho = EqPolynomial::<F>::evals(&point.r, None);
        let dot = |col: &[F]| {
            col.iter()
                .zip(eq_rho.iter())
                .fold(F::zero(), |acc, (x, e)| acc + *x * *e)
        };

        let eq_eval = EqPolynomial::<F>::mle(&self.params.r_address, &point.r);
        let io_mask_eval = dot(&self.io_mask_col);
        let val_io_eval = dot(&self.val_io_col);
        let (_, val_final_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );

        eq_eval * io_mask_eval * (val_final_claim - val_io_eval)
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

    /// Build an honest instance: `val_io = val_final` on `[io_start, io_end)`, else 0; `io_mask`
    /// the matching indicator. Then `Σ eq·io_mask·(val_final − val_io) = 0`.
    fn build(
        rng: &mut Rng,
        log_k: usize,
        io_start: usize,
        io_end: usize,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let k = 1usize << log_k;
        let val_final = rand_vec(rng, k);
        let mut val_io = vec![F::from_u64(0); k];
        let mut io_mask = vec![F::from_u64(0); k];
        for idx in io_start..io_end {
            val_io[idx] = val_final[idx];
            io_mask[idx] = F::from_u64(1);
        }
        (val_final, val_io, io_mask)
    }

    fn round_trip(seed: u64, log_k: usize, io_start: usize, io_end: usize) {
        let mut rng = Rng(seed);
        let (val_final, val_io, io_mask) = build(&mut rng, log_k, io_start, io_end);
        let r_address = rand_vec(&mut rng, log_k);

        // Prover
        let mut prover_acc = Openings::<F>::new(log_k);
        let params = RamOutputCheckParams::new(r_address.clone());
        let mut prover = RamOutputCheck::new_prover(
            params.clone(),
            val_final.clone(),
            val_io.clone(),
            io_mask.clone(),
        );
        let input_claim = prover.input_claim(&prover_acc);
        assert_eq!(
            input_claim,
            F::from_u64(0),
            "honest output check is a zero-check"
        );
        let mut prover_t = Blake2bTranscript::<F>::new(b"ram-output-check");
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        // Verifier
        let mut verifier_acc = Openings::<F>::new(log_k);
        let verifier = RamOutputCheck::new_verifier(params, val_io.clone(), io_mask.clone());
        let claim = SumcheckClaim {
            num_vars: log_k,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"ram-output-check");
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("output check must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        let (_, vf_rho) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );
        verifier_acc.append_virtual(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            OpeningPoint::new(point.clone()),
            vf_rho,
        );
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match eq·io_mask·(val_final − val_io)"
        );

        // Cached Val_final(ρ) equals its direct MLE at ρ = reverse(challenges).
        let mut rho = point.clone();
        rho.reverse();
        let eq_rho = EqPolynomial::<F>::evals(&rho, None);
        let vf_mle = val_final
            .iter()
            .zip(eq_rho.iter())
            .fold(F::from_u64(0), |a, (x, e)| a + *x * *e);
        assert_eq!(vf_rho, vf_mle, "RamValFinal(ρ) matches direct MLE");
    }

    #[test]
    fn ram_output_check_round_trip() {
        // I/O region [2,6) within K=8 (block-aligned), and other sizes.
        round_trip(0x0C01, 3, 2, 6);
        round_trip(0x0C02, 4, 4, 12);
        round_trip(0x0C03, 5, 0, 16);
        round_trip(0x0C04, 6, 8, 40);
    }

    #[test]
    fn tampered_proof_rejected() {
        let log_k = 4;
        let mut rng = Rng(0x0CFE);
        let (val_final, val_io, io_mask) = build(&mut rng, log_k, 4, 12);
        let r_address = rand_vec(&mut rng, log_k);

        let mut acc = Openings::<F>::new(log_k);
        let params = RamOutputCheckParams::new(r_address);
        let mut prover = RamOutputCheck::new_prover(params, val_final, val_io, io_mask);
        let input_claim = prover.input_claim(&acc);
        let mut prover_t = Blake2bTranscript::<F>::new(b"t");
        let (mut proof, _) = prove(&mut prover, &mut acc, &mut prover_t);

        proof.round_polynomials[0] = UnivariatePoly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let claim = SumcheckClaim {
            num_vars: log_k,
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
