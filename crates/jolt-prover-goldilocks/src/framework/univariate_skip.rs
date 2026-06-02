//! Univariate-skip first round — collapses the first `log2(DOMAIN_SIZE)` binary sumcheck rounds
//! (over the constraint-row domain) into a single high-degree univariate message
//! `s1(Y) = L̃(τ_high, Y) · t1(Y)` over a symmetric integer window. Used by Spartan's outer/product
//! sumchecks; a separate proof object + verify path from the batched cycle rounds.
//!
//! Vendored from jolt-core `subprotocols/univariate_skip.rs` (the parity oracle), retargeted to the
//! lean [`jolt_field::Field`]: `JoltField → Field`, `tau_high: F::Challenge → F`, `UniPoly → `
//! [`jolt_poly::UnivariatePoly`], the `#[cfg(zk)]` Pedersen variant dropped, and the opening-
//! accumulator `flush_to_transcript` dropped (this branch does not append openings to the FS
//! transcript). The window is the symmetric integer grid (field-agnostic; see [`super::lagrange`]).
//!
//! Parameter relations (all checked): `DOMAIN_SIZE = DEGREE + 1`, `EXTENDED_SIZE = 2·DEGREE + 1`,
//! `NUM_COEFFS = 3·DEGREE + 1` (so `s1` has degree `≤ 3·DEGREE`).

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, Transcript};

use super::lagrange::{check_sum_evals, LagrangePolynomial};
use super::sumcheck::SumcheckInstance;
use crate::framework::accumulator::Openings;

/// Univariate-skip verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniSkipError {
    /// The first-round polynomial exceeded the instance's degree bound.
    DegreeBound { got: usize, max: usize },
    /// The symmetric-window sum of `s1` did not equal the input claim.
    SumCheck,
    /// `s1(r0)` did not equal the instance's recomputed output claim.
    OutputClaim,
}

impl std::fmt::Display for UniSkipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegreeBound { got, max } => {
                write!(f, "uni-skip first-round degree {got} exceeds bound {max}")
            }
            Self::SumCheck => write!(f, "uni-skip first-round window sum != input claim"),
            Self::OutputClaim => write!(f, "uni-skip first-round s1(r0) != expected output claim"),
        }
    }
}

impl std::error::Error for UniSkipError {}

/// Interleaved symmetric univariate-skip target indices outside the base window.
///
/// Base window: the symmetric grid of size `DOMAIN_SIZE`, indices `start..start+DOMAIN_SIZE-1` with
/// `start = -⌊(DOMAIN_SIZE-1)/2⌋`. Targets are the extended points `{−DEGREE..−1} ∪ {1..DEGREE}`,
/// interleaved as `[start-1, end+1, start-2, end+2, …]` until `DEGREE` points are produced.
#[inline]
pub const fn uniskip_targets<const DOMAIN_SIZE: usize, const DEGREE: usize>() -> [i64; DEGREE] {
    let d: i64 = DEGREE as i64;
    let ext_left: i64 = -d;
    let ext_right: i64 = d;
    let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
    let base_right: i64 = base_left + (DOMAIN_SIZE as i64) - 1;

    let mut targets: [i64; DEGREE] = [0; DEGREE];
    let mut idx = 0usize;
    let mut n = base_left - 1;
    let mut p = base_right + 1;

    while n >= ext_left && p <= ext_right && idx < DEGREE {
        targets[idx] = n;
        idx += 1;
        if idx >= DEGREE {
            break;
        }
        targets[idx] = p;
        idx += 1;
        n -= 1;
        p += 1;
    }
    while idx < DEGREE && n >= ext_left {
        targets[idx] = n;
        idx += 1;
        n -= 1;
    }
    while idx < DEGREE && p <= ext_right {
        targets[idx] = p;
        idx += 1;
        p += 1;
    }
    targets
}

/// Build the uni-skip first-round polynomial `s1(Y) = L̃(τ_high, Y) · t1(Y)` from base + extended
/// evaluations of `t1`. `L̃` is the degree-`(DOMAIN_SIZE-1)` interpolant of the Lagrange basis values
/// `[L_i(τ_high)]` on the base window; `t1` (degree `≤ 2·DEGREE`) is reconstructed from its evals on
/// the extended symmetric window. `s1` has degree `≤ 3·DEGREE` (`NUM_COEFFS` coefficients).
///
/// `base_evals = None` treats the base evaluations as all zero. `extended_evals` is `t1` on the
/// targets of [`uniskip_targets`].
pub fn build_uniskip_first_round_poly<
    F: Field,
    const DOMAIN_SIZE: usize,
    const DEGREE: usize,
    const EXTENDED_SIZE: usize,
    const NUM_COEFFS: usize,
>(
    base_evals: Option<&[F; DOMAIN_SIZE]>,
    extended_evals: &[F; DEGREE],
    tau_high: F,
) -> UnivariatePoly<F> {
    debug_assert_eq!(EXTENDED_SIZE, 2 * DEGREE + 1);
    debug_assert_eq!(NUM_COEFFS, 3 * DEGREE + 1);
    debug_assert_eq!(DOMAIN_SIZE, DEGREE + 1);

    let targets: [i64; DEGREE] = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
    let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];

    if let Some(base) = base_evals {
        let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
        for (i, &val) in base.iter().enumerate() {
            let z = base_left + (i as i64);
            let pos = (z + (DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }
    }
    for (idx, &val) in extended_evals.iter().enumerate() {
        let z = targets[idx];
        let pos = (z + (DEGREE as i64)) as usize;
        t1_vals[pos] = val;
    }

    let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<EXTENDED_SIZE>(&t1_vals);
    let lagrange_values = LagrangePolynomial::<F>::evals::<DOMAIN_SIZE>(&tau_high);
    let lagrange_coeffs =
        LagrangePolynomial::<F>::interpolate_coeffs::<DOMAIN_SIZE>(&lagrange_values);

    let mut s1_coeffs: [F; NUM_COEFFS] = [F::zero(); NUM_COEFFS];
    for (i, &a) in lagrange_coeffs.iter().enumerate() {
        for (j, &b) in t1_coeffs.iter().enumerate() {
            s1_coeffs[i + j] += a * b;
        }
    }
    UnivariatePoly::new(s1_coeffs.to_vec())
}

/// The uni-skip first-round proof: the (full) high-degree univariate sent in round 0.
#[derive(Clone, Debug)]
pub struct UniSkipFirstRoundProof<F: Field> {
    pub uni_poly: UnivariatePoly<F>,
}

/// Prove the uni-skip first round (non-ZK): read the input claim, build `s1` via the instance's
/// `compute_message(0, ·)`, absorb its coefficients, squeeze `r0`, cache the instance's openings at
/// `[r0]`. Returns the proof and `r0` (the caller continues with the batched cycle rounds).
pub fn prove_uniskip_round<F, I, T>(
    instance: &mut I,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> (UniSkipFirstRoundProof<F>, F)
where
    F: Field,
    I: SumcheckInstance<F>,
    T: Transcript<Challenge = F>,
{
    let input_claim = instance.input_claim(&*accumulator);
    let uni_poly = instance.compute_message(0, input_claim);
    for coeff in uni_poly.coefficients() {
        coeff.append_to_transcript(transcript);
    }
    let r0 = transcript.challenge();
    instance.cache_openings(accumulator, &[r0]);
    (UniSkipFirstRoundProof { uni_poly }, r0)
}

/// Verify the uni-skip first round: degree bound, then absorb `s1` and squeeze `r0` (matching the
/// prover), check the symmetric-`N`-window sum equals the input claim, and check `s1(r0)` equals the
/// instance's recomputed output claim. `N` is the base window size (`DOMAIN_SIZE`); `NUM_COEFFS` is
/// `s1`'s coefficient count. Returns `r0` on success.
pub fn verify_uniskip_round<F, I, T, const N: usize, const NUM_COEFFS: usize>(
    proof: &UniSkipFirstRoundProof<F>,
    instance: &I,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<F, UniSkipError>
where
    F: Field,
    I: SumcheckInstance<F>,
    T: Transcript<Challenge = F>,
{
    let degree_bound = instance.degree();
    let got = UnivariatePolynomial::degree(&proof.uni_poly);
    if got > degree_bound {
        return Err(UniSkipError::DegreeBound {
            got,
            max: degree_bound,
        });
    }

    for coeff in proof.uni_poly.coefficients() {
        coeff.append_to_transcript(transcript);
    }
    let r0 = transcript.challenge();

    let input_claim = instance.input_claim(&*accumulator);
    if !check_sum_evals::<F, N, NUM_COEFFS>(&proof.uni_poly, input_claim) {
        return Err(UniSkipError::SumCheck);
    }

    instance.cache_openings(accumulator, &[r0]);
    let expected_output = proof.uni_poly.evaluate(r0);
    let claimed_output = instance.expected_output_claim(&*accumulator, &[r0]);
    if claimed_output != expected_output {
        return Err(UniSkipError::OutputClaim);
    }
    Ok(r0)
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::accumulator::OpeningAccumulator;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_transcript::Blake2bTranscript;

    // Concrete uni-skip parameters for the tests: DOMAIN_SIZE = DEGREE + 1.
    const DEGREE: usize = 2;
    const DOMAIN_SIZE: usize = 3;
    const EXTENDED_SIZE: usize = 5; // 2*DEGREE + 1
    const NUM_COEFFS: usize = 7; // 3*DEGREE + 1

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

    #[test]
    fn targets_are_interleaved_symmetric() {
        assert_eq!(uniskip_targets::<3, 2>(), [-2, 2]);
        assert_eq!(uniskip_targets::<5, 4>(), [-3, 3, -4, 4]);
        assert_eq!(uniskip_targets::<2, 1>(), [-1]); // base {0,1}, ext {-1}
    }

    /// `build_uniskip_first_round_poly` produces `s1 = L̃(τ_high,·)·t1(·)` as polynomials, so
    /// `s1(z) == L̃(z)·t1(z)` at arbitrary points (checked against `LagrangePolynomial` directly).
    #[test]
    fn build_poly_is_lagrange_kernel_times_t1() {
        let mut rng = Rng(0x5111);
        let base_evals: [F; DOMAIN_SIZE] = core::array::from_fn(|_| F::from_u64(rng.next()));
        let extended_evals: [F; DEGREE] = core::array::from_fn(|_| F::from_u64(rng.next()));
        let tau_high = F::from_u64(rng.next());

        let s1 = build_uniskip_first_round_poly::<F, DOMAIN_SIZE, DEGREE, EXTENDED_SIZE, NUM_COEFFS>(
            Some(&base_evals),
            &extended_evals,
            tau_high,
        );

        // Reconstruct t1's evals on the symmetric EXTENDED window and L̃'s basis values.
        let t1_vals = t1_window_vals(&base_evals, &extended_evals);
        let lagrange_values = LagrangePolynomial::<F>::evals::<DOMAIN_SIZE>(&tau_high);

        for z in [-5i64, -1, 0, 3, 7] {
            let zf = F::from_i64(z);
            let l_tilde = LagrangePolynomial::<F>::evaluate::<DOMAIN_SIZE>(&lagrange_values, &zf);
            let t1 = LagrangePolynomial::<F>::evaluate::<EXTENDED_SIZE>(&t1_vals, &zf);
            assert_eq!(
                s1.evaluate(zf),
                l_tilde * t1,
                "s1(z) == L̃(z)·t1(z) at z={z}"
            );
        }
    }

    /// `t1`'s evaluations on the symmetric EXTENDED window, matching the layout
    /// `build_uniskip_first_round_poly` uses (`pos = z + DEGREE`; base in the centre, ext at targets).
    fn t1_window_vals(
        base_evals: &[F; DOMAIN_SIZE],
        extended_evals: &[F; DEGREE],
    ) -> [F; EXTENDED_SIZE] {
        let mut t1_vals = [F::from_u64(0); EXTENDED_SIZE];
        let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
        for (i, &v) in base_evals.iter().enumerate() {
            let pos = (base_left + i as i64 + DEGREE as i64) as usize;
            t1_vals[pos] = v;
        }
        let targets = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
        for (idx, &v) in extended_evals.iter().enumerate() {
            let pos = (targets[idx] + DEGREE as i64) as usize;
            t1_vals[pos] = v;
        }
        t1_vals
    }

    /// A synthetic uni-skip instance: round-0 message is `s1 = L̃(τ_high,·)·t1(·)`; the input claim is
    /// the symmetric-`DOMAIN_SIZE`-window sum of `s1`; the output claim recomputes `L̃(r0)·t1(r0)`
    /// independently (so a correct `s1` round-trips, and the verify checks tie back to the math).
    struct UniSkipTestInstance {
        base_evals: [F; DOMAIN_SIZE],
        extended_evals: [F; DEGREE],
        tau_high: F,
        input_claim: F,
        lagrange_values: [F; DOMAIN_SIZE],
        t1_vals: [F; EXTENDED_SIZE],
    }

    impl UniSkipTestInstance {
        fn new(rng: &mut Rng) -> Self {
            let base_evals: [F; DOMAIN_SIZE] = core::array::from_fn(|_| F::from_u64(rng.next()));
            let extended_evals: [F; DEGREE] = core::array::from_fn(|_| F::from_u64(rng.next()));
            let tau_high = F::from_u64(rng.next());
            let s1 = build_uniskip_first_round_poly::<
                F,
                DOMAIN_SIZE,
                DEGREE,
                EXTENDED_SIZE,
                NUM_COEFFS,
            >(Some(&base_evals), &extended_evals, tau_high);
            // input claim = Σ_{t in symmetric DOMAIN_SIZE-window} s1(t).
            let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
            let input_claim = (0..DOMAIN_SIZE)
                .map(|i| s1.evaluate(F::from_i64(base_left + i as i64)))
                .fold(F::from_u64(0), |a, b| a + b);
            let lagrange_values = LagrangePolynomial::<F>::evals::<DOMAIN_SIZE>(&tau_high);
            let t1_vals = t1_window_vals(&base_evals, &extended_evals);
            Self {
                base_evals,
                extended_evals,
                tau_high,
                input_claim,
                lagrange_values,
                t1_vals,
            }
        }
    }

    impl SumcheckInstance<F> for UniSkipTestInstance {
        fn num_rounds(&self) -> usize {
            1
        }
        fn degree(&self) -> usize {
            NUM_COEFFS - 1
        }
        fn input_claim(&self, _acc: &dyn OpeningAccumulator<F>) -> F {
            self.input_claim
        }
        fn compute_message(&mut self, _round: usize, _prev: F) -> UnivariatePoly<F> {
            build_uniskip_first_round_poly::<F, DOMAIN_SIZE, DEGREE, EXTENDED_SIZE, NUM_COEFFS>(
                Some(&self.base_evals),
                &self.extended_evals,
                self.tau_high,
            )
        }
        fn bind(&mut self, _r: F, _round: usize) {}
        fn cache_openings(&self, _acc: &mut Openings<F>, _challenges: &[F]) {}
        fn expected_output_claim(&self, _acc: &dyn OpeningAccumulator<F>, challenges: &[F]) -> F {
            let r0 = challenges[0];
            let l_tilde =
                LagrangePolynomial::<F>::evaluate::<DOMAIN_SIZE>(&self.lagrange_values, &r0);
            let t1 = LagrangePolynomial::<F>::evaluate::<EXTENDED_SIZE>(&self.t1_vals, &r0);
            l_tilde * t1
        }
    }

    #[test]
    fn prove_verify_round_trip() {
        let mut rng = Rng(0x5151);
        let mut instance = UniSkipTestInstance::new(&mut rng);
        let mut prover_acc = Openings::<F>::new(4);
        let mut prover_t = Blake2bTranscript::<F>::new(b"uniskip");
        let (proof, r0_prover) = prove_uniskip_round(&mut instance, &mut prover_acc, &mut prover_t);

        let verifier_instance = UniSkipTestInstance::new(&mut Rng(0x5151));
        let mut verifier_acc = Openings::<F>::new(4);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"uniskip");
        let r0 = verify_uniskip_round::<F, _, _, DOMAIN_SIZE, NUM_COEFFS>(
            &proof,
            &verifier_instance,
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("uni-skip first round must verify");
        assert_eq!(r0, r0_prover, "verifier r0 matches prover r0");
    }

    #[test]
    fn tampered_first_round_poly_rejected() {
        let mut rng = Rng(0x5252);
        let mut instance = UniSkipTestInstance::new(&mut rng);
        let mut prover_acc = Openings::<F>::new(4);
        let mut prover_t = Blake2bTranscript::<F>::new(b"uniskip");
        let (mut proof, _) = prove_uniskip_round(&mut instance, &mut prover_acc, &mut prover_t);

        // Perturb the constant coefficient: the window sum no longer equals the input claim.
        let mut coeffs = proof.uni_poly.coefficients().to_vec();
        coeffs[0] += F::from_u64(1);
        proof.uni_poly = UnivariatePoly::new(coeffs);

        let verifier_instance = UniSkipTestInstance::new(&mut Rng(0x5252));
        let mut verifier_acc = Openings::<F>::new(4);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"uniskip");
        let result = verify_uniskip_round::<F, _, _, DOMAIN_SIZE, NUM_COEFFS>(
            &proof,
            &verifier_instance,
            &mut verifier_acc,
            &mut verifier_t,
        );
        assert!(
            result.is_err(),
            "tampered uni-skip first round must be rejected"
        );
    }
}
