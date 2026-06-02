//! Prover-side sumcheck-instance trait + batched driver, retargeted from jolt-core's
//! `subprotocols/sumcheck_prover.rs` to the lean field/transcript and bridged to the workspace
//! **verifier** (`jolt-sumcheck`): the driver emits a [`jolt_sumcheck::SumcheckProof`] (a
//! `Vec<UnivariatePoly<F>>`) that `jolt_sumcheck::SumcheckVerifier` checks unchanged. Challenges
//! are plain `F` (the `C = F = Fp3` convention; jolt-core's `F::Challenge` collapses to `F`).
//!
//! This is the *clear* (non-ZK) path. The opening-accumulator coupling (jolt-core's
//! `input_claim(accumulator)` / `cache_openings`) and the BlindFold committed path land with the
//! accumulator subsystem; here `input_claim`/`expected_output_claim` are plain values so the
//! driver↔verifier bridge can be validated standalone.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    BatchedSumcheckVerifier, EvaluationClaim, RoundProof, SumcheckClaim, SumcheckError,
    SumcheckProof, SumcheckVerifier,
};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::framework::accumulator::{
    OpeningAccumulator, OpeningPoint, Openings, BIG_ENDIAN, LITTLE_ENDIAN,
};

/// A prover-side sumcheck instance: one batched claim reduced over [`Self::num_rounds`] rounds.
/// Mirrors the jolt-core `SumcheckInstanceProver`/`SumcheckInstanceParams` surface, minus the
/// `#[cfg(zk)]` BlindFold constraint methods (non-ZK this phase).
pub trait SumcheckInstance<F: Field> {
    /// Number of sumcheck rounds (variables bound).
    fn num_rounds(&self) -> usize;

    /// Degree bound of each round polynomial.
    fn degree(&self) -> usize;

    /// The claimed sum `Σ_x g(x)`, computed from prior openings in the accumulator.
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;

    /// The round-`round` univariate message, given the running claim. Must have degree
    /// `≤ self.degree()` and satisfy `s(0) + s(1) = previous_claim`.
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UnivariatePoly<F>;

    /// Bind the round's variable to `r`.
    fn bind(&mut self, r: F, round: usize);

    /// Store this instance's output openings (claims/points) into the accumulator after the
    /// sumcheck completes.
    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]);

    /// The expected final evaluation `g(r_1, …, r_n)` the verifier's reduced claim must match,
    /// computed from the cached output openings + challenges.
    fn expected_output_claim(&self, accumulator: &dyn OpeningAccumulator<F>, challenges: &[F])
        -> F;

    /// Map the sumcheck challenges (little-endian, round order) to the canonical big-endian
    /// opening point used to key cached openings. Matches jolt-core's default.
    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    /// The global round at which this instance becomes active in a batched (front-loaded) sumcheck
    /// of `max_num_rounds` rounds. Default = `max_num_rounds - num_rounds` (shorter instances active
    /// in the last `num_rounds` rounds), matching jolt-core's `round_offset`.
    fn round_offset(&self, max_num_rounds: usize) -> usize {
        max_num_rounds - self.num_rounds()
    }

    /// End-of-protocol hook (after the last challenge is bound, before `cache_openings`) for
    /// instances with delayed bindings. Default no-op. Mirrors jolt-core's `finalize`.
    fn finalize(&mut self) {}
}

/// Drive a single sumcheck instance to completion, emitting a workspace-verifiable proof and the
/// squeezed challenge point. Reads the input claim from `accumulator`, absorbs each round
/// polynomial through the same `RoundProof` path the verifier replays (so
/// `jolt_sumcheck::SumcheckVerifier::verify` accepts the result), then caches output openings.
pub fn prove<F, I, T>(
    instance: &mut I,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> (SumcheckProof<F>, Vec<F>)
where
    F: Field,
    I: SumcheckInstance<F>,
    T: Transcript<Challenge = F>,
{
    let n = instance.num_rounds();
    let mut claim = instance.input_claim(&*accumulator);
    let mut round_polynomials = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for round in 0..n {
        let poly = instance.compute_message(round, claim);
        <UnivariatePoly<F> as RoundProof<F>>::append_to_transcript(&poly, transcript);
        let r = transcript.challenge();
        instance.bind(r, round);
        claim = poly.evaluate(r);
        challenges.push(r);
        round_polynomials.push(poly);
    }

    instance.cache_openings(accumulator, &challenges);
    (SumcheckProof { round_polynomials }, challenges)
}

/// Verify a single-instance proof via the workspace verifier, returning the reduced evaluation
/// claim `g(r) = v`. The caller discharges `v` against [`SumcheckInstance::expected_output_claim`]
/// (and, later, the opening accumulator).
pub fn verify<F, T>(
    claim: &SumcheckClaim<F>,
    proof: &SumcheckProof<F>,
    transcript: &mut T,
) -> Result<EvaluationClaim<F>, SumcheckError<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    SumcheckVerifier::verify(claim, &proof.round_polynomials, transcript)
}

/// Drive a **front-loaded batched** sumcheck over many instances of (possibly) differing round
/// counts and degrees, emitting a proof the workspace [`BatchedSumcheckVerifier`] accepts. Mirrors
/// jolt-core `BatchedSumcheck::prove` (non-ZK), retargeted to the lean field/transcript.
///
/// Transcript order (must match [`BatchedSumcheckVerifier`]): absorb each instance's `input_claim`
/// in order, squeeze the batching challenge `α`, then per round absorb the combined round polynomial
/// and squeeze the round challenge. Batching coefficients are `α^j` (NOT jolt-core's independent
/// `challenge_vector` — the workspace verifier uses the running power, so the prover matches it).
///
/// Front-loading: instance `j` is active in rounds `[offset_j, offset_j + num_rounds_j)` with
/// `offset_j = round_offset(max)` (default `max − num_rounds_j`); in its gap rounds it contributes a
/// constant dummy `claim_j / 2` (so `H(0)+H(1) = claim_j`), and its claim is pre-scaled by
/// `2^{max − num_rounds_j}` so it equals the true `input_claim` at activation.
///
/// Returns `(proof, challenges, batching_coeffs)` — `challenges` has length `max_num_rounds`;
/// `batching_coeffs[j] = α^j`. The verifier reduces to `Σ_j α^j · expected_output_claim_j` at each
/// instance's `challenges[offset_j .. offset_j + num_rounds_j]` slice.
#[expect(
    clippy::unwrap_used,
    reason = "instances is asserted non-empty (so max() is Some); 2 is invertible in any prime field"
)]
pub fn prove_batched<F, T>(
    mut instances: Vec<&mut dyn SumcheckInstance<F>>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> (SumcheckProof<F>, Vec<F>, Vec<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    assert!(
        !instances.is_empty(),
        "batched sumcheck needs at least one instance"
    );
    let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();

    // Fiat-Shamir: absorb input claims (BatchedSumcheckVerifier order), squeeze α.
    let input_claims: Vec<F> = instances
        .iter()
        .map(|s| s.input_claim(&*accumulator))
        .collect();
    for c in &input_claims {
        c.append_to_transcript(transcript);
    }
    let alpha: F = transcript.challenge();
    let mut batching_coeffs = Vec::with_capacity(instances.len());
    let mut power = F::one();
    for _ in 0..instances.len() {
        batching_coeffs.push(power);
        power *= alpha;
    }

    // Front-loaded scaling: claim_j *= 2^{max − num_rounds_j}.
    let mut individual_claims: Vec<F> = instances
        .iter()
        .zip(input_claims.iter())
        .map(|(s, &c)| c.mul_pow_2(max_num_rounds - s.num_rounds()))
        .collect();

    let two_inv = F::from_u64(2).inverse().unwrap();
    let mut round_polynomials = Vec::with_capacity(max_num_rounds);
    let mut challenges = Vec::with_capacity(max_num_rounds);

    for round in 0..max_num_rounds {
        let polys: Vec<UnivariatePoly<F>> = instances
            .iter_mut()
            .zip(individual_claims.iter())
            .map(|(s, &prev)| {
                let num_rounds = s.num_rounds();
                let offset = s.round_offset(max_num_rounds);
                if round >= offset && round < offset + num_rounds {
                    s.compute_message(round - offset, prev)
                } else {
                    UnivariatePoly::new(vec![prev * two_inv])
                }
            })
            .collect();

        // Combined round polynomial Σ_j α^j · poly_j.
        let mut batched = &polys[0] * batching_coeffs[0];
        for (poly, &coeff) in polys.iter().zip(batching_coeffs.iter()).skip(1) {
            batched += &(poly * coeff);
        }

        <UnivariatePoly<F> as RoundProof<F>>::append_to_transcript(&batched, transcript);
        let r = transcript.challenge();
        challenges.push(r);

        for (claim, poly) in individual_claims.iter_mut().zip(polys.iter()) {
            *claim = poly.evaluate(r);
        }
        for s in &mut instances {
            let num_rounds = s.num_rounds();
            let offset = s.round_offset(max_num_rounds);
            if round >= offset && round < offset + num_rounds {
                s.bind(r, round - offset);
            }
        }
        round_polynomials.push(batched);
    }

    for s in &mut instances {
        s.finalize();
    }
    for s in &instances {
        let offset = s.round_offset(max_num_rounds);
        let r_slice = &challenges[offset..offset + s.num_rounds()];
        s.cache_openings(accumulator, r_slice);
    }

    (
        SumcheckProof { round_polynomials },
        challenges,
        batching_coeffs,
    )
}

/// Verify a batched proof via the workspace [`BatchedSumcheckVerifier`], returning the combined
/// reduced claim `{point, value}`. The caller discharges `value` against
/// `Σ_j α^j · expected_output_claim_j` at each instance's challenge slice.
pub fn verify_batched<F, T>(
    claims: &[SumcheckClaim<F>],
    proof: &SumcheckProof<F>,
    transcript: &mut T,
) -> Result<EvaluationClaim<F>, SumcheckError<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    BatchedSumcheckVerifier::verify(claims, &proof.round_polynomials, transcript)
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::poly::MultilinearPolynomial;
    use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
    use jolt_poly::BindingOrder;
    use jolt_transcript::Blake2bTranscript;

    /// Proves `Σ_x A(x)·B(x)` over two dense polynomials (degree-2 round messages).
    struct ProductInstance<F: Field> {
        a: MultilinearPolynomial<F>,
        b: MultilinearPolynomial<F>,
        num_rounds: usize,
        claim: F,
    }

    impl<F: Field> ProductInstance<F> {
        fn new(a: Vec<F>, b: Vec<F>) -> Self {
            assert_eq!(a.len(), b.len());
            let num_rounds = a.len().trailing_zeros() as usize;
            let claim = a
                .iter()
                .zip(b.iter())
                .fold(F::zero(), |acc, (x, y)| acc + *x * *y);
            Self {
                a: MultilinearPolynomial::from(a),
                b: MultilinearPolynomial::from(b),
                num_rounds,
                claim,
            }
        }
    }

    impl<F: Field> SumcheckInstance<F> for ProductInstance<F> {
        fn num_rounds(&self) -> usize {
            self.num_rounds
        }
        fn degree(&self) -> usize {
            2
        }
        fn input_claim(&self, _acc: &dyn OpeningAccumulator<F>) -> F {
            self.claim
        }
        fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
            let half = self.a.len() / 2;
            // round message at points 0, 1, 2 (degree-2 product of two linear factors)
            let evals = (0..half).fold([F::zero(); 3], |mut acc, i| {
                let ae = self.a.sumcheck_evals_array::<3>(i, BindingOrder::LowToHigh);
                let be = self.b.sumcheck_evals_array::<3>(i, BindingOrder::LowToHigh);
                for k in 0..3 {
                    acc[k] += ae[k] * be[k];
                }
                acc
            });
            UnivariatePoly::from_evals(&evals)
        }
        fn bind(&mut self, r: F, _round: usize) {
            self.a.bind_parallel(r, BindingOrder::LowToHigh);
            self.b.bind_parallel(r, BindingOrder::LowToHigh);
        }
        fn cache_openings(&self, _acc: &mut Openings<F>, _challenges: &[F]) {}
        fn expected_output_claim(&self, _acc: &dyn OpeningAccumulator<F>, _challenges: &[F]) -> F {
            self.a.final_sumcheck_claim() * self.b.final_sumcheck_claim()
        }
    }

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

    fn product_round_trip<F: Field>(seed: u64, log_len: usize) {
        let mut rng = Rng(seed);
        let len = 1usize << log_len;
        let a: Vec<F> = (0..len).map(|_| F::from_u64(rng.next())).collect();
        let b: Vec<F> = (0..len).map(|_| F::from_u64(rng.next())).collect();

        let mut instance = ProductInstance::new(a, b);
        let mut acc = Openings::<F>::new(log_len);
        let input_claim = instance.input_claim(&acc);
        let degree = instance.degree();

        let mut prover_t = Blake2bTranscript::<F>::new(b"framework-sumcheck-test");
        let (proof, challenges) = prove(&mut instance, &mut acc, &mut prover_t);
        let output = instance.expected_output_claim(&acc, &challenges);

        let claim = SumcheckClaim {
            num_vars: log_len,
            degree,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"framework-sumcheck-test");
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("proof must verify");

        assert_eq!(
            point, challenges,
            "verifier point must match prover challenges"
        );
        assert_eq!(value, output, "reduced claim must equal A(r)·B(r)");
    }

    #[test]
    fn product_sumcheck_round_trip_goldilocks() {
        for log_len in 1..=8 {
            product_round_trip::<Goldilocks>(0xD000 + log_len as u64, log_len);
        }
    }

    #[test]
    fn product_sumcheck_round_trip_fp3() {
        for log_len in 1..=8 {
            product_round_trip::<GoldilocksFp3>(0xE000 + log_len as u64, log_len);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let mut instance = ProductInstance::<Goldilocks>::new(
            (1..=8).map(Goldilocks::from_u64).collect(),
            (1..=8).map(|x| Goldilocks::from_u64(x + 3)).collect(),
        );
        let mut acc = Openings::<Goldilocks>::new(3);
        let input_claim = instance.input_claim(&acc);
        let mut prover_t = Blake2bTranscript::<Goldilocks>::new(b"t");
        let (mut proof, _) = prove(&mut instance, &mut acc, &mut prover_t);
        // Corrupt the first round polynomial.
        proof.round_polynomials[0] = UnivariatePoly::new(vec![
            Goldilocks::from_u64(1),
            Goldilocks::from_u64(2),
            Goldilocks::from_u64(3),
        ]);
        let claim = SumcheckClaim {
            num_vars: 3,
            degree: 2,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<Goldilocks>::new(b"t");
        assert!(
            verify(&claim, &proof, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }

    fn product_instance(rng: &mut Rng, log_len: usize) -> ProductInstance<Goldilocks> {
        let len = 1usize << log_len;
        let a = (0..len).map(|_| Goldilocks::from_u64(rng.next())).collect();
        let b = (0..len).map(|_| Goldilocks::from_u64(rng.next())).collect();
        ProductInstance::new(a, b)
    }

    /// Front-loaded batched sumcheck over three instances of differing round counts (4, 3, 2) —
    /// exercises the gap-round dummies — round-tripped against the workspace `BatchedSumcheckVerifier`.
    #[test]
    fn batched_round_trip_with_gap_rounds() {
        let mut rng = Rng(0xBA7C);
        let mut i0 = product_instance(&mut rng, 4);
        let mut i1 = product_instance(&mut rng, 3);
        let mut i2 = product_instance(&mut rng, 2);

        let mut acc = Openings::<Goldilocks>::new(4);
        let claims = vec![
            SumcheckClaim {
                num_vars: 4,
                degree: 2,
                claimed_sum: i0.input_claim(&acc),
            },
            SumcheckClaim {
                num_vars: 3,
                degree: 2,
                claimed_sum: i1.input_claim(&acc),
            },
            SumcheckClaim {
                num_vars: 2,
                degree: 2,
                claimed_sum: i2.input_claim(&acc),
            },
        ];

        let mut prover_t = Blake2bTranscript::<Goldilocks>::new(b"batched");
        let instances: Vec<&mut dyn SumcheckInstance<Goldilocks>> = vec![&mut i0, &mut i1, &mut i2];
        let (proof, challenges, coeffs) = prove_batched(instances, &mut acc, &mut prover_t);
        assert_eq!(challenges.len(), 4, "challenges has length max_num_rounds");

        let mut verifier_t = Blake2bTranscript::<Goldilocks>::new(b"batched");
        let EvaluationClaim { point, value } =
            verify_batched(&claims, &proof, &mut verifier_t).expect("batched proof must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        // Combined reduced claim = Σ_j α^j · A_j(r_slice)·B_j(r_slice) at each instance's active slice
        // (offset_j = max − num_rounds_j: 0, 1, 2).
        let v0 = coeffs[0] * i0.expected_output_claim(&acc, &challenges[0..4]);
        let v1 = coeffs[1] * i1.expected_output_claim(&acc, &challenges[1..4]);
        let v2 = coeffs[2] * i2.expected_output_claim(&acc, &challenges[2..4]);
        assert_eq!(
            value,
            v0 + v1 + v2,
            "batched reduced claim must equal Σ α^j · A_j(r)·B_j(r)"
        );
    }

    #[test]
    fn tampered_batched_rejected() {
        let mut rng = Rng(0xBADB);
        let mut i0 = product_instance(&mut rng, 4);
        let mut i1 = product_instance(&mut rng, 3);
        let mut acc = Openings::<Goldilocks>::new(4);
        let claims = vec![
            SumcheckClaim {
                num_vars: 4,
                degree: 2,
                claimed_sum: i0.input_claim(&acc),
            },
            SumcheckClaim {
                num_vars: 3,
                degree: 2,
                claimed_sum: i1.input_claim(&acc),
            },
        ];
        let mut prover_t = Blake2bTranscript::<Goldilocks>::new(b"batched-t");
        let (mut proof, _, _) = prove_batched(vec![&mut i0, &mut i1], &mut acc, &mut prover_t);
        proof.round_polynomials[0] = UnivariatePoly::new(vec![
            Goldilocks::from_u64(1),
            Goldilocks::from_u64(2),
            Goldilocks::from_u64(3),
        ]);
        let mut verifier_t = Blake2bTranscript::<Goldilocks>::new(b"batched-t");
        assert!(
            verify_batched(&claims, &proof, &mut verifier_t).is_err(),
            "tampered batched proof must be rejected"
        );
    }
}
