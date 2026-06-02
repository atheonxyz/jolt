//! Prover-side sumcheck-instance trait + batched driver over the single spongefish NARG.
//!
//! Round polynomials are written into the NARG byte string as prover messages
//! ([`ProverFs::observe`]) and read back by the verifier ([`VerifierFs::read_coeffs`]) —
//! there is no separate `SumcheckProof` carrier (the round polys live in the proof's
//! NARG, the same sponge WHIR commit/open drive). The workspace `SumcheckVerifier`
//! bridge is dropped; `verify` is an inline round loop. Challenges are plain `F` (the
//! `C = F = Fp3` convention). `SumcheckClaim`/`EvaluationClaim`/`SumcheckError` are
//! kept as pure value types.
//!
//! To keep prover and verifier in lockstep over the NARG, each round writes a FIXED
//! number of coefficients — `degree + 1` (single) / `max_degree + 1` (batched) — padding
//! with zeros, so the verifier reads exactly that many regardless of the actual
//! (possibly lower-degree, e.g. gap-round dummy) round polynomial.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim, SumcheckError};

use crate::framework::accumulator::{
    OpeningAccumulator, OpeningPoint, Openings, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::framework::transcript::{ProverFs, VerifierFs};

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

/// Write a round polynomial's coefficients into the NARG, padded with zeros to `n_coeffs`
/// (= the declared degree + 1) so the verifier always reads a fixed count.
pub(crate) fn write_round_poly<F, T>(transcript: &mut T, poly: &UnivariatePoly<F>, n_coeffs: usize)
where
    F: Field,
    T: ProverFs<F>,
{
    let coeffs = poly.coefficients();
    debug_assert!(
        coeffs.len() <= n_coeffs,
        "round polynomial degree {} exceeds declared bound {}",
        coeffs.len().saturating_sub(1),
        n_coeffs - 1
    );
    for i in 0..n_coeffs {
        transcript.observe(coeffs.get(i).copied().unwrap_or_else(F::zero));
    }
}

/// Read one round polynomial (`n_coeffs` coefficients) back out of the NARG and check round
/// consistency `s(0) + s(1) == running`; returns the reconstructed polynomial.
fn read_and_check_round<F, T>(
    transcript: &mut T,
    n_coeffs: usize,
    running: F,
    round: usize,
    num_vars: usize,
) -> Result<UnivariatePoly<F>, SumcheckError<F>>
where
    F: Field,
    T: VerifierFs<F>,
{
    let coeffs = transcript
        .read_coeffs(n_coeffs)
        .ok_or(SumcheckError::WrongNumberOfRounds {
            expected: num_vars,
            got: round,
        })?;
    let poly = UnivariatePoly::new(coeffs);
    let sum = poly.evaluate(F::zero()) + poly.evaluate(F::one());
    if sum != running {
        return Err(SumcheckError::RoundCheckFailed {
            round,
            expected: running,
            actual: sum,
        });
    }
    Ok(poly)
}

/// Drive a single sumcheck instance to completion, writing each round polynomial into the NARG
/// and returning the squeezed challenge point. Reads the input claim from `accumulator`, then
/// caches output openings.
pub fn prove<F, I, T>(instance: &mut I, accumulator: &mut Openings<F>, transcript: &mut T) -> Vec<F>
where
    F: Field,
    I: SumcheckInstance<F>,
    T: ProverFs<F>,
{
    let n = instance.num_rounds();
    let n_coeffs = instance.degree() + 1;
    let mut claim = instance.input_claim(&*accumulator);
    let mut challenges = Vec::with_capacity(n);

    for round in 0..n {
        let poly = instance.compute_message(round, claim);
        write_round_poly(transcript, &poly, n_coeffs);
        let r = transcript.challenge();
        instance.bind(r, round);
        claim = poly.evaluate(r);
        challenges.push(r);
    }

    instance.cache_openings(accumulator, &challenges);
    challenges
}

/// Verify a single-instance proof by replaying the round loop over the verifier's NARG, returning
/// the reduced evaluation claim `g(r) = v`. The caller discharges `v` against
/// [`SumcheckInstance::expected_output_claim`] (and the opening accumulator).
pub fn verify<F, T>(
    claim: &SumcheckClaim<F>,
    transcript: &mut T,
) -> Result<EvaluationClaim<F>, SumcheckError<F>>
where
    F: Field,
    T: VerifierFs<F>,
{
    let n_coeffs = claim.degree + 1;
    let mut running = claim.claimed_sum;
    let mut challenges = Vec::with_capacity(claim.num_vars);

    for round in 0..claim.num_vars {
        let poly = read_and_check_round(transcript, n_coeffs, running, round, claim.num_vars)?;
        let r = transcript.challenge();
        running = poly.evaluate(r);
        challenges.push(r);
    }

    Ok(EvaluationClaim {
        point: challenges,
        value: running,
    })
}

/// Drive a **front-loaded batched** sumcheck over many instances of (possibly) differing round
/// counts and degrees. Mirrors jolt-core `BatchedSumcheck::prove` (non-ZK).
///
/// Transcript order: observe each instance's `input_claim` into the NARG, squeeze the batching
/// challenge `α`, then per round write the combined round polynomial and squeeze the round
/// challenge. Batching coefficients are `α^j` (the running power).
///
/// Front-loading: instance `j` is active in rounds `[offset_j, offset_j + num_rounds_j)` with
/// `offset_j = round_offset(max)` (default `max − num_rounds_j`); in its gap rounds it contributes a
/// constant dummy `claim_j / 2` (so `H(0)+H(1) = claim_j`), and its claim is pre-scaled by
/// `2^{max − num_rounds_j}` so it equals the true `input_claim` at activation.
///
/// Returns `(challenges, batching_coeffs)` — `challenges` has length `max_num_rounds`;
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
) -> (Vec<F>, Vec<F>)
where
    F: Field,
    T: ProverFs<F>,
{
    assert!(
        !instances.is_empty(),
        "batched sumcheck needs at least one instance"
    );
    let max_num_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
    let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();
    let n_coeffs = max_degree + 1;

    // Fiat-Shamir: observe input claims into the NARG, squeeze α. A wrong claim desyncs the
    // verifier's round-0 `s(0)+s(1) == combined_sum` check (combined_sum uses the true claims).
    let input_claims: Vec<F> = instances
        .iter()
        .map(|s| s.input_claim(&*accumulator))
        .collect();
    for c in &input_claims {
        transcript.observe(*c);
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

        write_round_poly(transcript, &batched, n_coeffs);
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
    }

    for s in &mut instances {
        s.finalize();
    }
    for s in &instances {
        let offset = s.round_offset(max_num_rounds);
        let r_slice = &challenges[offset..offset + s.num_rounds()];
        s.cache_openings(accumulator, r_slice);
    }

    (challenges, batching_coeffs)
}

/// Verify a batched proof, returning the combined reduced claim `{point, value}` **and** the
/// batching coefficients `α^j`. The caller discharges `value` against
/// `Σ_j α^j · expected_output_claim_j` at each instance's challenge slice — so it needs the `α^j`.
///
/// Reads the prover's observed input claims back out of the NARG (to keep the sponge in lockstep),
/// then squeezes `α`, scales each *known* claim by `2^{max−n_j}`, combines with `α^j`, and replays
/// the round loop on the combined claim. The combined sum uses the passed-in (true) claims, so a
/// prover that observed a wrong claim is caught at round 0.
pub fn verify_batched<F, T>(
    claims: &[SumcheckClaim<F>],
    transcript: &mut T,
) -> Result<(EvaluationClaim<F>, Vec<F>), SumcheckError<F>>
where
    F: Field,
    T: VerifierFs<F>,
{
    let (first, rest) = claims.split_first().ok_or(SumcheckError::EmptyClaims)?;
    let max_num_vars = rest
        .iter()
        .fold(first.num_vars, |acc, c| acc.max(c.num_vars));
    let max_degree = rest.iter().fold(first.degree, |acc, c| acc.max(c.degree));

    // Advance the sponge over the prover's observed input claims (values discarded — the
    // combined sum below is computed from the verifier's own claims).
    let _claims = transcript
        .read_coeffs(claims.len())
        .ok_or(SumcheckError::EmptyClaims)?;
    let alpha: F = transcript.challenge();

    let mut batching_coeffs = Vec::with_capacity(claims.len());
    let mut power = F::one();
    let mut combined_sum = F::zero();
    for claim in claims {
        batching_coeffs.push(power);
        let scaled = claim.claimed_sum.mul_pow_2(max_num_vars - claim.num_vars);
        combined_sum += power * scaled;
        power *= alpha;
    }

    let n_coeffs = max_degree + 1;
    let mut running = combined_sum;
    let mut challenges = Vec::with_capacity(max_num_vars);
    for round in 0..max_num_vars {
        let poly = read_and_check_round(transcript, n_coeffs, running, round, max_num_vars)?;
        let r = transcript.challenge();
        running = poly.evaluate(r);
        challenges.push(r);
    }

    Ok((
        EvaluationClaim {
            point: challenges,
            value: running,
        },
        batching_coeffs,
    ))
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::poly::MultilinearPolynomial;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_field::Field;
    use jolt_poly::BindingOrder;

    /// Proves `Σ_x A(x)·B(x)` over two dense polynomials (degree-2 round messages).
    struct ProductInstance {
        a: MultilinearPolynomial<F>,
        b: MultilinearPolynomial<F>,
        num_rounds: usize,
        claim: F,
    }

    impl ProductInstance {
        fn new(a: Vec<F>, b: Vec<F>) -> Self {
            assert_eq!(a.len(), b.len());
            let num_rounds = a.len().trailing_zeros() as usize;
            let claim = a
                .iter()
                .zip(b.iter())
                .fold(F::from_u64(0), |acc, (x, y)| acc + *x * *y);
            Self {
                a: MultilinearPolynomial::from(a),
                b: MultilinearPolynomial::from(b),
                num_rounds,
                claim,
            }
        }
    }

    impl SumcheckInstance<F> for ProductInstance {
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
            let evals = (0..half).fold([F::from_u64(0); 3], |mut acc, i| {
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

    fn product_round_trip(seed: u64, log_len: usize) {
        let mut rng = Rng(seed);
        let len = 1usize << log_len;
        let a: Vec<F> = (0..len).map(|_| F::from_u64(rng.next())).collect();
        let b: Vec<F> = (0..len).map(|_| F::from_u64(rng.next())).collect();

        let mut instance = ProductInstance::new(a, b);
        let mut acc = Openings::<F>::new(log_len);
        let input_claim = instance.input_claim(&acc);
        let degree = instance.degree();

        let mut prover_t = ProverTranscript::new("framework-sumcheck-test");
        let challenges = prove(&mut instance, &mut acc, &mut prover_t);
        let output = instance.expected_output_claim(&acc, &challenges);
        let proof = prover_t.into_proof();

        let claim = SumcheckClaim {
            num_vars: log_len,
            degree,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("framework-sumcheck-test", &proof);
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("proof must verify");

        assert_eq!(
            point, challenges,
            "verifier point must match prover challenges"
        );
        assert_eq!(value, output, "reduced claim must equal A(r)·B(r)");
    }

    #[test]
    fn product_sumcheck_round_trip_fp3() {
        for log_len in 1..=8 {
            product_round_trip(0xE000 + log_len as u64, log_len);
        }
    }

    #[test]
    fn tampered_proof_rejected() {
        let mut instance = ProductInstance::new(
            (1..=8).map(F::from_u64).collect(),
            (1..=8).map(|x| F::from_u64(x + 3)).collect(),
        );
        let mut acc = Openings::<F>::new(3);
        let input_claim = instance.input_claim(&acc);
        let mut prover_t = ProverTranscript::new("t");
        let _ = prove(&mut instance, &mut acc, &mut prover_t);
        let mut proof = prover_t.into_proof();
        // Corrupt the first round polynomial's bytes in the NARG.
        proof.narg_string[0] ^= 0x01;
        let claim = SumcheckClaim {
            num_vars: 3,
            degree: 2,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("t", &proof);
        assert!(
            verify(&claim, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }

    fn product_instance(rng: &mut Rng, log_len: usize) -> ProductInstance {
        let len = 1usize << log_len;
        let a = (0..len).map(|_| F::from_u64(rng.next())).collect();
        let b = (0..len).map(|_| F::from_u64(rng.next())).collect();
        ProductInstance::new(a, b)
    }

    /// Front-loaded batched sumcheck over three instances of differing round counts (4, 3, 2) —
    /// exercises the gap-round dummies — round-tripped over the NARG.
    #[test]
    fn batched_round_trip_with_gap_rounds() {
        let mut rng = Rng(0xBA7C);
        let mut i0 = product_instance(&mut rng, 4);
        let mut i1 = product_instance(&mut rng, 3);
        let mut i2 = product_instance(&mut rng, 2);

        let mut acc = Openings::<F>::new(4);
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

        let mut prover_t = ProverTranscript::new("batched");
        let instances: Vec<&mut dyn SumcheckInstance<F>> = vec![&mut i0, &mut i1, &mut i2];
        let (challenges, coeffs) = prove_batched(instances, &mut acc, &mut prover_t);
        assert_eq!(challenges.len(), 4, "challenges has length max_num_rounds");
        let proof = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("batched", &proof);
        let (EvaluationClaim { point, value }, vcoeffs) =
            verify_batched(&claims, &mut verifier_t).expect("batched proof must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );
        assert_eq!(
            vcoeffs, coeffs,
            "verifier recomputes the prover's α^j coeffs"
        );

        // Combined reduced claim = Σ_j α^j · A_j(r_slice)·B_j(r_slice) at each instance's active
        // slice (offset_j = max − num_rounds_j: 0, 1, 2).
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
        let mut acc = Openings::<F>::new(4);
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
        let mut prover_t = ProverTranscript::new("batched-t");
        let _ = prove_batched(vec![&mut i0, &mut i1], &mut acc, &mut prover_t);
        let mut proof = prover_t.into_proof();
        // Corrupt a round polynomial byte (past the two observed input-claim scalars).
        let off = proof.narg_string.len() / 2;
        proof.narg_string[off] ^= 0x01;
        let mut verifier_t = VerifierTranscript::new("batched-t", &proof);
        assert!(
            verify_batched(&claims, &mut verifier_t).is_err(),
            "tampered batched proof must be rejected"
        );
    }
}
