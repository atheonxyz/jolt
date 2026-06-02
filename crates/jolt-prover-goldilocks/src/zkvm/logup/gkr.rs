//! Per-family fan-in-2 fractional GKR prover + verifier — the Figure-1 pushforward opening.
//!
//! Ported from `crates/whir-pcs-bench/src/gkr.rs::{prove_family_gkr, prove_single_instance_sumcheck}`
//! and given the **verifier** the prototype lacks. Per family, two single-instance fractional GKRs
//! prove the LogUp\* identity
//!
//! ```text
//! Σ_j eq(bits(j), r_M_row) / (α − M*[j])  ==  Σ_k P^F[k] / (α − k),
//! ```
//!
//! the A-circuit over the `T·d` leaves `(eq_m_row, α − M*)` (depth `log_t + log_d`) and the
//! B-circuit over the `2^log_m` leaves `(P^F, α − k)` (depth `log_m`). Each circuit is reduced
//! top-down: send the root fraction, squeeze `(α_c, β_c)`, then a [`GkrLayer`] sumcheck per layer,
//! interleaved with sending the four leaf values and squeezing the merge challenge `t` + fresh
//! `(α_c, β_c)`. At the leaves the protocol yields two committed openings — `M̃*(r*_A) = α − D̃_A`
//! and `P̃^F(r*_B) = Ñ_B` — plus the §4.5.2 `P̃^F(r_col)` claim; all three go to the accumulator.
//!
//! **Single-NARG model:** the layer sumcheck round polynomials (via [`sumcheck::prove`]), the circuit
//! roots, and the per-layer leaf values are all written into the shared transcript — the prover
//! `observe`s them, the verifier `read`s them back — so [`GkrProof`] carries no data (an empty marker
//! kept only to preserve the per-chunk driver's `Vec<GkrProof>` shape).
//!
//! The three prototype `assert!`s become fallible [`GkrError`] checks (one prover-side, two
//! verifier-side):
//! - eq. 5 main identity — checked in [`pushforward::prepare_family`] (prover only; the verifier
//!   discharges it via the GKR + the M8 WHIR open of `P^F(r_col)`).
//! - root histogram `N_A·D_B == N_B·D_A` — [`verify_family_gkr`].
//! - per-layer consistency `value == eq(point, r')·F(NL,NR,DL,DR)` — [`verify_circuit`].
//! - plus two leaf structural checks tying the GKR leaves to the public `eq_m_row` (A) / index
//!   polynomial `α − k` (B).

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

use crate::framework::accumulator::{CommittedPolynomial, OpeningPoint, Openings, SumcheckId};
use crate::framework::sumcheck;
use crate::framework::transcript::{ProverFs, VerifierFs};

use super::circuit::Circuit;
use super::layer::{f_combine, GkrLayer, DEGREE};
use super::pushforward::{PushforwardData, VerifierView};
use super::{idx_mle_lsb, GkrError};

/// The per-family pushforward-GKR proof. Carries no data — all round polynomials, circuit roots, and
/// layer leaf values live in the shared NARG. An empty marker so the per-chunk driver keeps its
/// `Vec<GkrProof>` shape.
#[derive(Clone, Debug, Default)]
pub struct GkrProof<F: Field> {
    _marker: PhantomData<F>,
}

/// The GKR circuit sumchecks bind LSB-first, so the leaf points `r*` are LSB-first. The framework's
/// cached opening points (and the WHIR stage-8 open) are BIG_ENDIAN, so reverse before caching — the
/// claim is unchanged (`MLE_BIG_ENDIAN(col, reverse(r)) = MLE_LSB(col, r)`); the structural leaf
/// checks above use the raw LSB-first point.
#[inline]
fn rev<F: Field>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

/// Prove one circuit top-down: write the root + each layer's sumcheck (round polys) + leaf values into
/// the NARG. Returns the leaf point `r*` and the leaf fraction `(Ñ(r*), D̃(r*))`.
fn prove_circuit<F, T>(
    circuit: &Circuit<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> (Vec<F>, F, F)
where
    F: Field,
    T: ProverFs<F>,
{
    let log_size = circuit.log_size();
    let (n_root, d_root) = circuit.root();
    transcript.observe(n_root);
    transcript.observe(d_root);
    let mut alpha: F = transcript.challenge();
    let mut beta: F = transcript.challenge();

    let mut claim = alpha * n_root + beta * d_root;
    let mut point: Vec<F> = Vec::new();
    let mut leaf_n = n_root;
    let mut leaf_d = d_root;

    for k in 0..log_size {
        let mut instance = GkrLayer::new(circuit.level(k + 1), point.clone(), alpha, beta, claim);
        let r_prime = sumcheck::prove(&mut instance, accumulator, transcript);
        let (nl, nr, dl, dr) = instance.leaf_values();
        transcript.observe(nl);
        transcript.observe(nr);
        transcript.observe(dl);
        transcript.observe(dr);

        let t: F = transcript.challenge();
        let new_alpha: F = transcript.challenge();
        let new_beta: F = transcript.challenge();

        let n_comb = nl + t * (nr - nl);
        let d_comb = dl + t * (dr - dl);
        let mut new_point = Vec::with_capacity(r_prime.len() + 1);
        new_point.push(t);
        new_point.extend(r_prime);

        point = new_point;
        claim = new_alpha * n_comb + new_beta * d_comb;
        alpha = new_alpha;
        beta = new_beta;
        leaf_n = n_comb;
        leaf_d = d_comb;
    }

    (point, leaf_n, leaf_d)
}

/// Verify one circuit top-down by reading its root, per-layer sumchecks, and leaf values out of the
/// NARG. Returns `(leaf point, Ñ(r*), D̃(r*), n_root, d_root)`; performs the per-layer consistency
/// check (`tag` identifies the circuit in any [`GkrError`]).
fn verify_circuit<F, T>(
    log_size: usize,
    tag: char,
    transcript: &mut T,
) -> Result<(Vec<F>, F, F, F, F), GkrError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let roots = transcript.read_coeffs(2).ok_or(GkrError::Sumcheck)?;
    let (n_root, d_root) = (roots[0], roots[1]);
    let mut alpha: F = transcript.challenge();
    let mut beta: F = transcript.challenge();

    let mut claim = alpha * n_root + beta * d_root;
    let mut point: Vec<F> = Vec::new();
    let mut leaf_n = n_root;
    let mut leaf_d = d_root;

    for k in 0..log_size {
        let sumcheck_claim = SumcheckClaim {
            num_vars: k,
            degree: DEGREE,
            claimed_sum: claim,
        };
        let EvaluationClaim {
            point: r_prime,
            value,
        } = sumcheck::verify(&sumcheck_claim, transcript).map_err(|_| GkrError::Sumcheck)?;

        let leaf = transcript.read_coeffs(4).ok_or(GkrError::Sumcheck)?;
        let (nl, nr, dl, dr) = (leaf[0], leaf[1], leaf[2], leaf[3]);

        // Per-layer consistency (prototype assert #3): the reduced claim must be the bound eq factor
        // times the gate value at the bound leaf values.
        let eq_bound = EqPolynomial::<F>::mle(&point, &r_prime);
        if value != eq_bound * f_combine(alpha, beta, nl, nr, dl, dr) {
            return Err(GkrError::LayerConsistency {
                circuit: tag,
                layer: k,
            });
        }

        let t: F = transcript.challenge();
        let new_alpha: F = transcript.challenge();
        let new_beta: F = transcript.challenge();

        let n_comb = nl + t * (nr - nl);
        let d_comb = dl + t * (dr - dl);
        let mut new_point = Vec::with_capacity(r_prime.len() + 1);
        new_point.push(t);
        new_point.extend(r_prime);

        point = new_point;
        claim = new_alpha * n_comb + new_beta * d_comb;
        alpha = new_alpha;
        beta = new_beta;
        leaf_n = n_comb;
        leaf_d = d_comb;
    }

    Ok((point, leaf_n, leaf_d, n_root, d_root))
}

/// Prove the per-family pushforward GKR and append the three leaf/reduction openings to the
/// accumulator. `family_index` keys the `RaDense`/`Pushforward` openings (0 = Instruction, 1 =
/// Bytecode, 2 = Ram, per the caller's convention).
pub fn prove_family_gkr<F, T>(
    data: &PushforwardData<F>,
    family_index: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> GkrProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let circuit_a = Circuit::build(
        data.eq_m_row.clone(),
        data.leaf_denom_a.clone(),
        data.log_size_a(),
    );
    let circuit_b = Circuit::build(
        data.pushforward.clone(),
        data.leaf_denom_b.clone(),
        data.log_size_b(),
    );

    let (point_a, _leaf_n_a, leaf_d_a) = prove_circuit(&circuit_a, accumulator, transcript);
    let (point_b, leaf_n_b, _leaf_d_b) = prove_circuit(&circuit_b, accumulator, transcript);

    // ra_dense (M*) opening: D̃_A(r*_A) = α − M̃*(r*_A) ⇒ M̃*(r*_A) = α − D̃_A. Cached BIG_ENDIAN.
    accumulator.append_dense(
        CommittedPolynomial::RaDense(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(rev(&point_a)),
        data.alpha - leaf_d_a,
    );
    // P^F opening (GKR leaf): Ñ_B(r*_B) = P̃^F(r*_B).
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(rev(&point_b)),
        leaf_n_b,
    );
    // P^F opening (§4.5.2 reduction): P̃^F(r_col) = combined_claim.
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardReduction,
        OpeningPoint::new(rev(&data.r_col)),
        data.combined_claim,
    );

    GkrProof {
        _marker: PhantomData,
    }
}

/// Verify the per-family pushforward GKR, performing the root-histogram, per-layer-consistency, and
/// leaf-structural checks, and appending the reconstructed openings to the accumulator.
pub fn verify_family_gkr<F, T>(
    view: &VerifierView<F>,
    _proof: &GkrProof<F>,
    family_index: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), GkrError>
where
    F: Field,
    T: VerifierFs<F>,
{
    // Circuits are read in the order the prover wrote them (A then B); the root-histogram check
    // (prototype assert #2) runs once both roots have been read out of the NARG.
    let (point_a, leaf_n_a, leaf_d_a, na, da) = verify_circuit(view.log_size_a(), 'A', transcript)?;
    let (point_b, leaf_n_b, _leaf_d_b, nb, db) =
        verify_circuit(view.log_size_b(), 'B', transcript)?;

    if na * db != nb * da {
        return Err(GkrError::RootHistogram);
    }

    // A-circuit numerator leaf is the public eq weighting eq(r_M_row, r*_A).
    if leaf_n_a != EqPolynomial::<F>::mle(&view.r_m_row, &point_a) {
        return Err(GkrError::LeafStructural { circuit: 'A' });
    }
    // B-circuit denominator leaf is the public index polynomial α − k at r*_B.
    if _leaf_d_b != view.alpha - idx_mle_lsb(&point_b) {
        return Err(GkrError::LeafStructural { circuit: 'B' });
    }

    accumulator.append_dense(
        CommittedPolynomial::RaDense(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(rev(&point_a)),
        view.alpha - leaf_d_a,
    );
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(rev(&point_b)),
        leaf_n_b,
    );
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardReduction,
        OpeningPoint::new(rev(&view.r_col)),
        view.combined_claim,
    );

    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::accumulator::OpeningAccumulator;
    use crate::zkvm::logup::pushforward::{
        claim_eval, prepare_family, prepare_family_verifier, Family,
    };
    use jolt_field::goldilocks::GoldilocksFp3 as F;

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

    fn synth_family(rng: &mut Rng, log_t: usize, log_d: usize, log_m: usize) -> Family<F> {
        let t = 1usize << log_t;
        let d = 1usize << log_d;
        let k_logical = 1u32 << log_m.min(4);
        let indices: Vec<Vec<u32>> = (0..d)
            .map(|_| (0..t).map(|_| (rng.next() as u32) % k_logical).collect())
            .collect();
        Family {
            name: "synth",
            log_t,
            log_d,
            log_m,
            r_row: rand_vec(rng, log_t),
            r_col: rand_vec(rng, log_m),
            indices,
        }
    }

    /// Full per-family round-trip: prepare → prove → verify, and confirm the three accumulator
    /// openings match between prover and verifier.
    fn family_round_trip(seed: u64, log_t: usize, log_d: usize, log_m: usize) {
        let mut rng = Rng(seed);
        let family = synth_family(&mut rng, log_t, log_d, log_m);
        let claims = claim_eval(&family);

        // Prover
        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = ProverTranscript::new("logup-gkr");
        let data = prepare_family(&family, &claims, &mut prover_t).expect("prep");
        let proof = prove_family_gkr(&data, 0, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        // Verifier
        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = VerifierTranscript::new("logup-gkr", &narg);
        let view = prepare_family_verifier(
            log_t,
            log_d,
            log_m,
            &family.r_row,
            &family.r_col,
            &claims,
            &mut verifier_t,
        );
        verify_family_gkr(&view, &proof, 0, &mut verifier_acc, &mut verifier_t)
            .expect("gkr must verify");

        // The three openings agree between prover and verifier.
        for (poly, sc) in [
            (CommittedPolynomial::RaDense(0), SumcheckId::PushforwardGkr),
            (
                CommittedPolynomial::Pushforward(0),
                SumcheckId::PushforwardGkr,
            ),
            (
                CommittedPolynomial::Pushforward(0),
                SumcheckId::PushforwardReduction,
            ),
        ] {
            let (pp, pc) = prover_acc.get_committed_polynomial_opening(poly, sc);
            let (vp, vc) = verifier_acc.get_committed_polynomial_opening(poly, sc);
            assert_eq!(pp, vp, "opening point agrees for {poly:?}/{sc:?}");
            assert_eq!(pc, vc, "opening claim agrees for {poly:?}/{sc:?}");
        }

        // The §4.5.2 reduction opening equals the directly-computed P̃^F(r_col).
        let (_, red) = verifier_acc.get_committed_polynomial_opening(
            CommittedPolynomial::Pushforward(0),
            SumcheckId::PushforwardReduction,
        );
        assert_eq!(red, data.combined_claim);
    }

    #[test]
    fn pushforward_gkr_round_trip() {
        family_round_trip(0x6001, 6, 2, 4);
        family_round_trip(0x6002, 5, 1, 4);
        family_round_trip(0x6003, 4, 3, 5);
        family_round_trip(0x6004, 7, 2, 3);
        family_round_trip(0x6005, 3, 2, 5);
    }

    /// Tampering a layer round polynomial in the NARG trips the verifier (sumcheck/consistency).
    #[test]
    fn tampered_layer_proof_rejected() {
        let mut rng = Rng(0x6FEE);
        let (log_t, log_d, log_m) = (5, 2, 4);
        let family = synth_family(&mut rng, log_t, log_d, log_m);
        let claims = claim_eval(&family);
        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = ProverTranscript::new("logup-gkr");
        let data = prepare_family(&family, &claims, &mut prover_t).expect("prep");
        let proof = prove_family_gkr(&data, 0, &mut prover_acc, &mut prover_t);
        let mut narg = prover_t.into_proof();

        // Corrupt a byte midway through the NARG (inside the A-circuit layer data).
        let off = narg.narg_string.len() / 3;
        narg.narg_string[off] ^= 0x01;

        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = VerifierTranscript::new("logup-gkr", &narg);
        let view = prepare_family_verifier(
            log_t,
            log_d,
            log_m,
            &family.r_row,
            &family.r_col,
            &claims,
            &mut verifier_t,
        );
        assert!(verify_family_gkr(&view, &proof, 0, &mut verifier_acc, &mut verifier_t).is_err());
    }

    /// Tampering a committed root fraction (a NARG byte) is rejected.
    #[test]
    fn tampered_root_rejected() {
        let mut rng = Rng(0x6FAB);
        let (log_t, log_d, log_m) = (5, 2, 4);
        let family = synth_family(&mut rng, log_t, log_d, log_m);
        let claims = claim_eval(&family);
        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = ProverTranscript::new("logup-gkr");
        let data = prepare_family(&family, &claims, &mut prover_t).expect("prep");
        let proof = prove_family_gkr(&data, 0, &mut prover_acc, &mut prover_t);
        let mut narg = prover_t.into_proof();

        // Corrupt the very first observed value (circuit A's numerator root).
        narg.narg_string[0] ^= 0x01;

        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = VerifierTranscript::new("logup-gkr", &narg);
        let view = prepare_family_verifier(
            log_t,
            log_d,
            log_m,
            &family.r_row,
            &family.r_col,
            &claims,
            &mut verifier_t,
        );
        assert!(verify_family_gkr(&view, &proof, 0, &mut verifier_acc, &mut verifier_t).is_err());
    }
}
