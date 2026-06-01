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
//! The three prototype `assert!`s become fallible [`GkrError`] checks (one prover-side, two
//! verifier-side):
//! - eq. 5 main identity — checked in [`pushforward::prepare_family`] (prover only; the verifier
//!   discharges it via the GKR + the M8 WHIR open of `P^F(r_col)`).
//! - root histogram `N_A·D_B == N_B·D_A` — [`verify_family_gkr`].
//! - per-layer consistency `value == eq(point, r')·F(NL,NR,DL,DR)` — [`verify_circuit`].
//! - plus two leaf structural checks tying the GKR leaves to the public `eq_m_row` (A) / index
//!   polynomial `α − k` (B).
//!
//! TODO(M8): the `K = 2³²` streaming-pyramid memory optimization (design §13-Q4) — this builds the
//! full pyramid densely.

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim, SumcheckProof};
use jolt_transcript::Transcript;

use crate::framework::accumulator::{CommittedPolynomial, OpeningPoint, Openings, SumcheckId};
use crate::framework::sumcheck;

use super::circuit::Circuit;
use super::layer::{f_combine, GkrLayer, DEGREE};
use super::pushforward::{PushforwardData, VerifierView};
use super::{idx_mle_lsb, GkrError};

/// A proven GKR layer: the framework sumcheck proof + the four leaf values `(NL,NR,DL,DR)(r')`.
type ProvenLayer<F> = (SumcheckProof<F>, (F, F, F, F));

/// One circuit's GKR transcript: the root fraction + per-layer `(sumcheck proof, leaf values)`.
#[derive(Clone, Debug)]
struct CircuitProof<F: Field> {
    root: (F, F),
    layers: Vec<ProvenLayer<F>>,
}

/// The per-family pushforward-GKR proof (A-circuit over `ra_dense`, B-circuit over `P^F`).
#[derive(Clone, Debug)]
pub struct GkrProof<F: Field> {
    a: CircuitProof<F>,
    b: CircuitProof<F>,
}

/// Prove one circuit top-down. Returns its proof, the leaf point `r*`, and the leaf fraction
/// `(Ñ(r*), D̃(r*))` (the merged numerator/denominator at the leaves).
fn prove_circuit<F, T>(
    circuit: &Circuit<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> (CircuitProof<F>, Vec<F>, F, F)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let log_size = circuit.log_size();
    let (n_root, d_root) = circuit.root();
    transcript.append(&n_root);
    transcript.append(&d_root);
    let mut alpha: F = transcript.challenge();
    let mut beta: F = transcript.challenge();

    let mut claim = alpha * n_root + beta * d_root;
    let mut point: Vec<F> = Vec::new();
    let mut leaf_n = n_root;
    let mut leaf_d = d_root;
    let mut layers = Vec::with_capacity(log_size);

    for _k in 0..log_size {
        let mut instance = GkrLayer::new(circuit.level(_k + 1), point.clone(), alpha, beta, claim);
        let (proof, r_prime) = sumcheck::prove(&mut instance, accumulator, transcript);
        let leaf_vals @ (nl, nr, dl, dr) = instance.leaf_values();
        transcript.append(&nl);
        transcript.append(&nr);
        transcript.append(&dl);
        transcript.append(&dr);

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
        layers.push((proof, leaf_vals));
    }

    (
        CircuitProof {
            root: (n_root, d_root),
            layers,
        },
        point,
        leaf_n,
        leaf_d,
    )
}

/// Verify one circuit top-down, returning the leaf point + leaf fraction. Performs the per-layer
/// consistency check (`tag` identifies the circuit in any [`GkrError`]).
fn verify_circuit<F, T>(
    proof: &CircuitProof<F>,
    log_size: usize,
    tag: char,
    transcript: &mut T,
) -> Result<(Vec<F>, F, F), GkrError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.layers.len() != log_size {
        return Err(GkrError::Sumcheck);
    }
    let (n_root, d_root) = proof.root;
    transcript.append(&n_root);
    transcript.append(&d_root);
    let mut alpha: F = transcript.challenge();
    let mut beta: F = transcript.challenge();

    let mut claim = alpha * n_root + beta * d_root;
    let mut point: Vec<F> = Vec::new();
    let mut leaf_n = n_root;
    let mut leaf_d = d_root;

    for (k, (layer_proof, leaf_vals)) in proof.layers.iter().enumerate() {
        let sumcheck_claim = SumcheckClaim {
            num_vars: k,
            degree: DEGREE,
            claimed_sum: claim,
        };
        let EvaluationClaim {
            point: r_prime,
            value,
        } = sumcheck::verify(&sumcheck_claim, layer_proof, transcript)
            .map_err(|_| GkrError::Sumcheck)?;

        let (nl, nr, dl, dr) = *leaf_vals;
        // Per-layer consistency (prototype assert #3): the reduced claim must be the bound eq factor
        // times the gate value at the bound leaf values.
        let eq_bound = EqPolynomial::<F>::mle(&point, &r_prime);
        if value != eq_bound * f_combine(alpha, beta, nl, nr, dl, dr) {
            return Err(GkrError::LayerConsistency {
                circuit: tag,
                layer: k,
            });
        }

        transcript.append(&nl);
        transcript.append(&nr);
        transcript.append(&dl);
        transcript.append(&dr);
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

    Ok((point, leaf_n, leaf_d))
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
    T: Transcript<Challenge = F>,
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

    let (a, point_a, _leaf_n_a, leaf_d_a) = prove_circuit(&circuit_a, accumulator, transcript);
    let (b, point_b, leaf_n_b, _leaf_d_b) = prove_circuit(&circuit_b, accumulator, transcript);

    // ra_dense (M*) opening: D̃_A(r*_A) = α − M̃*(r*_A) ⇒ M̃*(r*_A) = α − D̃_A.
    accumulator.append_dense(
        CommittedPolynomial::RaDense(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(point_a),
        data.alpha - leaf_d_a,
    );
    // P^F opening (GKR leaf): Ñ_B(r*_B) = P̃^F(r*_B).
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(point_b),
        leaf_n_b,
    );
    // P^F opening (§4.5.2 reduction): P̃^F(r_col) = combined_claim.
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardReduction,
        OpeningPoint::new(data.r_col.clone()),
        data.combined_claim,
    );

    GkrProof { a, b }
}

/// Verify the per-family pushforward GKR, performing the root-histogram, per-layer-consistency, and
/// leaf-structural checks, and appending the reconstructed openings to the accumulator.
pub fn verify_family_gkr<F, T>(
    view: &VerifierView<F>,
    proof: &GkrProof<F>,
    family_index: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), GkrError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    // Root histogram (prototype assert #2): the two fractional sums agree.
    let (na, da) = proof.a.root;
    let (nb, db) = proof.b.root;
    if na * db != nb * da {
        return Err(GkrError::RootHistogram);
    }

    let (point_a, leaf_n_a, leaf_d_a) =
        verify_circuit(&proof.a, view.log_size_a(), 'A', transcript)?;
    let (point_b, leaf_n_b, leaf_d_b) =
        verify_circuit(&proof.b, view.log_size_b(), 'B', transcript)?;

    // A-circuit numerator leaf is the public eq weighting eq(r_M_row, r*_A).
    if leaf_n_a != EqPolynomial::<F>::mle(&view.r_m_row, &point_a) {
        return Err(GkrError::LeafStructural { circuit: 'A' });
    }
    // B-circuit denominator leaf is the public index polynomial α − k at r*_B.
    if leaf_d_b != view.alpha - idx_mle_lsb(&point_b) {
        return Err(GkrError::LeafStructural { circuit: 'B' });
    }

    accumulator.append_dense(
        CommittedPolynomial::RaDense(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(point_a),
        view.alpha - leaf_d_a,
    );
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardGkr,
        OpeningPoint::new(point_b),
        leaf_n_b,
    );
    accumulator.append_dense(
        CommittedPolynomial::Pushforward(family_index),
        SumcheckId::PushforwardReduction,
        OpeningPoint::new(view.r_col.clone()),
        view.combined_claim,
    );

    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::accumulator::OpeningAccumulator;
    use crate::zkvm::logup::pushforward::{
        claim_eval, prepare_family, prepare_family_verifier, Family,
    };
    use jolt_field::goldilocks::GoldilocksFp3 as F;
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
        let mut prover_t = Blake2bTranscript::<F>::new(b"logup-gkr");
        let data = prepare_family(&family, &claims, &mut prover_t).expect("prep");
        let proof = prove_family_gkr(&data, 0, &mut prover_acc, &mut prover_t);

        // Verifier
        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"logup-gkr");
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

    /// Tampering a layer round polynomial trips the framework sumcheck verifier (→ `Sumcheck`).
    #[test]
    fn tampered_layer_proof_rejected() {
        let mut rng = Rng(0x6FEE);
        let (log_t, log_d, log_m) = (5, 2, 4);
        let family = synth_family(&mut rng, log_t, log_d, log_m);
        let claims = claim_eval(&family);
        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = Blake2bTranscript::<F>::new(b"logup-gkr");
        let data = prepare_family(&family, &claims, &mut prover_t).expect("prep");
        let mut proof = prove_family_gkr(&data, 0, &mut prover_acc, &mut prover_t);

        // Corrupt the deepest A-circuit layer's round polynomial.
        let last = proof.a.layers.len() - 1;
        proof.a.layers[last].0.round_polynomials[0] = jolt_poly::UnivariatePoly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);

        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"logup-gkr");
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

    /// Tampering a committed root fraction trips the root-histogram check.
    #[test]
    fn tampered_root_rejected() {
        let mut rng = Rng(0x6FAB);
        let (log_t, log_d, log_m) = (5, 2, 4);
        let family = synth_family(&mut rng, log_t, log_d, log_m);
        let claims = claim_eval(&family);
        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = Blake2bTranscript::<F>::new(b"logup-gkr");
        let data = prepare_family(&family, &claims, &mut prover_t).expect("prep");
        let mut proof = prove_family_gkr(&data, 0, &mut prover_acc, &mut prover_t);

        proof.b.root.0 += F::from_u64(1);

        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"logup-gkr");
        let view = prepare_family_verifier(
            log_t,
            log_d,
            log_m,
            &family.r_row,
            &family.r_col,
            &claims,
            &mut verifier_t,
        );
        assert_eq!(
            verify_family_gkr(&view, &proof, 0, &mut verifier_acc, &mut verifier_t),
            Err(GkrError::RootHistogram),
        );
    }
}
