//! Sanity round-trip: commit a base-Goldilocks column, open it at a multilinear
//! point, and verify — the `make_whir_things` flow on our base-field data. This
//! confirms the field→WHIR seam and the commit/open/verify path are well-formed.
//!
//! Uses `pow_bits = 0` (no grinding) so the round-trip is fast; this only tests
//! functional correctness, not the production query/grinding security.

use std::borrow::Cow;

use whir::algebra::embedding::Basefield;
use whir::algebra::fields::{Field64, Field64_3};
use whir::algebra::linear_form::{Evaluate, LinearForm, MultilinearExtension};
use whir::parameters::ProtocolParameters;
use whir::protocols::whir::Config;
use whir::transcript::codecs::Empty;
use whir::transcript::{DomainSeparator, ProverState, VerifierState};

use jolt_field::goldilocks::Goldilocks;

use crate::convert::column_to_field64;

/// A deterministic *pseudo-random* multilinear evaluation point in `Fp3`
/// (splitmix64, no rng dependency). A structured point can hit a zero quotient
/// denominator in the verifier; real WHIR openings use Fiat-Shamir-random points.
fn eval_point(num_vars: usize) -> Vec<Field64_3> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..num_vars)
        .map(|_| {
            Field64_3::new(
                Field64::from(next()),
                Field64::from(next()),
                Field64::from(next()),
            )
        })
        .collect()
}

/// Commit `values` (length a power of two `> 1`), open at a fixed point, verify.
/// Returns `true` iff the full commit → open → verify → final-claim check passes.
pub fn sanity_roundtrip(values: &[Goldilocks]) -> bool {
    let n = values.len();
    assert!(
        n.is_power_of_two() && n > 1,
        "sanity_roundtrip expects a power-of-two length > 1"
    );
    let num_vars = n.trailing_zeros() as usize;

    // Functional-correctness config: a low security level (like WHIR's own
    // round-trip tests) keeps small inputs non-degenerate and fast. The
    // production query/grinding security lives in `params::whir_params` and is
    // exercised by the real commit; an honest proof verifies at any level.
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        ..crate::params::whir_params()
    };
    let config = Config::<Basefield<Field64_3>>::new(n, &params);

    let vector = column_to_field64(values);
    let point = eval_point(num_vars);
    let claimed_eval = MultilinearExtension {
        point: point.clone(),
    }
    .evaluate(config.embedding(), &vector);

    let ds = DomainSeparator::protocol(&config)
        .session(&"jolt-whir/sanity")
        .instance(&Empty);

    // Prover: commit + open.
    let mut prover_state = ProverState::new_std(&ds);
    let witness = config.commit(&mut prover_state, &[vector.as_slice()]);
    let prove_forms: Vec<Box<dyn LinearForm<Field64_3>>> = vec![Box::new(MultilinearExtension {
        point: point.clone(),
    })];
    let _ = config.prove(
        &mut prover_state,
        vec![Cow::from(vector)],
        vec![Cow::Owned(witness)],
        prove_forms,
        Cow::Owned(vec![claimed_eval]),
    );
    let proof = prover_state.proof();

    // Verifier: receive commitment + verify + complete the deferred MLE check.
    let mut verifier_state = VerifierState::new_std(&ds, &proof);
    let Ok(commitment) = config.receive_commitment(&mut verifier_state) else {
        return false;
    };
    let Ok(final_claim) = config.verify(&mut verifier_state, &[&commitment], &[claimed_eval])
    else {
        return false;
    };
    let verify_form = MultilinearExtension { point };
    final_claim
        .verify([&verify_form as &dyn LinearForm<Field64_3>])
        .is_ok()
}
