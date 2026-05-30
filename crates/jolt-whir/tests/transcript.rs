//! M1 gate: the shared spongefish transcript.
//!
//! Verifies (a) `from_field64_3` is the exact inverse of `to_field64_3`, and
//! (b) the jolt-whir [`ProverTranscript`] draws the *same* `Fp3` challenge as a
//! raw WHIR sponge driven with identical seeding and absorbs — i.e. the wrapper
//! does not perturb the shared Fiat-Shamir schedule, so Jolt's sumcheck rounds
//! and WHIR's commit/open agree on every challenge.

#![cfg(feature = "goldilocks")]

use whir::algebra::fields::Field64_3;
use whir::transcript::codecs::Empty;
use whir::transcript::{DomainSeparator, ProverState, VerifierMessage};

use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
use jolt_field::Field;
use jolt_whir::convert::{from_field64_3, to_field64_3};
use jolt_whir::ProverTranscript;

/// Deterministic splitmix64 (no rng dependency).
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

fn rand_fp3(r: &mut Rng) -> GoldilocksFp3 {
    GoldilocksFp3::new(
        Goldilocks::from_u64(r.next()),
        Goldilocks::from_u64(r.next()),
        Goldilocks::from_u64(r.next()),
    )
}

#[test]
fn from_field64_3_inverts_to_field64_3() {
    let mut r = Rng(0x1357_9BDF_2468_ACE0);
    for _ in 0..5000 {
        let x = rand_fp3(&mut r);
        assert_eq!(from_field64_3(to_field64_3(x)), x, "Fp3 → Field64_3 → Fp3");
    }
}

#[test]
fn shared_sponge_challenge_matches_raw_whir() {
    // Our ProverTranscript and a raw WHIR sponge, seeded identically and fed the
    // same absorbs, must squeeze the identical Fp3 challenge.
    let mut r = Rng(0x0F0E_0D0C_0B0A_0908);
    let msgs: Vec<GoldilocksFp3> = (0..8).map(|_| rand_fp3(&mut r)).collect();

    // (a) via the jolt-whir wrapper.
    let mut t = ProverTranscript::new("xcheck");
    for m in &msgs {
        t.observe_ext(*m);
    }
    let ours = t.challenge_fp3();

    // (b) via raw WHIR, replicating the wrapper's seeding (PROTOCOL/session) and
    // absorb sequence exactly.
    let ds = DomainSeparator::protocol(&"jolt-whir/goldilocks")
        .session(&"xcheck")
        .instance(&Empty);
    let mut raw = ProverState::new_std(&ds);
    for m in &msgs {
        raw.prover_message(&to_field64_3(*m));
    }
    let theirs = from_field64_3(raw.verifier_message::<Field64_3>());

    assert_eq!(ours, theirs, "shared-sponge challenge divergence");
}
