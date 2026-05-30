//! M2 gate: `WhirScheme` commit → open → verify over the **shared** spongefish
//! transcript (one `ProverTranscript` for commit+open, one `VerifierTranscript`
//! replaying the single proof) — the inherent-API analogue of a
//! `CommitmentScheme` round-trip, and an upgrade over Phase-1 `sanity_roundtrip`'s
//! two-separate-`ProverState` flow.

#![cfg(feature = "goldilocks")]
#![expect(clippy::expect_used)]

use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
use jolt_field::Field;
use jolt_whir::{ProverTranscript, VerifierTranscript, WhirScheme};

/// Deterministic splitmix64.
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

/// Non-degenerate column (spread across all bits; WHIR's open divides by the
/// evaluation, which is 0 for the zero polynomial).
fn column(size: usize, seed: u64) -> Vec<Goldilocks> {
    let mut r = Rng(seed);
    (0..size).map(|_| Goldilocks::from_u64(r.next())).collect()
}

fn point(num_vars: usize, seed: u64) -> Vec<GoldilocksFp3> {
    let mut r = Rng(seed);
    (0..num_vars)
        .map(|_| {
            GoldilocksFp3::new(
                Goldilocks::from_u64(r.next()),
                Goldilocks::from_u64(r.next()),
                Goldilocks::from_u64(r.next()),
            )
        })
        .collect()
}

#[test]
fn whir_scheme_shared_transcript_round_trip() {
    let num_vars = 6;
    let size = 1 << num_vars;
    let col = column(size, 0xA11CE);
    let pt = point(num_vars, 0xB0B);

    let config = WhirScheme::config(size);
    let eval = WhirScheme::evaluate(&config, &col, &pt);

    // Prover: one shared transcript for commit + open.
    let mut t = ProverTranscript::new("scheme-rt");
    let hint = WhirScheme::commit(&mut t, &config, &col);
    WhirScheme::open(&mut t, &config, &col, hint, &pt, eval);
    let proof = t.into_proof();

    // Verifier: replay the single proof.
    let mut vt = VerifierTranscript::new("scheme-rt", &proof);
    let commitment = WhirScheme::receive_commitment(&mut vt, &config).expect("receive_commitment");
    WhirScheme::verify(&mut vt, &config, &commitment, &pt, eval).expect("verify");
}

#[test]
fn whir_scheme_multi_commit_then_open_one() {
    // Commit several columns on one transcript (as the witness commit phase does),
    // then open + verify one of them. The verifier must receive all commitments in
    // order so the shared sponge stays in lockstep.
    let num_vars = 6;
    let size = 1 << num_vars;
    let n = 4;
    let cols: Vec<Vec<Goldilocks>> = (0..n).map(|k| column(size, 0x100 + k as u64)).collect();
    let pt = point(num_vars, 0xCAFE);

    let config = WhirScheme::config(size);
    let open_idx = 0;
    let eval = WhirScheme::evaluate(&config, &cols[open_idx], &pt);

    let mut t = ProverTranscript::new("scheme-multi");
    let mut hints = Vec::new();
    for c in &cols {
        hints.push(WhirScheme::commit(&mut t, &config, c));
    }
    WhirScheme::open(
        &mut t,
        &config,
        &cols[open_idx],
        hints.swap_remove(open_idx),
        &pt,
        eval,
    );
    let proof = t.into_proof();

    let mut vt = VerifierTranscript::new("scheme-multi", &proof);
    let mut commitments = Vec::new();
    for _ in 0..n {
        commitments.push(WhirScheme::receive_commitment(&mut vt, &config).expect("receive"));
    }
    WhirScheme::verify(&mut vt, &config, &commitments[open_idx], &pt, eval).expect("verify");
}
