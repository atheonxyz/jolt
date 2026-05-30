//! M2 gate: `WhirScheme` commit → open → verify over the **shared** spongefish
//! transcript (one `ProverTranscript` for commit+open, one `VerifierTranscript`
//! replaying the single proof) — the inherent-API analogue of a
//! `CommitmentScheme` round-trip, and an upgrade over Phase-1 `sanity_roundtrip`'s
//! two-separate-`ProverState` flow.

#![cfg(feature = "goldilocks")]
#![expect(clippy::expect_used)]

use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
use jolt_field::Field;
use jolt_whir::{ProverTranscript, VerifierTranscript, WhirCommitment, WhirScheme};

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

#[test]
fn whir_scheme_batch_open_two_size_classes() {
    // Class A: length 2^6, 3 columns at 2 points. Class B: length 2^4, 2 columns at 1 point.
    let (nv_a, n_a, m_a) = (6usize, 3usize, 2usize);
    let (nv_b, n_b, m_b) = (4usize, 2usize, 1usize);
    let cols_a: Vec<Vec<Goldilocks>> = (0..n_a)
        .map(|k| column(1 << nv_a, 0x2000 + k as u64))
        .collect();
    let cols_b: Vec<Vec<Goldilocks>> = (0..n_b)
        .map(|k| column(1 << nv_b, 0x3000 + k as u64))
        .collect();
    let pts_a: Vec<Vec<GoldilocksFp3>> = (0..m_a).map(|f| point(nv_a, 0x4000 + f as u64)).collect();
    let pts_b: Vec<Vec<GoldilocksFp3>> = (0..m_b).map(|f| point(nv_b, 0x5000 + f as u64)).collect();

    let cfg_a = WhirScheme::config(1 << nv_a);
    let cfg_b = WhirScheme::config(1 << nv_b);

    // form-major evals: evals[f * N + v] = columns[v](points[f]).
    let mut evals_a = Vec::with_capacity(m_a * n_a);
    for pt in &pts_a {
        for c in &cols_a {
            evals_a.push(WhirScheme::evaluate(&cfg_a, c, pt));
        }
    }
    let mut evals_b = Vec::with_capacity(m_b * n_b);
    for pt in &pts_b {
        for c in &cols_b {
            evals_b.push(WhirScheme::evaluate(&cfg_b, c, pt));
        }
    }

    // Prover: commit all (A then B) on one transcript, then batch-open per class.
    let mut t = ProverTranscript::new("batch");
    let hints_a: Vec<_> = cols_a
        .iter()
        .map(|c| WhirScheme::commit(&mut t, &cfg_a, c))
        .collect();
    let hints_b: Vec<_> = cols_b
        .iter()
        .map(|c| WhirScheme::commit(&mut t, &cfg_b, c))
        .collect();
    let crefs_a: Vec<&[Goldilocks]> = cols_a.iter().map(Vec::as_slice).collect();
    let crefs_b: Vec<&[Goldilocks]> = cols_b.iter().map(Vec::as_slice).collect();
    WhirScheme::open_batch(&mut t, &cfg_a, &crefs_a, hints_a, &pts_a, &evals_a);
    WhirScheme::open_batch(&mut t, &cfg_b, &crefs_b, hints_b, &pts_b, &evals_b);
    let proof = t.into_proof();

    // Verifier: receive all (A then B) in order, then verify per class.
    let mut vt = VerifierTranscript::new("batch", &proof);
    let comms_a: Vec<WhirCommitment> = (0..n_a)
        .map(|_| WhirScheme::receive_commitment(&mut vt, &cfg_a).expect("recv a"))
        .collect();
    let comms_b: Vec<WhirCommitment> = (0..n_b)
        .map(|_| WhirScheme::receive_commitment(&mut vt, &cfg_b).expect("recv b"))
        .collect();
    let cref_a: Vec<&WhirCommitment> = comms_a.iter().collect();
    let cref_b: Vec<&WhirCommitment> = comms_b.iter().collect();
    WhirScheme::verify_batch(&mut vt, &cfg_a, &cref_a, &pts_a, &evals_a).expect("verify a");
    WhirScheme::verify_batch(&mut vt, &cfg_b, &cref_b, &pts_b, &evals_b).expect("verify b");
}
