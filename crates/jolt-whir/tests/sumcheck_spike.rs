//! De-risking spike for the single-spongefish transcript migration (piece T).
//!
//! Validates the new proof MODEL: sumcheck round polynomials are written into the
//! NARG byte string via [`ProverTranscript::observe_ext`] and read back by the
//! verifier via the new [`VerifierTranscript::read_exts`] — NOT carried in a side
//! `SumcheckProof` struct. A complete small product-sumcheck round-trips (prover →
//! `into_proof` → verifier) on ONE shared sponge: the verifier replays the
//! `s(0)+s(1)=claim` consistency check each round, both sides squeeze identical
//! challenges, and `check_eof` confirms the NARG is fully consumed. This is the
//! minimum program that forces `read_ext`/`read_exts`/`check_eof` into existence
//! and proves the write→narg→read round-trip the full migration depends on.

#![cfg(feature = "goldilocks")]
#![expect(clippy::expect_used)]

use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
use jolt_field::Field;
use jolt_whir::{ProverTranscript, VerifierTranscript};

/// Deterministic splitmix64 (no rng dependency), mirroring `tests/transcript.rs`.
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

#[inline]
fn zero() -> GoldilocksFp3 {
    GoldilocksFp3::from_u64(0)
}

#[inline]
fn one() -> GoldilocksFp3 {
    GoldilocksFp3::from_u64(1)
}

const NUM_VARS: usize = 3;
const DEGREE: usize = 2; // product of two multilinears → quadratic round poly
const NUM_COEFFS: usize = DEGREE + 1; // [c0, c1, c2]

/// Evaluate a univariate given in coefficient form `[c0, c1, …]` at `x` (Horner).
fn eval_coeffs(c: &[GoldilocksFp3], x: GoldilocksFp3) -> GoldilocksFp3 {
    c.iter().rev().fold(zero(), |acc, coeff| acc * x + *coeff)
}

/// One honest product-sumcheck round over the high variable: given current tables
/// `a`, `b` (length `2·half`), return `s(X) = Σ_j (a0+a1 X)(b0+b1 X)` in coefficient
/// form `[Σ a0 b0, Σ(a0 b1 + a1 b0), Σ a1 b1]` (`a1 = a_hi − a_lo`).
fn round_poly(a: &[GoldilocksFp3], b: &[GoldilocksFp3]) -> [GoldilocksFp3; NUM_COEFFS] {
    let half = a.len() / 2;
    let mut c = [zero(); NUM_COEFFS];
    for j in 0..half {
        let a0 = a[j];
        let a1 = a[j + half] - a[j];
        let b0 = b[j];
        let b1 = b[j + half] - b[j];
        c[0] += a0 * b0;
        c[1] += a0 * b1 + a1 * b0;
        c[2] += a1 * b1;
    }
    c
}

/// Bind the high variable of `t` at `r`: `t'[j] = t[j] + r·(t[j+half] − t[j])`.
fn bind_high(t: &[GoldilocksFp3], r: GoldilocksFp3) -> Vec<GoldilocksFp3> {
    let half = t.len() / 2;
    (0..half).map(|j| t[j] + r * (t[j + half] - t[j])).collect()
}

fn sum_of_products(a: &[GoldilocksFp3], b: &[GoldilocksFp3]) -> GoldilocksFp3 {
    a.iter().zip(b).fold(zero(), |acc, (x, y)| acc + *x * *y)
}

#[test]
fn product_sumcheck_round_trips_through_narg() {
    let n = 1usize << NUM_VARS;
    let mut rng = Rng(0xA5A5_5A5A_1234_5678);
    let mut a: Vec<GoldilocksFp3> = (0..n).map(|_| rand_fp3(&mut rng)).collect();
    let mut b: Vec<GoldilocksFp3> = (0..n).map(|_| rand_fp3(&mut rng)).collect();

    let claim = sum_of_products(&a, &b);

    // Prover: write each round poly into the NARG; squeeze challenges from the sponge.
    let mut pt = ProverTranscript::new("sumcheck-spike");
    let mut prover_challenges = Vec::with_capacity(NUM_VARS);
    let mut running = claim;
    for _ in 0..NUM_VARS {
        let c = round_poly(&a, &b);
        // honest sumcheck invariant: s(0)+s(1) = running claim
        debug_assert_eq!(c[0] + eval_coeffs(&c, one()), running);
        for coeff in &c {
            pt.observe_ext(*coeff);
        }
        let r = pt.challenge_fp3();
        prover_challenges.push(r);
        running = eval_coeffs(&c, r);
        a = bind_high(&a, r);
        b = bind_high(&b, r);
    }
    // After NUM_VARS rounds the tables are length 1; the running claim is A(r)·B(r).
    let (a_r, b_r) = (a[0], b[0]);
    assert_eq!(running, a_r * b_r, "prover: final claim ≠ A(r)·B(r)");
    pt.observe_ext(a_r);
    pt.observe_ext(b_r);

    let proof = pt.into_proof();

    // Verifier: read each round poly back out of the NARG; replay the checks.
    let mut vt = VerifierTranscript::new("sumcheck-spike", &proof);
    let mut verifier_challenges = Vec::with_capacity(NUM_VARS);
    let mut running = claim;
    for _ in 0..NUM_VARS {
        let c = vt.read_exts(NUM_COEFFS).expect("read round poly");
        let s0 = c[0];
        let s1 = eval_coeffs(&c, one());
        assert_eq!(s0 + s1, running, "sumcheck consistency s(0)+s(1)=claim");
        let r = vt.challenge_fp3();
        verifier_challenges.push(r);
        running = eval_coeffs(&c, r);
    }
    let a_r = vt.read_ext().expect("read A(r)");
    let b_r = vt.read_ext().expect("read B(r)");
    assert_eq!(running, a_r * b_r, "verifier: final claim ≠ A(r)·B(r)");

    vt.check_eof().expect("NARG fully consumed at end of proof");

    assert_eq!(
        prover_challenges, verifier_challenges,
        "prover/verifier challenge divergence over the shared sponge"
    );
}

/// A corrupted round-poly coefficient must break the verifier's sumcheck consistency
/// check — confirms the round polys genuinely travel through (and are read from) the
/// NARG, not recomputed on both sides.
#[test]
fn tampered_round_poly_is_rejected() {
    let n = 1usize << NUM_VARS;
    let mut rng = Rng(0x0BAD_F00D_DEAD_BEEF);
    let mut a: Vec<GoldilocksFp3> = (0..n).map(|_| rand_fp3(&mut rng)).collect();
    let mut b: Vec<GoldilocksFp3> = (0..n).map(|_| rand_fp3(&mut rng)).collect();
    let claim = sum_of_products(&a, &b);

    let mut pt = ProverTranscript::new("sumcheck-spike-tamper");
    let mut running = claim;
    for round in 0..NUM_VARS {
        let mut c = round_poly(&a, &b);
        if round == 0 {
            // Tamper the first round's constant coeff; s(0)+s(1) no longer equals claim.
            c[0] += one();
        }
        for coeff in &c {
            pt.observe_ext(*coeff);
        }
        let r = pt.challenge_fp3();
        running = eval_coeffs(&c, r);
        a = bind_high(&a, r);
        b = bind_high(&b, r);
    }
    let proof = pt.into_proof();

    let mut vt = VerifierTranscript::new("sumcheck-spike-tamper", &proof);
    let c = vt.read_exts(NUM_COEFFS).expect("read round poly");
    let s_sum = c[0] + eval_coeffs(&c, one());
    assert_ne!(
        s_sum, claim,
        "tampered round poly must fail s(0)+s(1)=claim"
    );
    let _ = running;
}
