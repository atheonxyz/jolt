//! Value ↔ base-Goldilocks limb primitives for the RV64 witness.
//!
//! Goldilocks holds `[0, p) = [0, 2^64 − 2^32]`, so a full 64-bit (or signed
//! 64-bit-magnitude) value is **not** a single canonical element — it is stored
//! as two 32-bit limbs `v = lo + hi·2^32`, each `< 2^32 < p` (the `DWordWL`
//! shape from lambda_vm). Signed increments carry a separate sign bit.
//!
//! Phase 1 only *decomposes and commits* these limbs; the recomposition /
//! `2^-32` carry **constraints** that tie limbs back together are Phase 2.
//! The recompose helpers here exist for round-trip tests.

use super::base::{Goldilocks, INV_TWO_POW_32};
use crate::Field;

/// Bits per limb (`DWordWL`: two 32-bit words per 64-bit value).
pub const LIMB_BITS: u32 = 32;
const LIMB_MASK: u64 = 0xFFFF_FFFF;

/// `2^32` as a base-field element (the limb recomposition weight).
#[inline]
fn two_pow_32() -> Goldilocks {
    Goldilocks::from_u64(1 << LIMB_BITS)
}

/// Split a `u64` into two base-field limbs `[lo, hi]` with `v = lo + hi·2^32`.
#[inline]
pub fn u64_to_limbs(v: u64) -> [Goldilocks; 2] {
    [
        Goldilocks::from_u64(v & LIMB_MASK),
        Goldilocks::from_u64(v >> LIMB_BITS),
    ]
}

/// Recompose two limbs into a `u64` (`lo + hi·2^32`). For tests / verification.
///
/// Each limb is taken modulo `2^32` (its valid range); a well-formed witness
/// limb is already `< 2^32`.
#[inline]
pub fn limbs_to_u64(limbs: [Goldilocks; 2]) -> u64 {
    let lo = limbs[0].to_u64().unwrap_or(0) & LIMB_MASK;
    let hi = limbs[1].to_u64().unwrap_or(0) & LIMB_MASK;
    lo | (hi << LIMB_BITS)
}

/// Sign-magnitude decomposition of a signed increment (`RdInc`/`RamInc`).
///
/// A register/RAM delta `post − pre` of two `u64`s has magnitude `< 2^64`, so
/// the magnitude fits exactly two 32-bit limbs. Returns `(sign, [lo, hi])` where
/// `sign = 1` for negative, else `0`; the value is `(1 − 2·sign)·(lo + hi·2^32)`.
#[inline]
pub fn i128_to_sign_limbs(v: i128) -> (Goldilocks, [Goldilocks; 2]) {
    debug_assert!(
        v.unsigned_abs() < (1u128 << 64),
        "increment magnitude must fit 64 bits (2 limbs)"
    );
    let sign = if v < 0 {
        Goldilocks::from_u64(1)
    } else {
        Goldilocks::from_u64(0)
    };
    let mag = v.unsigned_abs() as u64;
    (sign, u64_to_limbs(mag))
}

/// Recompose `(sign, [lo, hi])` into an `i128`. For tests / verification.
#[inline]
pub fn sign_limbs_to_i128(sign: Goldilocks, limbs: [Goldilocks; 2]) -> i128 {
    let mag = limbs_to_u64(limbs) as i128;
    if sign.to_u64().unwrap_or(0) == 1 {
        -mag
    } else {
        mag
    }
}

/// **Signed** two-limb decomposition of an i65 increment (`RdInc`/`RamInc`):
/// `v = lo + hi·2^32` where `lo ∈ [0, 2^32)` and `hi` is the *signed* high limb
/// (`hi = ⌊v / 2^32⌋`, magnitude `< 2^32`, stored as `p − |hi|` when negative).
///
/// This is the Phase-2 representation: recomposition [`signed_limbs_recompose`] is
/// **linear** in the two committed columns (no separate sign factor), so the
/// `Val = Σ inc·wa·LT` sumcheck stays degree-3.
#[inline]
pub fn i128_to_signed_limbs(v: i128) -> [Goldilocks; 2] {
    debug_assert!(
        v.unsigned_abs() < (1u128 << 64),
        "increment magnitude must fit 64 bits (signed 2 limbs)"
    );
    let lo = v.rem_euclid(1i128 << LIMB_BITS) as u64; // [0, 2^32)
    let hi = v.div_euclid(1i128 << LIMB_BITS); // signed, |hi| < 2^32
    [Goldilocks::from_u64(lo), Goldilocks::from_i128(hi)]
}

/// Linear field recomposition `lo + hi·2^32` of a signed 2-limb value. Equals
/// `Goldilocks::from_i128(v)` for any `v` produced by [`i128_to_signed_limbs`].
#[inline]
pub fn signed_limbs_recompose(limbs: [Goldilocks; 2]) -> Goldilocks {
    limbs[0] + limbs[1] * two_pow_32()
}

/// The Boolean carry of a 32-bit limb addition via the `2^-32` trick:
/// `carry = 2^-32·(a + b − sum)`. For `a, b < 2^32` and `sum = (a+b) mod 2^32`
/// this is exactly the carry-out bit `⌊(a+b)/2^32⌋ ∈ {0, 1}` (exact because
/// `2^32 | p−1`). The constraint side range-checks the result is Boolean.
#[inline]
pub fn carry_bit(a: Goldilocks, b: Goldilocks, sum: Goldilocks) -> Goldilocks {
    (a + b - sum) * INV_TWO_POW_32
}

/// Split a `u128` (e.g. a 128-bit MUL product) into four 32-bit base-field limbs
/// `[p0, p1, p2, p3]` with `v = Σ p_i·2^{32i}`.
#[inline]
pub fn u128_to_limbs(v: u128) -> [Goldilocks; 4] {
    let mask = u128::from(LIMB_MASK);
    [
        Goldilocks::from_u64((v & mask) as u64),
        Goldilocks::from_u64(((v >> 32) & mask) as u64),
        Goldilocks::from_u64(((v >> 64) & mask) as u64),
        Goldilocks::from_u64((v >> 96) as u64),
    ]
}

/// Recompose four 32-bit limbs into a `u128`. For tests / verification.
#[inline]
pub fn limbs_to_u128(limbs: [Goldilocks; 4]) -> u128 {
    let l = |i: usize| u128::from(limbs[i].to_u64().unwrap_or(0) & LIMB_MASK);
    l(0) | (l(1) << 32) | (l(2) << 64) | (l(3) << 96)
}
