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

use super::base::Goldilocks;
use crate::Field;

/// Bits per limb (`DWordWL`: two 32-bit words per 64-bit value).
pub const LIMB_BITS: u32 = 32;
const LIMB_MASK: u64 = 0xFFFF_FFFF;

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
