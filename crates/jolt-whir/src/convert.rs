//! The single WHIR seam: convert Jolt's hand-coded Goldilocks elements into the
//! arkworks Goldilocks types WHIR commits over.
//!
//! `jolt-field`'s `Goldilocks` is Montgomery-free; WHIR's `Field64` is arkworks
//! Montgomery `Fp64`. Both represent the same field, so conversion is a cheap
//! canonical-`u64` round-trip. This keeps `jolt-field` free of any `whir`/arkworks
//! dependency — the boundary lives here.

use ark_ff::PrimeField;
use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
use jolt_field::Field;
use whir::algebra::fields::{Field64, Field64_3};

/// Base Goldilocks → WHIR's arkworks `Field64` (commit alphabet).
#[inline]
#[expect(clippy::expect_used)]
pub fn to_field64(x: Goldilocks) -> Field64 {
    // `to_u64()` returns the canonical representative for Goldilocks (always `Some`).
    Field64::from(x.to_u64().expect("Goldilocks always fits u64"))
}

/// WHIR's arkworks `Field64` → base Goldilocks (the inverse of [`to_field64`]).
///
/// `into_bigint()` returns the canonical `[0, p)` representative as a single
/// limb (`Field64` is `Fp64`, one 64-bit limb).
#[inline]
pub fn from_field64(x: Field64) -> Goldilocks {
    Goldilocks::from_u64(x.into_bigint().0[0])
}

/// `Fp3` → WHIR's arkworks `Field64_3` (challenge/eval field). Phase-2 / cross-checks.
#[inline]
pub fn to_field64_3(x: GoldilocksFp3) -> Field64_3 {
    let c = x.coeffs();
    Field64_3::new(to_field64(c[0]), to_field64(c[1]), to_field64(c[2]))
}

/// WHIR's `Field64_3` → `Fp3` (the inverse of [`to_field64_3`]). Used to decode
/// Fiat-Shamir challenges drawn on the shared spongefish transcript so the
/// Goldilocks prover and WHIR agree byte-for-byte on every challenge.
#[inline]
pub fn from_field64_3(x: Field64_3) -> GoldilocksFp3 {
    GoldilocksFp3::new(from_field64(x.c0), from_field64(x.c1), from_field64(x.c2))
}

/// Convert a base-Goldilocks column to a WHIR `Field64` commit vector.
#[inline]
pub fn column_to_field64(values: &[Goldilocks]) -> Vec<Field64> {
    values.iter().copied().map(to_field64).collect()
}
