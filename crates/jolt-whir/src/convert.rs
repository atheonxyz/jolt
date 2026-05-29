//! The single WHIR seam: convert Jolt's hand-coded Goldilocks elements into the
//! arkworks Goldilocks types WHIR commits over.
//!
//! `jolt-field`'s `Goldilocks` is Montgomery-free; WHIR's `Field64` is arkworks
//! Montgomery `Fp64`. Both represent the same field, so conversion is a cheap
//! canonical-`u64` round-trip. This keeps `jolt-field` free of any `whir`/arkworks
//! dependency — the boundary lives here.

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

/// `Fp3` → WHIR's arkworks `Field64_3` (challenge/eval field). Phase-2 / cross-checks.
#[inline]
pub fn to_field64_3(x: GoldilocksFp3) -> Field64_3 {
    let c = x.coeffs();
    Field64_3::new(to_field64(c[0]), to_field64(c[1]), to_field64(c[2]))
}

/// Convert a base-Goldilocks column to a WHIR `Field64` commit vector.
#[inline]
pub fn column_to_field64(values: &[Goldilocks]) -> Vec<Field64> {
    values.iter().copied().map(to_field64).collect()
}
