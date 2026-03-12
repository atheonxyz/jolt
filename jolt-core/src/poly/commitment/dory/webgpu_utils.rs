//! Shared BN254 field element ↔ u32 limb conversions for WebGPU bridge modules.
//!
//! All values are in Montgomery form (matching the GPU shader representation)
//! unless noted otherwise (e.g., `fr_to_raw_limbs` returns de-Montgomery-ified bits).

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ff::biginteger::BigInt;
use ark_ff::{PrimeField, Zero};

pub(crate) const FQ_LIMBS: usize = 8;

/// Deserialize an Fq element from 8 u32 limbs (Montgomery form, little-endian).
#[inline(always)]
pub(crate) fn limbs8_to_fq(limbs: &[u32]) -> Fq {
    let bigint = BigInt::<4>::new([
        (limbs[1] as u64) << 32 | limbs[0] as u64,
        (limbs[3] as u64) << 32 | limbs[2] as u64,
        (limbs[5] as u64) << 32 | limbs[4] as u64,
        (limbs[7] as u64) << 32 | limbs[6] as u64,
    ]);
    Fq::new_unchecked(bigint)
}

/// Serialize an Fq element to 8 u32 limbs (Montgomery form, little-endian).
#[inline(always)]
pub(crate) fn fq_to_limbs(f: &Fq) -> [u32; FQ_LIMBS] {
    let mut out = [0u32; FQ_LIMBS];
    let words = (f.0).0;
    for i in 0..4 {
        out[i * 2] = words[i] as u32;
        out[i * 2 + 1] = (words[i] >> 32) as u32;
    }
    out
}

/// Serialize an Fr scalar to raw bit representation (non-Montgomery) as 8 u32s.
/// GPU kernels extract bit windows directly from the scalar, so raw bits are
/// needed (via `into_bigint` which de-Montgomery-ifies).
pub(crate) fn fr_to_raw_limbs(scalar: &Fr) -> [u32; FQ_LIMBS] {
    let mut out = [0u32; FQ_LIMBS];
    let bigint = scalar.into_bigint();
    let words = bigint.0;
    for i in 0..4 {
        out[i * 2] = words[i] as u32;
        out[i * 2 + 1] = (words[i] >> 32) as u32;
    }
    out
}

/// Serialize a G1Affine point to 16 u32s (x:8 + y:8, Montgomery form).
pub(crate) fn g1_affine_to_limbs(point: &G1Affine) -> [u32; 16] {
    let mut out = [0u32; 16];
    let x_words = (point.x.0).0;
    let y_words = (point.y.0).0;
    for i in 0..4 {
        out[i * 2] = x_words[i] as u32;
        out[i * 2 + 1] = (x_words[i] >> 32) as u32;
    }
    for i in 0..4 {
        out[8 + i * 2] = y_words[i] as u32;
        out[8 + i * 2 + 1] = (y_words[i] >> 32) as u32;
    }
    out
}

/// Deserialize a Jacobian G1 result from 24 u32s (x:8, y:8, z:8 in Montgomery form).
/// Returns the identity point if z == 0.
#[allow(dead_code)] // Used by wasm32-gated code and tests
pub(crate) fn jacobian_from_limbs(limbs: &[u32]) -> G1Projective {
    let x = limbs8_to_fq(&limbs[0..8]);
    let y = limbs8_to_fq(&limbs[8..16]);
    let z = limbs8_to_fq(&limbs[16..24]);
    if z.is_zero() {
        return G1Projective::default();
    }
    G1Projective::new_unchecked(x, y, z)
}
