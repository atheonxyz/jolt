//! WebGPU-accelerated batch MSM bridge for WASM builds.
//!
//! Provides serialization of G1 points to u32 limbs for GPU transfer,
//! deserialization of Jacobian G1 results, and a CPU fallback MSM using arkworks.

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ff::biginteger::BigInt;
use ark_ff::{Field, Fp, MontConfig, PrimeField, Zero};
use std::sync::OnceLock;

const NUM_LIMBS: usize = 8;

/// Convert Fq (base field) from 8 u32 limbs (little-endian)
#[inline(always)]
fn limbs8_to_fq(limbs: &[u32]) -> Fq {
    let bigint = BigInt::<4>::new([
        (limbs[1] as u64) << 32 | limbs[0] as u64,
        (limbs[3] as u64) << 32 | limbs[2] as u64,
        (limbs[5] as u64) << 32 | limbs[4] as u64,
        (limbs[7] as u64) << 32 | limbs[6] as u64,
    ]);
    Fp(bigint, std::marker::PhantomData)
}

/// Serialize G1 affine point to 16 u32s (x:8 + y:8, Montgomery form)
fn g1_affine_to_limbs(point: &G1Affine) -> [u32; 16] {
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

/// Deserialize Jacobian G1 result from 24 u32s (x:8, y:8, z:8 in Montgomery form)
fn jacobian_from_limbs(limbs: &[u32]) -> G1Projective {
    let x = limbs8_to_fq(&limbs[0..8]);
    let y = limbs8_to_fq(&limbs[8..16]);
    let z = limbs8_to_fq(&limbs[16..24]);
    G1Projective::new_unchecked(x, y, z)
}

/// Precomputed R^{-1} mod q for Montgomery form correction.
/// When scalars are in Montgomery form (Fr), the MSM result is scaled by R.
/// Multiply by R^{-1} to get the correct result.
fn get_r_inv() -> &'static Fr {
    static R_INV: OnceLock<Fr> = OnceLock::new();
    R_INV.get_or_init(|| {
        let r_bigint = <ark_bn254::FrConfig as MontConfig<4>>::R;
        let r_as_fr = Fr::from_bigint(r_bigint).expect("R must be a valid scalar field element");
        r_as_fr
            .inverse()
            .expect("R is invertible in the scalar field")
    })
}

/// Apply R_inv correction: multiply a G1 point by R^{-1} scalar
fn apply_r_inv_correction(point: G1Projective) -> G1Projective {
    if point.is_zero() {
        return point;
    }
    let r_inv = get_r_inv();
    point * r_inv
}

#[cfg(target_arch = "wasm32")]
mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// Check if WebGPU MSM is available (initialized by JS)
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_msm_available")]
        pub fn js_gpu_msm_available() -> bool;
    }
}

/// Check if WebGPU MSM acceleration is available in the current runtime.
#[cfg(target_arch = "wasm32")]
pub fn is_gpu_msm_available() -> bool {
    use std::panic;
    panic::catch_unwind(js_bridge::js_gpu_msm_available).unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_gpu_msm_available() -> bool {
    false
}

/// CPU batch MSM using arkworks `VariableBaseMSM::msm_serial` with rayon
/// parallelism across batch rows — **exactly** matching `commit_tier_1`.
///
/// Points: affine, 16 u32s each (x:8 + y:8, Montgomery form).
/// Scalars: 8 u32s each (raw 256-bit integers — NOT Montgomery form).
/// Returns: Jacobian results, 24 u32s each (x:8 + y:8 + z:8, Montgomery form).
pub fn cpu_batch_msm_from_limbs(
    points_flat: &[u32],
    scalars_flat: &[u32],
    num_points: usize,
    batch_size: usize,
) -> Vec<u32> {
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkMSM;
    use rayon::prelude::*;

    // Deserialize affine bases from u32 limbs
    let bases: Vec<G1Affine> = (0..num_points)
        .map(|i| {
            let off = i * 16;
            let x = limbs8_to_fq(&points_flat[off..off + 8]);
            let y = limbs8_to_fq(&points_flat[off + 8..off + 16]);
            G1Affine::new_unchecked(x, y)
        })
        .collect();

    // rayon par_iter across batch rows — same as commit_tier_1's par_chunks.
    // Each row: msm_serial (no internal threading) — same as msm_field_elements.
    let results: Vec<G1Projective> = (0..batch_size)
        .into_par_iter()
        .map(|b| {
            let row_offset = b * num_points * NUM_LIMBS;
            let scalars: Vec<Fr> = (0..num_points)
                .map(|i| {
                    let off = row_offset + i * NUM_LIMBS;
                    let limbs = &scalars_flat[off..off + NUM_LIMBS];
                    // Convert u32 limbs to little-endian bytes, then reduce mod r.
                    // from_le_bytes_mod_order handles scalars >= Fr::MODULUS correctly
                    // (matching the GPU which reduces via EC group-order arithmetic).
                    let mut bytes = [0u8; 32];
                    for j in 0..NUM_LIMBS {
                        bytes[j * 4..j * 4 + 4].copy_from_slice(&limbs[j].to_le_bytes());
                    }
                    Fr::from_le_bytes_mod_order(&bytes)
                })
                .collect();

            <G1Projective as ArkMSM>::msm_serial(&bases, &scalars).unwrap_or(G1Projective::zero())
        })
        .collect();

    // Serialize Jacobian results to flat u32 array (24 u32s per point)
    let mut out = Vec::with_capacity(batch_size * 24);
    for result in &results {
        let x_words = (result.x.0).0;
        let y_words = (result.y.0).0;
        let z_words = (result.z.0).0;
        for i in 0..4 {
            out.push(x_words[i] as u32);
            out.push((x_words[i] >> 32) as u32);
        }
        for i in 0..4 {
            out.push(y_words[i] as u32);
            out.push((y_words[i] >> 32) as u32);
        }
        for i in 0..4 {
            out.push(z_words[i] as u32);
            out.push((z_words[i] >> 32) as u32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
    use ark_ff::{Field, MontConfig, One, PrimeField, Zero};

    /// Print MSM reference values as u32 limbs (JS format) for use in browser tests.
    /// Computes scalar multiples of the generator and verifies MSM results.
    #[test]
    fn msm_reference_values_correctness() {
        let g = G1Affine::generator();
        let g_proj = G1Projective::from(g);

        // Compute multiples: 2G, 3G, 4G, 5G
        let g2 = (g_proj + g_proj).into_affine();
        let g3 = (g_proj + g_proj + g_proj).into_affine();
        let g5 = (g_proj + g_proj + g_proj + g_proj + g_proj).into_affine();

        // Print G1 generator as JS limbs
        let g_limbs = g1_affine_to_limbs(&g);

        // Print 2G
        let g2_limbs = g1_affine_to_limbs(&g2);

        // Print 3G
        let g3_limbs = g1_affine_to_limbs(&g3);

        // Print 5G
        let g5_limbs = g1_affine_to_limbs(&g5);

        // MSM test 1: 2*G + 3*(2G) + 1*(3G) = 2G + 6G + 3G = 11G
        let bases_1 = vec![g, g2, g3];
        let scalars_1 = vec![Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let msm_1 = G1Projective::msm(&bases_1, &scalars_1).unwrap();
        let expected_11g = g_proj * Fr::from(11u64);
        assert_eq!(
            msm_1, expected_11g,
            "MSM(2*G + 3*2G + 1*3G) should equal 11G"
        );

        let msm_1_affine = msm_1.into_affine();
        let msm_1_limbs = g1_affine_to_limbs(&msm_1_affine);

        // MSM test 2: 1*G + 2*(2G) + 3*(3G) + 4*(4G) + 5*(5G)
        //           = 1G + 4G + 9G + 16G + 25G = 55G
        let g4 = (g_proj + g_proj + g_proj + g_proj).into_affine();
        let bases_2 = vec![g, g2, g3, g4, g5];
        let scalars_2 = vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(5u64),
        ];
        let msm_2 = G1Projective::msm(&bases_2, &scalars_2).unwrap();
        let expected_55g = g_proj * Fr::from(55u64);
        assert_eq!(
            msm_2, expected_55g,
            "MSM(1*G + 2*2G + 3*3G + 4*4G + 5*5G) should equal 55G"
        );

        let msm_2_affine = msm_2.into_affine();
        let msm_2_limbs = g1_affine_to_limbs(&msm_2_affine);
    }

    /// Verify G1 point serialization/deserialization roundtrip.
    /// Serializes with g1_affine_to_limbs, deserializes with limbs8_to_fq.
    #[test]
    fn verify_serialization_roundtrip() {
        // Test with G1 generator
        let g = G1Affine::generator();
        let limbs = g1_affine_to_limbs(&g);

        let x_back = limbs8_to_fq(&limbs[0..8]);
        let y_back = limbs8_to_fq(&limbs[8..16]);
        assert_eq!(x_back, g.x, "G1 generator x roundtrip failed");
        assert_eq!(y_back, g.y, "G1 generator y roundtrip failed");

        // Test with 2G
        let g2 = (G1Projective::from(g) + G1Projective::from(g)).into_affine();
        let limbs2 = g1_affine_to_limbs(&g2);

        let x2_back = limbs8_to_fq(&limbs2[0..8]);
        let y2_back = limbs8_to_fq(&limbs2[8..16]);
        assert_eq!(x2_back, g2.x, "2G x roundtrip failed");
        assert_eq!(y2_back, g2.y, "2G y roundtrip failed");

        // Test Jacobian roundtrip: create Jacobian with z=1, recover affine
        let mut jac_limbs = [0u32; 24];
        jac_limbs[0..16].copy_from_slice(&limbs);
        // z = Fq::one() in Montgomery form
        let one_fq = Fq::one();
        let one_words = (one_fq.0).0;
        for i in 0..4 {
            jac_limbs[16 + i * 2] = one_words[i] as u32;
            jac_limbs[16 + i * 2 + 1] = (one_words[i] >> 32) as u32;
        }
        let g_back = jacobian_from_limbs(&jac_limbs);
        assert_eq!(g_back.into_affine(), g, "Jacobian roundtrip failed");
    }

    /// Compare MSM computed via arkworks VariableBaseMSM against expected scalar multiples.
    #[test]
    fn compare_scalar_msm_vs_arkworks() {
        let g = G1Affine::generator();
        let g_proj = G1Projective::from(g);

        // Test 1: Single point MSM — 7*G = 7G
        let result = G1Projective::msm(&[g], &[Fr::from(7u64)]).unwrap();
        let expected = g_proj * Fr::from(7u64);
        assert_eq!(result, expected, "Single-point MSM: 7*G should equal 7G");

        // Test 2: Two identical bases — 3*G + 5*G = 8G
        let result2 = G1Projective::msm(&[g, g], &[Fr::from(3u64), Fr::from(5u64)]).unwrap();
        let expected2 = g_proj * Fr::from(8u64);
        assert_eq!(
            result2, expected2,
            "Two-point MSM: 3*G + 5*G should equal 8G"
        );

        // Test 3: Different bases — 2*G + 3*(2G) = 2G + 6G = 8G
        let g2 = (g_proj + g_proj).into_affine();
        let result3 = G1Projective::msm(&[g, g2], &[Fr::from(2u64), Fr::from(3u64)]).unwrap();
        let expected3 = g_proj * Fr::from(8u64);
        assert_eq!(
            result3, expected3,
            "Mixed-base MSM: 2*G + 3*(2G) should equal 8G"
        );

        // Test 4: Zero scalar — 0*G + 5*G = 5G
        let result4 = G1Projective::msm(&[g, g], &[Fr::zero(), Fr::from(5u64)]).unwrap();
        let expected4 = g_proj * Fr::from(5u64);
        assert_eq!(
            result4, expected4,
            "MSM with zero scalar: 0*G + 5*G should equal 5G"
        );

        // Test 5: Larger MSM — sum of i*(iG) for i=1..5 = 1+4+9+16+25 = 55G
        let g3 = (g_proj * Fr::from(3u64)).into_affine();
        let g4 = (g_proj * Fr::from(4u64)).into_affine();
        let g5 = (g_proj * Fr::from(5u64)).into_affine();
        let bases = vec![g, g2, g3, g4, g5];
        let scalars = vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(5u64),
        ];
        let result5 = G1Projective::msm(&bases, &scalars).unwrap();
        let expected5 = g_proj * Fr::from(55u64);
        assert_eq!(
            result5, expected5,
            "Larger MSM: sum i*(iG) for i=1..5 should equal 55G"
        );
    }

    /// Verify R_inv correction math for Montgomery-form scalar MSMs.
    /// When scalars are in Montgomery form, the MSM result is scaled by R.
    /// Multiplying by R^{-1} should recover the correct result.
    #[test]
    fn verify_r_inv_correction() {
        let g = G1Affine::generator();
        let g_proj = G1Projective::from(g);

        // Compute R and R^{-1} inline
        let r_bigint = <ark_bn254::FrConfig as MontConfig<4>>::R;
        let r_fr = Fr::from_bigint(r_bigint).unwrap();
        let r_inv = r_fr.inverse().unwrap();

        // Verify R * R^{-1} = 1
        assert_eq!(r_fr * r_inv, Fr::one(), "R * R_inv should equal 1");

        // Simulate GPU Montgomery-form MSM:
        // We want: scalar * G
        // GPU sees Montgomery repr (scalar * R) as the raw integer, computes (scalar * R) * G
        // Correction: (scalar * R * G) * R^{-1} = scalar * G
        let scalar = Fr::from(42u64);
        let expected = g_proj * scalar;

        // Simulate GPU result: uses Montgomery representation as raw integer
        let gpu_result = g_proj * (scalar * r_fr);
        let corrected = gpu_result * r_inv;
        assert_eq!(
            corrected, expected,
            "R_inv correction should recover correct MSM result"
        );

        // Verify apply_r_inv_correction produces the same result
        let corrected2 = apply_r_inv_correction(gpu_result);
        assert_eq!(
            corrected2, expected,
            "apply_r_inv_correction should match manual R_inv"
        );

        // Test with zero point — should remain zero
        let zero_point = G1Projective::zero();
        let zero_corrected = apply_r_inv_correction(zero_point);
        assert!(
            zero_corrected.is_zero(),
            "R_inv correction of zero should be zero"
        );

        // Print R and R_inv as u32 limbs for JS reference
    }
}
