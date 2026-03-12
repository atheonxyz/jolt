//! WebGPU-accelerated batch MSM bridge for WASM builds.
//!
//! Provides serialization of G1 points to u32 limbs for GPU transfer,
//! deserialization of Jacobian G1 results, and a CPU fallback MSM using arkworks.

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ff::biginteger::BigInt;
use ark_ff::{Fp, Zero};
use std::sync::OnceLock;

const SCALAR_LIMBS: usize = 4;

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

/// Precomputed 2^128 mod r (BN254 scalar field order) for two's complement correction.
/// When encoding negative i128 as two's complement (2^128 + val), the GPU result
/// includes an extra `2^128 * P_i` term for each negative scalar. This constant
/// is used to subtract the correction: result - (2^128 mod r) * correction_sum.
fn get_two_pow_128_mod_r() -> &'static Fr {
    static TWO_POW_128: OnceLock<Fr> = OnceLock::new();
    TWO_POW_128.get_or_init(|| {
        let mut val = Fr::from(1u64);
        for _ in 0..128 {
            val = val + val;
        }
        val
    })
}

#[cfg(target_arch = "wasm32")]
mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// Check if WebGPU MSM is available (initialized by JS)
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_msm_available")]
        pub fn js_gpu_msm_available() -> bool;

        /// Dispatch a batched MSM to the GPU. All MSMs share the same bases.
        /// Returns a Promise that resolves to a Uint32Array of Jacobian results
        /// (24 u32s per result: x:8 + y:8 + z:8, Montgomery form).
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_batch_msm")]
        pub fn js_gpu_batch_msm(
            points_flat: &[u32],
            scalars_flat: &[u32],
            num_points: u32,
            scalar_bit_width: u32,
            batch_size: u32,
        ) -> JsValue;
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

/// Convert an i128 scalar to 4 u32 limbs (128-bit two's complement).
///
/// Positive values: direct binary representation (fits in 128 bits).
/// Negative values: two's complement = 2^128 + val (unsigned 128-bit).
/// The GPU decomposes these as unsigned integers. For negative scalars, the result
/// includes an extra `2^128 * P_i` term that must be corrected after the GPU returns.
#[inline]
pub fn i128_to_4limbs(val: i128) -> [u32; 4] {
    // Two's complement representation is just the raw bytes of i128
    let bits = val as u128;
    [
        bits as u32,
        (bits >> 32) as u32,
        (bits >> 64) as u32,
        (bits >> 96) as u32,
    ]
}

/// Compute the two's complement correction for a batch of MSMs.
///
/// When negative i128 scalars are encoded as two's complement (2^128 + val),
/// each negative scalar contributes an extra `2^128 * P_i` to the MSM result.
/// This function computes `(2^128 mod r) * Σ_{i: val_i < 0} P_i` per batch item,
/// which must be subtracted from the GPU result.
///
/// `neg_sums` should contain one G1Projective per batch item: the sum of bases
/// where the scalar was negative. If a batch item has no negative scalars, pass zero.
pub fn apply_twos_complement_corrections(
    gpu_results: &mut [G1Projective],
    neg_sums: &[G1Projective],
) {
    let two_pow_128 = get_two_pow_128_mod_r();
    for (result, neg_sum) in gpu_results.iter_mut().zip(neg_sums.iter()) {
        if !neg_sum.is_zero() {
            *result -= *neg_sum * two_pow_128;
        }
    }
}

/// Serialize G1 affine bases to a flat u32 buffer (16 u32s per point, Montgomery form).
pub fn serialize_g1_bases(bases: &[G1Affine]) -> Vec<u32> {
    let mut out = Vec::with_capacity(bases.len() * 16);
    for b in bases {
        out.extend_from_slice(&g1_affine_to_limbs(b));
    }
    out
}

/// Handle for an in-flight GPU batch MSM computation.
#[cfg(target_arch = "wasm32")]
pub struct GpuBatchMsmHandle {
    gpu_future: wasm_bindgen_futures::JsFuture,
    batch_size: usize,
}

/// Dispatch a batched MSM to the GPU. All MSMs share the same bases (points_flat).
/// Each MSM has `num_points` scalars in `scalars_flat` (4 u32s per scalar, 128-bit
/// two's complement). Returns a handle for later collection with `resolve_gpu_batch_msm`.
#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_batch_msm(
    points_flat: &[u32],
    scalars_flat: &[u32],
    num_points: usize,
    batch_size: usize,
) -> GpuBatchMsmHandle {
    let promise = js_bridge::js_gpu_batch_msm(
        points_flat,
        scalars_flat,
        num_points as u32,
        128,
        batch_size as u32,
    );
    let gpu_future = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise));
    GpuBatchMsmHandle {
        gpu_future,
        batch_size,
    }
}

/// Await GPU batch MSM results and deserialize to G1 projective points.
/// Applies two's complement correction for negative scalars.
///
/// `neg_sums`: one G1Projective per batch item — the sum of bases where the scalar
/// was negative. Pass an empty slice to skip correction (all-positive scalars).
#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_batch_msm(
    handle: GpuBatchMsmHandle,
    neg_sums: &[G1Projective],
) -> Vec<super::wrappers::ArkG1> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let result = handle.gpu_future.await.expect("GPU batch MSM failed");
    let u32_array = result
        .dyn_into::<Uint32Array>()
        .expect("GPU MSM result should be Uint32Array");
    let flat: Vec<u32> = u32_array.to_vec();

    assert_eq!(
        flat.len(),
        handle.batch_size * 24,
        "GPU MSM result size mismatch: expected {} u32s, got {}",
        handle.batch_size * 24,
        flat.len()
    );

    // Raw integer scalars → no Montgomery R factor in the result.
    // Only need two's complement correction for negative scalars.
    let mut results: Vec<G1Projective> = flat
        .chunks_exact(24)
        .map(|chunk| jacobian_from_limbs(chunk))
        .collect();

    if !neg_sums.is_empty() {
        apply_twos_complement_corrections(&mut results, neg_sums);
    }

    results.into_iter().map(super::wrappers::ArkG1).collect()
}

/// CPU batch MSM using arkworks `VariableBaseMSM::msm_serial` with rayon
/// parallelism across batch rows — **exactly** matching `commit_tier_1`.
///
/// Points: affine, 16 u32s each (x:8 + y:8, Montgomery form).
/// Scalars: 4 u32s each (128-bit two's complement raw integers).
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
    let results: Vec<G1Projective> = (0..batch_size)
        .into_par_iter()
        .map(|b| {
            let row_offset = b * num_points * SCALAR_LIMBS;
            let scalars: Vec<Fr> = (0..num_points)
                .map(|i| {
                    let off = row_offset + i * SCALAR_LIMBS;
                    let limbs = &scalars_flat[off..off + SCALAR_LIMBS];
                    // Convert 4 u32 limbs (128-bit two's complement) to Fr.
                    let mut bytes = [0u8; 16];
                    for j in 0..SCALAR_LIMBS {
                        bytes[j * 4..j * 4 + 4].copy_from_slice(&limbs[j].to_le_bytes());
                    }
                    let val = i128::from_le_bytes(bytes);
                    if val >= 0 {
                        Fr::from(val as u128)
                    } else {
                        -Fr::from((-val) as u128)
                    }
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
    use ark_ff::{One, Zero};

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

    /// Verify two's complement encoding and correction math.
    /// For negative i128, encoding as 128-bit two's complement (2^128 + val) introduces
    /// an extra 2^128 * P term that apply_twos_complement_corrections removes.
    #[test]
    fn verify_twos_complement_correction() {
        let g = G1Affine::generator();
        let g_proj = G1Projective::from(g);
        let g2 = (g_proj + g_proj).into_affine();
        let g3 = (g_proj + g_proj + g_proj).into_affine();

        // i128_to_4limbs: positive values
        let limbs_5 = i128_to_4limbs(5);
        assert_eq!(limbs_5, [5, 0, 0, 0]);

        // i128_to_4limbs: negative values (two's complement)
        let limbs_neg5 = i128_to_4limbs(-5);
        let expected_neg5 = (-5i128) as u128;
        assert_eq!(limbs_neg5[0], expected_neg5 as u32);

        // Simulate GPU MSM with two's complement scalars:
        // Want: 5*G + (-3)*2G = 5G - 6G = -G
        // Two's complement encodes -3 as 2^128-3.
        // GPU computes: 5*G + (2^128-3)*2G = 5G - 6G + 2^128*2G = -G + 2^128*2G
        let expected = g_proj * Fr::from(5u64) - G1Projective::from(g2) * Fr::from(3u64);

        let two_pow_128 = *get_two_pow_128_mod_r();
        let gpu_simulated =
            g_proj * Fr::from(5u64) + G1Projective::from(g2) * Fr::from((-3i128 as u128));
        let neg_sum = G1Projective::from(g2); // only g2 had a negative scalar

        let mut results = vec![gpu_simulated];
        apply_twos_complement_corrections(&mut results, &[neg_sum]);
        assert_eq!(
            results[0], expected,
            "Two's complement correction should recover correct MSM"
        );

        // Test with all-positive (no correction needed)
        let pos_only = g_proj * Fr::from(7u64);
        let mut results2 = vec![pos_only];
        apply_twos_complement_corrections(&mut results2, &[G1Projective::zero()]);
        assert_eq!(results2[0], pos_only, "Zero correction should be identity");
    }
}
