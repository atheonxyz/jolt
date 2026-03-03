//! WebGPU-accelerated batch MSM bridge for WASM builds.
//!
//! When the `webgpu-pairing` feature is enabled in a WASM build, this module provides:
//! - Serialization of G1 points and various scalar types to u32 limbs for GPU transfer
//! - Deserialization of Jacobian G1 results from u32 limbs
//! - A JS bridge via `wasm_bindgen` extern imports to call the WebGPU CUZK Pippenger MSM
//! - R^{-1} correction for Montgomery-form scalar MSMs

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ff::biginteger::{BigInt, S128};
use ark_ff::{Field, Fp, MontConfig, PrimeField, Zero};
use std::sync::OnceLock;

const NUM_LIMBS: usize = 8;

// ---------------------------------------------------------------------------
// Limb conversion helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Scalar serialization — convert various types to 8 × u32 limbs
// ---------------------------------------------------------------------------

/// Serialize a Fr scalar to 8 u32 limbs (in Montgomery form, as Arkworks stores them).
/// No into_bigint() — we keep Montgomery form and correct with R^{-1} after MSM.
fn fr_to_limbs(scalar: &Fr) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    let words = (scalar.0).0;
    for i in 0..4 {
        out[i * 2] = words[i] as u32;
        out[i * 2 + 1] = (words[i] >> 32) as u32;
    }
    out
}

/// Serialize small scalar types to 8 u32 limbs (zero-padded).
/// These are NOT in Montgomery form, so no R_inv correction needed.
fn u8_to_limbs(s: u8) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    out[0] = s as u32;
    out
}

fn u16_to_limbs(s: u16) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    out[0] = s as u32;
    out
}

fn u32_to_limbs(s: u32) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    out[0] = s;
    out
}

fn u64_to_limbs(s: u64) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    out[0] = s as u32;
    out[1] = (s >> 32) as u32;
    out
}

fn u128_to_limbs(s: u128) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    out[0] = s as u32;
    out[1] = (s >> 32) as u32;
    out[2] = (s >> 64) as u32;
    out[3] = (s >> 96) as u32;
    out
}

fn i64_to_limbs(s: i64) -> [u32; NUM_LIMBS] {
    // Convert signed to field element, then serialize as Fr (Montgomery form)
    let fr: Fr = if s >= 0 {
        Fr::from(s as u64)
    } else {
        -Fr::from((-s) as u64)
    };
    fr_to_limbs(&fr)
}

fn i128_to_limbs(s: i128) -> [u32; NUM_LIMBS] {
    let fr: Fr = if s >= 0 {
        Fr::from(s as u128)
    } else {
        -Fr::from((-s) as u128)
    };
    fr_to_limbs(&fr)
}

fn s128_to_limbs(s: &S128) -> [u32; NUM_LIMBS] {
    // S128 has to_i128() for small values, and magnitude_as_u128() + is_positive for large
    if let Some(val) = s.to_i128() {
        i128_to_limbs(val)
    } else {
        // Exceeds i128 range: convert via magnitude + sign
        let fr: Fr = if s.is_positive {
            Fr::from(s.magnitude_as_u128())
        } else {
            -Fr::from(s.magnitude_as_u128())
        };
        fr_to_limbs(&fr)
    }
}

// ---------------------------------------------------------------------------
// R_inv correction for Montgomery-form scalar MSMs
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// JS Bridge — extern imports via wasm_bindgen
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// Called from Rust to invoke the WebGPU batch MSM pipeline in JS.
        /// Takes flattened G1 points (16 u32s per point), flattened scalars (8 u32s per scalar),
        /// numPoints, scalarBitWidth, and batchSize.
        /// Returns a Promise<Uint32Array> containing Jacobian results (24 u32s per MSM).
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_batch_msm")]
        pub async fn js_gpu_batch_msm(
            points_flat: &[u32],
            scalars_flat: &[u32],
            num_points: u32,
            scalar_bit_width: u32,
            batch_size: u32,
        ) -> JsValue;

        /// Non-async version: dispatches GPU work and returns the raw Promise (JsValue).
        /// The JS function body runs synchronously up to queue.submit(), starting
        /// GPU execution immediately. The Promise resolves when buffer mapping completes.
        /// Use this to overlap GPU work with CPU work before awaiting.
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_batch_msm")]
        pub fn js_gpu_batch_msm_fire(
            points_flat: &[u32],
            scalars_flat: &[u32],
            num_points: u32,
            scalar_bit_width: u32,
            batch_size: u32,
        ) -> JsValue;

        /// Check if WebGPU MSM is available (initialized by JS)
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_msm_available")]
        pub fn js_gpu_msm_available() -> bool;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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

/// Scalar type descriptor for determining bit width and R_inv correction needs
#[derive(Clone, Copy)]
pub enum ScalarKind {
    U8,
    U16,
    U32,
    U64,
    U128,
    I64,
    I128,
    S128,
    Fr, // 256-bit Montgomery form — needs R_inv correction
}

impl ScalarKind {
    pub fn bit_width(&self) -> u32 {
        match self {
            ScalarKind::U8 => 8,
            ScalarKind::U16 => 16,
            ScalarKind::U32 => 32,
            ScalarKind::U64 => 64,
            ScalarKind::U128 => 128,
            // Signed types are converted to Fr (Montgomery form) so they are 256-bit
            ScalarKind::I64 | ScalarKind::I128 | ScalarKind::S128 | ScalarKind::Fr => 256,
        }
    }

    pub fn needs_r_inv(&self) -> bool {
        // Any type that gets serialized as Montgomery-form Fr needs R_inv correction
        match self {
            ScalarKind::U8
            | ScalarKind::U16
            | ScalarKind::U32
            | ScalarKind::U64
            | ScalarKind::U128 => false,
            // i64/i128/S128 are converted to Fr (Montgomery form) during serialization
            ScalarKind::I64 | ScalarKind::I128 | ScalarKind::S128 | ScalarKind::Fr => true,
        }
    }
}

/// GPU-accelerated batch MSM for a batch of MSMs sharing the same bases.
///
/// Each MSM computes `Σ scalars[i] * bases[i]` for a row of scalars.
///
/// `bases`: G1Affine points shared by all MSMs in the batch
/// `scalar_rows`: Each row is a set of scalars for one MSM (same length as bases)
/// `scalar_kind`: Type of scalars (determines bit width and R_inv correction)
///
/// Returns one G1Projective result per row (i.e., per MSM in the batch).
#[cfg(target_arch = "wasm32")]
pub async fn gpu_batch_msm(
    bases: &[G1Affine],
    scalar_rows: &[Vec<u32>], // Already serialized to NUM_LIMBS u32s per scalar
    scalar_kind: ScalarKind,
    num_points_per_row: usize,
) -> Vec<G1Projective> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let batch_size = scalar_rows.len();
    let num_points = num_points_per_row;

    // Serialize bases to flat u32 array (16 u32s per point)
    let mut points_flat = Vec::with_capacity(num_points * 16);
    for base in bases.iter().take(num_points) {
        points_flat.extend_from_slice(&g1_affine_to_limbs(base));
    }

    // Flatten all scalar rows into a single contiguous array
    // Layout: [row0_scalar0..row0_scalarN, row1_scalar0..row1_scalarN, ...]
    let mut scalars_flat = Vec::with_capacity(batch_size * num_points * NUM_LIMBS);
    for row in scalar_rows {
        scalars_flat.extend_from_slice(row);
    }

    let bit_width = scalar_kind.bit_width();

    // Call JS WebGPU MSM
    let result_js = js_bridge::js_gpu_batch_msm(
        &points_flat,
        &scalars_flat,
        num_points as u32,
        bit_width,
        batch_size as u32,
    )
    .await;

    // Convert JsValue → Uint32Array → Vec<u32>
    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU MSM");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    // Deserialize Jacobian results and optionally apply R_inv correction
    let needs_r_inv = scalar_kind.needs_r_inv();
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let start = i * 24; // 3 * NUM_LIMBS = 24 u32s per Jacobian point
        let point = jacobian_from_limbs(&result_words[start..start + 24]);
        if needs_r_inv {
            results.push(apply_r_inv_correction(point));
        } else {
            results.push(point);
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Scalar serialization helpers for commit_tier_1
// ---------------------------------------------------------------------------

/// Serialize a slice of u8 scalars into a Vec of u32 limbs (NUM_LIMBS per scalar)
pub fn serialize_u8_scalars(scalars: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for &s in scalars {
        out.extend_from_slice(&u8_to_limbs(s));
    }
    out
}

pub fn serialize_u16_scalars(scalars: &[u16]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for &s in scalars {
        out.extend_from_slice(&u16_to_limbs(s));
    }
    out
}

pub fn serialize_u32_scalars(scalars: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for &s in scalars {
        out.extend_from_slice(&u32_to_limbs(s));
    }
    out
}

pub fn serialize_u64_scalars(scalars: &[u64]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for &s in scalars {
        out.extend_from_slice(&u64_to_limbs(s));
    }
    out
}

pub fn serialize_u128_scalars(scalars: &[u128]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for &s in scalars {
        out.extend_from_slice(&u128_to_limbs(s));
    }
    out
}

pub fn serialize_i64_scalars(scalars: &[i64]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for &s in scalars {
        out.extend_from_slice(&i64_to_limbs(s));
    }
    out
}

pub fn serialize_i128_scalars(scalars: &[i128]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for &s in scalars {
        out.extend_from_slice(&i128_to_limbs(s));
    }
    out
}

pub fn serialize_s128_scalars(scalars: &[S128]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for s in scalars {
        out.extend_from_slice(&s128_to_limbs(s));
    }
    out
}

pub fn serialize_fr_scalars(scalars: &[Fr]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for s in scalars {
        out.extend_from_slice(&fr_to_limbs(s));
    }
    out
}

// ---------------------------------------------------------------------------
// Non-blocking dispatch/resolve for GPU MSM overlap
// ---------------------------------------------------------------------------

/// Handle for an in-flight GPU MSM computation.
/// Created by `dispatch_gpu_batch_msm`, consumed by `resolve_gpu_batch_msm`.
#[cfg(target_arch = "wasm32")]
pub struct GpuMsmHandle {
    promise: wasm_bindgen::JsValue,
    batch_size: usize,
    needs_r_inv: bool,
}

/// Dispatch a GPU batch MSM without awaiting the result.
/// The JS function body runs synchronously up to queue.submit(), so
/// the GPU hardware starts executing immediately. Call `resolve_gpu_batch_msm()`
/// later to await the results.
///
/// This enables overlapping GPU MSM with CPU work (e.g., OneHot commit_rows).
#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_batch_msm(
    bases: &[G1Affine],
    scalar_rows: &[Vec<u32>],
    scalar_kind: ScalarKind,
    num_points_per_row: usize,
) -> GpuMsmHandle {
    let batch_size = scalar_rows.len();
    let num_points = num_points_per_row;

    let mut points_flat = Vec::with_capacity(num_points * 16);
    for base in bases.iter().take(num_points) {
        points_flat.extend_from_slice(&g1_affine_to_limbs(base));
    }

    let mut scalars_flat = Vec::with_capacity(batch_size * num_points * NUM_LIMBS);
    for row in scalar_rows {
        scalars_flat.extend_from_slice(row);
    }

    let bit_width = scalar_kind.bit_width();

    // Call JS function synchronously — GPU work starts executing before this returns.
    // The returned JsValue is a Promise that resolves when buffer mapping completes.
    let promise = js_bridge::js_gpu_batch_msm_fire(
        &points_flat,
        &scalars_flat,
        num_points as u32,
        bit_width,
        batch_size as u32,
    );

    GpuMsmHandle {
        promise,
        batch_size,
        needs_r_inv: scalar_kind.needs_r_inv(),
    }
}

/// Await a previously dispatched GPU batch MSM and return the results.
#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_batch_msm(handle: GpuMsmHandle) -> Vec<G1Projective> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let result_js = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(handle.promise))
        .await
        .expect("GPU MSM promise rejected");

    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU MSM");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    let mut results = Vec::with_capacity(handle.batch_size);
    for i in 0..handle.batch_size {
        let start = i * 24;
        let point = jacobian_from_limbs(&result_words[start..start + 24]);
        if handle.needs_r_inv {
            results.push(apply_r_inv_correction(point));
        } else {
            results.push(point);
        }
    }

    results
}

// ---------------------------------------------------------------------------
// CPU batch MSM — mirrors the commit_tier_1 path for browser comparison tests
// ---------------------------------------------------------------------------

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
    use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
    use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
    use ark_ff::{Field, MontConfig, One, PrimeField, Zero};

    /// Helper: print a slice of u32 limbs in JS `new Uint32Array([...])` format.
    fn print_js_u32_array(name: &str, limbs: &[u32]) {
        print!("const {} = new Uint32Array([", name);
        for (i, l) in limbs.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            if i % 8 == 0 {
                print!("\n  ");
            }
            print!("0x{:08x}", l);
        }
        println!("\n]);");
    }

    /// Print MSM reference values as u32 limbs (JS format) for use in browser tests.
    /// Computes scalar multiples of the generator and verifies MSM results.
    #[test]
    fn print_msm_reference_values() {
        let g = G1Affine::generator();
        let g_proj = G1Projective::from(g);

        // Compute multiples: 2G, 3G, 4G, 5G
        let g2 = (g_proj + g_proj).into_affine();
        let g3 = (g_proj + g_proj + g_proj).into_affine();
        let g5 = (g_proj + g_proj + g_proj + g_proj + g_proj).into_affine();

        // Print G1 generator as JS limbs
        let g_limbs = g1_affine_to_limbs(&g);
        println!("G1 generator:");
        print_js_u32_array("REF_G_X", &g_limbs[0..8]);
        print_js_u32_array("REF_G_Y", &g_limbs[8..16]);

        // Print 2G
        let g2_limbs = g1_affine_to_limbs(&g2);
        println!("\n2G:");
        print_js_u32_array("REF_2G_X", &g2_limbs[0..8]);
        print_js_u32_array("REF_2G_Y", &g2_limbs[8..16]);

        // Print 3G
        let g3_limbs = g1_affine_to_limbs(&g3);
        println!("\n3G:");
        print_js_u32_array("REF_3G_X", &g3_limbs[0..8]);
        print_js_u32_array("REF_3G_Y", &g3_limbs[8..16]);

        // Print 5G
        let g5_limbs = g1_affine_to_limbs(&g5);
        println!("\n5G:");
        print_js_u32_array("REF_5G_X", &g5_limbs[0..8]);
        print_js_u32_array("REF_5G_Y", &g5_limbs[8..16]);

        // MSM test 1: 2*G + 3*(2G) + 1*(3G) = 2G + 6G + 3G = 11G
        let bases_1 = vec![g, g2, g3];
        let scalars_1 = vec![Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let msm_1 = G1Projective::msm(&bases_1, &scalars_1).unwrap();
        let expected_11g = g_proj * Fr::from(11u64);
        assert_eq!(
            msm_1, expected_11g,
            "MSM(2*G + 3*2G + 1*3G) should equal 11G"
        );
        println!("\nMSM test 1: 2*G + 3*(2G) + 1*(3G) = 11G OK");

        let msm_1_affine = msm_1.into_affine();
        let msm_1_limbs = g1_affine_to_limbs(&msm_1_affine);
        print_js_u32_array("REF_11G_X", &msm_1_limbs[0..8]);
        print_js_u32_array("REF_11G_Y", &msm_1_limbs[8..16]);

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
        println!("\nMSM test 2: 1*G + 2*(2G) + 3*(3G) + 4*(4G) + 5*(5G) = 55G OK");

        let msm_2_affine = msm_2.into_affine();
        let msm_2_limbs = g1_affine_to_limbs(&msm_2_affine);
        print_js_u32_array("REF_55G_X", &msm_2_limbs[0..8]);
        print_js_u32_array("REF_55G_Y", &msm_2_limbs[8..16]);
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
        println!("G1 generator serialization roundtrip: OK");

        // Test with 2G
        let g2 = (G1Projective::from(g) + G1Projective::from(g)).into_affine();
        let limbs2 = g1_affine_to_limbs(&g2);

        let x2_back = limbs8_to_fq(&limbs2[0..8]);
        let y2_back = limbs8_to_fq(&limbs2[8..16]);
        assert_eq!(x2_back, g2.x, "2G x roundtrip failed");
        assert_eq!(y2_back, g2.y, "2G y roundtrip failed");
        println!("2G serialization roundtrip: OK");

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
        println!("Jacobian serialization roundtrip: OK");
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
        println!("MSM test: 7*G = 7G OK");

        // Test 2: Two identical bases — 3*G + 5*G = 8G
        let result2 = G1Projective::msm(&[g, g], &[Fr::from(3u64), Fr::from(5u64)]).unwrap();
        let expected2 = g_proj * Fr::from(8u64);
        assert_eq!(
            result2, expected2,
            "Two-point MSM: 3*G + 5*G should equal 8G"
        );
        println!("MSM test: 3*G + 5*G = 8G OK");

        // Test 3: Different bases — 2*G + 3*(2G) = 2G + 6G = 8G
        let g2 = (g_proj + g_proj).into_affine();
        let result3 = G1Projective::msm(&[g, g2], &[Fr::from(2u64), Fr::from(3u64)]).unwrap();
        let expected3 = g_proj * Fr::from(8u64);
        assert_eq!(
            result3, expected3,
            "Mixed-base MSM: 2*G + 3*(2G) should equal 8G"
        );
        println!("MSM test: 2*G + 3*(2G) = 8G OK");

        // Test 4: Zero scalar — 0*G + 5*G = 5G
        let result4 = G1Projective::msm(&[g, g], &[Fr::zero(), Fr::from(5u64)]).unwrap();
        let expected4 = g_proj * Fr::from(5u64);
        assert_eq!(
            result4, expected4,
            "MSM with zero scalar: 0*G + 5*G should equal 5G"
        );
        println!("MSM test: 0*G + 5*G = 5G OK");

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
        println!("MSM test: sum i*(iG) for i=1..5 = 55G OK");
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
        println!("R * R^{{-1}} = 1 OK");

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
        println!("R_inv correction for scalar=42: OK");

        // Verify apply_r_inv_correction produces the same result
        let corrected2 = apply_r_inv_correction(gpu_result);
        assert_eq!(
            corrected2, expected,
            "apply_r_inv_correction should match manual R_inv"
        );
        println!("apply_r_inv_correction matches manual computation OK");

        // Test with zero point — should remain zero
        let zero_point = G1Projective::zero();
        let zero_corrected = apply_r_inv_correction(zero_point);
        assert!(
            zero_corrected.is_zero(),
            "R_inv correction of zero should be zero"
        );
        println!("R_inv correction of zero point: OK");

        // Print R and R_inv as u32 limbs for JS reference
        let r_limbs = fr_to_limbs(&r_fr);
        let r_inv_limbs = fr_to_limbs(&r_inv);
        print_js_u32_array("MONT_R", &r_limbs);
        print_js_u32_array("MONT_R_INV", &r_inv_limbs);
    }
}
