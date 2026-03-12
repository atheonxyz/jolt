//! WebGPU-accelerated G2 fixed-base scalar multiplication bridge for WASM builds.
//!
//! When the `webgpu-pairing` feature is enabled in a WASM build, this module provides:
//! - Precomputed table construction for fixed-base windowed scalar mul (w=5)
//! - Table caching: table is built once per base point and reused across calls
//! - GPU buffer caching: JS side keeps the table GPUBuffer alive across invocations
//! - Serialization of scalars to u32 limbs for GPU transfer
//! - Deserialization of G2 Jacobian results from u32 limbs
//! - A JS bridge via `wasm_bindgen` extern imports to call the WebGPU pipeline
//! - Public API: `dispatch_gpu_g2_scalar_mul` / `resolve_gpu_g2_scalar_mul` for
//!   overlapping GPU work with CPU computation

use ark_bn254::{Fq2, Fr, G2Affine, G2Projective};
use ark_ec::CurveGroup;
use ark_ff::AdditiveGroup;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
use super::webgpu_utils::limbs8_to_fq;
use super::webgpu_utils::{fq_to_limbs, fr_to_raw_limbs, FQ_LIMBS};
#[cfg(target_arch = "wasm32")]
use super::wrappers::ArkG2;
#[cfg(target_arch = "wasm32")]
use ark_ff::{Field, Zero};

const NUM_LIMBS: usize = FQ_LIMBS;
const WINDOW_SIZE: usize = 5;
const NUM_WINDOWS: usize = 51; // ceil(254 / 5)
const TABLE_ENTRIES: usize = 31; // 2^5 - 1
const G2_AFFINE_WORDS: usize = 4 * NUM_LIMBS; // x.c0, x.c1, y.c0, y.c1
#[cfg(target_arch = "wasm32")]
const G2_JACOBIAN_WORDS: usize = 6 * NUM_LIMBS; // x.c0, x.c1, y.c0, y.c1, z.c0, z.c1

fn fq2_to_limbs(f: &Fq2) -> [u32; 2 * NUM_LIMBS] {
    let mut out = [0u32; 2 * NUM_LIMBS];
    let c0 = fq_to_limbs(&f.c0);
    let c1 = fq_to_limbs(&f.c1);
    out[..NUM_LIMBS].copy_from_slice(&c0);
    out[NUM_LIMBS..].copy_from_slice(&c1);
    out
}

fn g2_affine_to_limbs(point: &G2Affine) -> [u32; G2_AFFINE_WORDS] {
    let mut out = [0u32; G2_AFFINE_WORDS];
    let x_limbs = fq2_to_limbs(&point.x);
    let y_limbs = fq2_to_limbs(&point.y);
    out[..2 * NUM_LIMBS].copy_from_slice(&x_limbs);
    out[2 * NUM_LIMBS..].copy_from_slice(&y_limbs);
    out
}

/// Build the precomputed table for a given base point.
///
/// Table layout: for each window i (0..NUM_WINDOWS), for each digit j (1..=TABLE_ENTRIES):
///   table[i * TABLE_ENTRIES + (j-1)] = j * 2^(WINDOW_SIZE * i) * base
///
/// All points stored in affine coordinates (Montgomery form).
pub fn build_fixed_base_table(base: &G2Affine) -> Vec<u32> {
    let total_points = NUM_WINDOWS * TABLE_ENTRIES;
    let mut table = Vec::with_capacity(total_points * G2_AFFINE_WORDS);

    let base_proj = G2Projective::from(*base);

    // power = 2^(WINDOW_SIZE * i) * base
    let mut power = base_proj;

    for _window in 0..NUM_WINDOWS {
        // For this window, compute j * power for j = 1..TABLE_ENTRIES
        let mut multiple = power;
        for _j in 0..TABLE_ENTRIES {
            let affine = multiple.into_affine();
            table.extend_from_slice(&g2_affine_to_limbs(&affine));
            multiple += power;
        }

        // Advance to next window: power *= 2^WINDOW_SIZE
        for _ in 0..WINDOW_SIZE {
            power = power.double();
        }
    }

    assert_eq!(table.len(), total_points * G2_AFFINE_WORDS);
    table
}

/// Serialize a slice of Fr scalars to raw bit representation for GPU.
pub fn serialize_fr_scalars_raw(scalars: &[Fr]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for s in scalars {
        out.extend_from_slice(&fr_to_raw_limbs(s));
    }
    out
}

#[cfg(target_arch = "wasm32")]
fn parse_g2_jacobian(limbs: &[u32]) -> G2Projective {
    let x_c0 = limbs8_to_fq(&limbs[0..NUM_LIMBS]);
    let x_c1 = limbs8_to_fq(&limbs[NUM_LIMBS..2 * NUM_LIMBS]);
    let y_c0 = limbs8_to_fq(&limbs[2 * NUM_LIMBS..3 * NUM_LIMBS]);
    let y_c1 = limbs8_to_fq(&limbs[3 * NUM_LIMBS..4 * NUM_LIMBS]);
    let z_c0 = limbs8_to_fq(&limbs[4 * NUM_LIMBS..5 * NUM_LIMBS]);
    let z_c1 = limbs8_to_fq(&limbs[5 * NUM_LIMBS..6 * NUM_LIMBS]);

    let x = Fq2::new(x_c0, x_c1);
    let y = Fq2::new(y_c0, y_c1);
    let z = Fq2::new(z_c0, z_c1);

    if z.is_zero() {
        return G2Projective::default(); // identity
    }

    // Convert from Jacobian (X, Y, Z) to affine (X/Z^2, Y/Z^3)
    let z_inv = z.inverse().unwrap();
    let z_inv2 = z_inv * z_inv;
    let z_inv3 = z_inv2 * z_inv;
    let aff_x = x * z_inv2;
    let aff_y = y * z_inv3;

    let affine = G2Affine::new_unchecked(aff_x, aff_y);
    G2Projective::from(affine)
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Cached table limbs and the base point fingerprint (first 8 u32 limbs of x.c0).
    /// If the fingerprint matches, reuse the cached table.
    static G2_TABLE_CACHE: RefCell<Option<G2TableCacheEntry>> = RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
struct G2TableCacheEntry {
    fingerprint: [u32; NUM_LIMBS],
    table_limbs: Vec<u32>,
}

#[cfg(target_arch = "wasm32")]
fn get_base_fingerprint(base: &G2Affine) -> [u32; NUM_LIMBS] {
    fq_to_limbs(&base.x.c0)
}

#[cfg(target_arch = "wasm32")]
fn get_or_build_table(base: &G2Affine) -> (Vec<u32>, bool) {
    let fp = get_base_fingerprint(base);
    G2_TABLE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(entry) = cache.as_ref() {
            if entry.fingerprint == fp {
                return (entry.table_limbs.clone(), true);
            }
        }
        let table = build_fixed_base_table(base);
        *cache = Some(G2TableCacheEntry {
            fingerprint: fp,
            table_limbs: table.clone(),
        });
        (table, false)
    })
}

#[cfg(target_arch = "wasm32")]
mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// Upload the precomputed table to a persistent GPU buffer.
        /// Called once (or when the base point changes).
        /// JS side caches the GPUBuffer for reuse across scalar mul calls.
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_g2_upload_table")]
        pub fn js_gpu_g2_upload_table(table: &[u32]);

        /// Run scalar multiplication using the previously-uploaded cached table.
        /// Only sends scalars to GPU — the table buffer is already on device.
        /// Returns a Promise<Uint32Array> containing G2Jacobian results.
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_g2_scalar_mul_cached")]
        pub async fn js_gpu_g2_scalar_mul_cached(scalars: &[u32], num_scalars: u32) -> JsValue;

        /// Legacy: full table + scalar mul in one call (kept for compatibility).
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_g2_scalar_mul")]
        pub async fn js_gpu_g2_scalar_mul(
            table: &[u32],
            scalars: &[u32],
            num_scalars: u32,
        ) -> JsValue;

        /// Check if WebGPU G2 scalar mul is available (initialized by JS)
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_g2_available")]
        pub fn js_gpu_g2_available() -> bool;
    }
}

/// Check if WebGPU G2 scalar mul acceleration is available in the current runtime.
#[cfg(target_arch = "wasm32")]
pub fn is_gpu_g2_available() -> bool {
    use std::panic;
    panic::catch_unwind(js_bridge::js_gpu_g2_available).unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_gpu_g2_available() -> bool {
    false
}

/// Handle for a dispatched GPU G2 scalar multiplication that can be resolved later.
/// Allows CPU work to overlap with the GPU computation.
#[cfg(target_arch = "wasm32")]
pub struct GpuG2ScalarMulHandle {
    future: wasm_bindgen_futures::JsFuture,
    n: usize,
}

/// Dispatch GPU G2 fixed-base scalar multiplication without waiting for results.
///
/// Serializes scalars and submits the GPU compute pass. Returns a handle that
/// can be resolved later with `resolve_gpu_g2_scalar_mul`. This allows the caller
/// to do useful CPU work while the GPU processes the scalar multiplications.
#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_g2_scalar_mul(
    base: &ark_bn254::G2Projective,
    scalars: &[Fr],
) -> GpuG2ScalarMulHandle {
    use js_sys::{Function, Reflect, Uint32Array};
    use wasm_bindgen::JsCast;

    let n = scalars.len();
    assert!(
        n > 0,
        "dispatch_gpu_g2_scalar_mul requires non-empty scalars"
    );

    let base_affine = base.into_affine();
    let (table_limbs, was_cached) = get_or_build_table(&base_affine);

    if was_cached {
        tracing::debug!("G2 table cache hit — skipping CPU table build + GPU upload");
    } else {
        tracing::info!(
            "G2 table built on CPU and cached ({} u32s)",
            table_limbs.len()
        );
        js_bridge::js_gpu_g2_upload_table(&table_limbs);
    }

    let scalar_limbs = serialize_fr_scalars_raw(scalars);

    // Call the JS function directly via js_sys to get the raw Promise.
    // This executes the JS function synchronously (dispatching the GPU compute pass
    // and calling buffer.mapAsync()), then returns the Promise without awaiting it.
    let future = {
        let global = js_sys::global();
        let func: Function = Reflect::get(&global, &"__jolt_gpu_g2_scalar_mul_cached".into())
            .expect("__jolt_gpu_g2_scalar_mul_cached not found on globalThis")
            .dyn_into()
            .expect("__jolt_gpu_g2_scalar_mul_cached is not a function");
        let scalars_js = Uint32Array::from(scalar_limbs.as_slice());
        let promise: js_sys::Promise = func
            .call2(
                &wasm_bindgen::JsValue::UNDEFINED,
                &scalars_js,
                &wasm_bindgen::JsValue::from(n as u32),
            )
            .expect("__jolt_gpu_g2_scalar_mul_cached call failed")
            .dyn_into()
            .expect("expected Promise from __jolt_gpu_g2_scalar_mul_cached");
        wasm_bindgen_futures::JsFuture::from(promise)
    };

    GpuG2ScalarMulHandle { future, n }
}

/// Await the result of a previously dispatched GPU G2 scalar multiplication.
#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_g2_scalar_mul(handle: GpuG2ScalarMulHandle) -> Vec<ArkG2> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let result_js = handle.future.await.expect("GPU G2 scalar mul failed");

    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU G2 scalar mul");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    let mut results = Vec::with_capacity(handle.n);
    for i in 0..handle.n {
        let start = i * G2_JACOBIAN_WORDS;
        let end = start + G2_JACOBIAN_WORDS;
        let point = parse_g2_jacobian(&result_words[start..end]);
        results.push(ArkG2(point));
    }

    results
}
