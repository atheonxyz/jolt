//! WebGPU-accelerated G2 fixed-base scalar multiplication bridge for WASM builds.
//!
//! When the `webgpu-pairing` feature is enabled in a WASM build, this module provides:
//! - Precomputed table construction for fixed-base windowed scalar mul (w=5)
//! - Table caching: table is built once per base point and reused across calls
//! - GPU buffer caching: JS side keeps the table GPUBuffer alive across invocations
//! - Serialization of scalars to u32 limbs for GPU transfer
//! - Deserialization of G2 Jacobian results from u32 limbs
//! - A JS bridge via `wasm_bindgen` extern imports to call the WebGPU pipeline
//! - Public API: `gpu_g2_fixed_base_scalar_mul(base, scalars)` async fn

use ark_bn254::{Fq, Fq2, Fr, G2Affine, G2Projective};
use ark_ec::CurveGroup;
use ark_ff::biginteger::BigInt;
use ark_ff::{AdditiveGroup, Field, PrimeField, Zero};

use super::wrappers::ArkG2;

use std::cell::RefCell;

const NUM_LIMBS: usize = 8;
const WINDOW_SIZE: usize = 5;
const NUM_WINDOWS: usize = 51; // ceil(254 / 5)
const TABLE_ENTRIES: usize = 31; // 2^5 - 1
const G2_AFFINE_WORDS: usize = 4 * NUM_LIMBS; // x.c0, x.c1, y.c0, y.c1
const G2_JACOBIAN_WORDS: usize = 6 * NUM_LIMBS; // x.c0, x.c1, y.c0, y.c1, z.c0, z.c1

// ---------------------------------------------------------------------------
// Limb conversion helpers
// ---------------------------------------------------------------------------

/// Serialize an Fq element to 8 u32 limbs (Montgomery form, little-endian).
/// Directly splits the internal u64 limbs into u32 pairs.
#[inline(always)]
fn fq_to_limbs(f: &Fq) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    let words = (f.0).0;
    for i in 0..4 {
        out[i * 2] = words[i] as u32;
        out[i * 2 + 1] = (words[i] >> 32) as u32;
    }
    out
}

/// Deserialize an Fq element from 8 u32 limbs (Montgomery form).
/// Uses `Fp(bigint, PhantomData)` to directly set Montgomery representation
/// without re-Montgomery-ifying (same as Fq::new_unchecked).
#[inline(always)]
fn limbs8_to_fq(limbs: &[u32]) -> Fq {
    let bigint = BigInt::<4>::new([
        (limbs[1] as u64) << 32 | limbs[0] as u64,
        (limbs[3] as u64) << 32 | limbs[2] as u64,
        (limbs[5] as u64) << 32 | limbs[4] as u64,
        (limbs[7] as u64) << 32 | limbs[6] as u64,
    ]);
    Fq::new_unchecked(bigint)
}

/// Serialize an Fq2 element to 16 u32 limbs: [c0_limbs..., c1_limbs...]
fn fq2_to_limbs(f: &Fq2) -> [u32; 2 * NUM_LIMBS] {
    let mut out = [0u32; 2 * NUM_LIMBS];
    let c0 = fq_to_limbs(&f.c0);
    let c1 = fq_to_limbs(&f.c1);
    out[..NUM_LIMBS].copy_from_slice(&c0);
    out[NUM_LIMBS..].copy_from_slice(&c1);
    out
}

/// Serialize a G2Affine point to limbs: [x.c0, x.c1, y.c0, y.c1]
fn g2_affine_to_limbs(point: &G2Affine) -> [u32; G2_AFFINE_WORDS] {
    let mut out = [0u32; G2_AFFINE_WORDS];
    let x_limbs = fq2_to_limbs(&point.x);
    let y_limbs = fq2_to_limbs(&point.y);
    out[..2 * NUM_LIMBS].copy_from_slice(&x_limbs);
    out[2 * NUM_LIMBS..].copy_from_slice(&y_limbs);
    out
}

/// Convert a scalar (Fr) to raw bit representation (non-Montgomery) as NUM_LIMBS u32s.
/// The GPU kernel extracts bit windows directly from the scalar representation,
/// so we need raw bits (via into_bigint which de-Montgomery-ifies).
fn fr_to_raw_limbs(scalar: &Fr) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    let bigint = scalar.into_bigint();
    let words = bigint.0;
    for i in 0..4 {
        out[i * 2] = words[i] as u32;
        out[i * 2 + 1] = (words[i] >> 32) as u32;
    }
    out
}

// ---------------------------------------------------------------------------
// Table construction
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Scalar serialization
// ---------------------------------------------------------------------------

/// Serialize a slice of Fr scalars to raw bit representation for GPU.
pub fn serialize_fr_scalars_raw(scalars: &[Fr]) -> Vec<u32> {
    let mut out = Vec::with_capacity(scalars.len() * NUM_LIMBS);
    for s in scalars {
        out.extend_from_slice(&fr_to_raw_limbs(s));
    }
    out
}

// ---------------------------------------------------------------------------
// Result deserialization
// ---------------------------------------------------------------------------

/// Parse a G2 Jacobian result from u32 limbs back to G2Projective.
///
/// The GPU returns Jacobian coordinates (X:Y:Z) in Montgomery form.
/// We reconstruct the Arkworks G2Projective: convert Jacobian to affine first.
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

// ---------------------------------------------------------------------------
// Table cache: build once per base point, reuse across all calls.
//
// The base point is always dory_setup.h2.0 (the G2 generator from setup),
// so this cache effectively means the table is built once at first use.
// Uses thread_local because WASM is single-threaded.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// JS Bridge — extern imports via wasm_bindgen
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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

/// GPU-accelerated G2 fixed-base scalar multiplication with table caching.
///
/// Computes `[s_0 * base, s_1 * base, ..., s_{n-1} * base]` using a precomputed
/// table on WebGPU. The table is built on CPU once and cached in both Rust
/// (thread_local) and JS (persistent GPUBuffer). Subsequent calls with the same
/// base point skip table construction and GPU upload entirely.
///
/// This is an async function because WebGPU buffer mapping is inherently async.
#[cfg(target_arch = "wasm32")]
pub async fn gpu_g2_fixed_base_scalar_mul(
    base: &ark_bn254::G2Projective,
    scalars: &[Fr],
) -> Vec<ArkG2> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let n = scalars.len();
    if n == 0 {
        return vec![];
    }

    // Get or build the precomputed table (cached after first call)
    let base_affine = base.into_affine();
    let (table_limbs, was_cached) = get_or_build_table(&base_affine);

    if was_cached {
        tracing::debug!("G2 table cache hit — skipping CPU table build + GPU upload");
    } else {
        tracing::info!(
            "G2 table built on CPU and cached ({} u32s)",
            table_limbs.len()
        );
        // Upload table to persistent GPU buffer (JS caches the GPUBuffer)
        js_bridge::js_gpu_g2_upload_table(&table_limbs);
    }

    // Serialize scalars to raw bits (non-Montgomery)
    let scalar_limbs = serialize_fr_scalars_raw(scalars);

    // Submit GPU work using cached table buffer
    let result_js = js_bridge::js_gpu_g2_scalar_mul_cached(&scalar_limbs, n as u32).await;

    // Convert JsValue → Uint32Array → Vec<u32>
    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU G2 scalar mul");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    // Deserialize G2 Jacobian results
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * G2_JACOBIAN_WORDS;
        let end = start + G2_JACOBIAN_WORDS;
        let point = parse_g2_jacobian(&result_words[start..end]);
        results.push(ArkG2(point));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::AffineRepr;
    use ark_ff::One;

    /// Mulberry32 PRNG — must match the JS implementation exactly.
    fn mulberry32(seed: u32) -> impl FnMut() -> u32 {
        let mut t = seed;
        move || {
            t = t.wrapping_add(0x6d2b79f5);
            let mut x = t ^ (t >> 15);
            x = x.wrapping_mul(t | 1);
            x ^= x.wrapping_add((x ^ (x >> 7)).wrapping_mul(x | 61));
            x ^ (x >> 14)
        }
    }

    /// Convert raw u32 limbs to Fr.
    /// Clears bit 255 first (GPU only processes bits 0-254),
    /// then uses from_le_bytes_mod_order.
    fn fr_from_raw_limbs_gpu(limbs: &[u32]) -> Fr {
        let mut bytes = [0u8; 32];
        for i in 0..8 {
            let le = limbs[i].to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&le);
        }
        // Clear bit 255 (MSB of last byte) — GPU processes 255 bits (0-254 only)
        bytes[31] &= 0x7F;
        Fr::from_le_bytes_mod_order(&bytes)
    }

    fn affine_to_32_limbs(pt: &G2Affine) -> [u32; 32] {
        g2_affine_to_limbs(pt)
    }

    fn format_u32_array_js(limbs: &[u32]) -> String {
        let strs: Vec<String> = limbs.iter().map(|x| x.to_string()).collect();
        format!("new Uint32Array([{}])", strs.join(", "))
    }

    #[test]
    fn generate_g2_reference_values() {
        let r#gen = G2Affine::generator();
        let gen_proj = G2Projective::from(r#gen);

        // scalar = 1
        {
            let fr = Fr::one();
            let result = (gen_proj * fr).into_affine();
            let limbs = affine_to_32_limbs(&result);
            println!("\"scalar = 1\": [{}],", format_u32_array_js(&limbs));
        }

        // scalar = 2
        {
            let fr = Fr::from(2u64);
            let result = (gen_proj * fr).into_affine();
            let limbs = affine_to_32_limbs(&result);
            println!("\"scalar = 2\": [{}],", format_u32_array_js(&limbs));
        }

        // scalar = 0
        println!("\"scalar = 0\": [null],");

        // small scalars 1-8
        {
            print!("\"small scalars 1-8\": [");
            for k in 1..=8u64 {
                let fr = Fr::from(k);
                let result = (gen_proj * fr).into_affine();
                let limbs = affine_to_32_limbs(&result);
                if k > 1 {
                    print!(", ");
                }
                print!("{}", format_u32_array_js(&limbs));
            }
            println!("],");
        }

        // random 256-bit scalars (seed 0xC0FFEE01) — 16 scalars
        // IMPORTANT: clear bit 255 to match GPU behavior (255 bits only)
        {
            let mut rng = mulberry32(0xC0FFEE01);
            print!("\"random 256-bit scalars (seed 0xC0FFEE01)\": [");
            for k in 0..16 {
                let mut raw_limbs = [0u32; 8];
                for i in 0..8 {
                    raw_limbs[i] = rng();
                }
                // Clear bit 255 — GPU processes bits 0-254 only
                raw_limbs[7] &= 0x7FFFFFFF;
                let fr = fr_from_raw_limbs_gpu(&raw_limbs);
                let result = (gen_proj * fr).into_affine();
                let limbs = affine_to_32_limbs(&result);
                if k > 0 {
                    print!(", ");
                }
                print!("{}", format_u32_array_js(&limbs));
            }
            println!("],");
        }

        // near-modulus scalar (Fr - 1)
        {
            let fr = -Fr::one();
            let result = (gen_proj * fr).into_affine();
            let limbs = affine_to_32_limbs(&result);
            println!(
                "\"near-modulus scalar (Fr - 1)\": [{}],",
                format_u32_array_js(&limbs)
            );
        }
    }
}
