//! WebGPU-accelerated batch multi-pairing bridge for WASM builds.
//!
//! When the `webgpu-pairing` feature is enabled in a WASM build, this module provides:
//! - Serialization of G1/G2 points to u32 limbs for GPU transfer
//! - Deserialization of Fp12 Miller loop results from u32 limbs
//! - A JS bridge via `wasm_bindgen` extern imports to call the WebGPU pipeline
//! - A hybrid 95/5 GPU/CPU batch multi-pairing: GPU handles 95% of Miller loops,
//!   CPU computes 5% in parallel (via Rayon worker threads), results are combined
//!   before final exponentiation

use ark_bn254::{Bn254, Fq, Fq12, Fq2, Fq6, Fr, G1Affine, G1Projective, G2Affine};
use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_ec::CurveGroup;
use ark_ff::biginteger::{BigInt, BigInteger};
use ark_ff::{Field, PrimeField, Zero};

use super::wrappers::{ArkG1, ArkG2, ArkGT};

const NUM_LIMBS: usize = 8;
const LOG_LIMB_SIZE: u32 = 32;
const FP12_WORDS: usize = 12 * NUM_LIMBS; // 96
const G1_JACOBIAN_WORDS: usize = 24;

fn bigint4_to_limbs(val: &BigInt<4>) -> [u32; NUM_LIMBS] {
    let mut result = [0u32; NUM_LIMBS];
    let bytes = val.to_bytes_le();
    let mut v = 0u64;
    let mut bits = 0u32;
    let mut idx = 0;
    for &byte in bytes.iter() {
        if idx >= NUM_LIMBS {
            break;
        }
        v |= (byte as u64) << bits;
        bits += 8;
        while bits >= LOG_LIMB_SIZE && idx < NUM_LIMBS {
            result[idx] = v as u32;
            v >>= LOG_LIMB_SIZE;
            bits -= LOG_LIMB_SIZE;
            idx += 1;
        }
    }
    if bits > 0 && idx < NUM_LIMBS {
        result[idx] = v as u32;
    }
    result
}

fn bigint4_from_limbs(limbs: &[u32]) -> BigInt<4> {
    let mut result = [0u64; 4];
    let mut accumulated_bits = 0usize;
    let mut current_u64 = 0u64;
    let mut result_idx = 0;
    for &limb in limbs {
        current_u64 |= (limb as u64) << accumulated_bits;
        accumulated_bits += 32;
        while accumulated_bits >= 64 && result_idx < 4 {
            result[result_idx] = current_u64;
            current_u64 = (limb as u64) >> (32 - (accumulated_bits - 64));
            accumulated_bits -= 64;
            result_idx += 1;
        }
    }
    if accumulated_bits > 0 && result_idx < 4 {
        result[result_idx] = current_u64;
    }
    BigInt::<4>::new(result)
}

fn batch_projective_to_affine_g1(points: &[ArkG1]) -> Vec<G1Affine> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }

    if n == 1 {
        return vec![points[0].0.into_affine()];
    }

    let identity_affine = ark_bn254::G1Projective::zero().into_affine();
    let mut result: Vec<G1Affine> = vec![identity_affine; n];

    let mut needs_inv: Vec<usize> = Vec::with_capacity(n);
    for (i, p) in points.iter().enumerate() {
        if !p.0.z.is_zero() {
            needs_inv.push(i);
        }
    }

    if needs_inv.is_empty() {
        return result;
    }

    let m = needs_inv.len();

    let mut prefix: Vec<Fq> = Vec::with_capacity(m);
    let mut acc = points[needs_inv[0]].0.z;
    prefix.push(acc);
    for j in 1..m {
        acc *= points[needs_inv[j]].0.z;
        prefix.push(acc);
    }

    let mut inv = acc
        .inverse()
        .expect("Product of non-zero Z values must be invertible");

    for j in (0..m).rev() {
        let idx = needs_inv[j];
        let z_inv = if j == 0 { inv } else { inv * prefix[j - 1] };

        let p = &points[idx].0;
        let z_inv2 = z_inv * z_inv;
        result[idx] = G1Affine::new_unchecked(p.x * z_inv2, p.y * z_inv2 * z_inv);

        inv *= p.z;
    }

    result
}

pub fn serialize_g1_points(points: &[ArkG1]) -> Vec<u32> {
    let affine_points = batch_projective_to_affine_g1(points);
    let mut out = Vec::with_capacity(points.len() * 2 * NUM_LIMBS);
    for affine in &affine_points {
        out.extend_from_slice(&bigint4_to_limbs(&affine.x.0));
        out.extend_from_slice(&bigint4_to_limbs(&affine.y.0));
    }
    out
}

pub fn serialize_g2_points(points: &[ArkG2]) -> Vec<u32> {
    let mut out = Vec::with_capacity(points.len() * 4 * NUM_LIMBS);
    for point in points {
        let affine: G2Affine = point.0.into_affine();
        out.extend_from_slice(&bigint4_to_limbs(&affine.x.c0.0));
        out.extend_from_slice(&bigint4_to_limbs(&affine.x.c1.0));
        out.extend_from_slice(&bigint4_to_limbs(&affine.y.c0.0));
        out.extend_from_slice(&bigint4_to_limbs(&affine.y.c1.0));
    }
    out
}

pub fn deserialize_fq12(words: &[u32]) -> Fq12 {
    let w: Vec<BigInt<4>> = words.chunks(NUM_LIMBS).map(bigint4_from_limbs).collect();
    Fq12::new(
        Fq6::new(
            Fq2::new(Fq::new_unchecked(w[0]), Fq::new_unchecked(w[1])),
            Fq2::new(Fq::new_unchecked(w[2]), Fq::new_unchecked(w[3])),
            Fq2::new(Fq::new_unchecked(w[4]), Fq::new_unchecked(w[5])),
        ),
        Fq6::new(
            Fq2::new(Fq::new_unchecked(w[6]), Fq::new_unchecked(w[7])),
            Fq2::new(Fq::new_unchecked(w[8]), Fq::new_unchecked(w[9])),
            Fq2::new(Fq::new_unchecked(w[10]), Fq::new_unchecked(w[11])),
        ),
    )
}

#[cfg(target_arch = "wasm32")]
mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = globalThis)]
        pub fn gpuCombineHintsFire(
            points_flat: &[u32],
            scalars_flat: &[u32],
            num_rows: u32,
            num_polys: u32,
        ) -> JsValue;

        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_batch_pairing")]
        pub async fn js_gpu_batch_pairing(
            g1_flat: &[u32],
            g2_flat: &[u32],
            group_sizes: &[u32],
            group_offsets: &[u32],
        ) -> JsValue;

        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_pairing_available")]
        pub fn js_gpu_pairing_available() -> bool;

        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_batch_pairing")]
        pub fn js_gpu_batch_pairing_fire(
            g1_flat: &[u32],
            g2_flat: &[u32],
            group_sizes: &[u32],
            group_offsets: &[u32],
        ) -> JsValue;

        #[wasm_bindgen(js_namespace = globalThis)]
        pub fn gpuBatchMultiPairingFromBufferFire(
            onehot_buffer: &JsValue,
            onehot_buffer_size: u32,
            poly_layout_flat: &[u32],
            total_affine_points: u32,
            g2_flat: &[u32],
            group_sizes: &[u32],
            group_offsets: &[u32],
        ) -> JsValue;

    }
}

#[cfg(target_arch = "wasm32")]
pub fn is_gpu_combine_hints_available() -> bool {
    std::panic::catch_unwind(js_bridge::js_gpu_pairing_available).unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_gpu_combine_hints_available() -> bool {
    false
}

#[cfg(target_arch = "wasm32")]
pub struct GpuCombineHintsHandle {
    gpu_future: wasm_bindgen_futures::JsFuture,
    num_rows: usize,
}

#[cfg(target_arch = "wasm32")]
fn fq_to_limbs(f: &Fq) -> [u32; NUM_LIMBS] {
    let mut out = [0u32; NUM_LIMBS];
    let words = (f.0).0;
    for i in 0..4 {
        out[i * 2] = words[i] as u32;
        out[i * 2 + 1] = (words[i] >> 32) as u32;
    }
    out
}

#[cfg(target_arch = "wasm32")]
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

#[cfg(target_arch = "wasm32")]
fn limbs8_to_fq(limbs: &[u32]) -> Fq {
    let bigint = BigInt::<4>::new([
        ((limbs[1] as u64) << 32) | (limbs[0] as u64),
        ((limbs[3] as u64) << 32) | (limbs[2] as u64),
        ((limbs[5] as u64) << 32) | (limbs[4] as u64),
        ((limbs[7] as u64) << 32) | (limbs[6] as u64),
    ]);
    Fq::new_unchecked(bigint)
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_combine_hints(
    hints: &[Vec<ArkG1>],
    coeffs: &[ark_bn254::Fr],
) -> GpuCombineHintsHandle {
    use wasm_bindgen::JsCast;

    let num_rows = super::DoryGlobals::get_max_num_rows();
    let num_polys = hints.len();

    let mut points_flat = vec![0u32; num_polys * num_rows * G1_JACOBIAN_WORDS];
    for (i, rows) in hints.iter().enumerate() {
        for j in 0..num_rows {
            let p = rows
                .get(j)
                .copied()
                .unwrap_or_else(|| ArkG1(G1Projective::zero()));
            let base = (i * num_rows + j) * G1_JACOBIAN_WORDS;
            let x = fq_to_limbs(&p.0.x);
            let y = fq_to_limbs(&p.0.y);
            let z = fq_to_limbs(&p.0.z);
            points_flat[base..base + NUM_LIMBS].copy_from_slice(&x);
            points_flat[base + NUM_LIMBS..base + 2 * NUM_LIMBS].copy_from_slice(&y);
            points_flat[base + 2 * NUM_LIMBS..base + 3 * NUM_LIMBS].copy_from_slice(&z);
        }
    }

    let mut scalars_flat = Vec::with_capacity(coeffs.len() * NUM_LIMBS);
    for coeff in coeffs {
        scalars_flat.extend_from_slice(&fr_to_raw_limbs(coeff));
    }

    let promise_js = js_bridge::gpuCombineHintsFire(
        &points_flat,
        &scalars_flat,
        num_rows as u32,
        num_polys as u32,
    );
    let promise: js_sys::Promise = promise_js.unchecked_into();
    let gpu_future = wasm_bindgen_futures::JsFuture::from(promise);

    GpuCombineHintsHandle {
        gpu_future,
        num_rows,
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_combine_hints(handle: GpuCombineHintsHandle) -> Vec<ArkG1> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let result_js = handle
        .gpu_future
        .await
        .expect("GPU combine_hints Promise rejected");
    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU combine_hints");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    let mut out = Vec::with_capacity(handle.num_rows);
    for j in 0..handle.num_rows {
        let base = j * G1_JACOBIAN_WORDS;
        let x = limbs8_to_fq(&result_words[base..base + NUM_LIMBS]);
        let y = limbs8_to_fq(&result_words[base + NUM_LIMBS..base + 2 * NUM_LIMBS]);
        let z = limbs8_to_fq(&result_words[base + 2 * NUM_LIMBS..base + 3 * NUM_LIMBS]);
        out.push(ArkG1(G1Projective::new_unchecked(x, y, z)));
    }
    out
}

#[cfg(target_arch = "wasm32")]
pub fn is_gpu_pairing_available() -> bool {
    use std::panic;
    panic::catch_unwind(js_bridge::js_gpu_pairing_available).unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_gpu_pairing_available() -> bool {
    false
}

#[cfg(target_arch = "wasm32")]
pub struct GpuBatchPairingHandle {
    gpu_future: wasm_bindgen_futures::JsFuture,
    cpu_millers: Vec<Option<Fq12>>,
    num_groups: usize,
}

#[cfg(target_arch = "wasm32")]
pub struct GpuMultiGroupPairingHandle {
    gpu_future: wasm_bindgen_futures::JsFuture,
    cpu_millers: Vec<Option<Fq12>>,
    num_groups: usize,
}

/// Fire a multi-group GPU pairing dispatch where each group has its OWN G2 vector.
/// Unlike `dispatch_gpu_batch_pairing` (shared G2), this flattens per-group G2 into
/// the same buffer layout — the GPU shader processes (g1[i], g2[i]) pairs independently.
/// Returns a handle for later collection with `resolve_gpu_multi_group_pairing`.
#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_multi_group_pairing(
    groups: &[(&[ArkG1], &[ArkG2])],
) -> GpuMultiGroupPairingHandle {
    use rayon::prelude::*;
    use wasm_bindgen::JsCast;

    const GPU_RATIO: f64 = 0.95;
    const MIN_HYBRID_PAIRS: usize = 20;

    let num_groups = groups.len();

    let mut gpu_counts = Vec::with_capacity(num_groups);
    let mut gpu_group_sizes = Vec::with_capacity(num_groups);
    let mut gpu_group_offsets = Vec::with_capacity(num_groups);
    let mut total_gpu_pairs = 0usize;

    for (g1, g2) in groups {
        assert_eq!(
            g1.len(),
            g2.len(),
            "G1 and G2 lengths must match within group"
        );
        let n = g1.len();
        let gpu_count = if n < MIN_HYBRID_PAIRS {
            n
        } else {
            ((n as f64) * GPU_RATIO)
                .round()
                .max(1.0)
                .min((n - 1) as f64) as usize
        };
        gpu_counts.push(gpu_count);
        gpu_group_sizes.push(gpu_count as u32);
        gpu_group_offsets.push(total_gpu_pairs as u32);
        total_gpu_pairs += gpu_count;
    }

    let per_group_g1: Vec<Vec<u32>> = groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|((g1, _), &gc)| serialize_g1_points(&g1[..gc]))
        .collect();

    let per_group_g2: Vec<Vec<u32>> = groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|((_, g2), &gc)| serialize_g2_points(&g2[..gc]))
        .collect();

    let mut g1_flat = Vec::with_capacity(total_gpu_pairs * 2 * NUM_LIMBS);
    let mut g2_flat = Vec::with_capacity(total_gpu_pairs * 4 * NUM_LIMBS);
    for g in 0..num_groups {
        g1_flat.extend_from_slice(&per_group_g1[g]);
        g2_flat.extend_from_slice(&per_group_g2[g]);
    }

    let promise_js = js_bridge::js_gpu_batch_pairing_fire(
        &g1_flat,
        &g2_flat,
        &gpu_group_sizes,
        &gpu_group_offsets,
    );
    let promise: js_sys::Promise = promise_js.unchecked_into();
    let gpu_future = wasm_bindgen_futures::JsFuture::from(promise);

    let cpu_millers: Vec<Option<Fq12>> = groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|((g1, g2), &gc)| {
            let n = g1.len();
            if gc >= n {
                None
            } else {
                let miller = Bn254::multi_miller_loop(
                    g1[gc..].iter().map(|p| p.0.into_affine()),
                    g2[gc..].iter().map(|p| p.0.into_affine()),
                );
                Some(miller.0)
            }
        })
        .collect();

    GpuMultiGroupPairingHandle {
        gpu_future,
        cpu_millers,
        num_groups,
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_multi_group_pairing(handle: GpuMultiGroupPairingHandle) -> Vec<ArkGT> {
    use js_sys::Uint32Array;
    use rayon::prelude::*;
    use wasm_bindgen::JsCast;

    let result_js = handle
        .gpu_future
        .await
        .expect("GPU multi-group pairing Promise rejected");
    let result_u32_array: Uint32Array =
        result_js.dyn_into().expect("Expected Uint32Array from GPU");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    (0..handle.num_groups)
        .into_par_iter()
        .map(|g| {
            let start = g * FP12_WORDS;
            let end = start + FP12_WORDS;
            let gpu_miller = deserialize_fq12(&result_words[start..end]);

            let combined_miller = match &handle.cpu_millers[g] {
                Some(cpu_fp12) => gpu_miller * cpu_fp12,
                None => gpu_miller,
            };

            let pairing_result =
                Bn254::final_exponentiation(MillerLoopOutput::<Bn254>(combined_miller))
                    .expect("Final exponentiation failed");
            ArkGT(pairing_result.0)
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub struct GpuBatchPairingFromBufferHandle {
    gpu_future: wasm_bindgen_futures::JsFuture,
    cpu_millers: Vec<Option<Fq12>>,
    num_groups: usize,
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_batch_pairing(
    g1_groups: &[&[ArkG1]],
    g2_bases: &[ArkG2],
) -> GpuBatchPairingHandle {
    use rayon::prelude::*;
    use wasm_bindgen::JsCast;

    const GPU_RATIO: f64 = 0.95;
    const MIN_HYBRID_PAIRS: usize = 20;

    let num_groups = g1_groups.len();

    let mut gpu_counts = Vec::with_capacity(num_groups);
    let mut gpu_group_sizes = Vec::with_capacity(num_groups);
    let mut gpu_group_offsets = Vec::new();
    let mut total_gpu_pairs = 0usize;
    let mut max_gpu_count = 0usize;

    for group in g1_groups {
        let n = group.len();
        let gpu_count = if n < MIN_HYBRID_PAIRS {
            n
        } else {
            ((n as f64) * GPU_RATIO)
                .round()
                .max(1.0)
                .min((n - 1) as f64) as usize
        };
        gpu_counts.push(gpu_count);
        gpu_group_sizes.push(gpu_count as u32);
        gpu_group_offsets.push(total_gpu_pairs as u32);
        max_gpu_count = max_gpu_count.max(gpu_count);
        total_gpu_pairs += gpu_count;
    }

    let g2_max_serialized = serialize_g2_points(&g2_bases[..max_gpu_count]);

    let per_group_g1: Vec<Vec<u32>> = g1_groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|(group, &gc)| serialize_g1_points(&group[..gc]))
        .collect();

    let mut g1_flat = Vec::with_capacity(total_gpu_pairs * 2 * NUM_LIMBS);
    let mut g2_flat = Vec::with_capacity(total_gpu_pairs * 4 * NUM_LIMBS);
    for (g, &gc) in gpu_counts.iter().enumerate() {
        g1_flat.extend_from_slice(&per_group_g1[g]);
        let g2_words = gc * 4 * NUM_LIMBS;
        g2_flat.extend_from_slice(&g2_max_serialized[..g2_words]);
    }

    let promise_js = js_bridge::js_gpu_batch_pairing_fire(
        &g1_flat,
        &g2_flat,
        &gpu_group_sizes,
        &gpu_group_offsets,
    );
    let promise: js_sys::Promise = promise_js.unchecked_into();
    let gpu_future = wasm_bindgen_futures::JsFuture::from(promise);

    let cpu_millers: Vec<Option<Fq12>> = g1_groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|(group, &gc)| {
            let n = group.len();
            if gc >= n {
                None
            } else {
                let miller = Bn254::multi_miller_loop(
                    group[gc..].iter().map(|p| p.0.into_affine()),
                    g2_bases[gc..n].iter().map(|p| p.0.into_affine()),
                );
                Some(miller.0)
            }
        })
        .collect();

    GpuBatchPairingHandle {
        gpu_future,
        cpu_millers,
        num_groups,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_batch_pairing_from_buffer(
    gpu_buffer: &wasm_bindgen::JsValue,
    gpu_buffer_size: usize,
    poly_layout: &[(usize, usize)],
    g2_bases: &[ArkG2],
    g1_groups: &[&[ArkG1]],
) -> GpuBatchPairingFromBufferHandle {
    use rayon::prelude::*;
    use wasm_bindgen::JsCast;

    const GPU_RATIO: f64 = 0.95;
    const MIN_HYBRID_PAIRS: usize = 20;

    let num_groups = g1_groups.len();
    assert_eq!(
        poly_layout.len(),
        num_groups,
        "poly_layout must match g1_groups"
    );

    let mut poly_layout_flat = Vec::with_capacity(num_groups * 4);
    let mut jac_offset = 0usize;
    let mut affine_offset = 0usize;
    let mut total_affine_points = 0usize;
    let mut full_group_sizes = Vec::with_capacity(num_groups);
    for &(num_chunks, k) in poly_layout {
        let points = num_chunks * k;
        full_group_sizes.push(points);
        poly_layout_flat.push(num_chunks as u32);
        poly_layout_flat.push(k as u32);
        poly_layout_flat.push(jac_offset as u32);
        poly_layout_flat.push(affine_offset as u32);
        jac_offset += points;
        affine_offset += points;
        total_affine_points += points;
    }

    let mut gpu_counts = Vec::with_capacity(num_groups);
    let mut gpu_group_sizes = Vec::with_capacity(num_groups);
    let mut gpu_group_offsets = Vec::with_capacity(num_groups);
    let mut total_gpu_pairs = 0usize;
    let mut max_gpu_count = 0usize;
    for (group, &expected_points) in g1_groups.iter().zip(full_group_sizes.iter()) {
        assert_eq!(
            group.len(),
            expected_points,
            "g1 group len must match poly layout points"
        );
        let n = group.len();
        let gpu_count = if n < MIN_HYBRID_PAIRS {
            n
        } else {
            ((n as f64) * GPU_RATIO)
                .round()
                .max(1.0)
                .min((n - 1) as f64) as usize
        };
        gpu_counts.push(gpu_count);
        gpu_group_sizes.push(gpu_count as u32);
        gpu_group_offsets.push(total_gpu_pairs as u32);
        max_gpu_count = max_gpu_count.max(gpu_count);
        total_gpu_pairs += gpu_count;
    }

    let g2_max_serialized = serialize_g2_points(&g2_bases[..max_gpu_count]);
    let mut g2_flat = Vec::with_capacity(total_gpu_pairs * 4 * NUM_LIMBS);
    for &gc in gpu_counts.iter() {
        let g2_words = gc * 4 * NUM_LIMBS;
        g2_flat.extend_from_slice(&g2_max_serialized[..g2_words]);
    }

    let promise_js = js_bridge::gpuBatchMultiPairingFromBufferFire(
        gpu_buffer,
        gpu_buffer_size as u32,
        &poly_layout_flat,
        total_affine_points as u32,
        &g2_flat,
        &gpu_group_sizes,
        &gpu_group_offsets,
    );
    let promise: js_sys::Promise = promise_js.unchecked_into();
    let gpu_future = wasm_bindgen_futures::JsFuture::from(promise);

    let cpu_millers: Vec<Option<Fq12>> = g1_groups
        .par_iter()
        .zip(gpu_counts.par_iter())
        .map(|(group, &gc)| {
            let n = group.len();
            if gc >= n {
                None
            } else {
                let miller = Bn254::multi_miller_loop(
                    group[gc..].iter().map(|p| p.0.into_affine()),
                    g2_bases[gc..n].iter().map(|p| p.0.into_affine()),
                );
                Some(miller.0)
            }
        })
        .collect();

    GpuBatchPairingFromBufferHandle {
        gpu_future,
        cpu_millers,
        num_groups,
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_batch_pairing(handle: GpuBatchPairingHandle) -> Vec<ArkGT> {
    use js_sys::Uint32Array;
    use rayon::prelude::*;
    use wasm_bindgen::JsCast;

    let result_js = handle
        .gpu_future
        .await
        .expect("GPU batch pairing Promise rejected");
    let result_u32_array: Uint32Array =
        result_js.dyn_into().expect("Expected Uint32Array from GPU");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    (0..handle.num_groups)
        .into_par_iter()
        .map(|g| {
            let start = g * FP12_WORDS;
            let end = start + FP12_WORDS;
            let gpu_miller = deserialize_fq12(&result_words[start..end]);

            let combined_miller = match &handle.cpu_millers[g] {
                Some(cpu_fp12) => gpu_miller * cpu_fp12,
                None => gpu_miller,
            };

            let pairing_result =
                Bn254::final_exponentiation(MillerLoopOutput::<Bn254>(combined_miller))
                    .expect("Final exponentiation failed");
            ArkGT(pairing_result.0)
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_batch_pairing_from_buffer(
    handle: GpuBatchPairingFromBufferHandle,
) -> Vec<ArkGT> {
    use js_sys::Uint32Array;
    use rayon::prelude::*;
    use wasm_bindgen::JsCast;

    let result_js = handle
        .gpu_future
        .await
        .expect("GPU batch pairing-from-buffer Promise rejected");
    let result_u32_array: Uint32Array =
        result_js.dyn_into().expect("Expected Uint32Array from GPU");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    (0..handle.num_groups)
        .into_par_iter()
        .map(|g| {
            let start = g * FP12_WORDS;
            let end = start + FP12_WORDS;
            let gpu_miller = deserialize_fq12(&result_words[start..end]);

            let combined_miller = match &handle.cpu_millers[g] {
                Some(cpu_fp12) => gpu_miller * cpu_fp12,
                None => gpu_miller,
            };

            let pairing_result =
                Bn254::final_exponentiation(MillerLoopOutput::<Bn254>(combined_miller))
                    .expect("Final exponentiation failed");
            ArkGT(pairing_result.0)
        })
        .collect()
}
