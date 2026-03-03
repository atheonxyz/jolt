//! WebGPU-accelerated OneHot batch G1 addition bridge for WASM builds.
//!
//! Gather approach: Rust builds per-(chunk,ki) gather lists with rayon,
//! sends pre-built lists to JS for direct GPU dispatch. Each GPU thread
//! only loads the base points it needs. ~7x faster than direct-scan.
//!
//! Architecture (optimized for 30+ OneHot polys):
//!   1. Rust: serialize bases ONCE (shared across all polys)
//!   2. Rust: build gather lists per-poly with rayon (parallel)
//!   3. JS: dispatch each poly with pre-built lists (no JS preprocessing)
//!   4. JS: cache bases GPU buffer (upload once, reuse)
//!   5. Rust: resolve all GPU results, then par_iter tier-2 pairings
//!
//! This is the WebGPU counterpart of the Metal `metal_onehot_hybrid` gather kernel.

use ark_bn254::{Fq, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::biginteger::BigInt;
use ark_ff::{Fp, Zero};

const NUM_LIMBS: usize = 8;
const G1_AFFINE_WORDS: usize = 2 * NUM_LIMBS; // 16 u32s per G1Affine
const G1_JACOBIAN_WORDS: usize = 3 * NUM_LIMBS; // 24 u32s per G1Jacobian

fn g1_affine_to_limbs(point: &G1Affine) -> [u32; G1_AFFINE_WORDS] {
    let mut out = [0u32; G1_AFFINE_WORDS];
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

fn jacobian_from_limbs(limbs: &[u32]) -> G1Projective {
    let x = limbs8_to_fq(&limbs[0..8]);
    let y = limbs8_to_fq(&limbs[8..16]);
    let z = limbs8_to_fq(&limbs[16..24]);
    if z.is_zero() {
        return G1Projective::default();
    }
    G1Projective::new_unchecked(x, y, z)
}

fn pack_indices_chunk(indices: &[Option<u8>], row_len: usize) -> Vec<u32> {
    let words_per_chunk = (row_len + 3) / 4;
    let mut packed = Vec::with_capacity(words_per_chunk);
    for word_idx in 0..words_per_chunk {
        let mut word = 0u32;
        for byte_pos in 0..4 {
            let col = word_idx * 4 + byte_pos;
            let byte_val = if col < indices.len() {
                indices[col].map_or(0xFF_u8, |v| v)
            } else {
                0xFF_u8
            };
            word |= (byte_val as u32) << (byte_pos * 8);
        }
        packed.push(word);
    }
    packed
}

fn pack_all_indices(all_indices: &[Option<u8>], num_chunks: usize, row_len: usize) -> Vec<u32> {
    let words_per_chunk = (row_len + 3) / 4;
    let mut packed = Vec::with_capacity(num_chunks * words_per_chunk);
    for c in 0..num_chunks {
        let start = c * row_len;
        let end = (start + row_len).min(all_indices.len());
        let chunk_slice = &all_indices[start..end];
        packed.extend_from_slice(&pack_indices_chunk(chunk_slice, row_len));
    }
    packed
}

#[cfg(target_arch = "wasm32")]
mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_onehot_batch_g1_add")]
        pub async fn js_gpu_onehot_batch_g1_add(
            bases_flat: &[u32],
            packed_indices: &[u32],
            num_chunks: u32,
            k: u32,
            row_len: u32,
        ) -> JsValue;

        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_onehot_batch_g1_add")]
        pub fn js_gpu_onehot_batch_g1_add_fire(
            bases_flat: &[u32],
            packed_indices: &[u32],
            num_chunks: u32,
            k: u32,
            row_len: u32,
        ) -> JsValue;

        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_onehot_gather_direct")]
        pub fn js_gpu_onehot_gather_direct_fire(
            bases_flat: &[u32],
            gather_cols: &[u32],
            jobs: &[u32],
            num_jobs: u32,
        ) -> JsValue;

        #[wasm_bindgen(js_namespace = globalThis)]
        pub fn gpuOnehotGatherDirectRetainBufferFire(
            bases_flat: &[u32],
            gather_cols: &[u32],
            jobs: &[u32],
            num_jobs: u32,
        ) -> JsValue;

        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_onehot_available")]
        pub fn js_gpu_onehot_available() -> bool;
    }
}

#[cfg(target_arch = "wasm32")]
pub fn is_gpu_onehot_available() -> bool {
    use std::panic;
    panic::catch_unwind(js_bridge::js_gpu_onehot_available).unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_gpu_onehot_available() -> bool {
    false
}

#[cfg(target_arch = "wasm32")]
pub struct GpuOnehotHandle {
    promise: wasm_bindgen::JsValue,
    num_chunks: usize,
    k: usize,
}

pub fn serialize_onehot_bases(bases: &[G1Affine], row_len: usize) -> Vec<u32> {
    serialize_bases(bases, row_len)
}

fn serialize_bases(bases: &[G1Affine], row_len: usize) -> Vec<u32> {
    let mut bases_flat = Vec::with_capacity(row_len * G1_AFFINE_WORDS);
    for i in 0..row_len {
        if bases[i].is_zero() {
            bases_flat.extend_from_slice(&[0u32; G1_AFFINE_WORDS]);
        } else {
            bases_flat.extend_from_slice(&g1_affine_to_limbs(&bases[i]));
        }
    }
    bases_flat
}

fn deserialize_results(
    result_words: &[u32],
    num_chunks: usize,
    k: usize,
) -> Vec<Vec<G1Projective>> {
    let mut output = Vec::with_capacity(num_chunks);
    for c in 0..num_chunks {
        let mut row = Vec::with_capacity(k);
        for ki in 0..k {
            let idx = c * k + ki;
            let start = idx * G1_JACOBIAN_WORDS;
            let end = start + G1_JACOBIAN_WORDS;
            if end <= result_words.len() {
                row.push(jacobian_from_limbs(&result_words[start..end]));
            } else {
                row.push(G1Projective::default());
            }
        }
        output.push(row);
    }
    output
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_onehot(
    bases: &[G1Affine],
    all_indices: &[Option<u8>],
    num_chunks: usize,
    k: usize,
    row_len: usize,
) -> GpuOnehotHandle {
    let bases_flat = serialize_bases(bases, row_len);
    let packed_indices = pack_all_indices(all_indices, num_chunks, row_len);
    let promise = js_bridge::js_gpu_onehot_batch_g1_add_fire(
        &bases_flat,
        &packed_indices,
        num_chunks as u32,
        k as u32,
        row_len as u32,
    );
    GpuOnehotHandle {
        promise,
        num_chunks,
        k,
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_onehot(handle: GpuOnehotHandle) -> Vec<Vec<G1Projective>> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let result_js = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(handle.promise))
        .await
        .expect("GPU OneHot promise rejected");

    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU OneHot");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    deserialize_results(&result_words, handle.num_chunks, handle.k)
}

pub fn build_gather_lists(
    all_indices: &[Option<u8>],
    num_chunks: usize,
    k: usize,
    row_len: usize,
) -> (Vec<u32>, Vec<u32>, u32) {
    let total_slots = num_chunks * k;

    let mut counts = vec![0u32; total_slots];
    for c in 0..num_chunks {
        let base = c * row_len;
        for col in 0..row_len {
            let idx = base + col;
            if idx < all_indices.len() {
                if let Some(v) = all_indices[idx] {
                    let ki = v as usize;
                    if ki < k {
                        counts[c * k + ki] += 1;
                    }
                }
            }
        }
    }

    let mut offsets = vec![0u32; total_slots];
    let mut total_cols = 0u32;
    for i in 0..total_slots {
        offsets[i] = total_cols;
        total_cols += counts[i];
    }

    let mut gather_cols = vec![0u32; total_cols as usize];
    let mut cursors = offsets.clone();
    for c in 0..num_chunks {
        let base = c * row_len;
        for col in 0..row_len {
            let idx = base + col;
            if idx < all_indices.len() {
                if let Some(v) = all_indices[idx] {
                    let ki = v as usize;
                    if ki < k {
                        let slot = c * k + ki;
                        gather_cols[cursors[slot] as usize] = col as u32;
                        cursors[slot] += 1;
                    }
                }
            }
        }
    }

    let mut jobs = vec![0u32; total_slots * 3];
    for i in 0..total_slots {
        jobs[i * 3] = offsets[i];
        jobs[i * 3 + 1] = counts[i];
        jobs[i * 3 + 2] = i as u32;
    }

    (gather_cols, jobs, total_slots as u32)
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_onehot_direct(
    bases_flat: &[u32],
    gather_cols: &[u32],
    jobs: &[u32],
    num_jobs: u32,
    num_chunks: usize,
    k: usize,
) -> GpuOnehotHandle {
    let promise =
        js_bridge::js_gpu_onehot_gather_direct_fire(bases_flat, gather_cols, jobs, num_jobs);
    GpuOnehotHandle {
        promise,
        num_chunks,
        k,
    }
}

#[cfg(target_arch = "wasm32")]
pub struct GpuOnehotBatchHandle {
    promise: wasm_bindgen::JsValue,
    poly_layout: Vec<(usize, usize, usize)>,
}

#[cfg(target_arch = "wasm32")]
pub struct GpuOnehotBatchResultWithBuffer {
    pub cpu_results: Vec<Vec<Vec<G1Projective>>>,
    pub gpu_buffer: wasm_bindgen::JsValue,
    pub gpu_buffer_size: usize,
    pub poly_layout: Vec<(usize, usize)>,
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_onehot_batch(
    bases_flat: &[u32],
    poly_data: &[(Vec<u32>, Vec<u32>, u32, usize, usize)],
) -> GpuOnehotBatchHandle {
    let total_gather: usize = poly_data.iter().map(|(gc, _, _, _, _)| gc.len()).sum();
    let total_jobs: usize = poly_data.iter().map(|(_, _, nj, _, _)| *nj as usize).sum();

    let mut all_gather_cols = Vec::with_capacity(total_gather);
    let mut all_jobs = Vec::with_capacity(total_jobs * 3);
    let mut poly_layout = Vec::with_capacity(poly_data.len());
    let mut gather_offset = 0u32;
    let mut output_offset = 0usize;

    for (gather_cols, jobs, num_jobs, num_chunks, k) in poly_data {
        let nj = *num_jobs as usize;
        poly_layout.push((*num_chunks, *k, output_offset));
        all_gather_cols.extend_from_slice(gather_cols);
        for i in 0..nj {
            all_jobs.push(jobs[i * 3] + gather_offset);
            all_jobs.push(jobs[i * 3 + 1]);
            all_jobs.push(jobs[i * 3 + 2] + output_offset as u32);
        }
        gather_offset += gather_cols.len() as u32;
        output_offset += nj;
    }

    let promise = js_bridge::js_gpu_onehot_gather_direct_fire(
        bases_flat,
        &all_gather_cols,
        &all_jobs,
        total_jobs as u32,
    );

    GpuOnehotBatchHandle {
        promise,
        poly_layout,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_gpu_onehot_batch_retain_buffer(
    bases_flat: &[u32],
    poly_data: &[(Vec<u32>, Vec<u32>, u32, usize, usize)],
) -> GpuOnehotBatchHandle {
    let total_gather: usize = poly_data.iter().map(|(gc, _, _, _, _)| gc.len()).sum();
    let total_jobs: usize = poly_data.iter().map(|(_, _, nj, _, _)| *nj as usize).sum();

    let mut all_gather_cols = Vec::with_capacity(total_gather);
    let mut all_jobs = Vec::with_capacity(total_jobs * 3);
    let mut poly_layout = Vec::with_capacity(poly_data.len());
    let mut gather_offset = 0u32;
    let mut output_offset = 0usize;

    for (gather_cols, jobs, num_jobs, num_chunks, k) in poly_data {
        let nj = *num_jobs as usize;
        poly_layout.push((*num_chunks, *k, output_offset));
        all_gather_cols.extend_from_slice(gather_cols);
        for i in 0..nj {
            all_jobs.push(jobs[i * 3] + gather_offset);
            all_jobs.push(jobs[i * 3 + 1]);
            all_jobs.push(jobs[i * 3 + 2] + output_offset as u32);
        }
        gather_offset += gather_cols.len() as u32;
        output_offset += nj;
    }

    let promise = js_bridge::gpuOnehotGatherDirectRetainBufferFire(
        bases_flat,
        &all_gather_cols,
        &all_jobs,
        total_jobs as u32,
    );

    GpuOnehotBatchHandle {
        promise,
        poly_layout,
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_onehot_batch(handle: GpuOnehotBatchHandle) -> Vec<Vec<Vec<G1Projective>>> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    let result_js = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(handle.promise))
        .await
        .expect("GPU OneHot batch promise rejected");

    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU OneHot batch");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    handle
        .poly_layout
        .iter()
        .map(|&(num_chunks, k, output_offset)| {
            let start_word = output_offset * G1_JACOBIAN_WORDS;
            let num_outputs = num_chunks * k;
            let end_word = start_word + num_outputs * G1_JACOBIAN_WORDS;
            deserialize_results(
                &result_words[start_word..end_word.min(result_words.len())],
                num_chunks,
                k,
            )
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub async fn resolve_gpu_onehot_batch_with_buffer(
    handle: GpuOnehotBatchHandle,
) -> GpuOnehotBatchResultWithBuffer {
    use js_sys::{Reflect, Uint32Array};
    use wasm_bindgen::JsCast;

    let result_js = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(handle.promise))
        .await
        .expect("GPU OneHot batch promise rejected");

    let cpu_data_key = wasm_bindgen::JsValue::from_str("cpuData");
    let gpu_buffer_key = wasm_bindgen::JsValue::from_str("gpuBuffer");
    let gpu_buffer_size_key = wasm_bindgen::JsValue::from_str("gpuBufferSize");

    let cpu_data_js = Reflect::get(&result_js, &cpu_data_key)
        .expect("Missing cpuData field in GPU OneHot batch result");
    let gpu_buffer = Reflect::get(&result_js, &gpu_buffer_key)
        .expect("Missing gpuBuffer field in GPU OneHot batch result");
    let gpu_buffer_size_js = Reflect::get(&result_js, &gpu_buffer_size_key)
        .expect("Missing gpuBufferSize field in GPU OneHot batch result");

    let result_u32_array: Uint32Array = cpu_data_js
        .dyn_into()
        .expect("Expected Uint32Array cpuData from GPU OneHot batch");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    let cpu_results: Vec<Vec<Vec<G1Projective>>> = handle
        .poly_layout
        .iter()
        .map(|&(num_chunks, k, output_offset)| {
            let start_word = output_offset * G1_JACOBIAN_WORDS;
            let num_outputs = num_chunks * k;
            let end_word = start_word + num_outputs * G1_JACOBIAN_WORDS;
            deserialize_results(
                &result_words[start_word..end_word.min(result_words.len())],
                num_chunks,
                k,
            )
        })
        .collect();

    let gpu_buffer_size = gpu_buffer_size_js
        .as_f64()
        .expect("gpuBufferSize must be a number") as usize;

    let poly_layout = handle
        .poly_layout
        .iter()
        .map(|&(num_chunks, k, _)| (num_chunks, k))
        .collect();

    GpuOnehotBatchResultWithBuffer {
        cpu_results,
        gpu_buffer,
        gpu_buffer_size,
        poly_layout,
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn gpu_onehot_batch_g1_add(
    bases: &[G1Affine],
    all_indices: &[Option<u8>],
    num_chunks: usize,
    k: usize,
    row_len: usize,
) -> Vec<Vec<G1Projective>> {
    use js_sys::Uint32Array;
    use wasm_bindgen::JsCast;

    if num_chunks == 0 || k == 0 {
        return Vec::new();
    }

    let bases_flat = serialize_bases(bases, row_len);
    let packed_indices = pack_all_indices(all_indices, num_chunks, row_len);

    let result_js = js_bridge::js_gpu_onehot_batch_g1_add(
        &bases_flat,
        &packed_indices,
        num_chunks as u32,
        k as u32,
        row_len as u32,
    )
    .await;

    let result_u32_array: Uint32Array = result_js
        .dyn_into()
        .expect("Expected Uint32Array from GPU OneHot");
    let result_words: Vec<u32> = result_u32_array.to_vec();

    deserialize_results(&result_words, num_chunks, k)
}
