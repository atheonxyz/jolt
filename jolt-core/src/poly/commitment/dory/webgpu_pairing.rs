use ark_bn254::{Fq, Fr, G1Projective};
use ark_ff::Zero;
use ark_ff::{biginteger::BigInt, PrimeField};

use super::wrappers::ArkG1;

const NUM_LIMBS: usize = 8;
const G1_JACOBIAN_WORDS: usize = 24;

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

        #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "__jolt_gpu_pairing_available")]
        pub fn js_gpu_pairing_available() -> bool;
    }
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
pub fn is_gpu_combine_hints_available() -> bool {
    std::panic::catch_unwind(js_bridge::js_gpu_pairing_available).unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_gpu_combine_hints_available() -> bool {
    false
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
