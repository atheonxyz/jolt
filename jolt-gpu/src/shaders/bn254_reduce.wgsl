// Fp12 tree reduction kernel entry point
// This file is concatenated after bn254_common.wgsl by the host

@group(0) @binding(0) var<storage, read_write> data: array<u32>;

struct ReduceParams {
    stride: u32,
    num_threads: u32,
    element_offset: u32,
    _pad0: u32,
}
@group(0) @binding(1) var<uniform> reduce_params: ReduceParams;

fn load_fp12_from_data(base: u32) -> Fp12 {
    var a: Fp12;
    for (var i = 0u; i < 8u; i = i + 1u) {
        a.c0.c0.c0.limbs[i] = data[base + 0u * 8u + i];
        a.c0.c0.c1.limbs[i] = data[base + 1u * 8u + i];
        a.c0.c1.c0.limbs[i] = data[base + 2u * 8u + i];
        a.c0.c1.c1.limbs[i] = data[base + 3u * 8u + i];
        a.c0.c2.c0.limbs[i] = data[base + 4u * 8u + i];
        a.c0.c2.c1.limbs[i] = data[base + 5u * 8u + i];
        a.c1.c0.c0.limbs[i] = data[base + 6u * 8u + i];
        a.c1.c0.c1.limbs[i] = data[base + 7u * 8u + i];
        a.c1.c1.c0.limbs[i] = data[base + 8u * 8u + i];
        a.c1.c1.c1.limbs[i] = data[base + 9u * 8u + i];
        a.c1.c2.c0.limbs[i] = data[base + 10u * 8u + i];
        a.c1.c2.c1.limbs[i] = data[base + 11u * 8u + i];
    }
    return a;
}

fn store_fp12_to_data(base: u32, a: Fp12) {
    for (var i = 0u; i < 8u; i = i + 1u) {
        data[base + 0u * 8u + i] = a.c0.c0.c0.limbs[i];
        data[base + 1u * 8u + i] = a.c0.c0.c1.limbs[i];
        data[base + 2u * 8u + i] = a.c0.c1.c0.limbs[i];
        data[base + 3u * 8u + i] = a.c0.c1.c1.limbs[i];
        data[base + 4u * 8u + i] = a.c0.c2.c0.limbs[i];
        data[base + 5u * 8u + i] = a.c0.c2.c1.limbs[i];
        data[base + 6u * 8u + i] = a.c1.c0.c0.limbs[i];
        data[base + 7u * 8u + i] = a.c1.c0.c1.limbs[i];
        data[base + 8u * 8u + i] = a.c1.c1.c0.limbs[i];
        data[base + 9u * 8u + i] = a.c1.c1.c1.limbs[i];
        data[base + 10u * 8u + i] = a.c1.c2.c0.limbs[i];
        data[base + 11u * 8u + i] = a.c1.c2.c1.limbs[i];
    }
}

@compute @workgroup_size(256)
fn fp12_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= reduce_params.num_threads) { return; }

    let idx = tid + reduce_params.element_offset;
    let FP12_WORDS = 12u * 8u;

    let offset_a = idx * FP12_WORDS;
    let offset_b = (idx + reduce_params.stride) * FP12_WORDS;

    let a = load_fp12_from_data(offset_a);
    let b = load_fp12_from_data(offset_b);
    let result = fp12_mul(a, b);
    store_fp12_to_data(offset_a, result);
}
