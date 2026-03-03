// Miller loop kernel entry point
// This file is concatenated after bn254_common.wgsl by the host

@group(0) @binding(0) var<storage, read> g1_points: array<u32>;
@group(0) @binding(1) var<storage, read> g2_points: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

struct MillerParams {
    num_pairs: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: MillerParams;

@compute @workgroup_size(128)
fn miller_loop_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_pairs) { return; }

    let g1_offset = tid * 2u * 8u;
    var p_x: BigInt;
    var p_y: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        p_x.limbs[i] = g1_points[g1_offset + i];
        p_y.limbs[i] = g1_points[g1_offset + 8u + i];
    }

    let g2_offset = tid * 4u * 8u;
    var Q: G2Affine;
    for (var i = 0u; i < 8u; i = i + 1u) {
        Q.x.c0.limbs[i] = g2_points[g2_offset + i];
        Q.x.c1.limbs[i] = g2_points[g2_offset + 8u + i];
        Q.y.c0.limbs[i] = g2_points[g2_offset + 16u + i];
        Q.y.c1.limbs[i] = g2_points[g2_offset + 24u + i];
    }

    var f: Fp12;
    if (is_bigint_zero(p_x) && is_bigint_zero(p_y)) {
        f = fp12_one();
    } else {
        f = miller_loop_single(p_x, p_y, Q);
    }

    let out_offset = tid * 12u * 8u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        results[out_offset + 0u * 8u + i] = f.c0.c0.c0.limbs[i];
        results[out_offset + 1u * 8u + i] = f.c0.c0.c1.limbs[i];
        results[out_offset + 2u * 8u + i] = f.c0.c1.c0.limbs[i];
        results[out_offset + 3u * 8u + i] = f.c0.c1.c1.limbs[i];
        results[out_offset + 4u * 8u + i] = f.c0.c2.c0.limbs[i];
        results[out_offset + 5u * 8u + i] = f.c0.c2.c1.limbs[i];
        results[out_offset + 6u * 8u + i] = f.c1.c0.c0.limbs[i];
        results[out_offset + 7u * 8u + i] = f.c1.c0.c1.limbs[i];
        results[out_offset + 8u * 8u + i] = f.c1.c1.c0.limbs[i];
        results[out_offset + 9u * 8u + i] = f.c1.c1.c1.limbs[i];
        results[out_offset + 10u * 8u + i] = f.c1.c2.c0.limbs[i];
        results[out_offset + 11u * 8u + i] = f.c1.c2.c1.limbs[i];
    }
}
