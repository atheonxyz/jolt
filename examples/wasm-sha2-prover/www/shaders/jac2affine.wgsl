// Jacobian → Affine conversion + index transposition kernel
// This file is concatenated after bn254_common.wgsl + msm_g1_curve.wgsl by the host
//
// Reads Jacobian G1 points from OneHot GPU output buffer.
// Converts to Affine via Z^(-1) (Fermat's little theorem using ff_invert).
// Writes Affine points with transposed indices matching the pairing layout.
//
// OneHot layout:  per-poly contiguous, index = c * k + ki
// Pairing layout: per-poly contiguous, index = c + ki * num_chunks (transposed)

@group(0) @binding(0) var<storage, read> jac_points: array<u32>;
@group(0) @binding(1) var<storage, read_write> affine_points: array<u32>;
@group(0) @binding(2) var<storage, read> poly_layout: array<u32>;
@group(0) @binding(3) var<uniform> params: Jac2AffineParams;

struct Jac2AffineParams {
    total_points: u32,
    num_polys: u32,
    _pad0: u32,
    _pad1: u32,
}

@compute @workgroup_size(128)
fn jac2affine_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.total_points) { return; }

    // Find which poly this thread belongs to via linear scan of layout table.
    // poly_layout: [num_chunks_0, k_0, jac_offset_0, affine_offset_0,
    //               num_chunks_1, k_1, jac_offset_1, affine_offset_1, ...]
    var poly_idx = 0u;
    var remaining = tid;
    for (var p = 0u; p < params.num_polys; p = p + 1u) {
        let num_outputs = poly_layout[p * 4u] * poly_layout[p * 4u + 1u];
        if (remaining < num_outputs) {
            poly_idx = p;
            break;
        }
        remaining = remaining - num_outputs;
    }

    let num_chunks = poly_layout[poly_idx * 4u];
    let k = poly_layout[poly_idx * 4u + 1u];
    let jac_offset = poly_layout[poly_idx * 4u + 2u];
    let affine_offset = poly_layout[poly_idx * 4u + 3u];

    // Decode (chunk, ki) from remaining index within this poly
    let c = remaining / k;
    let ki = remaining % k;

    // Read Jacobian point: 24 u32 limbs (X:8, Y:8, Z:8)
    let jac_idx = (jac_offset + c * k + ki) * 24u;
    var X: BigInt;
    var Y: BigInt;
    var Z: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        X.limbs[i] = jac_points[jac_idx + i];
        Y.limbs[i] = jac_points[jac_idx + 8u + i];
        Z.limbs[i] = jac_points[jac_idx + 16u + i];
    }

    // Transposed output index: row_idx = c + ki * num_chunks
    let out_idx = (affine_offset + c + ki * num_chunks) * 16u;

    // Identity point (Z == 0): write (0, 0)
    if (is_bigint_zero(Z)) {
        for (var i = 0u; i < 16u; i = i + 1u) {
            affine_points[out_idx + i] = 0u;
        }
        return;
    }

    // Compute Z^(-1) via Fermat's little theorem: Z^(p-2)
    let z_inv = ff_invert(Z);
    let z_inv2 = mont_mul_cios(z_inv, z_inv);
    let z_inv3 = mont_mul_cios(z_inv2, z_inv);

    // Affine coordinates: x = X * Z^(-2), y = Y * Z^(-3)
    let affine_x = mont_mul_cios(X, z_inv2);
    let affine_y = mont_mul_cios(Y, z_inv3);

    // Write Affine point: 16 u32 limbs (x:8, y:8)
    for (var i = 0u; i < 8u; i = i + 1u) {
        affine_points[out_idx + i] = affine_x.limbs[i];
        affine_points[out_idx + 8u + i] = affine_y.limbs[i];
    }
}
