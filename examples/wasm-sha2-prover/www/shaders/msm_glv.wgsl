override WG_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> points_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> points_out: array<u32>;
@group(0) @binding(2) var<storage, read> negate_flags: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

const PT_STRIDE: u32 = 16u;
const GLV_BETA_MONT: array<u32, 8> = array<u32, 8>(
    0x607cfd48u, 0xe4bd44e5u, 0xbb966e3du, 0xc28f069fu,
    0xe0acccb0u, 0x5e6dd9e7u, 0xe131a029u, 0x30644e72u
);

fn load_beta_mont() -> BigInt {
    var beta: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        beta.limbs[i] = GLV_BETA_MONT[i];
    }
    return beta;
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn apply_endomorphism(@builtin(global_invocation_id) gid: vec3<u32>) {
    let point_idx = gid.x;
    let num_points = params[0];
    if (point_idx >= num_points) { return; }

    let base = point_idx * PT_STRIDE;
    var x: BigInt;
    var y: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        x.limbs[i] = points_in[base + i];
        y.limbs[i] = points_in[base + 8u + i];
    }

    let phi_x = mont_mul_cios(x, load_beta_mont());

    var phi_y = y;
    if (negate_flags[point_idx] != 0u && !is_bigint_zero(phi_y)) {
        phi_y = ff_sub(modulus(), phi_y);
    }

    for (var i = 0u; i < 8u; i = i + 1u) {
        points_out[base + i] = phi_x.limbs[i];
        points_out[base + 8u + i] = phi_y.limbs[i];
    }
}
