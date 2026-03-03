override WG_SIZE: u32 = 64;

@group(0) @binding(0) var<storage, read> points_raw: array<u32>;
@group(0) @binding(1) var<storage, read> scalars_raw: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<BigInt>;

const PT_STRIDE: u32 = 16u;
const SCALAR_LIMBS: u32 = 8u;

fn load_affine_point(point_idx: u32) -> G1Affine {
    let base = point_idx * PT_STRIDE;
    var out: G1Affine;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        out.x.limbs[i] = points_raw[base + i];
        out.y.limbs[i] = points_raw[base + 8u + i];
    }
    return out;
}

fn get_scalar_bit(scalar_idx: u32, bit_idx: u32) -> u32 {
    let limb = bit_idx / 32u;
    let shift = bit_idx % 32u;
    if (limb >= SCALAR_LIMBS) {
        return 0u;
    }
    let base = scalar_idx * SCALAR_LIMBS;
    return (scalars_raw[base + limb] >> shift) & 1u;
}

fn scalar_mul_affine(point: G1Affine, scalar_idx: u32, scalar_bit_width: u32) -> G1Jacobian {
    if (scalar_bit_width == 0u) {
        return g1_zero_mont();
    }

    var base: G1Jacobian;
    base.x = point.x;
    base.y = point.y;
    base.z = g1_one_mont_z();

    var acc = g1_zero_mont();
    for (var bit = i32(scalar_bit_width) - 1; bit >= 0; bit = bit - 1) {
        if (!is_g1_zero(acc)) {
            acc = g1_dbl(acc);
        }
        if (get_scalar_bit(scalar_idx, u32(bit)) == 1u) {
            if (is_g1_zero(acc)) {
                acc = base;
            } else {
                acc = g1_add(acc, base);
            }
        }
    }

    return acc;
}

fn store_result(msm_idx: u32, value: G1Jacobian) {
    let out_base = 3u * msm_idx;
    results[out_base] = value.x;
    results[out_base + 1u] = value.y;
    results[out_base + 2u] = value.z;
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn naive_msm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let msm_idx = gid.x;
    let num_points = params[0];
    let batch_size = params[1];
    let scalar_bit_width = params[2];
    if (msm_idx >= batch_size) {
        return;
    }

    var acc = g1_zero_mont();
    var initialized = false;

    for (var point_idx = 0u; point_idx < num_points; point_idx = point_idx + 1u) {
        let scalar_idx = msm_idx * num_points + point_idx;
        let point = load_affine_point(point_idx);
        let term = scalar_mul_affine(point, scalar_idx, scalar_bit_width);
        if (is_g1_zero(term)) {
            continue;
        }
        if (!initialized) {
            acc = term;
            initialized = true;
        } else {
            acc = g1_add(acc, term);
        }
    }

    store_result(msm_idx, acc);
}
