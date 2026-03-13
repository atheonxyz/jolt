@group(0) @binding(0) var<storage, read> ch_points: array<u32>;
@group(0) @binding(1) var<storage, read> ch_scalars: array<u32>;
@group(0) @binding(2) var<storage, read_write> ch_results: array<u32>;
@group(0) @binding(3) var<uniform> ch_params: CombineHintsParams;

struct CombineHintsParams {
    num_rows: u32,
    num_polys: u32,
    _pad0: u32,
    _pad1: u32,
}

fn read_g1j_from_points(base: u32) -> G1Jacobian {
    var out: G1Jacobian;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out.x.limbs[i] = ch_points[base + i];
        out.y.limbs[i] = ch_points[base + 8u + i];
        out.z.limbs[i] = ch_points[base + 16u + i];
    }
    return out;
}

fn write_g1j_to_results(base: u32, pt: G1Jacobian) {
    for (var i = 0u; i < 8u; i = i + 1u) {
        ch_results[base + i] = pt.x.limbs[i];
        ch_results[base + 8u + i] = pt.y.limbs[i];
        ch_results[base + 16u + i] = pt.z.limbs[i];
    }
}

fn read_scalar(scalar_idx: u32) -> BigInt {
    var out: BigInt;
    let base = scalar_idx * 8u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out.limbs[i] = ch_scalars[base + i];
    }
    return out;
}

fn get_scalar_bit(s: BigInt, bit: u32) -> bool {
    let limb = s.limbs[bit / 32u];
    return ((limb >> (bit % 32u)) & 1u) == 1u;
}

@compute @workgroup_size(128)
fn combine_hints_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= ch_params.num_rows) {
        return;
    }

    var acc = g1_zero_mont();

    var bit: i32 = 253;
    while (bit >= 0) {
        acc = g1_dbl(acc);

        for (var i = 0u; i < ch_params.num_polys; i = i + 1u) {
            let scalar = read_scalar(i);
            if (get_scalar_bit(scalar, u32(bit))) {
                let point_idx = i * ch_params.num_rows + row;
                let point_base = point_idx * 24u;
                let pt = read_g1j_from_points(point_base);
                acc = g1_add(acc, pt);
            }
        }

        bit = bit - 1;
    }

    write_g1j_to_results(row * 24u, acc);
}
