const G2_AFFINE_WORDS: u32 = 32u;
const G2_JACOBIAN_WORDS: u32 = 48u;
const G2_SCALAR_WORDS: u32 = 8u;

@group(0) @binding(0) var<storage, read> g2_srs_affine: array<u32>;
@group(0) @binding(1) var<storage, read> g2_addends: array<u32>;
@group(0) @binding(2) var<storage, read_write> g2_results: array<u32>;
@group(0) @binding(3) var<storage, read> g2_scalar: array<u32>;

struct G2FoldParams {
    count: u32,
}
@group(0) @binding(4) var<uniform> g2_fold_params: G2FoldParams;

fn g2j_is_zero(a: G2Jacobian) -> bool {
    return fp2_is_zero(a.z);
}

fn g2j_zero() -> G2Jacobian {
    var r: G2Jacobian;
    r.x = fp2_one();
    r.y = fp2_one();
    r.z = fp2_zero();
    return r;
}

fn g2_affine_to_jacobian(a: G2Affine) -> G2Jacobian {
    var r: G2Jacobian;
    r.x = a.x;
    r.y = a.y;
    r.z = fp2_one();
    return r;
}

fn g2j_dbl(pt: G2Jacobian) -> G2Jacobian {
    let x = pt.x;
    let y = pt.y;
    let z = pt.z;

    let a = fp2_sqr(x);
    let b = fp2_sqr(y);
    let c = fp2_sqr(b);

    let x1b = fp2_add(x, b);
    let x1b2 = fp2_sqr(x1b);
    let ac = fp2_add(a, c);
    let x1b2ac = fp2_sub(x1b2, ac);
    let d = fp2_double(x1b2ac);

    let a2 = fp2_double(a);
    let e = fp2_add(a2, a);
    let f = fp2_sqr(e);

    let d2 = fp2_double(d);
    let x3 = fp2_sub(f, d2);

    let c2 = fp2_double(c);
    let c4 = fp2_double(c2);
    let c8 = fp2_double(c4);

    let dx3 = fp2_sub(d, x3);
    let edx3 = fp2_mul(e, dx3);
    let y3 = fp2_sub(edx3, c8);

    let y1z1 = fp2_mul(y, z);
    let z3 = fp2_double(y1z1);

    var result: G2Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

fn g2j_add(a_pt: G2Jacobian, b_pt: G2Jacobian) -> G2Jacobian {
    if (g2j_is_zero(a_pt)) { return b_pt; }
    if (g2j_is_zero(b_pt)) { return a_pt; }

    let z1z1 = fp2_sqr(a_pt.z);
    let z2z2 = fp2_sqr(b_pt.z);
    let u1 = fp2_mul(a_pt.x, z2z2);
    let u2 = fp2_mul(b_pt.x, z1z1);
    let s1 = fp2_mul(fp2_mul(a_pt.y, b_pt.z), z2z2);
    let s2 = fp2_mul(fp2_mul(b_pt.y, a_pt.z), z1z1);

    let h = fp2_sub(u2, u1);
    if (fp2_is_zero(h)) {
        let s_diff = fp2_sub(s2, s1);
        if (fp2_is_zero(s_diff)) {
            return g2j_dbl(a_pt);
        }
        return g2j_zero();
    }

    let i_val = fp2_double(fp2_double(fp2_sqr(h)));
    let j = fp2_mul(h, i_val);
    let r_val = fp2_double(fp2_sub(s2, s1));
    let v = fp2_mul(u1, i_val);

    let x3 = fp2_sub(fp2_sub(fp2_sqr(r_val), j), fp2_double(v));
    let y3 = fp2_sub(
        fp2_mul(r_val, fp2_sub(v, x3)),
        fp2_double(fp2_mul(s1, j)),
    );
    let z1_plus_z2 = fp2_add(a_pt.z, b_pt.z);
    let z3 = fp2_mul(fp2_sub(fp2_sub(fp2_sqr(z1_plus_z2), z1z1), z2z2), h);

    var result: G2Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

fn g2j_add_mixed(a_pt: G2Jacobian, b_pt: G2Affine) -> G2Jacobian {
    if (g2j_is_zero(a_pt)) {
        return g2_affine_to_jacobian(b_pt);
    }

    let z1z1 = fp2_sqr(a_pt.z);
    let u2 = fp2_mul(b_pt.x, z1z1);
    let s2 = fp2_mul(fp2_mul(b_pt.y, a_pt.z), z1z1);
    let h = fp2_sub(u2, a_pt.x);
    let y_diff = fp2_sub(s2, a_pt.y);
    let r_val = fp2_double(y_diff);

    if (fp2_is_zero(h)) {
        if (fp2_is_zero(r_val)) {
            return g2j_dbl(a_pt);
        }
        return g2j_zero();
    }

    let hh = fp2_sqr(h);
    let i_val = fp2_double(fp2_double(hh));
    let j = fp2_mul(h, i_val);
    let v = fp2_mul(a_pt.x, i_val);
    let x3 = fp2_sub(fp2_sub(fp2_sqr(r_val), j), fp2_double(v));
    let y3 = fp2_sub(
        fp2_mul(r_val, fp2_sub(v, x3)),
        fp2_double(fp2_mul(a_pt.y, j)),
    );
    let z1_plus_h = fp2_add(a_pt.z, h);
    let z3 = fp2_sub(fp2_sub(fp2_sqr(z1_plus_h), z1z1), hh);

    var result: G2Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

fn g2_read_srs_affine(idx: u32) -> G2Affine {
    let offset = idx * G2_AFFINE_WORDS;
    var pt: G2Affine;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        pt.x.c0.limbs[i] = g2_srs_affine[offset + 0u * NUM_LIMBS + i];
        pt.x.c1.limbs[i] = g2_srs_affine[offset + 1u * NUM_LIMBS + i];
        pt.y.c0.limbs[i] = g2_srs_affine[offset + 2u * NUM_LIMBS + i];
        pt.y.c1.limbs[i] = g2_srs_affine[offset + 3u * NUM_LIMBS + i];
    }
    return pt;
}

fn g2_read_addend(idx: u32) -> G2Jacobian {
    let offset = idx * G2_JACOBIAN_WORDS;
    var pt: G2Jacobian;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        pt.x.c0.limbs[i] = g2_addends[offset + 0u * NUM_LIMBS + i];
        pt.x.c1.limbs[i] = g2_addends[offset + 1u * NUM_LIMBS + i];
        pt.y.c0.limbs[i] = g2_addends[offset + 2u * NUM_LIMBS + i];
        pt.y.c1.limbs[i] = g2_addends[offset + 3u * NUM_LIMBS + i];
        pt.z.c0.limbs[i] = g2_addends[offset + 4u * NUM_LIMBS + i];
        pt.z.c1.limbs[i] = g2_addends[offset + 5u * NUM_LIMBS + i];
    }
    return pt;
}

fn g2_write_result(idx: u32, pt: G2Jacobian) {
    let offset = idx * G2_JACOBIAN_WORDS;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        g2_results[offset + 0u * NUM_LIMBS + i] = pt.x.c0.limbs[i];
        g2_results[offset + 1u * NUM_LIMBS + i] = pt.x.c1.limbs[i];
        g2_results[offset + 2u * NUM_LIMBS + i] = pt.y.c0.limbs[i];
        g2_results[offset + 3u * NUM_LIMBS + i] = pt.y.c1.limbs[i];
        g2_results[offset + 4u * NUM_LIMBS + i] = pt.z.c0.limbs[i];
        g2_results[offset + 5u * NUM_LIMBS + i] = pt.z.c1.limbs[i];
    }
}

fn scalar_window(window_idx: u32) -> u32 {
    let bit_index = window_idx * 4u;
    let limb_idx = bit_index / 32u;
    let bit_off = bit_index % 32u;
    if (limb_idx >= G2_SCALAR_WORDS) {
        return 0u;
    }
    return (g2_scalar[limb_idx] >> bit_off) & 0xFu;
}

@compute @workgroup_size(64)
fn g2_srs_fold_opt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= g2_fold_params.count) {
        return;
    }

    let base_affine = g2_read_srs_affine(tid);
    let addend = g2_read_addend(tid);

    var table: array<G2Jacobian, 15>;
    table[0] = g2_affine_to_jacobian(base_affine);
    table[1] = g2j_dbl(table[0]);
    for (var j = 2u; j < 15u; j = j + 1u) {
        table[j] = g2j_add_mixed(table[j - 1u], base_affine);
    }

    var acc = g2j_zero();

    let top2 = (g2_scalar[7] >> 28u) & 0x3u;
    if (top2 == 1u) {
        acc = g2j_add_mixed(acc, base_affine);
    } else if (top2 > 1u) {
        acc = g2j_add(acc, table[top2 - 1u]);
    }

    for (var w: i32 = 62; w >= 0; w = w - 1) {
        acc = g2j_dbl(acc);
        acc = g2j_dbl(acc);
        acc = g2j_dbl(acc);
        acc = g2j_dbl(acc);

        let digit = scalar_window(u32(w));
        if (digit == 1u) {
            acc = g2j_add_mixed(acc, base_affine);
        } else if (digit > 1u) {
            acc = g2j_add(acc, table[digit - 1u]);
        }
    }

    let out = g2j_add(acc, addend);
    g2_write_result(tid, out);
}
