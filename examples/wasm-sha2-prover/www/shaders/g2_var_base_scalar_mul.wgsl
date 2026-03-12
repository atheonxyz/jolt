// G2 variable-base scalar multiplication + add kernel for BN254.
//
// Computes, for each i in [0, count):
//   result[i] = scalar * points[i] + addends[i]
//
// All inputs and outputs are G2 Jacobian in Montgomery form.
// Scalar is a single Fr element in raw bit representation (8 u32 limbs, LE).

const G2_JACOBIAN_WORDS: u32 = 48u; // 6 * NUM_LIMBS
const G2_SCALAR_WORDS: u32 = 8u;

@group(0) @binding(0) var<storage, read> g2_points: array<u32>;
@group(0) @binding(1) var<storage, read> g2_addends: array<u32>;
@group(0) @binding(2) var<storage, read_write> g2_results: array<u32>;
@group(0) @binding(3) var<storage, read> g2_scalar: array<u32>;

struct G2VBParams {
    count: u32,
}
@group(0) @binding(4) var<uniform> g2_vb_params: G2VBParams;

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

// Full Jacobian addition: Jacobian + Jacobian -> Jacobian
// Formula (add-2007-bl): 11M + 5S
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

fn g2_read_jacobian(buf: u32, offset: u32) -> G2Jacobian {
    var pt: G2Jacobian;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        if (buf == 0u) {
            pt.x.c0.limbs[i] = g2_points[offset + 0u * NUM_LIMBS + i];
            pt.x.c1.limbs[i] = g2_points[offset + 1u * NUM_LIMBS + i];
            pt.y.c0.limbs[i] = g2_points[offset + 2u * NUM_LIMBS + i];
            pt.y.c1.limbs[i] = g2_points[offset + 3u * NUM_LIMBS + i];
            pt.z.c0.limbs[i] = g2_points[offset + 4u * NUM_LIMBS + i];
            pt.z.c1.limbs[i] = g2_points[offset + 5u * NUM_LIMBS + i];
        } else {
            pt.x.c0.limbs[i] = g2_addends[offset + 0u * NUM_LIMBS + i];
            pt.x.c1.limbs[i] = g2_addends[offset + 1u * NUM_LIMBS + i];
            pt.y.c0.limbs[i] = g2_addends[offset + 2u * NUM_LIMBS + i];
            pt.y.c1.limbs[i] = g2_addends[offset + 3u * NUM_LIMBS + i];
            pt.z.c0.limbs[i] = g2_addends[offset + 4u * NUM_LIMBS + i];
            pt.z.c1.limbs[i] = g2_addends[offset + 5u * NUM_LIMBS + i];
        }
    }
    return pt;
}

fn g2_write_jacobian(offset: u32, pt: G2Jacobian) {
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        g2_results[offset + 0u * NUM_LIMBS + i] = pt.x.c0.limbs[i];
        g2_results[offset + 1u * NUM_LIMBS + i] = pt.x.c1.limbs[i];
        g2_results[offset + 2u * NUM_LIMBS + i] = pt.y.c0.limbs[i];
        g2_results[offset + 3u * NUM_LIMBS + i] = pt.y.c1.limbs[i];
        g2_results[offset + 4u * NUM_LIMBS + i] = pt.z.c0.limbs[i];
        g2_results[offset + 5u * NUM_LIMBS + i] = pt.z.c1.limbs[i];
    }
}

fn scalar_bit(bit_index: u32) -> u32 {
    let limb_idx = bit_index / 32u;
    let bit_off = bit_index % 32u;
    if (limb_idx >= G2_SCALAR_WORDS) {
        return 0u;
    }
    return (g2_scalar[limb_idx] >> bit_off) & 1u;
}

@compute @workgroup_size(64)
fn g2_var_base_scalar_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= g2_vb_params.count) {
        return;
    }

    let in_offset = tid * G2_JACOBIAN_WORDS;
    let out_offset = tid * G2_JACOBIAN_WORDS;

    let point = g2_read_jacobian(0u, in_offset);
    let addend = g2_read_jacobian(1u, in_offset);

    var acc = g2j_zero();
    for (var bit: i32 = 253; bit >= 0; bit = bit - 1) {
        acc = g2j_dbl(acc);
        if (scalar_bit(u32(bit)) == 1u) {
            acc = g2j_add(acc, point);
        }
    }

    let out = g2j_add(acc, addend);
    g2_write_jacobian(out_offset, out);
}
