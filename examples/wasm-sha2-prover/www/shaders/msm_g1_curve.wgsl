const BN254_ZERO_XR: array<u32, 8> = array<u32, 8>(
    3314486685u, 3546104717u, 4123462461u, 175696680u,
    2021213740u, 1718526831u, 2584207151u, 235567041u
);
const BN254_ZERO_YR: array<u32, 8> = array<u32, 8>(
    3314486685u, 3546104717u, 4123462461u, 175696680u,
    2021213740u, 1718526831u, 2584207151u, 235567041u
);
const BN254_ZERO_ZR: array<u32, 8> = array<u32, 8>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);
const BN254_ONE_YR: array<u32, 8> = array<u32, 8>(
    2334006074u, 2797242139u, 3951957627u, 351393361u,
    4042427480u, 3437053662u, 873447006u, 471134083u
);

struct G1Jacobian {
    x: BigInt,
    y: BigInt,
    z: BigInt,
}

struct G1Affine {
    x: BigInt,
    y: BigInt,
}

fn g1_bigint_from_array(words: array<u32, 8>) -> BigInt {
    var out: BigInt;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        out.limbs[i] = words[i];
    }
    return out;
}

fn g1_zero_mont() -> G1Jacobian {
    var result: G1Jacobian;
    result.x = g1_bigint_from_array(BN254_ZERO_XR);
    result.y = g1_bigint_from_array(BN254_ZERO_YR);
    result.z = g1_bigint_from_array(BN254_ZERO_ZR);
    return result;
}

fn g1_one_mont_z() -> BigInt {
    var result: BigInt;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        result.limbs[i] = BN254_ONE_ZR[i];
    }
    return result;
}

fn is_g1_zero(a: G1Jacobian) -> bool {
    return is_bigint_zero(a.z);
}

fn g1_eq(a: G1Jacobian, b: G1Jacobian) -> bool {
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        if (a.x.limbs[i] != b.x.limbs[i]) { return false; }
        if (a.y.limbs[i] != b.y.limbs[i]) { return false; }
        if (a.z.limbs[i] != b.z.limbs[i]) { return false; }
    }
    return true;
}

fn g1_dbl(pt: G1Jacobian) -> G1Jacobian {
    let x = pt.x;
    let y = pt.y;
    let z = pt.z;
    let a = mont_mul_cios(x, x);
    let b = mont_mul_cios(y, y);
    let c = mont_mul_cios(b, b);
    let x1b = ff_add(x, b);
    let x1b2 = mont_mul_cios(x1b, x1b);
    let ac = ff_add(a, c);
    let x1b2ac = ff_sub(x1b2, ac);
    let d = ff_add(x1b2ac, x1b2ac);
    let a2 = ff_add(a, a);
    let e = ff_add(a2, a);
    let f = mont_mul_cios(e, e);
    let d2 = ff_add(d, d);
    let x3 = ff_sub(f, d2);
    let c2 = ff_add(c, c);
    let c4 = ff_add(c2, c2);
    let c8 = ff_add(c4, c4);
    let dx3 = ff_sub(d, x3);
    let edx3 = mont_mul_cios(e, dx3);
    let y3 = ff_sub(edx3, c8);
    let y1z1 = mont_mul_cios(y, z);
    let z3 = ff_add(y1z1, y1z1);

    var result: G1Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

fn g1_add(a: G1Jacobian, b: G1Jacobian) -> G1Jacobian {
    if (is_g1_zero(a)) { return b; }
    if (is_g1_zero(b)) { return a; }

    let x1 = a.x;
    let y1 = a.y;
    let z1 = a.z;
    let x2 = b.x;
    let y2 = b.y;
    let z2 = b.z;
    let z1z1 = mont_mul_cios(z1, z1);
    let z2z2 = mont_mul_cios(z2, z2);
    let u1 = mont_mul_cios(x1, z2z2);
    let u2 = mont_mul_cios(x2, z1z1);
    let y1z2 = mont_mul_cios(y1, z2);
    let s1 = mont_mul_cios(y1z2, z2z2);
    let y2z1 = mont_mul_cios(y2, z1);
    let s2 = mont_mul_cios(y2z1, z1z1);
    let h = ff_sub(u2, u1);

    // Degenerate case: u1 == u2 (projectively same x-coordinate)
    if (is_bigint_zero(h)) {
        let s_diff = ff_sub(s2, s1);
        if (is_bigint_zero(s_diff)) {
            // Same point projectively → double
            return g1_dbl(a);
        }
        // Inverse points → identity
        return g1_zero_mont();
    }

    let s2s1 = ff_sub(s2, s1);
    let r = ff_add(s2s1, s2s1);
    let h2 = ff_add(h, h);
    let i = mont_mul_cios(h2, h2);
    let j = mont_mul_cios(h, i);
    let v = mont_mul_cios(u1, i);
    let v2 = ff_add(v, v);
    let r2 = mont_mul_cios(r, r);
    let jv2 = ff_add(j, v2);
    let x3 = ff_sub(r2, jv2);
    let vx3 = ff_sub(v, x3);
    let rvx3 = mont_mul_cios(r, vx3);
    let s12 = ff_add(s1, s1);
    let s12j = mont_mul_cios(s12, j);
    let y3 = ff_sub(rvx3, s12j);
    let z1z2 = mont_mul_cios(z1, z2);
    let z1z2h = mont_mul_cios(z1z2, h);
    let z3 = ff_add(z1z2h, z1z2h);

    var result: G1Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

fn g1_madd_fast(a: G1Jacobian, b: G1Affine) -> G1Jacobian {
    // Handle zero accumulator: convert affine point to Jacobian
    if (is_g1_zero(a)) {
        var result: G1Jacobian;
        result.x = b.x;
        result.y = b.y;
        result.z = g1_one_mont_z();
        return result;
    }

    let x1 = a.x;
    let y1 = a.y;
    let z1 = a.z;
    let x2 = b.x;
    let y2 = b.y;
    let z1z1 = mont_mul_cios(z1, z1);
    let u2 = mont_mul_cios(x2, z1z1);
    let temp_s2 = mont_mul_cios(y2, z1);
    let s2 = mont_mul_cios(temp_s2, z1z1);
    let h = ff_sub(u2, x1);
    if (is_bigint_zero(h)) {
        let diff = ff_sub(s2, y1);
        if (is_bigint_zero(diff)) {
            return g1_dbl(a);
        }
        return g1_zero_mont();
    }

    let z1h = mont_mul_cios(z1, h);
    let z3 = ff_add(z1h, z1h);
    let hh = mont_mul_cios(h, h);
    var i_val = ff_add(hh, hh);
    i_val = ff_add(i_val, i_val);
    let j = mont_mul_cios(h, i_val);
    let diff2 = ff_sub(s2, y1);
    let r = ff_add(diff2, diff2);
    let v = mont_mul_cios(x1, i_val);
    let r2 = mont_mul_cios(r, r);
    let v2 = ff_add(v, v);
    let jv2 = ff_add(j, v2);
    let x3 = ff_sub(r2, jv2);
    let v_minus_x3 = ff_sub(v, x3);
    let r_vmx3 = mont_mul_cios(r, v_minus_x3);
    let y1j = mont_mul_cios(y1, j);
    let y1j2 = ff_add(y1j, y1j);
    let y3 = ff_sub(r_vmx3, y1j2);

    var result: G1Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

fn g1_neg(pt: G1Jacobian) -> G1Jacobian {
    if (is_g1_zero(pt)) { return pt; }
    let p = modulus();
    let neg_y = ff_sub(p, pt.y);
    var result: G1Jacobian;
    result.x = pt.x;
    result.y = neg_y;
    result.z = pt.z;
    return result;
}

fn g1_scalar_mul(pt: G1Jacobian, scalar: u32) -> G1Jacobian {
    var result = g1_zero_mont();
    var s = scalar;
    var temp = pt;
    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = g1_add(result, temp);
        }
        temp = g1_dbl(temp);
        s = s >> 1u;
    }
    return result;
}
