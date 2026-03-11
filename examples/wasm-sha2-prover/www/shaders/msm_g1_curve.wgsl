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

// XYZZ extended Jacobian coordinates: stores ZZ=Z^2, ZZZ=Z^3 to avoid recomputation.
// Saves 1 mont_mul_cios per mixed-addition vs standard Jacobian (10M vs 11M).
struct G1XYZZ {
    x: BigInt,
    y: BigInt,
    zz: BigInt,
    zzz: BigInt,
}

fn xyzz_zero() -> G1XYZZ {
    var r: G1XYZZ;
    r.x = g1_bigint_from_array(BN254_ZERO_XR);
    r.y = g1_bigint_from_array(BN254_ZERO_YR);
    r.zz = bigint_zero();
    r.zzz = bigint_zero();
    return r;
}

fn is_xyzz_zero(a: G1XYZZ) -> bool {
    return is_bigint_zero(a.zz);
}

fn xyzz_neg(a: G1XYZZ) -> G1XYZZ {
    if (is_xyzz_zero(a)) { return a; }
    let p = modulus();
    var r: G1XYZZ;
    r.x = a.x;
    r.y = ff_sub(p, a.y);
    r.zz = a.zz;
    r.zzz = a.zzz;
    return r;
}

// Mixed addition: XYZZ + Affine → XYZZ (10M vs 11M Jacobian)
fn xyzz_madd(a: G1XYZZ, b: G1Affine) -> G1XYZZ {
    if (is_xyzz_zero(a)) {
        var r: G1XYZZ;
        r.x = b.x;
        r.y = b.y;
        r.zz = g1_one_mont_z();
        r.zzz = g1_one_mont_z();
        return r;
    }

    let U = mont_mul_cios(b.x, a.zz);    // X2*ZZ1
    let S = mont_mul_cios(b.y, a.zzz);   // Y2*ZZZ1
    let H = ff_sub(U, a.x);
    let R = ff_sub(S, a.y);

    if (is_bigint_zero(H)) {
        if (is_bigint_zero(R)) {
            return xyzz_dbl(a);
        }
        return xyzz_zero();
    }

    let HH = mont_mul_cios(H, H);        // H^2
    let HHH = mont_mul_cios(H, HH);      // H^3
    let Q = mont_mul_cios(a.x, HH);      // X1*H^2
    let RR = mont_mul_cios(R, R);         // R^2
    let Q2 = ff_add(Q, Q);
    let x3 = ff_sub(ff_sub(RR, HHH), Q2);
    let QmX3 = ff_sub(Q, x3);
    let RQmX3 = mont_mul_cios(R, QmX3);  // R*(Q-X3)
    let Y1HHH = mont_mul_cios(a.y, HHH); // Y1*H^3
    let y3 = ff_sub(RQmX3, Y1HHH);
    let zz3 = mont_mul_cios(a.zz, HH);   // ZZ1*H^2
    let zzz3 = mont_mul_cios(a.zzz, HHH); // ZZZ1*H^3

    var r: G1XYZZ;
    r.x = x3;
    r.y = y3;
    r.zz = zz3;
    r.zzz = zzz3;
    return r;
}

// Affine + Affine → XYZZ (6M). Used for the first two points in a bucket,
// where ZZ=ZZZ=1. Saves 4 mont_mul vs xyzz_madd.
fn xyzz_add_affine_affine(a: G1Affine, b: G1Affine) -> G1XYZZ {
    let H = ff_sub(b.x, a.x);
    let R = ff_sub(b.y, a.y);

    if (is_bigint_zero(H)) {
        if (is_bigint_zero(R)) {
            // Same point → double
            var xyzz_a: G1XYZZ;
            xyzz_a.x = a.x;
            xyzz_a.y = a.y;
            xyzz_a.zz = g1_one_mont_z();
            xyzz_a.zzz = g1_one_mont_z();
            return xyzz_dbl(xyzz_a);
        }
        return xyzz_zero();
    }

    let HH = mont_mul_cios(H, H);        // 1M
    let HHH = mont_mul_cios(H, HH);      // 2M
    let Q = mont_mul_cios(a.x, HH);      // 3M
    let RR = mont_mul_cios(R, R);         // 4M
    let Q2 = ff_add(Q, Q);
    let x3 = ff_sub(ff_sub(RR, HHH), Q2);
    let QmX3 = ff_sub(Q, x3);
    let RQmX3 = mont_mul_cios(R, QmX3);  // 5M
    let Y1HHH = mont_mul_cios(a.y, HHH); // 6M
    let y3 = ff_sub(RQmX3, Y1HHH);

    var r: G1XYZZ;
    r.x = x3;
    r.y = y3;
    r.zz = HH;   // ZZ = 1*HH = HH
    r.zzz = HHH; // ZZZ = 1*HHH = HHH
    return r;
}

// Full addition: XYZZ + XYZZ → XYZZ (14M vs 16M Jacobian)
fn xyzz_add(a: G1XYZZ, b: G1XYZZ) -> G1XYZZ {
    if (is_xyzz_zero(a)) { return b; }
    if (is_xyzz_zero(b)) { return a; }

    let U1 = mont_mul_cios(a.x, b.zz);    // X1*ZZ2
    let U2 = mont_mul_cios(b.x, a.zz);    // X2*ZZ1
    let S1 = mont_mul_cios(a.y, b.zzz);   // Y1*ZZZ2
    let S2 = mont_mul_cios(b.y, a.zzz);   // Y2*ZZZ1
    let H = ff_sub(U2, U1);
    let R = ff_sub(S2, S1);

    if (is_bigint_zero(H)) {
        if (is_bigint_zero(R)) {
            return xyzz_dbl(a);
        }
        return xyzz_zero();
    }

    let HH = mont_mul_cios(H, H);
    let HHH = mont_mul_cios(H, HH);
    let Q = mont_mul_cios(U1, HH);
    let RR = mont_mul_cios(R, R);
    let Q2 = ff_add(Q, Q);
    let x3 = ff_sub(ff_sub(RR, HHH), Q2);
    let QmX3 = ff_sub(Q, x3);
    let RQmX3 = mont_mul_cios(R, QmX3);
    let S1HHH = mont_mul_cios(S1, HHH);
    let y3 = ff_sub(RQmX3, S1HHH);
    let zz3 = mont_mul_cios(a.zz, mont_mul_cios(b.zz, HH));
    let zzz3 = mont_mul_cios(a.zzz, mont_mul_cios(b.zzz, HHH));

    var r: G1XYZZ;
    r.x = x3;
    r.y = y3;
    r.zz = zz3;
    r.zzz = zzz3;
    return r;
}

// Doubling: XYZZ → XYZZ (a=0 for BN254)
fn xyzz_dbl(a: G1XYZZ) -> G1XYZZ {
    if (is_xyzz_zero(a)) { return a; }
    let x = a.x;
    let y = a.y;
    let A = mont_mul_cios(x, x);     // X^2
    let B = mont_mul_cios(y, y);     // Y^2
    let C = mont_mul_cios(B, B);     // Y^4
    let xb = ff_add(x, B);
    let xb2 = mont_mul_cios(xb, xb); // (X+B)^2
    let ac = ff_add(A, C);
    let D = ff_add(ff_sub(xb2, ac), ff_sub(xb2, ac)); // 2*((X+B)^2 - A - C)
    let A2 = ff_add(A, A);
    let E = ff_add(A2, A);           // 3A
    let F = mont_mul_cios(E, E);     // (3A)^2
    let D2 = ff_add(D, D);
    let x3 = ff_sub(F, D2);          // F - 2D
    let C2 = ff_add(C, C);
    let C4 = ff_add(C2, C2);
    let C8 = ff_add(C4, C4);
    let DmX3 = ff_sub(D, x3);
    let y3 = ff_sub(mont_mul_cios(E, DmX3), C8); // E*(D-X3) - 8C
    // ZZ3 = 4*B*ZZ1, ZZZ3 = 8*B*Y*ZZZ1
    // But we need B, ZZ1 — compute ZZ3 = (2Y)^2 * ZZ1
    let Y2 = ff_add(y, y);
    let Y2sq = mont_mul_cios(Y2, Y2);  // 4Y^2 = 4B
    let zz3 = mont_mul_cios(Y2sq, a.zz);
    let zzz3 = mont_mul_cios(mont_mul_cios(Y2sq, Y2), a.zzz); // 8Y^3 * ZZZ1

    var r: G1XYZZ;
    r.x = x3;
    r.y = y3;
    r.zz = zz3;
    r.zzz = zzz3;
    return r;
}

fn xyzz_scalar_mul(pt: G1XYZZ, scalar: u32) -> G1XYZZ {
    var result = xyzz_zero();
    var s = scalar;
    var temp = pt;
    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = xyzz_add(result, temp);
        }
        temp = xyzz_dbl(temp);
        s = s >> 1u;
    }
    return result;
}

fn xyzz_to_jacobian(a: G1XYZZ) -> G1Jacobian {
    if (is_xyzz_zero(a)) { return g1_zero_mont(); }
    // From XYZZ (X, Y, ZZ, ZZZ) to Jacobian (X, Y, Z):
    // Z = ZZZ / ZZ (since ZZ=Z^2 and ZZZ=Z^3, Z = ZZZ/ZZ = Z)
    let Z = mont_mul_cios(a.zzz, ff_invert(a.zz));
    var r: G1Jacobian;
    r.x = a.x;
    r.y = a.y;
    r.z = Z;
    return r;
}
