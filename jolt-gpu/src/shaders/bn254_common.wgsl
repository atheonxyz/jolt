const NUM_LIMBS: u32 = 8u;
const NUM_LIMBS_WIDE: u32 = 9u;
const NUM_LIMBS_EXTRA_WIDE: u32 = 16u;
const LOG_LIMB_SIZE: u32 = 32u;
const N0: u32 = 3834012553u;
const ATE_LOOP_NAF_LEN: i32 = 65;

const BN254_BASEFIELD_MODULUS: array<u32, 8> = array<u32, 8>(
    3632069959u, 1008765974u, 1752287885u, 2541841041u,
    2172737629u, 3092268470u, 3778125865u, 811880050u
);

const MONT_R_SQUARED: array<u32, 8> = array<u32, 8>(
    1401617033u, 4079811675u, 3561292283u, 3051821329u,
    172064758u, 1202396927u, 3401069855u, 114859889u
);

const BN254_ONE_ZR: array<u32, 8> = array<u32, 8>(
    3314486685u, 3546104717u, 4123462461u, 175696680u,
    2021213740u, 1718526831u, 2584207151u, 235567041u
);

const ATE_LOOP_NAF: array<i32, 65> = array<i32, 65>(
    1, 1, 0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0, 1, 0, -1, 0, 0,
    -1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, -1,
    0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, 1,
    0, 1, 0, 0, 0
);

const FROBENIUS_COEFF_X_C0: array<u32, 8> = array<u32, 8>(
    1164159792u, 3044490000u, 2846516308u, 880775624u,
    606996881u, 2046849319u, 293737708u, 425114840u
);
const FROBENIUS_COEFF_X_C1: array<u32, 8> = array<u32, 8>(
    2695513943u, 1854185246u, 2314768705u, 2853993325u,
    4209035834u, 3068597197u, 1317202883u, 644435899u
);
const FROBENIUS_COEFF_Y_C0: array<u32, 8> = array<u32, 8>(
    691451433u, 3837517068u, 3778263755u, 3140546914u,
    4184101734u, 833212854u, 2768304349u, 624259262u
);
const FROBENIUS_COEFF_Y_C1: array<u32, 8> = array<u32, 8>(
    1610512327u, 2715253988u, 2015810011u, 128974097u,
    3145653355u, 1830206759u, 2245983948u, 747053058u
);
const FROBENIUS_COEFF_X2_C0: array<u32, 8> = array<u32, 8>(
    333974428u, 860932238u, 3680392889u, 2110674300u,
    3054851658u, 1610724536u, 33691616u, 646112791u
);
const FROBENIUS_COEFF_X2_C1: array<u32, 8> = array<u32, 8>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);
const FROBENIUS_COEFF_Y2_C0: array<u32, 8> = array<u32, 8>(
    317583274u, 1757628553u, 1923792719u, 2366144360u,
    151523889u, 1373741639u, 1193918714u, 576313009u
);
const FROBENIUS_COEFF_Y2_C1: array<u32, 8> = array<u32, 8>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);

const TWO_INV: array<u32, 8> = array<u32, 8>(
    1325794674u, 2277435346u, 790391525u, 3506252509u,
    4244459332u, 2405397650u, 1033682860u, 523723546u
);

const G2_COEFF_B_C0: array<u32, 8> = array<u32, 8>(
    2008548008u, 1006188771u, 909333341u, 34282279u,
    1232425568u, 649588208u, 1132767341u, 622118450u
);
const G2_COEFF_B_C1: array<u32, 8> = array<u32, 8>(
    3520921447u, 954723532u, 2479754558u, 1710273405u,
    581697706u, 3611939037u, 1248365901u, 21084622u
);

struct BigInt { limbs: array<u32, 8> }
struct BigIntWide { limbs: array<u32, 9> }
struct BigIntResult { value: BigInt, carry: u32 }
struct BigIntResultWide { value: BigIntWide, carry: u32 }

struct Fp2 { c0: BigInt, c1: BigInt }
struct Fp6 { c0: Fp2, c1: Fp2, c2: Fp2 }
struct Fp12 { c0: Fp6, c1: Fp6 }
struct G2Jacobian { x: Fp2, y: Fp2, z: Fp2 }
struct G2Affine { x: Fp2, y: Fp2 }
struct EllCoeffs { ell_0: Fp2, ell_vw: Fp2, ell_vv: Fp2 }
struct SparseResult01234 { v0: Fp2, v1: Fp2, v2: Fp2, v3: Fp2, v4: Fp2 }
struct G2DoubleResult { point: G2Jacobian, coeffs: EllCoeffs }
struct G2AddResult { point: G2Jacobian, coeffs: EllCoeffs }

fn bigint_zero() -> BigInt {
    var z: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) { z.limbs[i] = 0u; }
    return z;
}

fn bigint_zero_wide() -> BigIntWide {
    var z: BigIntWide;
    for (var i = 0u; i < 9u; i = i + 1u) { z.limbs[i] = 0u; }
    return z;
}

fn modulus() -> BigInt {
    var p: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) { p.limbs[i] = BN254_BASEFIELD_MODULUS[i]; }
    return p;
}

fn get_r_squared() -> BigInt {
    var r: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) { r.limbs[i] = MONT_R_SQUARED[i]; }
    return r;
}

fn add_with_carry(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum_ab = a + b;
    let c1 = select(0u, 1u, sum_ab < a);
    let sum = sum_ab + carry_in;
    let c2 = select(0u, 1u, sum < sum_ab);
    return vec2<u32>(sum, c1 + c2);
}

fn sub_with_borrow(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let t1 = a - b;
    let b1 = select(0u, 1u, a < b);
    let t2 = t1 - borrow_in;
    let b2 = select(0u, 1u, t1 < borrow_in);
    return vec2<u32>(t2, b1 + b2);
}

fn mulhi_u32(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let lo_lo = a_lo * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_lo = a_hi * b_lo;
    let hi_hi = a_hi * b_hi;
    let mid = (lo_lo >> 16u) + (lo_hi & 0xFFFFu) + (hi_lo & 0xFFFFu);
    return hi_hi + (lo_hi >> 16u) + (hi_lo >> 16u) + (mid >> 16u);
}

fn bigint_add_unsafe(lhs: BigInt, rhs: BigInt) -> BigIntResult {
    var res: BigIntResult;
    var carry = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let s = add_with_carry(lhs.limbs[i], rhs.limbs[i], carry);
        res.value.limbs[i] = s.x;
        carry = s.y;
    }
    res.carry = carry;
    return res;
}

fn bigint_add_wide(lhs: BigInt, rhs: BigInt) -> BigIntResultWide {
    var res: BigIntResultWide;
    var carry = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let s = add_with_carry(lhs.limbs[i], rhs.limbs[i], carry);
        res.value.limbs[i] = s.x;
        carry = s.y;
    }
    res.value.limbs[8] = carry;
    res.carry = carry;
    return res;
}

fn bigint_sub(lhs: BigInt, rhs: BigInt) -> BigIntResult {
    var res: BigIntResult;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let d = sub_with_borrow(lhs.limbs[i], rhs.limbs[i], borrow);
        res.value.limbs[i] = d.x;
        borrow = d.y;
    }
    res.carry = borrow;
    return res;
}

fn bigint_gte(lhs: BigInt, rhs: BigInt) -> bool {
    for (var idx = 0u; idx < 8u; idx = idx + 1u) {
        let i = 7u - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) { return false; }
        if (lhs.limbs[i] > rhs.limbs[i]) { return true; }
    }
    return true;
}

fn bigint_eq(lhs: BigInt, rhs: BigInt) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (lhs.limbs[i] != rhs.limbs[i]) { return false; }
    }
    return true;
}

fn is_bigint_zero(x: BigInt) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (x.limbs[i] != 0u) { return false; }
    }
    return true;
}

fn ff_reduce(a: BigInt) -> BigInt {
    let p = modulus();
    let res = bigint_sub(a, p);
    var out: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out.limbs[i] = select(res.value.limbs[i], a.limbs[i], res.carry == 1u);
    }
    return out;
}

fn ff_add(a: BigInt, b: BigInt) -> BigInt {
    return ff_reduce(bigint_add_unsafe(a, b).value);
}

fn ff_sub(a: BigInt, b: BigInt) -> BigInt {
    let p = modulus();
    var borrow = 0u;
    var diff: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let d = sub_with_borrow(a.limbs[i], b.limbs[i], borrow);
        diff.limbs[i] = d.x;
        borrow = d.y;
    }

    var carry = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let addend = p.limbs[i] & (0u - borrow);
        let s = add_with_carry(diff.limbs[i], addend, carry);
        diff.limbs[i] = s.x;
        carry = s.y;
    }
    return diff;
}

fn mont_mul_cios(x: BigInt, y: BigInt) -> BigInt {
    let p = modulus();
    var result: BigInt;
    var t: array<u32, 10>;
    for (var i = 0u; i < 10u; i = i + 1u) { t[i] = 0u; }

    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry = 0u;

        for (var j = 0u; j < 8u; j = j + 1u) {
            var prod_lo = x.limbs[j] * y.limbs[i];
            var prod_hi = mulhi_u32(x.limbs[j], y.limbs[i]);
            let s1 = prod_lo + t[j];
            prod_hi = prod_hi + select(0u, 1u, s1 < prod_lo);
            let s2 = s1 + carry;
            prod_hi = prod_hi + select(0u, 1u, s2 < s1);
            t[j] = s2;
            carry = prod_hi;
        }

        var s = t[8] + carry;
        t[9] = select(0u, 1u, s < t[8]);
        t[8] = s;

        let m = t[0] * N0;
        var c = mulhi_u32(m, p.limbs[0]) + select(0u, 1u, t[0] != 0u);

        for (var j = 1u; j < 8u; j = j + 1u) {
            let mlo = m * p.limbs[j];
            var mhi = mulhi_u32(m, p.limbs[j]);
            let a1 = mlo + c;
            mhi = mhi + select(0u, 1u, a1 < mlo);
            let a2 = t[j] + a1;
            mhi = mhi + select(0u, 1u, a2 < t[j]);
            t[j - 1u] = a2;
            c = mhi;
        }

        s = t[8] + c;
        t[7] = s;
        t[8] = t[9] + select(0u, 1u, s < c);
    }

    var borrow = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = t[i] - p.limbs[i] - borrow;
        borrow = select(0u, 1u, (t[i] < p.limbs[i]) || ((t[i] == p.limbs[i]) && (borrow == 1u)));
    }

    let mask = 0u - borrow;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = (t[i] & mask) | (result.limbs[i] & (~mask));
    }

    return result;
}

fn ff_mul(a: BigInt, b: BigInt) -> BigInt {
    return mont_mul_cios(a, b);
}

fn ff_invert(a: BigInt) -> BigInt {
    // Compute a^(p-2) mod p using 4-bit windowed exponentiation.
    // p-2 = 0x30644e72_e131a029_b85045b6_8181585d_97816a91_6871ca8d_3c208c16_d87cfd45
    // 4-bit window: precompute table[1..15], scan 64 nibbles.
    // 325 mont_mul vs 362 naive (10% fewer ops).
    let exp: array<u32, 8> = array<u32, 8>(
        0x30644e72u, 0xe131a029u, 0xb85045b6u, 0x8181585du,
        0x97816a91u, 0x6871ca8du, 0x3c208c16u, 0xd87cfd45u
    );

    // Precompute table[0..15]: table[i] = a^i (14 mont_mul)
    var table: array<BigInt, 16>;
    table[0] = bigint_zero(); // unused sentinel
    table[1] = a;
    for (var i = 2u; i < 16u; i = i + 1u) {
        table[i] = mont_mul_cios(table[i - 1u], a);
    }

    // First nibble is 0x3 — start result = a^3
    var result = table[3];

    // Process remaining 63 nibbles (252 bits)
    // Nibble layout: exp[0] has nibbles at bits [31:28], [27:24], ..., [3:0]
    // First nibble (bits [31:28] of exp[0]) is 0x3, already handled above.
    // Continue from bits [27:24] of exp[0].
    for (var ni = 1u; ni < 8u; ni = ni + 1u) {
        let nib = (exp[0] >> (28u - ni * 4u)) & 0xFu;
        result = mont_mul_cios(result, result);
        result = mont_mul_cios(result, result);
        result = mont_mul_cios(result, result);
        result = mont_mul_cios(result, result);
        if (nib != 0u) {
            result = mont_mul_cios(result, table[nib]);
        }
    }

    // Process remaining 7 limbs (56 nibbles = 224 bits)
    for (var limb = 1u; limb < 8u; limb = limb + 1u) {
        let e = exp[limb];
        for (var ni = 0u; ni < 8u; ni = ni + 1u) {
            let nib = (e >> (28u - ni * 4u)) & 0xFu;
            result = mont_mul_cios(result, result);
            result = mont_mul_cios(result, result);
            result = mont_mul_cios(result, result);
            result = mont_mul_cios(result, result);
            if (nib != 0u) {
                result = mont_mul_cios(result, table[nib]);
            }
        }
    }

    return result;
}

fn fp2_zero() -> Fp2 {
    var r: Fp2;
    r.c0 = bigint_zero();
    r.c1 = bigint_zero();
    return r;
}

fn fp2_one() -> Fp2 {
    var one_mont: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        one_mont.limbs[i] = BN254_ONE_ZR[i];
    }
    var r: Fp2;
    r.c0 = one_mont;
    r.c1 = bigint_zero();
    return r;
}

fn fp2_is_zero(a: Fp2) -> bool {
    return is_bigint_zero(a.c0) && is_bigint_zero(a.c1);
}

fn fp2_add(a: Fp2, b: Fp2) -> Fp2 {
    var r: Fp2;
    r.c0 = ff_add(a.c0, b.c0);
    r.c1 = ff_add(a.c1, b.c1);
    return r;
}

fn fp2_sub(a: Fp2, b: Fp2) -> Fp2 {
    var r: Fp2;
    r.c0 = ff_sub(a.c0, b.c0);
    r.c1 = ff_sub(a.c1, b.c1);
    return r;
}

fn fp2_neg(a: Fp2) -> Fp2 {
    let p = modulus();
    var r: Fp2;
    if (is_bigint_zero(a.c0)) { r.c0 = a.c0; } else { r.c0 = ff_sub(p, a.c0); }
    if (is_bigint_zero(a.c1)) { r.c1 = a.c1; } else { r.c1 = ff_sub(p, a.c1); }
    return r;
}

fn fp2_conjugate(a: Fp2) -> Fp2 {
    let p = modulus();
    var r: Fp2;
    r.c0 = a.c0;
    if (is_bigint_zero(a.c1)) { r.c1 = a.c1; } else { r.c1 = ff_sub(p, a.c1); }
    return r;
}

fn fp2_mul(a: Fp2, b: Fp2) -> Fp2 {
    let v0 = ff_mul(a.c0, b.c0);
    let v1 = ff_mul(a.c1, b.c1);

    let a01 = ff_add(a.c0, a.c1);
    let b01 = ff_add(b.c0, b.c1);
    var c1 = ff_mul(a01, b01);
    c1 = ff_sub(c1, v0);
    c1 = ff_sub(c1, v1);

    let c0 = ff_sub(v0, v1);

    var r: Fp2;
    r.c0 = c0;
    r.c1 = c1;
    return r;
}

fn fp2_sqr(a: Fp2) -> Fp2 {
    let sum = ff_add(a.c0, a.c1);
    let diff = ff_sub(a.c0, a.c1);
    let c0 = ff_mul(sum, diff);

    let prod = ff_mul(a.c0, a.c1);
    let c1 = ff_add(prod, prod);

    var r: Fp2;
    r.c0 = c0;
    r.c1 = c1;
    return r;
}

fn fp2_mul_by_fp(a: Fp2, s: BigInt) -> Fp2 {
    var r: Fp2;
    r.c0 = ff_mul(a.c0, s);
    r.c1 = ff_mul(a.c1, s);
    return r;
}

fn ff_mul_by_9(x: BigInt) -> BigInt {
    let x2 = ff_add(x, x);
    let x4 = ff_add(x2, x2);
    let x8 = ff_add(x4, x4);
    return ff_add(x8, x);
}

fn fp2_mul_by_nonresidue(a: Fp2) -> Fp2 {
    let nine_a0 = ff_mul_by_9(a.c0);
    let nine_a1 = ff_mul_by_9(a.c1);

    var r: Fp2;
    r.c0 = ff_sub(nine_a0, a.c1);
    r.c1 = ff_add(a.c0, nine_a1);
    return r;
}

fn fp2_inv(a: Fp2) -> Fp2 {
    let a0_sq = ff_mul(a.c0, a.c0);
    let a1_sq = ff_mul(a.c1, a.c1);
    let norm = ff_add(a0_sq, a1_sq);

    let norm_inv = ff_invert(norm);

    var r: Fp2;
    r.c0 = ff_mul(a.c0, norm_inv);
    let p = modulus();
    var neg_a1: BigInt;
    if (is_bigint_zero(a.c1)) { neg_a1 = a.c1; } else { neg_a1 = ff_sub(p, a.c1); }
    r.c1 = ff_mul(neg_a1, norm_inv);
    return r;
}

fn fp2_double(a: Fp2) -> Fp2 {
    var r: Fp2;
    r.c0 = ff_add(a.c0, a.c0);
    r.c1 = ff_add(a.c1, a.c1);
    return r;
}

fn fp6_zero() -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_zero();
    r.c1 = fp2_zero();
    r.c2 = fp2_zero();
    return r;
}

fn fp6_one() -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_one();
    r.c1 = fp2_zero();
    r.c2 = fp2_zero();
    return r;
}

fn fp6_is_zero(a: Fp6) -> bool {
    return fp2_is_zero(a.c0) && fp2_is_zero(a.c1) && fp2_is_zero(a.c2);
}

fn fp6_add(a: Fp6, b: Fp6) -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_add(a.c0, b.c0);
    r.c1 = fp2_add(a.c1, b.c1);
    r.c2 = fp2_add(a.c2, b.c2);
    return r;
}

fn fp6_sub(a: Fp6, b: Fp6) -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_sub(a.c0, b.c0);
    r.c1 = fp2_sub(a.c1, b.c1);
    r.c2 = fp2_sub(a.c2, b.c2);
    return r;
}

fn fp6_neg(a: Fp6) -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_neg(a.c0);
    r.c1 = fp2_neg(a.c1);
    r.c2 = fp2_neg(a.c2);
    return r;
}

fn fp6_double(a: Fp6) -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_double(a.c0);
    r.c1 = fp2_double(a.c1);
    r.c2 = fp2_double(a.c2);
    return r;
}

fn fp6_mul(a: Fp6, b: Fp6) -> Fp6 {
    let v0 = fp2_mul(a.c0, b.c0);
    let v1 = fp2_mul(a.c1, b.c1);
    let v2 = fp2_mul(a.c2, b.c2);

    var t0 = fp2_add(a.c1, a.c2);
    var t1 = fp2_add(b.c1, b.c2);
    var t2 = fp2_mul(t0, t1);
    t2 = fp2_sub(t2, v1);
    t2 = fp2_sub(t2, v2);
    t2 = fp2_mul_by_nonresidue(t2);
    let c0 = fp2_add(v0, t2);

    t0 = fp2_add(a.c0, a.c1);
    t1 = fp2_add(b.c0, b.c1);
    t2 = fp2_mul(t0, t1);
    t2 = fp2_sub(t2, v0);
    t2 = fp2_sub(t2, v1);
    let xi_v2 = fp2_mul_by_nonresidue(v2);
    let c1 = fp2_add(t2, xi_v2);

    t0 = fp2_add(a.c0, a.c2);
    t1 = fp2_add(b.c0, b.c2);
    t2 = fp2_mul(t0, t1);
    t2 = fp2_sub(t2, v0);
    t2 = fp2_sub(t2, v2);
    let c2 = fp2_add(t2, v1);

    var r: Fp6;
    r.c0 = c0;
    r.c1 = c1;
    r.c2 = c2;
    return r;
}

fn fp6_sqr(a: Fp6) -> Fp6 {
    let s0 = fp2_sqr(a.c0);
    let ab = fp2_mul(a.c0, a.c1);
    let s1 = fp2_double(ab);
    var t = fp2_add(a.c0, a.c2);
    t = fp2_sub(t, a.c1);
    let s2 = fp2_sqr(t);
    let bc = fp2_mul(a.c1, a.c2);
    let s3 = fp2_double(bc);
    let s4 = fp2_sqr(a.c2);

    let xi_s3 = fp2_mul_by_nonresidue(s3);
    let c0 = fp2_add(s0, xi_s3);

    let xi_s4 = fp2_mul_by_nonresidue(s4);
    let c1 = fp2_add(s1, xi_s4);

    var c2 = fp2_add(s1, s2);
    c2 = fp2_add(c2, s3);
    c2 = fp2_sub(c2, s0);
    c2 = fp2_sub(c2, s4);

    var r: Fp6;
    r.c0 = c0;
    r.c1 = c1;
    r.c2 = c2;
    return r;
}

fn fp6_mul_by_nonresidue(a: Fp6) -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_mul_by_nonresidue(a.c2);
    r.c1 = a.c0;
    r.c2 = a.c1;
    return r;
}

fn fp6_mul_by_01(a: Fp6, b0: Fp2, b1: Fp2) -> Fp6 {
    let v0 = fp2_mul(a.c0, b0);
    let v1 = fp2_mul(a.c1, b1);

    var t = fp2_add(a.c1, a.c2);
    t = fp2_mul(t, b1);
    t = fp2_sub(t, v1);
    t = fp2_mul_by_nonresidue(t);
    let c0 = fp2_add(v0, t);

    let a01 = fp2_add(a.c0, a.c1);
    let b01 = fp2_add(b0, b1);
    var c1 = fp2_mul(a01, b01);
    c1 = fp2_sub(c1, v0);
    c1 = fp2_sub(c1, v1);

    let a02 = fp2_add(a.c0, a.c2);
    var c2 = fp2_mul(a02, b0);
    c2 = fp2_sub(c2, v0);
    c2 = fp2_add(c2, v1);

    var r: Fp6;
    r.c0 = c0;
    r.c1 = c1;
    r.c2 = c2;
    return r;
}

fn fp6_mul_by_fp2(a: Fp6, b: Fp2) -> Fp6 {
    var r: Fp6;
    r.c0 = fp2_mul(a.c0, b);
    r.c1 = fp2_mul(a.c1, b);
    r.c2 = fp2_mul(a.c2, b);
    return r;
}

fn fp6_inv(a: Fp6) -> Fp6 {
    let c0_sq = fp2_sqr(a.c0);
    let c1_sq = fp2_sqr(a.c1);
    let c2_sq = fp2_sqr(a.c2);

    let c0c1 = fp2_mul(a.c0, a.c1);
    let c0c2 = fp2_mul(a.c0, a.c2);
    let c1c2 = fp2_mul(a.c1, a.c2);

    let A = fp2_sub(c0_sq, fp2_mul_by_nonresidue(c1c2));
    let B = fp2_sub(fp2_mul_by_nonresidue(c2_sq), c0c1);
    let C = fp2_sub(c1_sq, c0c2);

    let c2B = fp2_mul(a.c2, B);
    let c1C = fp2_mul(a.c1, C);
    var t = fp2_add(c2B, c1C);
    t = fp2_mul_by_nonresidue(t);
    var F = fp2_mul(a.c0, A);
    F = fp2_add(F, t);

    let F_inv = fp2_inv(F);

    var r: Fp6;
    r.c0 = fp2_mul(A, F_inv);
    r.c1 = fp2_mul(B, F_inv);
    r.c2 = fp2_mul(C, F_inv);
    return r;
}

fn fp12_zero() -> Fp12 {
    var r: Fp12;
    r.c0 = fp6_zero();
    r.c1 = fp6_zero();
    return r;
}

fn fp12_one() -> Fp12 {
    var r: Fp12;
    r.c0 = fp6_one();
    r.c1 = fp6_zero();
    return r;
}

fn fp12_mul(a: Fp12, b: Fp12) -> Fp12 {
    let v0 = fp6_mul(a.c0, b.c0);
    let v1 = fp6_mul(a.c1, b.c1);

    let a01 = fp6_add(a.c0, a.c1);
    let b01 = fp6_add(b.c0, b.c1);
    var c1 = fp6_mul(a01, b01);
    c1 = fp6_sub(c1, v0);
    c1 = fp6_sub(c1, v1);

    let c0 = fp6_add(v0, fp6_mul_by_nonresidue(v1));

    var r: Fp12;
    r.c0 = c0;
    r.c1 = c1;
    return r;
}

fn fp12_sqr(a: Fp12) -> Fp12 {
    let ab = fp6_mul(a.c0, a.c1);

    let v_a1 = fp6_mul_by_nonresidue(a.c1);
    let t0 = fp6_add(a.c0, a.c1);
    let t1 = fp6_add(a.c0, v_a1);
    var c0 = fp6_mul(t0, t1);
    c0 = fp6_sub(c0, ab);
    let v_ab = fp6_mul_by_nonresidue(ab);
    c0 = fp6_sub(c0, v_ab);

    let c1 = fp6_double(ab);

    var r: Fp12;
    r.c0 = c0;
    r.c1 = c1;
    return r;
}

fn fp12_mul_by_034(a: Fp12, c0: Fp2, c3: Fp2, c4: Fp2) -> Fp12 {
    let v0 = fp6_mul_by_fp2(a.c0, c0);
    let v1 = fp6_mul_by_01(a.c1, c3, c4);

    let b_sum_0 = fp2_add(c0, c3);
    let a_sum = fp6_add(a.c0, a.c1);
    var r_c1 = fp6_mul_by_01(a_sum, b_sum_0, c4);
    r_c1 = fp6_sub(r_c1, v0);
    r_c1 = fp6_sub(r_c1, v1);

    let r_c0 = fp6_add(v0, fp6_mul_by_nonresidue(v1));

    var r: Fp12;
    r.c0 = r_c0;
    r.c1 = r_c1;
    return r;
}

fn fp12_mul_034_by_034(d0: Fp2, d3: Fp2, d4: Fp2, c0: Fp2, c3: Fp2, c4: Fp2) -> SparseResult01234 {
    let x0 = fp2_mul(c0, d0);
    let x3 = fp2_mul(c3, d3);
    let x4 = fp2_mul(c4, d4);

    var x04 = fp2_mul(fp2_add(c0, c4), fp2_add(d0, d4));
    x04 = fp2_sub(fp2_sub(x04, x0), x4);

    var x03 = fp2_mul(fp2_add(c0, c3), fp2_add(d0, d3));
    x03 = fp2_sub(fp2_sub(x03, x0), x3);

    var x34 = fp2_mul(fp2_add(c3, c4), fp2_add(d3, d4));
    x34 = fp2_sub(fp2_sub(x34, x3), x4);

    let z00 = fp2_add(fp2_mul_by_nonresidue(x4), x0);

    var r: SparseResult01234;
    r.v0 = z00;
    r.v1 = x3;
    r.v2 = x34;
    r.v3 = x03;
    r.v4 = x04;
    return r;
}

fn fp12_mul_by_01234(f: Fp12, s: SparseResult01234) -> Fp12 {
    var bc0: Fp6;
    bc0.c0 = s.v0;
    bc0.c1 = s.v1;
    bc0.c2 = s.v2;

    var bc1: Fp6;
    bc1.c0 = s.v3;
    bc1.c1 = s.v4;
    bc1.c2 = fp2_zero();

    let a = fp6_mul(fp6_add(f.c0, f.c1), fp6_add(bc0, bc1));
    let b = fp6_mul(f.c0, bc0);
    let c = fp6_mul_by_01(f.c1, s.v3, s.v4);

    let new_c1 = fp6_sub(fp6_sub(a, b), c);
    let new_c0 = fp6_add(fp6_mul_by_nonresidue(c), b);

    var r: Fp12;
    r.c0 = new_c0;
    r.c1 = new_c1;
    return r;
}

fn g2_is_zero(a: G2Jacobian) -> bool {
    return fp2_is_zero(a.z);
}

fn g2_zero() -> G2Jacobian {
    var r: G2Jacobian;
    r.x = fp2_one();
    r.y = fp2_one();
    r.z = fp2_zero();
    return r;
}

fn load_coeff_b() -> Fp2 {
    var b: Fp2;
    for (var i = 0u; i < 8u; i = i + 1u) {
        b.c0.limbs[i] = G2_COEFF_B_C0[i];
        b.c1.limbs[i] = G2_COEFF_B_C1[i];
    }
    return b;
}

fn load_two_inv() -> BigInt {
    var r: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        r.limbs[i] = TWO_INV[i];
    }
    return r;
}

fn g2_double_eval(pt: G2Jacobian) -> G2DoubleResult {
    let two_inv = load_two_inv();
    let coeff_b = load_coeff_b();

    var a = fp2_mul(pt.x, pt.y);
    a = fp2_mul_by_fp(a, two_inv);

    let b = fp2_sqr(pt.y);
    let c = fp2_sqr(pt.z);

    let c3 = fp2_add(c, fp2_add(c, c));
    let e = fp2_mul(coeff_b, c3);

    let f = fp2_add(e, fp2_add(e, e));

    var g = fp2_add(b, f);
    g = fp2_mul_by_fp(g, two_inv);

    var h = fp2_add(pt.y, pt.z);
    h = fp2_sqr(h);
    h = fp2_sub(h, b);
    h = fp2_sub(h, c);

    let i_val = fp2_sub(e, b);
    let j = fp2_sqr(pt.x);

    let e_square = fp2_sqr(e);

    let new_x = fp2_mul(a, fp2_sub(b, f));
    let e_sq_3 = fp2_add(e_square, fp2_add(e_square, e_square));
    let new_y = fp2_sub(fp2_sqr(g), e_sq_3);
    let new_z = fp2_mul(b, h);

    var coeffs: EllCoeffs;
    coeffs.ell_0 = fp2_neg(h);
    coeffs.ell_vw = fp2_add(j, fp2_add(j, j));
    coeffs.ell_vv = i_val;

    var res: G2DoubleResult;
    res.point.x = new_x;
    res.point.y = new_y;
    res.point.z = new_z;
    res.coeffs = coeffs;
    return res;
}

fn g2_add_eval(T: G2Jacobian, Q: G2Affine) -> G2AddResult {
    let theta = fp2_sub(T.y, fp2_mul(Q.y, T.z));
    let lambda = fp2_sub(T.x, fp2_mul(Q.x, T.z));

    let c_val = fp2_sqr(theta);
    let d = fp2_sqr(lambda);
    let e = fp2_mul(lambda, d);
    let f_val = fp2_mul(T.z, c_val);
    let g = fp2_mul(T.x, d);
    let h = fp2_sub(fp2_add(e, f_val), fp2_double(g));

    let new_x = fp2_mul(lambda, h);
    let new_y = fp2_sub(fp2_mul(theta, fp2_sub(g, h)), fp2_mul(e, T.y));
    let new_z = fp2_mul(T.z, e);

    let j = fp2_sub(fp2_mul(theta, Q.x), fp2_mul(lambda, Q.y));

    var coeffs: EllCoeffs;
    coeffs.ell_0 = lambda;
    coeffs.ell_vw = fp2_neg(theta);
    coeffs.ell_vv = j;

    var res: G2AddResult;
    res.point.x = new_x;
    res.point.y = new_y;
    res.point.z = new_z;
    res.coeffs = coeffs;
    return res;
}

fn apply_line_to_f(f: Fp12, coeffs: EllCoeffs, x_P: BigInt, y_P: BigInt) -> Fp12 {
    let c0 = fp2_mul_by_fp(coeffs.ell_0, y_P);
    let c3 = fp2_mul_by_fp(coeffs.ell_vw, x_P);
    let c4 = coeffs.ell_vv;
    return fp12_mul_by_034(f, c0, c3, c4);
}

fn miller_loop_single(p_x: BigInt, p_y: BigInt, Q: G2Affine) -> Fp12 {
    var f = fp12_one();

    var T: G2Jacobian;
    T.x = Q.x;
    T.y = Q.y;
    T.z = fp2_one();

    for (var i: i32 = 1; i < ATE_LOOP_NAF_LEN; i = i + 1) {
        if (i > 1) {
            f = fp12_sqr(f);
        }

        let dbl_res = g2_double_eval(T);
        T = dbl_res.point;

        let naf_val = ATE_LOOP_NAF[i];
        if (naf_val == 1 || naf_val == -1) {
            var q_to_add: G2Affine;
            if (naf_val == 1) {
                q_to_add = Q;
            } else {
                q_to_add.x = Q.x;
                q_to_add.y = fp2_neg(Q.y);
            }
            let add_res = g2_add_eval(T, q_to_add);
            T = add_res.point;

            let c0_d = fp2_mul_by_fp(dbl_res.coeffs.ell_0, p_y);
            let c3_d = fp2_mul_by_fp(dbl_res.coeffs.ell_vw, p_x);
            let c4_d = dbl_res.coeffs.ell_vv;
            let c0_a = fp2_mul_by_fp(add_res.coeffs.ell_0, p_y);
            let c3_a = fp2_mul_by_fp(add_res.coeffs.ell_vw, p_x);
            let c4_a = add_res.coeffs.ell_vv;
            let sparse = fp12_mul_034_by_034(c0_d, c3_d, c4_d, c0_a, c3_a, c4_a);
            f = fp12_mul_by_01234(f, sparse);
        } else {
            f = apply_line_to_f(f, dbl_res.coeffs, p_x, p_y);
        }
    }

    var Q1: G2Affine;
    let q_x_conj = fp2_conjugate(Q.x);
    let q_y_conj = fp2_conjugate(Q.y);

    var frob_x: Fp2;
    var frob_y: Fp2;
    for (var i = 0u; i < 8u; i = i + 1u) {
        frob_x.c0.limbs[i] = FROBENIUS_COEFF_X_C0[i];
        frob_x.c1.limbs[i] = FROBENIUS_COEFF_X_C1[i];
        frob_y.c0.limbs[i] = FROBENIUS_COEFF_Y_C0[i];
        frob_y.c1.limbs[i] = FROBENIUS_COEFF_Y_C1[i];
    }

    Q1.x = fp2_mul(q_x_conj, frob_x);
    Q1.y = fp2_mul(q_y_conj, frob_y);

    let add_q1 = g2_add_eval(T, Q1);
    T = add_q1.point;

    var neg_Q2: G2Affine;
    var frob_x2: Fp2;
    var frob_y2: Fp2;
    for (var i = 0u; i < 8u; i = i + 1u) {
        frob_x2.c0.limbs[i] = FROBENIUS_COEFF_X2_C0[i];
        frob_x2.c1.limbs[i] = FROBENIUS_COEFF_X2_C1[i];
        frob_y2.c0.limbs[i] = FROBENIUS_COEFF_Y2_C0[i];
        frob_y2.c1.limbs[i] = FROBENIUS_COEFF_Y2_C1[i];
    }
    neg_Q2.x = fp2_mul(Q.x, frob_x2);
    neg_Q2.y = fp2_neg(fp2_mul(Q.y, frob_y2));

    let add_q2 = g2_add_eval(T, neg_Q2);

    let c0_q1 = fp2_mul_by_fp(add_q1.coeffs.ell_0, p_y);
    let c3_q1 = fp2_mul_by_fp(add_q1.coeffs.ell_vw, p_x);
    let c4_q1 = add_q1.coeffs.ell_vv;
    let c0_q2 = fp2_mul_by_fp(add_q2.coeffs.ell_0, p_y);
    let c3_q2 = fp2_mul_by_fp(add_q2.coeffs.ell_vw, p_x);
    let c4_q2 = add_q2.coeffs.ell_vv;
    let sparse_frob = fp12_mul_034_by_034(c0_q1, c3_q1, c4_q1, c0_q2, c3_q2, c4_q2);
    f = fp12_mul_by_01234(f, sparse_frob);

    return f;
}

