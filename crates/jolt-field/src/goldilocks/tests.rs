//! `num-bigint` oracle tests for the hand-coded Goldilocks field + Fp3.
//!
//! Every field operation is checked against an independent big-integer
//! reference (no crypto deps). This is the correctness de-risk for the
//! Montgomery-free arithmetic and the cubic-extension formulas.

#![expect(clippy::unwrap_used)]

use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use super::base::{Goldilocks, P};
use super::decompose::{i128_to_sign_limbs, limbs_to_u64, sign_limbs_to_i128, u64_to_limbs};
use super::ext3::GoldilocksFp3;
use crate::Field;

fn p_big() -> BigUint {
    BigUint::from(P)
}

fn gl_big(x: Goldilocks) -> BigUint {
    BigUint::from(x.to_u64().unwrap())
}

fn assert_gl_eq_big(got: Goldilocks, want: BigUint) {
    let want_mod = (&want % p_big()).to_u64_digits();
    let want_u64 = want_mod.first().copied().unwrap_or(0);
    assert_eq!(got.to_u64().unwrap(), want_u64, "got {got} want {want}");
}

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0x5EED_6017)
}

fn rand_gl(r: &mut ChaCha20Rng) -> Goldilocks {
    Goldilocks::from_u64(r.gen::<u64>())
}

// ---------- base field ----------

#[test]
fn base_arith_matches_bigint_oracle() {
    let mut r = rng();
    let p = p_big();
    for _ in 0..5000 {
        let a = rand_gl(&mut r);
        let b = rand_gl(&mut r);
        let ab = gl_big(a);
        let bb = gl_big(b);
        assert_gl_eq_big(a + b, &ab + &bb);
        assert_gl_eq_big(a - b, (&ab + &p - &bb) % &p);
        assert_gl_eq_big(a * b, &ab * &bb);
        assert_gl_eq_big(-a, (&p - &ab) % &p);
        assert_gl_eq_big(a.square(), &ab * &ab);
    }
}

#[test]
fn base_inverse_matches_oracle_and_property() {
    let mut r = rng();
    let p = p_big();
    let exp = &p - BigUint::from(2u32); // a^(p-2)
    for _ in 0..2000 {
        let a = rand_gl(&mut r);
        if a.is_zero() {
            continue;
        }
        let inv = a.inverse().unwrap();
        // a * a^{-1} == 1
        assert!((a * inv).is_one());
        // a^{-1} == a^(p-2) mod p
        assert_gl_eq_big(inv, gl_big(a).modpow(&exp, &p));
    }
    assert!(Goldilocks::zero().inverse().is_none());
}

#[test]
fn base_noncanonical_representatives_compare_equal() {
    // Values in [p, 2^64) alias to their canonical residue.
    assert_eq!(Goldilocks::from_raw(P), Goldilocks::zero());
    assert_eq!(Goldilocks::from_raw(P + 5), Goldilocks::from_u64(5));
    // u64::MAX = 2^64 - 1 ≡ 2^32 - 2 (mod p)
    assert_eq!(
        Goldilocks::from_raw(u64::MAX),
        Goldilocks::from_u64(0xFFFF_FFFE)
    );
    assert_eq!(Goldilocks::from_raw(P).to_u64().unwrap(), 0);
}

#[test]
fn base_from_int_conversions() {
    let p = p_big();
    assert_gl_eq_big(
        Goldilocks::from_u64(123_456_789),
        BigUint::from(123_456_789u64),
    );
    assert_gl_eq_big(Goldilocks::from_u128(u128::MAX), BigUint::from(u128::MAX));
    assert_gl_eq_big(Goldilocks::from_u64(u64::MAX), BigUint::from(u64::MAX));
    // negatives → p − |v|
    assert_gl_eq_big(Goldilocks::from_i64(-7), (&p - BigUint::from(7u32)) % &p);
    assert_gl_eq_big(
        Goldilocks::from_i128(-(1i128 << 100)),
        (&p - (BigUint::from(1u32) << 100u32) % &p) % &p,
    );
}

#[test]
fn base_mul_pow_2_matches_oracle() {
    let mut r = rng();
    let p = p_big();
    for _ in 0..1000 {
        let a = rand_gl(&mut r);
        let pow = r.gen_range(0..200usize);
        let want = (gl_big(a) * (BigUint::from(1u32) << pow as u32)) % &p;
        assert_gl_eq_big(a.mul_pow_2(pow), want);
    }
}

// ---------- Fp3 extension ----------

fn fp3_big(x: &GoldilocksFp3) -> [BigUint; 3] {
    let c = x.coeffs();
    [gl_big(c[0]), gl_big(c[1]), gl_big(c[2])]
}

/// Oracle multiply in Fp[x]/(x³ − 2).
fn fp3_oracle_mul(a: &[BigUint; 3], b: &[BigUint; 3], p: &BigUint) -> [BigUint; 3] {
    let two = BigUint::from(2u32);
    // r0 = a0 b0 + 2(a1 b2 + a2 b1)
    let r0 = (&a[0] * &b[0] + &two * (&a[1] * &b[2] + &a[2] * &b[1])) % p;
    // r1 = a0 b1 + a1 b0 + 2 a2 b2
    let r1 = (&a[0] * &b[1] + &a[1] * &b[0] + &two * (&a[2] * &b[2])) % p;
    // r2 = a0 b2 + a1 b1 + a2 b0
    let r2 = (&a[0] * &b[2] + &a[1] * &b[1] + &a[2] * &b[0]) % p;
    [r0, r1, r2]
}

fn assert_fp3_eq_big(got: GoldilocksFp3, want: [BigUint; 3]) {
    let g = fp3_big(&got);
    let p = p_big();
    for i in 0..3 {
        let w = (&want[i] % &p)
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0);
        assert_eq!(g[i].to_u64_digits().first().copied().unwrap_or(0), w);
    }
}

fn rand_fp3(r: &mut ChaCha20Rng) -> GoldilocksFp3 {
    GoldilocksFp3::new(rand_gl(r), rand_gl(r), rand_gl(r))
}

#[test]
fn fp3_arith_matches_bigint_oracle() {
    let mut r = rng();
    let p = p_big();
    for _ in 0..3000 {
        let a = rand_fp3(&mut r);
        let b = rand_fp3(&mut r);
        let (ab, bb) = (fp3_big(&a), fp3_big(&b));
        assert_fp3_eq_big(a + b, [&ab[0] + &bb[0], &ab[1] + &bb[1], &ab[2] + &bb[2]]);
        assert_fp3_eq_big(
            a - b,
            [
                (&ab[0] + &p - &bb[0]) % &p,
                (&ab[1] + &p - &bb[1]) % &p,
                (&ab[2] + &p - &bb[2]) % &p,
            ],
        );
        assert_fp3_eq_big(a * b, fp3_oracle_mul(&ab, &bb, &p));
        assert_fp3_eq_big(a.square(), fp3_oracle_mul(&ab, &ab, &p));
    }
}

#[test]
fn fp3_inverse_is_multiplicative_inverse() {
    let mut r = rng();
    for _ in 0..3000 {
        let a = rand_fp3(&mut r);
        if a.is_zero() {
            continue;
        }
        let inv = a.inverse().unwrap();
        assert!((a * inv).is_one(), "a={a} inv={inv}");
    }
    assert!(GoldilocksFp3::zero().inverse().is_none());
}

#[test]
fn fp3_mul_by_base_matches_full_mul() {
    let mut r = rng();
    for _ in 0..2000 {
        let a = rand_fp3(&mut r);
        let b = rand_gl(&mut r);
        assert_eq!(a.mul_by_base(b), a * GoldilocksFp3::from_base(b));
    }
}

#[test]
fn fp3_base_embedding() {
    let b = Goldilocks::from_u64(0xDEAD_BEEF);
    let e = GoldilocksFp3::from_base(b);
    assert_eq!(e.coeffs()[0], b);
    assert!(e.coeffs()[1].is_zero() && e.coeffs()[2].is_zero());
    assert_eq!(
        GoldilocksFp3::from_u64(42),
        GoldilocksFp3::from_base(Goldilocks::from_u64(42))
    );
}

// ---------- limb decomposition ----------

#[test]
fn limb_roundtrips() {
    let mut r = rng();
    for _ in 0..5000 {
        let v: u64 = r.gen();
        assert_eq!(limbs_to_u64(u64_to_limbs(v)), v);
    }
    // signed increments (|delta| < 2^64)
    for _ in 0..5000 {
        let v: i128 = (r.gen::<i64>()) as i128;
        let (sign, limbs) = i128_to_sign_limbs(v);
        assert_eq!(sign_limbs_to_i128(sign, limbs), v);
    }
    // boundary magnitudes
    for &mag in &[0u64, 1, u32::MAX as u64, 1u64 << 32, u64::MAX] {
        let (s, l) = i128_to_sign_limbs(mag as i128);
        assert_eq!(sign_limbs_to_i128(s, l), mag as i128);
        let (s, l) = i128_to_sign_limbs(-(mag as i128));
        assert_eq!(sign_limbs_to_i128(s, l), -(mag as i128));
    }
}
