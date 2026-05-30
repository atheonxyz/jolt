//! Goldilocks base field `Fp`, `p = 2^64 − 2^32 + 1`, Montgomery-free.
//!
//! Arithmetic follows the Plonky2 / lambda_vm design that exploits the prime's
//! structure (`2^64 ≡ 2^32 − 1`, `2^96 ≡ −1 (mod p)`): a 128-bit product reduces
//! with a handful of shifts/adds instead of a Montgomery REDC. Elements are
//! stored **non-canonically** in `[0, 2^64)` (the representative may be `≥ p` but
//! is always `< 2^64 = p + EPSILON`, so at most one conditional subtract
//! canonicalizes it). Equality, hashing, and serialization use the canonical
//! form so the non-canonical representation is invisible to callers.
//!
//! Correctness is guarded by the `num-bigint` oracle tests in `super::tests`.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::{One, Zero};
use rand_core::RngCore;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::accumulator::{GoldilocksAccumulator, GoldilocksScalarAccumulator};
use crate::Field;

/// The Goldilocks prime `p = 2^64 − 2^32 + 1`.
pub const P: u64 = 0xFFFF_FFFF_0000_0001;
/// `EPSILON = 2^32 − 1 = 2^64 mod p`.
const EPSILON: u64 = 0xFFFF_FFFF;

/// Goldilocks field element, stored non-canonically in `[0, 2^64)`.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct Goldilocks(pub(crate) u64);

impl Goldilocks {
    /// Reduce the internal representative to the canonical range `[0, p)`.
    ///
    /// Sound because every stored value is `< 2^64 = p + EPSILON < 2p`.
    #[inline(always)]
    pub(crate) const fn to_canonical_u64(self) -> u64 {
        let x = self.0;
        if x >= P {
            x - P
        } else {
            x
        }
    }

    /// Wrap a raw `u64` as a (non-canonical) field element. Any `u64` is a valid
    /// representative since `u64::MAX < 2p`.
    #[inline(always)]
    pub(crate) const fn from_raw(x: u64) -> Self {
        Self(x)
    }
}

/// `(a + b) mod p`, result non-canonical in `[0, 2^64)`.
#[inline(always)]
fn add_gl(a: u64, b: u64) -> u64 {
    // true sum = s1 + c1·2^64 ≡ s1 + c1·EPSILON (mod p).
    let (s1, c1) = a.overflowing_add(b);
    let (s2, c2) = s1.overflowing_add(if c1 { EPSILON } else { 0 });
    // A second overflow contributes another 2^64 ≡ EPSILON; cannot overflow a third time.
    if c2 {
        s2.wrapping_add(EPSILON)
    } else {
        s2
    }
}

/// `(a − b) mod p`, result non-canonical in `[0, 2^64)`.
#[inline(always)]
fn sub_gl(a: u64, b: u64) -> u64 {
    // true diff = d1 − c1·2^64 ≡ d1 − c1·EPSILON (mod p).
    let (d1, c1) = a.overflowing_sub(b);
    let (d2, c2) = d1.overflowing_sub(if c1 { EPSILON } else { 0 });
    if c2 {
        d2.wrapping_sub(EPSILON)
    } else {
        d2
    }
}

/// Reduce a 128-bit product to `[0, 2^64)` using `2^64 ≡ 2^32−1`, `2^96 ≡ −1`.
#[inline(always)]
fn reduce128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32; // coefficient of 2^96 ≡ −1
    let x_hi_lo = x_hi & EPSILON; // coefficient of 2^64 ≡ EPSILON

    // x_lo − x_hi_hi  (the 2^96 ≡ −1 term)
    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        t0 = t0.wrapping_sub(EPSILON);
    }
    // x_hi_lo · (2^32 − 1), computed as a shift+sub to avoid a multiply.
    // x_hi_lo < 2^32 so the shift does not overflow.
    let t1 = (x_hi_lo << 32) - x_hi_lo;
    add_gl(t0, t1)
}

/// Reduce a 192-bit integer `x = x[0] + x[1]·2^64 + x[2]·2^128` to `[0, 2^64)`.
///
/// Uses `2^128 ≡ −2^32 (mod p)` (from `2^96 ≡ −1`): the high limb's contribution
/// is `x[2]·2^128 ≡ −(x[2]·2^32)`, so we subtract it from the reduced low 128 bits.
/// The backing accumulator for `Goldilocks` deferred fmadd.
#[inline]
pub(crate) fn reduce192(x: [u64; 3]) -> u64 {
    let lo128 = (x[0] as u128) | ((x[1] as u128) << 64);
    let r_lo = reduce128(lo128);
    // x[2] < 2^64 ⇒ (x[2] << 32) < 2^96 fits a u128; reduce128 gives x[2]·2^32 mod p.
    let hi = reduce128((x[2] as u128) << 32);
    sub_gl(r_lo, hi)
}

/// Reduce a 256-bit integer `x = Σ x[i]·2^(64i)` to `[0, 2^64)`.
///
/// `2^192 ≡ 1 (mod p)` (from `2^96 ≡ −1`), so the top limb contributes `+ x[3]`.
/// The backing accumulator for the `Goldilocks` × raw-integer scalar fmadd.
#[inline]
pub(crate) fn reduce256(x: [u64; 4]) -> u64 {
    add_gl(reduce192([x[0], x[1], x[2]]), reduce128(x[3] as u128))
}

#[inline(always)]
fn mul_gl(a: u64, b: u64) -> u64 {
    reduce128((a as u128) * (b as u128))
}

#[inline(always)]
fn neg_gl(x: u64) -> u64 {
    let c = if x >= P { x - P } else { x };
    if c == 0 {
        0
    } else {
        P - c
    }
}

impl Goldilocks {
    #[inline(always)]
    fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self(1);
        while exp > 0 {
            if exp & 1 == 1 {
                acc = Self(mul_gl(acc.0, base.0));
            }
            base = Self(mul_gl(base.0, base.0));
            exp >>= 1;
        }
        acc
    }
}

impl PartialEq for Goldilocks {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_canonical_u64() == other.to_canonical_u64()
    }
}
impl Eq for Goldilocks {}

impl Hash for Goldilocks {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_canonical_u64().hash(state);
    }
}

impl PartialOrd for Goldilocks {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Goldilocks {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_canonical_u64().cmp(&other.to_canonical_u64())
    }
}

impl fmt::Debug for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_canonical_u64())
    }
}
impl fmt::Display for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_canonical_u64())
    }
}

impl Serialize for Goldilocks {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Canonical form so equal elements serialize identically.
        self.to_canonical_u64().serialize(s)
    }
}
impl<'de> Deserialize<'de> for Goldilocks {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = u64::deserialize(d)?;
        Ok(Self(v % P))
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for Goldilocks {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

impl Zero for Goldilocks {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.to_canonical_u64() == 0
    }
}
impl One for Goldilocks {
    #[inline(always)]
    fn one() -> Self {
        Self(1)
    }
    #[inline(always)]
    fn is_one(&self) -> bool {
        self.to_canonical_u64() == 1
    }
}

impl Neg for Goldilocks {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(neg_gl(self.0))
    }
}

macro_rules! bin_op {
    ($trait:ident, $method:ident, $f:expr) => {
        impl $trait for Goldilocks {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self($f(self.0, rhs.0))
            }
        }
        impl<'a> $trait<&'a Goldilocks> for Goldilocks {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: &'a Goldilocks) -> Self {
                Self($f(self.0, rhs.0))
            }
        }
    };
}
bin_op!(Add, add, add_gl);
bin_op!(Sub, sub, sub_gl);
bin_op!(Mul, mul, mul_gl);

impl Div for Goldilocks {
    type Output = Self;
    #[inline]
    #[expect(clippy::suspicious_arithmetic_impl, clippy::expect_used)]
    fn div(self, rhs: Self) -> Self {
        self * Field::inverse(&rhs).expect("division by zero in Goldilocks")
    }
}
impl<'a> Div<&'a Goldilocks> for Goldilocks {
    type Output = Self;
    #[inline]
    #[expect(clippy::suspicious_arithmetic_impl, clippy::expect_used)]
    fn div(self, rhs: &'a Goldilocks) -> Self {
        self * Field::inverse(rhs).expect("division by zero in Goldilocks")
    }
}

impl AddAssign for Goldilocks {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = add_gl(self.0, rhs.0);
    }
}
impl SubAssign for Goldilocks {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = sub_gl(self.0, rhs.0);
    }
}
impl MulAssign for Goldilocks {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = mul_gl(self.0, rhs.0);
    }
}

impl Sum for Goldilocks {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0), |a, b| a + b)
    }
}
impl<'a> Sum<&'a Goldilocks> for Goldilocks {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self(0), |a, b| a + *b)
    }
}
impl Product for Goldilocks {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(1), |a, b| a * b)
    }
}
impl<'a> Product<&'a Goldilocks> for Goldilocks {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self(1), |a, b| a * *b)
    }
}

impl From<u128> for Goldilocks {
    #[inline]
    fn from(v: u128) -> Self {
        <Self as Field>::from_u128(v)
    }
}

impl Field for Goldilocks {
    type Accumulator = GoldilocksAccumulator;
    type ScalarAccumulator = GoldilocksScalarAccumulator;

    const NUM_BYTES: usize = 8;

    #[inline]
    fn to_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[..8].copy_from_slice(&self.to_canonical_u64().to_le_bytes());
        out
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        // Rejection-sample for an unbiased element in [0, p).
        loop {
            let x = rng.next_u64();
            if x < P {
                return Self(x);
            }
        }
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        let n = bytes.len().min(8);
        buf[..n].copy_from_slice(&bytes[..n]);
        Self(u64::from_le_bytes(buf) % P)
    }

    #[inline(always)]
    fn to_u64(&self) -> Option<u64> {
        Some(self.to_canonical_u64())
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        64 - self.to_canonical_u64().leading_zeros()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        Self(mul_gl(self.0, self.0))
    }

    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        // a^(p-2). Square-and-multiply; correct and rarely on a hot path. The
        // Plonky2 addition chain (~72 muls) is a future perf specialization.
        Some(self.pow(P - 2))
    }

    #[inline(always)]
    fn from_u64(n: u64) -> Self {
        Self(n)
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self(val as u64)
        } else {
            Self(neg_gl((val.unsigned_abs()) % P))
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val >= 0 {
            Self(reduce128(val as u128))
        } else {
            Self(neg_gl(reduce128(val.unsigned_abs())))
        }
    }

    #[inline(always)]
    fn from_u128(val: u128) -> Self {
        Self(reduce128(val))
    }

    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        Self(mul_gl(self.0, n))
    }
}
