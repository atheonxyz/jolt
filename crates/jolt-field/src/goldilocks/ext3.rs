//! Goldilocks cubic extension `Fp3 = Fp[x]/(x³ − 2)` (~192-bit).
//!
//! This is the field the Phase-2 prover treats as its scalar (`JoltField`):
//! witnesses live in the base `Goldilocks`, challenges/folds in `Fp3`.
//! Elements are `c0 + c1·x + c2·x²` with `x³ = 2`. Multiplication is schoolbook
//! with the `x³ → 2` reduction (9 base muls); [`mul_by_base`](GoldilocksFp3::mul_by_base)
//! is the 3-mul base×ext fast path; inversion uses the cubic field norm
//! (1 base inverse + a few muls). Correctness is guarded by the `num-bigint`
//! oracle tests in `super::tests`.

use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::{One, Zero};
use rand_core::RngCore;
use serde::{Deserialize, Serialize};

use super::base::Goldilocks;
use crate::accumulator::{NaiveAccumulator, NaiveScalarAccumulator};
use crate::Field;

/// `Fp3 = Goldilocks[x]/(x³ − 2)`. Stored as `[c0, c1, c2]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct GoldilocksFp3(pub(crate) [Goldilocks; 3]);

#[inline(always)]
fn double(x: Goldilocks) -> Goldilocks {
    x + x
}

impl GoldilocksFp3 {
    /// The cubic non-residue `w = 2` (so `x³ = 2`).
    #[inline(always)]
    pub const fn new(c0: Goldilocks, c1: Goldilocks, c2: Goldilocks) -> Self {
        Self([c0, c1, c2])
    }

    /// Embed a base-field element as `b + 0·x + 0·x²`.
    #[inline(always)]
    pub const fn from_base(b: Goldilocks) -> Self {
        Self([b, Goldilocks::from_raw(0), Goldilocks::from_raw(0)])
    }

    /// The three base-field coefficients (LSB-first: `c0 + c1·x + c2·x²`).
    #[inline(always)]
    pub fn coeffs(&self) -> &[Goldilocks; 3] {
        &self.0
    }

    /// `self · b` for a base-field `b` — 3 base muls (the Phase-2 sumcheck hot path).
    #[inline(always)]
    pub fn mul_by_base(&self, b: Goldilocks) -> Self {
        Self([self.0[0] * b, self.0[1] * b, self.0[2] * b])
    }
}

#[inline(always)]
fn mul_fp3(a: &[Goldilocks; 3], b: &[Goldilocks; 3]) -> [Goldilocks; 3] {
    // (a0+a1x+a2x²)(b0+b1x+b2x²) mod (x³−2), with x³=2, x⁴=2x:
    //   r0 = a0 b0 + 2(a1 b2 + a2 b1)
    //   r1 = a0 b1 + a1 b0 + 2(a2 b2)
    //   r2 = a0 b2 + a1 b1 + a2 b0
    let (a0, a1, a2) = (a[0], a[1], a[2]);
    let (b0, b1, b2) = (b[0], b[1], b[2]);
    let r0 = a0 * b0 + double(a1 * b2 + a2 * b1);
    let r1 = a0 * b1 + a1 * b0 + double(a2 * b2);
    let r2 = a0 * b2 + a1 * b1 + a2 * b0;
    [r0, r1, r2]
}

impl fmt::Debug for GoldilocksFp3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp3({}, {}, {})", self.0[0], self.0[1], self.0[2])
    }
}
impl fmt::Display for GoldilocksFp3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}·x + {}·x²)", self.0[0], self.0[1], self.0[2])
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for GoldilocksFp3 {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

impl Zero for GoldilocksFp3 {
    #[inline(always)]
    fn zero() -> Self {
        Self([Goldilocks::zero(); 3])
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0[0].is_zero() && self.0[1].is_zero() && self.0[2].is_zero()
    }
}
impl One for GoldilocksFp3 {
    #[inline(always)]
    fn one() -> Self {
        Self([Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()])
    }
}

impl Neg for GoldilocksFp3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

macro_rules! ext_add_sub {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for GoldilocksFp3 {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self([self.0[0] $op rhs.0[0], self.0[1] $op rhs.0[1], self.0[2] $op rhs.0[2]])
            }
        }
        impl<'a> $trait<&'a GoldilocksFp3> for GoldilocksFp3 {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: &'a GoldilocksFp3) -> Self {
                Self([self.0[0] $op rhs.0[0], self.0[1] $op rhs.0[1], self.0[2] $op rhs.0[2]])
            }
        }
    };
}
ext_add_sub!(Add, add, +);
ext_add_sub!(Sub, sub, -);

impl Mul for GoldilocksFp3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(mul_fp3(&self.0, &rhs.0))
    }
}
impl<'a> Mul<&'a GoldilocksFp3> for GoldilocksFp3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &'a GoldilocksFp3) -> Self {
        Self(mul_fp3(&self.0, &rhs.0))
    }
}

impl Div for GoldilocksFp3 {
    type Output = Self;
    #[inline]
    #[expect(clippy::suspicious_arithmetic_impl, clippy::expect_used)]
    fn div(self, rhs: Self) -> Self {
        self * Field::inverse(&rhs).expect("division by zero in GoldilocksFp3")
    }
}
impl<'a> Div<&'a GoldilocksFp3> for GoldilocksFp3 {
    type Output = Self;
    #[inline]
    #[expect(clippy::suspicious_arithmetic_impl, clippy::expect_used)]
    fn div(self, rhs: &'a GoldilocksFp3) -> Self {
        self * Field::inverse(rhs).expect("division by zero in GoldilocksFp3")
    }
}

impl AddAssign for GoldilocksFp3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for GoldilocksFp3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for GoldilocksFp3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for GoldilocksFp3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}
impl<'a> Sum<&'a GoldilocksFp3> for GoldilocksFp3 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + *b)
    }
}
impl Product for GoldilocksFp3 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}
impl<'a> Product<&'a GoldilocksFp3> for GoldilocksFp3 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * *b)
    }
}

impl Field for GoldilocksFp3 {
    type Accumulator = NaiveAccumulator<Self>;
    type ScalarAccumulator = NaiveScalarAccumulator<Self>;

    const NUM_BYTES: usize = 24;

    #[inline]
    fn to_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, c) in self.0.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&c.to_canonical_u64().to_le_bytes());
        }
        out
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self([
            Goldilocks::random(rng),
            Goldilocks::random(rng),
            Goldilocks::random(rng),
        ])
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut c = [Goldilocks::zero(); 3];
        for (i, slot) in c.iter_mut().enumerate() {
            let lo = i * 8;
            if lo < bytes.len() {
                *slot = Goldilocks::from_bytes(&bytes[lo..bytes.len().min(lo + 8)]);
            }
        }
        Self(c)
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        if self.0[1].is_zero() && self.0[2].is_zero() {
            self.0[0].to_u64()
        } else {
            None
        }
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        if !self.0[2].is_zero() {
            128 + self.0[2].num_bits()
        } else if !self.0[1].is_zero() {
            64 + self.0[1].num_bits()
        } else {
            self.0[0].num_bits()
        }
    }

    #[inline(always)]
    fn square(&self) -> Self {
        Self(mul_fp3(&self.0, &self.0))
    }

    #[expect(clippy::expect_used)]
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        // Adjugate / norm inversion for Fp[x]/(x³−w), w = 2.
        // Cofactors of the multiply-by-self matrix's first row:
        let (a0, a1, a2) = (self.0[0], self.0[1], self.0[2]);
        let c00 = a0 * a0 - double(a1 * a2); // a0² − 2 a1 a2
        let c01 = double(a2 * a2) - a0 * a1; // 2 a2² − a0 a1
        let c02 = a1 * a1 - a0 * a2; // a1² − a0 a2
                                     // norm N = a0·c00 + 2·(a2·c01 + a1·c02) = a0³ + 2 a1³ + 4 a2³ − 6 a0 a1 a2
        let norm = a0 * c00 + double(a2 * c01 + a1 * c02);
        let norm_inv = Field::inverse(&norm).expect("nonzero Fp3 has nonzero norm");
        Some(Self([c00 * norm_inv, c01 * norm_inv, c02 * norm_inv]))
    }

    #[inline(always)]
    fn from_u64(n: u64) -> Self {
        Self::from_base(Goldilocks::from_u64(n))
    }
    #[inline(always)]
    fn from_i64(val: i64) -> Self {
        Self::from_base(Goldilocks::from_i64(val))
    }
    #[inline(always)]
    fn from_i128(val: i128) -> Self {
        Self::from_base(Goldilocks::from_i128(val))
    }
    #[inline(always)]
    fn from_u128(val: u128) -> Self {
        Self::from_base(Goldilocks::from_u128(val))
    }
}
