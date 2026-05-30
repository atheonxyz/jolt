//! Deferred-reduction accumulators for Goldilocks and its cubic extension.
//!
//! These mirror the BN254 [`WideAccumulator`](crate::arkworks::wide_accumulator)
//! pattern: the sumcheck inner loop runs `acc += a * b` hundreds of times per
//! output slot, and reducing mod `p` after every product dominates the CPU
//! prover. Instead we accumulate the *unreduced* integer sum of products in a
//! wide limb array and reduce once at the end.
//!
//! Reductions exploit the Goldilocks structure (`p = 2^64 − 2^32 + 1`):
//! `2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`, hence `2^128 ≡ −2^32` and `2^192 ≡ 1`. So a
//! 192- or 256-bit accumulator folds back to a single base element with a couple
//! of `reduce128` calls and one add/sub — no extra multiply. See
//! [`reduce192`](super::base::reduce192) / [`reduce256`](super::base::reduce256).
//!
//! Correctness is guarded by the `num-bigint` oracle tests in `super::tests`.

use crate::accumulator::{FieldAccumulator, FieldScalarAccumulator};
use crate::Limbs;

use super::base::{reduce192, reduce256, Goldilocks};
use super::ext3::GoldilocksFp3;

// Capacity note (both base accumulators): inputs are canonicalized to `< p < 2^64`
// in `fmadd`, so each product is `< p^2 < 2^128`. A 192-bit lane (`Limbs<3>`) thus
// holds up to `2^64` products before overflow; the scalar accumulator's 256-bit
// lane (`Limbs<4>`) holds `2^64` of the wider `< 2^192` `value × u128` products.
// No real sumcheck approaches `2^64` terms.

/// 192-bit deferred-reduction accumulator for [`Goldilocks`].
#[derive(Clone, Copy, Default)]
pub struct GoldilocksAccumulator {
    limbs: Limbs<3>,
}

impl FieldAccumulator for GoldilocksAccumulator {
    type Field = Goldilocks;

    #[inline(always)]
    fn fmadd(&mut self, a: Goldilocks, b: Goldilocks) {
        // Canonicalize so each operand is `< p`; the 128-bit product then fits
        // the capacity argument above.
        self.limbs.fmadd::<1, 1>(
            &Limbs::new([a.to_canonical_u64()]),
            &Limbs::new([b.to_canonical_u64()]),
        );
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.limbs.add_assign_trunc::<3>(&other.limbs);
    }

    #[inline]
    fn reduce(self) -> Goldilocks {
        Goldilocks::from_raw(reduce192(self.limbs.0))
    }
}

/// 256-bit deferred-reduction accumulator for [`Goldilocks`] × raw-integer scalars.
#[derive(Clone, Copy, Default)]
pub struct GoldilocksScalarAccumulator {
    limbs: Limbs<4>,
}

impl FieldScalarAccumulator for GoldilocksScalarAccumulator {
    type Field = Goldilocks;

    #[inline(always)]
    fn add(&mut self, value: Goldilocks) {
        self.limbs
            .add_assign_trunc::<1>(&Limbs::new([value.to_canonical_u64()]));
    }

    #[inline(always)]
    fn add_mul_u64(&mut self, value: Goldilocks, scalar: u64) {
        self.limbs.fmadd::<1, 1>(
            &Limbs::new([value.to_canonical_u64()]),
            &Limbs::new([scalar]),
        );
    }

    #[inline(always)]
    fn add_mul_u128(&mut self, value: Goldilocks, scalar: u128) {
        let s = Limbs::<2>::new([scalar as u64, (scalar >> 64) as u64]);
        self.limbs
            .fmadd::<1, 2>(&Limbs::new([value.to_canonical_u64()]), &s);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.limbs.add_assign_trunc::<4>(&other.limbs);
    }

    #[inline]
    fn reduce(self) -> Goldilocks {
        Goldilocks::from_raw(reduce256(self.limbs.0))
    }
}

/// Deferred-reduction accumulator for [`GoldilocksFp3`].
///
/// Holds three independent base accumulators, one per `Fp3` coordinate, fed by
/// the schoolbook `Fp3 × Fp3` cross terms (`x³ = 2`). The dominant Phase-2 inner
/// loop is `Fp3 × base` ([`fmadd_base`](Self::fmadd_base)), which touches each
/// lane once (3 base products) instead of the 9 of a full extension multiply.
#[derive(Clone, Copy, Default)]
pub struct Fp3Accumulator {
    lanes: [GoldilocksAccumulator; 3],
}

impl Fp3Accumulator {
    /// Fast lane: `self += a · b` for a base-field `b` — 3 base `fmadd`s, no
    /// cross terms. `(a0 + a1·x + a2·x²)·b = a0·b + a1·b·x + a2·b·x²`.
    #[inline(always)]
    pub fn fmadd_base(&mut self, a: GoldilocksFp3, b: Goldilocks) {
        let c = a.coeffs();
        self.lanes[0].fmadd(c[0], b);
        self.lanes[1].fmadd(c[1], b);
        self.lanes[2].fmadd(c[2], b);
    }
}

impl FieldAccumulator for Fp3Accumulator {
    type Field = GoldilocksFp3;

    #[inline(always)]
    fn fmadd(&mut self, a: GoldilocksFp3, b: GoldilocksFp3) {
        // (a0+a1x+a2x²)(b0+b1x+b2x²) mod (x³−2):
        //   r0 = a0 b0 + 2(a1 b2 + a2 b1)
        //   r1 = a0 b1 + a1 b0 + 2(a2 b2)
        //   r2 = a0 b2 + a1 b1 + a2 b0
        // The ×2 terms are doubled in-field before fmadd (stays `< 2^64`).
        let a = a.coeffs();
        let b = b.coeffs();
        let (a0, a1, a2) = (a[0], a[1], a[2]);
        let (b0, b1, b2) = (b[0], b[1], b[2]);

        self.lanes[0].fmadd(a0, b0);
        self.lanes[0].fmadd(a1 + a1, b2);
        self.lanes[0].fmadd(a2 + a2, b1);

        self.lanes[1].fmadd(a0, b1);
        self.lanes[1].fmadd(a1, b0);
        self.lanes[1].fmadd(a2 + a2, b2);

        self.lanes[2].fmadd(a0, b2);
        self.lanes[2].fmadd(a1, b1);
        self.lanes[2].fmadd(a2, b0);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.lanes[0].merge(other.lanes[0]);
        self.lanes[1].merge(other.lanes[1]);
        self.lanes[2].merge(other.lanes[2]);
    }

    #[inline]
    fn reduce(self) -> GoldilocksFp3 {
        GoldilocksFp3::new(
            self.lanes[0].reduce(),
            self.lanes[1].reduce(),
            self.lanes[2].reduce(),
        )
    }
}

/// Deferred-reduction accumulator for [`GoldilocksFp3`] × raw-integer scalars.
///
/// `(c0 + c1·x + c2·x²)·n = c0·n + c1·n·x + c2·n·x²`, so each coordinate
/// accumulates independently.
#[derive(Clone, Copy, Default)]
pub struct Fp3ScalarAccumulator {
    lanes: [GoldilocksScalarAccumulator; 3],
}

impl FieldScalarAccumulator for Fp3ScalarAccumulator {
    type Field = GoldilocksFp3;

    #[inline(always)]
    fn add(&mut self, value: GoldilocksFp3) {
        let c = value.coeffs();
        self.lanes[0].add(c[0]);
        self.lanes[1].add(c[1]);
        self.lanes[2].add(c[2]);
    }

    #[inline(always)]
    fn add_mul_u64(&mut self, value: GoldilocksFp3, scalar: u64) {
        let c = value.coeffs();
        self.lanes[0].add_mul_u64(c[0], scalar);
        self.lanes[1].add_mul_u64(c[1], scalar);
        self.lanes[2].add_mul_u64(c[2], scalar);
    }

    #[inline(always)]
    fn add_mul_u128(&mut self, value: GoldilocksFp3, scalar: u128) {
        let c = value.coeffs();
        self.lanes[0].add_mul_u128(c[0], scalar);
        self.lanes[1].add_mul_u128(c[1], scalar);
        self.lanes[2].add_mul_u128(c[2], scalar);
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.lanes[0].merge(other.lanes[0]);
        self.lanes[1].merge(other.lanes[1]);
        self.lanes[2].merge(other.lanes[2]);
    }

    #[inline]
    fn reduce(self) -> GoldilocksFp3 {
        GoldilocksFp3::new(
            self.lanes[0].reduce(),
            self.lanes[1].reduce(),
            self.lanes[2].reduce(),
        )
    }
}
