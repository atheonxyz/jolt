//! WHIR cross-check: the hand-coded Montgomery-free `Goldilocks` / `GoldilocksFp3`
//! arithmetic must agree, op-for-op, with WHIR's arkworks `Field64` / `Field64_3`
//! (Montgomery `Fp64` / `Fp3`). Both represent the same field — base `p = 2^64 −
//! 2^32 + 1`, cubic extension with nonresidue `2` (`x^3 = 2`, see
//! `whir/src/algebra/fields.rs`: `F3Config64::NONRESIDUE = 2`). This is the
//! external oracle that backs Phase 1's field correctness from the *commit* side,
//! complementing the in-crate `num-bigint` oracle in `jolt-field`.

#![cfg(feature = "goldilocks")]
#![expect(clippy::unwrap_used)]

use ark_ff::Field as ArkField;
use whir::algebra::fields::{Field64, Field64_3};

use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
use jolt_field::Field;
use jolt_whir::convert::{to_field64, to_field64_3};

/// Deterministic splitmix64 (no rng dependency), same generator as `sanity.rs`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn ark_base(x: Goldilocks) -> Field64 {
    Field64::from(x.to_u64().unwrap())
}

#[test]
fn base_arithmetic_matches_arkworks_field64() {
    // Edge values plus random draws across the full u64 range (exercises the
    // non-canonical `[p, 2^64)` aliasing band via `from_u64`'s reduction).
    let mut rng = Rng(0xC0DE_F00D_1234_5678);
    let edges: [u64; 6] = [
        0,
        1,
        0xFFFF_FFFF_0000_0000, // p - 1
        0xFFFF_FFFF_0000_0001, // p (aliases to 0)
        0xFFFF_FFFF_FFFF_FFFF, // 2^64 - 1 (in the aliasing band)
        0x0000_0001_0000_0000, // 2^32
    ];

    let mut samples: Vec<u64> = edges.to_vec();
    for _ in 0..2_000 {
        samples.push(rng.next());
    }

    for (i, &a_raw) in samples.iter().enumerate() {
        for &b_raw in samples.iter().step_by(7) {
            let a = Goldilocks::from_u64(a_raw);
            let b = Goldilocks::from_u64(b_raw);
            let aa = Field64::from(a_raw);
            let bb = Field64::from(b_raw);

            assert_eq!(to_field64(a + b), aa + bb, "add mismatch @ {i}");
            assert_eq!(to_field64(a - b), aa - bb, "sub mismatch @ {i}");
            assert_eq!(to_field64(a * b), aa * bb, "mul mismatch @ {i}");
            assert_eq!(to_field64(-a), -aa, "neg mismatch @ {i}");
            assert_eq!(to_field64(a.square()), aa.square(), "square mismatch @ {i}");
            // Convert-then-compare also catches any non-canonical leak: `to_field64`
            // must canonicalize so equal field values map to equal `Field64`.
            assert_eq!(ark_base(a), aa, "embed mismatch @ {i}");
        }
    }
}

#[test]
fn base_inverse_matches_arkworks_field64() {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    for _ in 0..1_000 {
        let a = Goldilocks::from_u64(rng.next());
        if a == Goldilocks::from_u64(0) {
            continue;
        }
        let inv = a.inverse().unwrap();
        let aa = Field64::from(a.to_u64().unwrap());
        let ark_inv = aa.inverse().unwrap();
        assert_eq!(to_field64(inv), ark_inv, "inverse mismatch");
        assert_eq!(a * inv, Goldilocks::from_u64(1), "a·a⁻¹ ≠ 1");
    }
}

#[test]
fn fp3_arithmetic_matches_arkworks_field64_3() {
    let mut rng = Rng(0xFEED_FACE_DEAD_BEEF);
    let mk = |r: &mut Rng| {
        let c = [r.next(), r.next(), r.next()];
        let ours = GoldilocksFp3::new(
            Goldilocks::from_u64(c[0]),
            Goldilocks::from_u64(c[1]),
            Goldilocks::from_u64(c[2]),
        );
        let ark = Field64_3::new(
            Field64::from(c[0]),
            Field64::from(c[1]),
            Field64::from(c[2]),
        );
        (ours, ark)
    };

    for _ in 0..2_000 {
        let (a, aa) = mk(&mut rng);
        let (b, bb) = mk(&mut rng);

        assert_eq!(to_field64_3(a + b), aa + bb, "fp3 add mismatch");
        assert_eq!(to_field64_3(a - b), aa - bb, "fp3 sub mismatch");
        assert_eq!(to_field64_3(a * b), aa * bb, "fp3 mul mismatch");
        assert_eq!(to_field64_3(-a), -aa, "fp3 neg mismatch");
        assert_eq!(to_field64_3(a.square()), aa.square(), "fp3 square mismatch");

        // `mul_by_base` (the Phase-2 hot path) == multiply by the base element
        // embedded into the extension.
        let s_raw = rng.next();
        let s = Goldilocks::from_u64(s_raw);
        let ark_embed = Field64_3::new(
            Field64::from(s_raw),
            Field64::from(0u64),
            Field64::from(0u64),
        );
        assert_eq!(
            to_field64_3(a.mul_by_base(s)),
            aa * ark_embed,
            "fp3 mul_by_base mismatch"
        );
    }
}

#[test]
fn fp3_inverse_matches_arkworks_field64_3() {
    let mut rng = Rng(0x0BAD_C0DE_F00D_2222);
    let zero = GoldilocksFp3::new(
        Goldilocks::from_u64(0),
        Goldilocks::from_u64(0),
        Goldilocks::from_u64(0),
    );
    for _ in 0..1_000 {
        let c = [rng.next(), rng.next(), rng.next()];
        let a = GoldilocksFp3::new(
            Goldilocks::from_u64(c[0]),
            Goldilocks::from_u64(c[1]),
            Goldilocks::from_u64(c[2]),
        );
        if a == zero {
            continue;
        }
        let aa = Field64_3::new(
            Field64::from(c[0]),
            Field64::from(c[1]),
            Field64::from(c[2]),
        );
        let inv = a.inverse().unwrap();
        assert_eq!(
            to_field64_3(inv),
            aa.inverse().unwrap(),
            "fp3 inverse mismatch"
        );
    }
}
