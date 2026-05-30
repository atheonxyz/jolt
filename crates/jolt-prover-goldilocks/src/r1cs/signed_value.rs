//! Derived signed value for a sign+magnitude operand (see `../../LIMBED_R1CS.md`,
//! "Dual-use operands"). `RIGHT_INSTRUCTION_INPUT` is stored as sign + unsigned
//! magnitude limbs `(sign, mlo, mhi)` for the MUL schoolbook, but the
//! lookup-operand eq-constraints need its *signed value* used **linearly**. This
//! derives `value = (1 − 2·sign)·(mlo + 2^32·mhi)` as a degree-2 relation:
//!
//! ```text
//! sign · mlo = sign_mlo            (product row)
//! sign · mhi = sign_mhi            (product row)
//! value = mlo + 2^32·mhi − 2·sign_mlo − 2^33·sign_mhi   (linear row)
//! ```
//!
//! so downstream eq-constraints reference `value` with degree-1, keeping the outer
//! Spartan sumcheck degree-2. Soundness needs `sign` Boolean and `mlo, mhi < 2^32`
//! (M6 range checks).

use jolt_field::Field;
use jolt_r1cs::constraint::SparseRow;

/// `z`-indices for one signed-value derivation. Generic so it composes into the
/// full RV64 layout.
#[derive(Clone, Copy, Debug)]
pub struct SignedValueVars {
    pub const_one: usize,
    pub sign: usize,
    pub mlo: usize,
    pub mhi: usize,
    /// Intermediates `sign·mlo`, `sign·mhi`.
    pub sign_mlo: usize,
    pub sign_mhi: usize,
    /// The derived signed value used linearly elsewhere.
    pub value: usize,
}

/// Number of constraint rows the derivation contributes.
pub const NUM_SIGNED_VALUE_ROWS: usize = 3;

/// Append the 3 derivation rows to (a, b, c).
pub fn push_signed_value_derivation<F: Field>(
    v: &SignedValueVars,
    a: &mut Vec<SparseRow<F>>,
    b: &mut Vec<SparseRow<F>>,
    c: &mut Vec<SparseRow<F>>,
) {
    let one = F::from_u64(1);
    let neg_one = F::from_i64(-1);
    let neg_two32 = F::from_i64(-(1i64 << 32));
    let two = F::from_u64(2);
    let two33 = F::from_u64(1u64 << 33);

    // sign · mlo = sign_mlo
    a.push(vec![(v.sign, one)]);
    b.push(vec![(v.mlo, one)]);
    c.push(vec![(v.sign_mlo, one)]);
    // sign · mhi = sign_mhi
    a.push(vec![(v.sign, one)]);
    b.push(vec![(v.mhi, one)]);
    c.push(vec![(v.sign_mhi, one)]);
    // 1 · (value − mlo − 2^32·mhi + 2·sign_mlo + 2^33·sign_mhi) = 0
    a.push(vec![(v.const_one, one)]);
    b.push(vec![
        (v.value, one),
        (v.mlo, neg_one),
        (v.mhi, neg_two32),
        (v.sign_mlo, two),
        (v.sign_mhi, two33),
    ]);
    c.push(Vec::new());
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::Goldilocks;
    use jolt_field::Field;
    use jolt_r1cs::ConstraintMatrices;

    fn vars() -> SignedValueVars {
        SignedValueVars {
            const_one: 0,
            sign: 1,
            mlo: 2,
            mhi: 3,
            sign_mlo: 4,
            sign_mhi: 5,
            value: 6,
        }
    }
    const NUM_VARS: usize = 7;

    fn matrices() -> ConstraintMatrices<Goldilocks> {
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        push_signed_value_derivation(&vars(), &mut a, &mut b, &mut c);
        ConstraintMatrices::new(NUM_SIGNED_VALUE_ROWS, NUM_VARS, a, b, c)
    }

    fn witness(mag: u64, neg: bool) -> Vec<Goldilocks> {
        let mask = 0xFFFF_FFFFu64;
        let (mlo, mhi) = (mag & mask, mag >> 32);
        let s = u64::from(neg);
        let v = vars();
        let mut w = vec![Goldilocks::from_u64(0); NUM_VARS];
        w[v.const_one] = Goldilocks::from_u64(1);
        w[v.sign] = Goldilocks::from_u64(s);
        w[v.mlo] = Goldilocks::from_u64(mlo);
        w[v.mhi] = Goldilocks::from_u64(mhi);
        w[v.sign_mlo] = Goldilocks::from_u64(if neg { mlo } else { 0 });
        w[v.sign_mhi] = Goldilocks::from_u64(if neg { mhi } else { 0 });
        // value = (1 − 2·sign)·mag
        w[v.value] = if neg {
            Goldilocks::from_i128(-(i128::from(mag)))
        } else {
            Goldilocks::from_u64(mag)
        };
        w
    }

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

    #[test]
    fn honest_signed_values_satisfy() {
        let m = matrices();
        for &mag in &[0u64, 1, u32::MAX as u64, 1 << 32, u64::MAX, u64::MAX - 1] {
            for neg in [false, true] {
                m.check_witness(&witness(mag, neg))
                    .expect("honest signed value must satisfy");
            }
        }
        let mut r = Rng(0x5160_4ED5_4A1D_0000);
        for _ in 0..3000 {
            let mag = r.next();
            m.check_witness(&witness(mag, r.next() & 1 == 1))
                .expect("random honest signed value must satisfy");
        }
    }

    #[test]
    fn tampered_value_or_sign_is_rejected() {
        let m = matrices();
        let v = vars();
        let mut w = witness(0x1234_5678_9abc_def0, true);
        w[v.value] += Goldilocks::from_u64(1);
        assert!(
            m.check_witness(&w).is_err(),
            "tampered value must be rejected"
        );

        // Claiming positive for a negative-magnitude value: sign_mlo/sign_mhi=0
        // but value still negative — the linear row breaks.
        let mut w2 = witness(42, true);
        w2[v.sign] = Goldilocks::from_u64(0);
        w2[v.sign_mlo] = Goldilocks::from_u64(0);
        w2[v.sign_mhi] = Goldilocks::from_u64(0);
        assert!(m.check_witness(&w2).is_err(), "sign flip must be rejected");
    }
}
