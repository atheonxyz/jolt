//! The 4-limb MUL schoolbook — the soundness-critical core of the limbed RV64
//! R1CS (see `../../LIMBED_R1CS.md`). The single BN254 row `Product = Left × Right`
//! becomes, over Goldilocks (where a 128-bit product can't be one field element),
//! a 2-limb × 2-limb = 4-limb schoolbook on **unsigned magnitudes** plus a sign
//! relation:
//!
//! ```text
//! magnitude(Left)  = Llo + 2^32·Lhi      magnitude(Right) = Rlo + 2^32·Rhi
//! partial products q0=Llo·Rlo  q1=Llo·Rhi  q2=Lhi·Rlo  q3=Lhi·Rhi   (each < p)
//! carry chain:  q0            = P0 + 2^32·c0
//!               q1 + q2 + c0  = P1 + 2^32·c1
//!               q3 + c1       = P2 + 2^32·c2
//!                               P3 = c2
//! sign:         Product.sign  = Left.sign ⊕ Right.sign
//! ```
//!
//! Emitted as R1CS rows: 4 product rows (`q_i = limbᵢ·limbⱼ`, degree-2), 4 linear
//! carry rows, 1 product row + 1 linear row for the XOR sign — 10 rows, all
//! degree ≤ 2. The partial products `q_i ≤ (2^32−1)² < p` fit one field element,
//! so the field equalities hold exactly even though the carry sums exceed `p`
//! (both sides reduce equally). Soundness additionally requires the M6 range
//! checks: `P0..P3, c0, c2 < 2^32`, `c1 < 2^33`, signs Boolean.

use jolt_field::Field;
use jolt_r1cs::constraint::SparseRow;

/// `z`-indices of the variables the MUL schoolbook touches. Generic over the
/// surrounding layout so it can be reused inside the full RV64 constraint set.
#[derive(Clone, Copy, Debug)]
pub struct MulVars {
    pub const_one: usize,
    /// Left operand magnitude limbs + sign.
    pub left_lo: usize,
    pub left_hi: usize,
    pub left_sign: usize,
    /// Right operand magnitude limbs + sign.
    pub right_lo: usize,
    pub right_hi: usize,
    pub right_sign: usize,
    /// Product magnitude limbs (P0..P3) + sign.
    pub p0: usize,
    pub p1: usize,
    pub p2: usize,
    pub p3: usize,
    pub product_sign: usize,
    /// Schoolbook intermediates: partial products and carries.
    pub q0: usize,
    pub q1: usize,
    pub q2: usize,
    pub q3: usize,
    pub c0: usize,
    pub c1: usize,
    pub c2: usize,
    /// `left_sign · right_sign` (for the XOR sign relation).
    pub sign_prod: usize,
}

/// Number of constraint rows the schoolbook contributes.
pub const NUM_MUL_ROWS: usize = 10;

/// Append the 10 MUL-schoolbook rows to the (a, b, c) row lists.
pub fn push_mul_constraints<F: Field>(
    v: &MulVars,
    a: &mut Vec<SparseRow<F>>,
    b: &mut Vec<SparseRow<F>>,
    c: &mut Vec<SparseRow<F>>,
) {
    let neg_one = F::from_i64(-1);
    let neg_two32 = F::from_i64(-(1i64 << 32));
    let one = F::from_u64(1);

    // Product rows q_i = limb · limb  (A·B = C).
    let prod = |x: usize,
                y: usize,
                q: usize,
                a: &mut Vec<SparseRow<F>>,
                b: &mut Vec<SparseRow<F>>,
                c: &mut Vec<SparseRow<F>>| {
        a.push(vec![(x, one)]);
        b.push(vec![(y, one)]);
        c.push(vec![(q, one)]);
    };
    prod(v.left_lo, v.right_lo, v.q0, a, b, c);
    prod(v.left_lo, v.right_hi, v.q1, a, b, c);
    prod(v.left_hi, v.right_lo, v.q2, a, b, c);
    prod(v.left_hi, v.right_hi, v.q3, a, b, c);

    // Linear carry rows: 1 · (lhs − rhs) = 0.
    let lin = |row: Vec<(usize, F)>,
               a: &mut Vec<SparseRow<F>>,
               b: &mut Vec<SparseRow<F>>,
               c: &mut Vec<SparseRow<F>>| {
        a.push(vec![(v.const_one, one)]);
        b.push(row);
        c.push(Vec::new());
    };
    // q0 = P0 + 2^32·c0
    lin(
        vec![(v.q0, one), (v.p0, neg_one), (v.c0, neg_two32)],
        a,
        b,
        c,
    );
    // q1 + q2 + c0 = P1 + 2^32·c1
    lin(
        vec![
            (v.q1, one),
            (v.q2, one),
            (v.c0, one),
            (v.p1, neg_one),
            (v.c1, neg_two32),
        ],
        a,
        b,
        c,
    );
    // q3 + c1 = P2 + 2^32·c2
    lin(
        vec![(v.q3, one), (v.c1, one), (v.p2, neg_one), (v.c2, neg_two32)],
        a,
        b,
        c,
    );
    // c2 = P3
    lin(vec![(v.c2, one), (v.p3, neg_one)], a, b, c);

    // Sign: product_sign = left_sign ⊕ right_sign = s_l + s_r − 2·(s_l·s_r).
    // (1) s_l · s_r = sign_prod
    a.push(vec![(v.left_sign, one)]);
    b.push(vec![(v.right_sign, one)]);
    c.push(vec![(v.sign_prod, one)]);
    // (2) 1 · (product_sign − s_l − s_r + 2·sign_prod) = 0
    a.push(vec![(v.const_one, one)]);
    b.push(vec![
        (v.product_sign, one),
        (v.left_sign, neg_one),
        (v.right_sign, neg_one),
        (v.sign_prod, F::from_u64(2)),
    ]);
    c.push(Vec::new());
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::decompose::limbs_to_u128;
    use jolt_field::goldilocks::Goldilocks;
    use jolt_field::Field;
    use jolt_r1cs::ConstraintMatrices;

    /// Minimal standalone layout for the schoolbook (just the MUL vars).
    fn vars() -> MulVars {
        MulVars {
            const_one: 0,
            left_lo: 1,
            left_hi: 2,
            left_sign: 3,
            right_lo: 4,
            right_hi: 5,
            right_sign: 6,
            p0: 7,
            p1: 8,
            p2: 9,
            p3: 10,
            product_sign: 11,
            q0: 12,
            q1: 13,
            q2: 14,
            q3: 15,
            c0: 16,
            c1: 17,
            c2: 18,
            sign_prod: 19,
        }
    }
    const NUM_VARS: usize = 20;

    fn matrices() -> ConstraintMatrices<Goldilocks> {
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        push_mul_constraints(&vars(), &mut a, &mut b, &mut c);
        ConstraintMatrices::new(NUM_MUL_ROWS, NUM_VARS, a, b, c)
    }

    /// Build an honest satisfying witness for `left_mag * right_mag` with the
    /// given signs.
    fn witness(left_mag: u64, right_mag: u64, ls: bool, rs: bool) -> Vec<Goldilocks> {
        let mask = 0xFFFF_FFFFu64;
        let (llo, lhi) = (left_mag & mask, left_mag >> 32);
        let (rlo, rhi) = (right_mag & mask, right_mag >> 32);
        let q0 = u128::from(llo) * u128::from(rlo);
        let q1 = u128::from(llo) * u128::from(rhi);
        let q2 = u128::from(lhi) * u128::from(rlo);
        let q3 = u128::from(lhi) * u128::from(rhi);
        let s0 = q0;
        let (p0, c0) = (s0 & u128::from(mask), s0 >> 32);
        let s1 = q1 + q2 + c0;
        let (p1, c1) = (s1 & u128::from(mask), s1 >> 32);
        let s2 = q3 + c1;
        let (p2, c2) = (s2 & u128::from(mask), s2 >> 32);
        let p3 = c2;
        let sign_prod = u64::from(ls && rs);
        let product_sign = u64::from(ls ^ rs);

        let g = Goldilocks::from_u128;
        let mut w = vec![Goldilocks::from_u64(0); NUM_VARS];
        let v = vars();
        w[v.const_one] = Goldilocks::from_u64(1);
        w[v.left_lo] = Goldilocks::from_u64(llo);
        w[v.left_hi] = Goldilocks::from_u64(lhi);
        w[v.left_sign] = Goldilocks::from_u64(u64::from(ls));
        w[v.right_lo] = Goldilocks::from_u64(rlo);
        w[v.right_hi] = Goldilocks::from_u64(rhi);
        w[v.right_sign] = Goldilocks::from_u64(u64::from(rs));
        w[v.p0] = g(p0);
        w[v.p1] = g(p1);
        w[v.p2] = g(p2);
        w[v.p3] = g(p3);
        w[v.product_sign] = Goldilocks::from_u64(product_sign);
        w[v.q0] = g(q0);
        w[v.q1] = g(q1);
        w[v.q2] = g(q2);
        w[v.q3] = g(q3);
        w[v.c0] = g(c0);
        w[v.c1] = g(c1);
        w[v.c2] = g(c2);
        w[v.sign_prod] = Goldilocks::from_u64(sign_prod);
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
    fn honest_products_satisfy() {
        let m = matrices();
        let mut r = Rng(0x4D55_4C5F_5345_4544);
        // edges + random full-width 64×64 products
        let edges = [0u64, 1, u32::MAX as u64, 1 << 32, u64::MAX, u64::MAX - 1];
        for &a in &edges {
            for &b in &edges {
                for (ls, rs) in [(false, false), (true, false), (false, true), (true, true)] {
                    m.check_witness(&witness(a, b, ls, rs))
                        .expect("honest product must satisfy");
                }
            }
        }
        for _ in 0..3000 {
            let (a, b) = (r.next(), r.next());
            m.check_witness(&witness(a, b, r.next() & 1 == 1, r.next() & 1 == 1))
                .expect("random honest product must satisfy");
        }
    }

    #[test]
    fn product_limbs_recompose_to_true_product() {
        let mut r = Rng(0x00C0_FFEE);
        for _ in 0..3000 {
            let (a, b) = (r.next(), r.next());
            let w = witness(a, b, false, false);
            let v = vars();
            let p = [w[v.p0], w[v.p1], w[v.p2], w[v.p3]];
            assert_eq!(limbs_to_u128(p), u128::from(a) * u128::from(b));
        }
    }

    #[test]
    fn tampered_product_is_rejected() {
        let m = matrices();
        let v = vars();
        let mut w = witness(0x1234_5678_9abc_def0, 0x0fed_cba9_8765_4321, true, false);
        // Corrupt one product magnitude limb.
        w[v.p1] += Goldilocks::from_u64(1);
        assert!(m.check_witness(&w).is_err(), "tampered P1 must be rejected");

        let mut w2 = witness(7, 9, false, false);
        w2[v.product_sign] = Goldilocks::from_u64(1); // 7*9 is positive; sign must be 0
        assert!(m.check_witness(&w2).is_err(), "wrong sign must be rejected");
    }
}
