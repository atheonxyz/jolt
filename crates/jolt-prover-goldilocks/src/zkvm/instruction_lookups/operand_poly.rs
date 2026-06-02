//! Operand-extraction multilinear extension for the instruction-lookup read-raf (P10 / IL-1).
//!
//! A lookup index `k` is the bit-**interleaving** of the two operands (`interleave_bits(left, right)`,
//! see [`jolt_lookup_tables::interleave_bits`]): the right operand occupies the even bit positions,
//! the left operand the odd ones. [`OperandPolynomial`] is the MLE that, evaluated at an index point
//! `r_addr` (length `2·m`), returns the left or right operand value — the verifier's operand eval in
//! the read-raf `expected_output_claim` (`RafVal = (1−raf)·(left + γ·right) + raf·γ·identity`).
//!
//! Ported from jolt-core `poly/identity_poly.rs::OperandPolynomial` over `jolt_field::Field`
//! (`JoltField → Field`, no `F::Challenge`). The identity-index MLE is reused as-is from
//! [`jolt_poly::IdentityPolynomial`]; the prefix/suffix decomposition of these polynomials (the
//! `prefix_polynomial`/`suffix_mle` halves) lands with the RAF Q-aggregator (IL-3).

use jolt_field::Field;

/// Which operand [`OperandPolynomial`] extracts from the interleaved index point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandSide {
    Left,
    Right,
}

/// The operand-extraction MLE over an interleaved index point of `num_vars = 2·m` variables:
/// `Right = Σ_{i<m} r[2i]·2^{m-1-i}`, `Left = Σ_{i<m} r[2i+1]·2^{m-1-i}`.
#[derive(Clone, Debug)]
pub struct OperandPolynomial {
    num_vars: usize,
    side: OperandSide,
}

impl OperandPolynomial {
    pub fn new(num_vars: usize, side: OperandSide) -> Self {
        debug_assert!(
            num_vars.is_multiple_of(2),
            "num_vars must be even (interleaved operands)"
        );
        Self { num_vars, side }
    }

    /// Evaluate the operand MLE at index point `r` (length `num_vars`).
    pub fn evaluate<F: Field>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), self.num_vars);
        let m = self.num_vars / 2;
        // Right operand = even bit positions (offset 0); left = odd positions (offset 1).
        let offset = match self.side {
            OperandSide::Right => 0,
            OperandSide::Left => 1,
        };
        (0..m).fold(F::from_u64(0), |acc, i| {
            acc + r[2 * i + offset] * F::from_u128(1u128 << (m - 1 - i))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_lookup_tables::uninterleave_bits;

    /// On the Boolean hypercube the operand MLE equals the de-interleaved operand value — matches
    /// jolt-core `operand_poly_boolean_hypercube` (big-endian point via the `reverse()`).
    #[test]
    fn operand_evaluate_matches_uninterleave() {
        const NUM_VARS: usize = 8;
        let right = OperandPolynomial::new(NUM_VARS, OperandSide::Right);
        let left = OperandPolynomial::new(NUM_VARS, OperandSide::Left);

        for i in 0u128..(1 << NUM_VARS) {
            let mut point = vec![F::from_u64(0); NUM_VARS];
            for (j, slot) in point.iter_mut().enumerate() {
                if (i >> j) & 1 == 1 {
                    *slot = F::from_u64(1);
                }
            }
            point.reverse();

            // `uninterleave_bits` returns (odd-k-position bits, even-k-position bits); in the
            // MSB-first point the even point-indices (offset 0 = Right) hold the former, the odd
            // indices (offset 1 = Left) the latter.
            let (u0, u1) = uninterleave_bits(i);
            assert_eq!(
                right.evaluate(&point),
                F::from_u64(u0),
                "offset-0 operand at index {i}"
            );
            assert_eq!(
                left.evaluate(&point),
                F::from_u64(u1),
                "offset-1 operand at index {i}"
            );
        }
    }
}
