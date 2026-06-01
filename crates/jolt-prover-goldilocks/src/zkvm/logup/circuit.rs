//! Fan-in-2 fractional-add circuit — the GKR pyramid for one LogUp\* fractional sum.
//!
//! Ported from `crates/whir-pcs-bench/src/gkr.rs::{Circuit, Level, build_circuit}`. Each gate holds
//! a `(numerator, denominator)` fraction; a level-`k` gate `i` combines its two level-`(k+1)`
//! children `(2i, 2i+1)` via fractional addition `n/d ⊕ n'/d' = (n·d' + n'·d)/(d·d')`. The root is
//! the full fractional sum `Σ_leaf num/denom`.
//!
//! `levels[0]` is the root (length 1); `levels[log_size]` are the leaves (length `2^log_size`).
//! The per-layer GKR sumcheck (see `layer`) reduces a claim at `levels[k]` to a claim at
//! `levels[k+1]`. Children pair on the **low bit** (`2i`/`2i+1`), matching the `LowToHigh` binding
//! convention used throughout the framework.
//!
//! Perf deferred (matches the prototype's deferred items): the AoS `PairVals` packing and the
//! streaming-pyramid memory optimization for `K = 2³²` (design §13-Q4). This builds the full pyramid
//! densely — fine for the decoupled tests; M8 wires the real-size families.

use jolt_field::Field;
use rayon::prelude::*;

/// A fan-in-2 fractional-add circuit, stored top-down: `levels[0]` is the root, `levels[L]` the
/// leaves. Each gate is a `(numerator, denominator)` pair.
#[derive(Clone, Debug)]
pub struct Circuit<F: Field> {
    levels: Vec<Vec<(F, F)>>,
}

impl<F: Field> Circuit<F> {
    /// Build the pyramid from `2^log_size` leaf numerators/denominators.
    pub fn build(leaves_num: Vec<F>, leaves_denom: Vec<F>, log_size: usize) -> Self {
        assert_eq!(leaves_num.len(), 1usize << log_size);
        assert_eq!(leaves_denom.len(), 1usize << log_size);

        let mut levels: Vec<Vec<(F, F)>> = Vec::with_capacity(log_size + 1);
        let leaves: Vec<(F, F)> = leaves_num
            .into_par_iter()
            .zip(leaves_denom.into_par_iter())
            .collect();
        levels.push(leaves);

        loop {
            let prev = &levels[levels.len() - 1];
            if prev.len() <= 1 {
                break;
            }
            let n = prev.len() / 2;
            let next: Vec<(F, F)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let (nl, dl) = prev[2 * i];
                    let (nr, dr) = prev[2 * i + 1];
                    (nl * dr + nr * dl, dl * dr)
                })
                .collect();
            levels.push(next);
        }

        // Built bottom-up (leaves at index 0); reverse so levels[0] is the root.
        levels.reverse();
        Self { levels }
    }

    /// The root fraction `(numerator, denominator) = Σ_leaf num/denom`.
    #[inline]
    pub fn root(&self) -> (F, F) {
        self.levels[0][0]
    }

    /// The gates at level `k` (`levels[0]` = root, `levels[log_size]` = leaves).
    #[inline]
    pub fn level(&self, k: usize) -> &[(F, F)] {
        &self.levels[k]
    }

    /// Depth `L`: `levels[L]` are the leaves.
    #[inline]
    pub fn log_size(&self) -> usize {
        self.levels.len() - 1
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;

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

    /// The root fraction equals the direct sum `Σ_i num_i / denom_i` (computed in the field).
    #[test]
    fn root_is_fractional_sum() {
        let mut rng = Rng(0xC1);
        for log_size in 0..=6 {
            let n = 1usize << log_size;
            let num: Vec<F> = (0..n).map(|_| F::from_u64(rng.next())).collect();
            let denom: Vec<F> = (0..n).map(|_| F::from_u64(rng.next() | 1)).collect();

            let direct = num
                .iter()
                .zip(denom.iter())
                .fold(F::from_u64(0), |acc, (&nu, &de)| {
                    acc + nu * de.inverse().unwrap()
                });

            let circuit = Circuit::build(num, denom, log_size);
            let (rn, rd) = circuit.root();
            assert_eq!(
                rn * rd.inverse().unwrap(),
                direct,
                "root fraction at log_size={log_size}"
            );
            assert_eq!(circuit.log_size(), log_size);
            assert_eq!(circuit.level(log_size).len(), n);
        }
    }

    /// Two circuits whose leaf fractions sum to the same value satisfy the histogram identity
    /// `N_A·D_B == N_B·D_A` at their roots (the GKR's root cross-check).
    #[test]
    fn equal_sums_satisfy_histogram_identity() {
        // Circuit A: {1/2, 1/3}. Circuit B: a single leaf {5/6} = 1/2 + 1/3.
        let a = Circuit::build(
            vec![F::from_u64(1), F::from_u64(1)],
            vec![F::from_u64(2), F::from_u64(3)],
            1,
        );
        let b = Circuit::build(vec![F::from_u64(5)], vec![F::from_u64(6)], 0);
        let (na, da) = a.root();
        let (nb, db) = b.root();
        assert_eq!(
            na * db,
            nb * da,
            "histogram identity for equal fractional sums"
        );
    }
}
