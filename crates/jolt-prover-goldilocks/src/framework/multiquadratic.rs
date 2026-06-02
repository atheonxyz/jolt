//! Multiquadratic polynomial over the `{0, 1, ∞}^num_vars` grid (base-3 layout, `z_0`
//! least-significant / fastest-varying) — the transient helper that compresses streaming-sumcheck
//! messages in Spartan's outer / product univariate-skip rounds. The `∞` coordinate stores the
//! slope (finite difference), so a pointwise product of two linear `{0,1,∞}` grids yields the
//! `{0,1,∞}` evaluations of the quadratic product `Az·Bz`.
//!
//! Vendored from jolt-core `poly/multiquadratic_poly.rs` (the parity oracle), retargeted to the lean
//! [`jolt_field::Field`]: `JoltField → Field`, `r: F::Challenge → r: F` (`C = F = Fp3`), the
//! `PolynomialBinding` trait impl becomes inherent methods, and the `Allocative`/`tracing` derives
//! are dropped. The arithmetic is **field-agnostic** (`+ − ×`, `r·(r−1)`), so it ports verbatim.

use jolt_field::Field;
use jolt_poly::BindingOrder;
use rayon::prelude::*;

/// Multiquadratic polynomial by its evaluations on `{0, 1, ∞}^num_vars` in base-3 layout
/// (`z_0` least-significant / fastest-varying).
#[derive(Clone, Debug)]
pub struct MultiquadraticPolynomial<F: Field> {
    num_vars: usize,
    evals: Vec<F>,
}

impl<F: Field> MultiquadraticPolynomial<F> {
    /// Construct from the full grid of evaluations (base-3 layout, `z_0` least-significant). The
    /// caller guarantees `evals.len() == 3^num_vars`.
    pub fn new(num_vars: usize, evals: Vec<F>) -> Self {
        let expected_len = 3usize.pow(num_vars as u32);
        debug_assert!(
            evals.len() == expected_len,
            "MultiquadraticPolynomial: expected {expected_len} evals, got {}",
            evals.len()
        );
        Self { num_vars, evals }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Underlying evaluations on `{0, 1, ∞}^num_vars`.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Expand evaluations of a degree-1 multilinear over `{0,1}^dim` to the multiquadratic grid over
    /// `{0,1,∞}^dim`. `input` is length `2^dim` (Boolean hypercube, last variable fastest);
    /// `output`/`tmp` are length `3^dim`. Each 1-D slice `(f0, f1)` becomes `(f0, f1, f1 − f0)` —
    /// the `∞` slot stores the slope.
    #[inline(always)]
    pub fn expand_linear_grid_to_multiquadratic(
        input: &[F],
        output: &mut [F],
        tmp: &mut [F],
        dim: usize,
    ) {
        let in_size = 1usize << dim;
        let out_size = 3usize.pow(dim as u32);

        debug_assert_eq!(input.len(), in_size);
        debug_assert_eq!(output.len(), out_size);
        debug_assert_eq!(tmp.len(), out_size);

        match dim {
            0 => {
                output[0] = input[0];
                return;
            }
            1 => {
                Self::expand_linear_dim1(input, output);
                return;
            }
            2 => {
                Self::expand_linear_dim2(input, output);
                return;
            }
            3 => {
                Self::expand_linear_dim3(input, output);
                return;
            }
            _ => {}
        }

        let (mut cur, mut next) = if dim % 2 == 1 {
            tmp[..input.len()].copy_from_slice(input);
            (tmp, output)
        } else {
            output[..input.len()].copy_from_slice(input);
            (output, tmp)
        };

        let mut in_stride = 1usize;
        let mut out_stride = 1usize;
        let mut blocks = 1 << (dim - 1);

        assert_eq!(cur.len(), out_size);
        assert_eq!(next.len(), out_size);
        assert_eq!(input.len(), in_size);

        for _ in 0..dim {
            for b in 0..blocks {
                let in_off = b * 2 * in_stride;
                let out_off = b * 3 * out_stride;

                for j in 0..in_stride {
                    let f0 = cur[in_off + j];
                    let f1 = cur[in_off + in_stride + j];
                    next[out_off + j] = f0;
                    next[out_off + out_stride + j] = f1;
                    next[out_off + 2 * out_stride + j] = f1 - f0;
                }
            }
            std::mem::swap(&mut cur, &mut next);
            in_stride *= 3;
            out_stride *= 3;
            blocks /= 2;
        }
    }

    #[inline(always)]
    fn expand_linear_dim1(input: &[F], output: &mut [F]) {
        debug_assert_eq!(input.len(), 2);
        debug_assert_eq!(output.len(), 3);
        let f0 = input[0];
        let f1 = input[1];
        output[0] = f0;
        output[1] = f1;
        output[2] = f1 - f0;
    }

    #[inline(always)]
    fn expand_linear_dim2(input: &[F], output: &mut [F]) {
        debug_assert_eq!(input.len(), 4);
        debug_assert_eq!(output.len(), 9);
        let f00 = input[0];
        let f01 = input[1];
        let f10 = input[2];
        let f11 = input[3];

        let a00 = f00;
        let a01 = f01;
        let a0_inf = f01 - f00;
        let a10 = f10;
        let a11 = f11;
        let a1_inf = f11 - f10;

        let inf0 = a10 - a00;
        let inf1 = a11 - a01;
        let inf_inf = a1_inf - a0_inf;

        output[0] = a00;
        output[1] = a01;
        output[2] = a0_inf;
        output[3] = a10;
        output[4] = a11;
        output[5] = a1_inf;
        output[6] = inf0;
        output[7] = inf1;
        output[8] = inf_inf;
    }

    #[inline(always)]
    fn expand_linear_dim3(input: &[F], output: &mut [F]) {
        debug_assert_eq!(input.len(), 8);
        debug_assert_eq!(output.len(), 27);
        let f000 = input[0];
        let f001 = input[1];
        let f010 = input[2];
        let f011 = input[3];
        let f100 = input[4];
        let f101 = input[5];
        let f110 = input[6];
        let f111 = input[7];

        let g000 = f000;
        let g001 = f001;
        let g00_inf = f001 - f000;
        let g010 = f010;
        let g011 = f011;
        let g01_inf = f011 - f010;
        let g100 = f100;
        let g101 = f101;
        let g10_inf = f101 - f100;
        let g110 = f110;
        let g111 = f111;
        let g11_inf = f111 - f110;

        let h0_0_0 = g000;
        let h0_1_0 = g010;
        let h0_inf_0 = g010 - g000;
        let h0_0_1 = g001;
        let h0_1_1 = g011;
        let h0_inf_1 = g011 - g001;
        let h0_0_inf = g00_inf;
        let h0_1_inf = g01_inf;
        let h0_inf_inf = g01_inf - g00_inf;

        let h1_0_0 = g100;
        let h1_1_0 = g110;
        let h1_inf_0 = g110 - g100;
        let h1_0_1 = g101;
        let h1_1_1 = g111;
        let h1_inf_1 = g111 - g101;
        let h1_0_inf = g10_inf;
        let h1_1_inf = g11_inf;
        let h1_inf_inf = g11_inf - g10_inf;

        output[0] = h0_0_0;
        output[9] = h1_0_0;
        output[18] = h1_0_0 - h0_0_0;
        output[1] = h0_0_1;
        output[10] = h1_0_1;
        output[19] = h1_0_1 - h0_0_1;
        output[2] = h0_0_inf;
        output[11] = h1_0_inf;
        output[20] = h1_0_inf - h0_0_inf;
        output[3] = h0_1_0;
        output[12] = h1_1_0;
        output[21] = h1_1_0 - h0_1_0;
        output[4] = h0_1_1;
        output[13] = h1_1_1;
        output[22] = h1_1_1 - h0_1_1;
        output[5] = h0_1_inf;
        output[14] = h1_1_inf;
        output[23] = h1_1_inf - h0_1_inf;
        output[6] = h0_inf_0;
        output[15] = h1_inf_0;
        output[24] = h1_inf_0 - h0_inf_0;
        output[7] = h0_inf_1;
        output[16] = h1_inf_1;
        output[25] = h1_inf_1 - h0_inf_1;
        output[8] = h0_inf_inf;
        output[17] = h1_inf_inf;
        output[26] = h1_inf_inf - h0_inf_inf;
    }

    /// Bind the first (least-significant) variable `z_0 := r`, reducing the dimension `w → w-1`.
    /// For each `(z_1, …, z_{w-1})` the three stored values `f(0,·), f(1,·), f(∞,·)` define the
    /// unique quadratic in `z_0`, evaluated at `r`: `f0·(1−r) + f1·r + f∞·r·(r−1)`.
    pub fn bind_first_variable(&mut self, r: F) {
        let w = self.num_vars;
        debug_assert!(w > 0);

        let new_size = 3_usize.pow((w - 1) as u32);
        let one = F::one();
        let r_term = r * (r - one);
        for new_idx in 0..new_size {
            let old_base_idx = new_idx * 3;
            let eval_at_0 = self.evals[old_base_idx];
            let eval_at_1 = self.evals[old_base_idx + 1];
            let eval_at_inf = self.evals[old_base_idx + 2];
            self.evals[new_idx] = eval_at_0 * (one - r) + eval_at_1 * r + eval_at_inf * r_term;
        }

        self.num_vars -= 1;
        self.evals.truncate(new_size);
    }

    /// Project to a univariate in `z_0` by summing against `E_active` over the remaining
    /// coordinates. `E_active[idx]` encodes, in binary, which of `z_1..z_{w-1}` take the "active"
    /// value (mapped to base-3 offset 1); `first_coord_val ∈ {0, 1, 2}` is `z_0` (2 = ∞).
    pub fn project_to_first_variable(&self, e_active: &[F], first_coord_val: usize) -> F {
        let w = self.num_vars;
        debug_assert!(w >= 1);
        let offset = first_coord_val;

        e_active
            .par_iter()
            .enumerate()
            .map(|(eq_active_idx, eq_active_val)| {
                let mut index = offset;
                let mut temp = eq_active_idx;
                let mut power = 3;
                for _ in 0..(w - 1) {
                    if temp & 1 == 1 {
                        index += power;
                    }
                    power *= 3;
                    temp >>= 1;
                }
                self.evals[index] * *eq_active_val
            })
            .sum()
    }

    pub fn is_bound(&self) -> bool {
        self.num_vars == 0 || self.evals.len() == 1
    }

    /// Bind the next variable to `r`. Only [`BindingOrder::LowToHigh`] is used by the outer/product
    /// streaming sumchecks (binds the least-significant variable).
    #[expect(
        clippy::panic,
        reason = "MultiquadraticPolynomial is bound LowToHigh only (outer/product streaming); HighToLow is never used"
    )]
    pub fn bind(&mut self, r: F, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bind_first_variable(r),
            BindingOrder::HighToLow => {
                panic!("HighToLow binding is not supported for MultiquadraticPolynomial")
            }
        }
    }

    /// Window sizes are small; this falls back to the sequential [`Self::bind`].
    pub fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        self.bind(r, order);
    }

    pub fn final_sumcheck_claim(&self) -> F {
        debug_assert!(self.is_bound());
        debug_assert_eq!(self.evals.len(), 1);
        self.evals[0]
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

    /// Base-3 index with digit `d[0]` most-significant, last variable least-significant
    /// (`idx = Σ d[i]·3^{dim-1-i}`), matching the explicit dim2/dim3 layout.
    fn base3_index(digits: &[usize]) -> usize {
        let dim = digits.len();
        digits.iter().enumerate().fold(0usize, |acc, (i, &d)| {
            acc + d * 3usize.pow((dim - 1 - i) as u32)
        })
    }

    /// Base-2 hypercube index, last variable fastest (`idx = Σ b[i]·2^{dim-1-i}`).
    fn base2_index(bits: &[usize]) -> usize {
        let dim = bits.len();
        bits.iter().enumerate().fold(0usize, |acc, (i, &b)| {
            acc + b * 2usize.pow((dim - 1 - i) as u32)
        })
    }

    fn expand(input: &[F], dim: usize) -> Vec<F> {
        let out_size = 3usize.pow(dim as u32);
        let mut output = vec![F::from_u64(0); out_size];
        let mut tmp = vec![F::from_u64(0); out_size];
        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
            input,
            &mut output,
            &mut tmp,
            dim,
        );
        output
    }

    /// Iterate over all `{0,1,∞}^dim` digit assignments (digits in `0..3`).
    fn all_digit_assignments(dim: usize, radix: usize) -> Vec<Vec<usize>> {
        let mut out = vec![vec![]];
        for _ in 0..dim {
            let mut next = Vec::new();
            for prefix in &out {
                for d in 0..radix {
                    let mut v = prefix.clone();
                    v.push(d);
                    next.push(v);
                }
            }
            out = next;
        }
        out
    }

    /// The expanded grid (a) embeds the Boolean hypercube on `{0,1}^dim` corners, and (b) stores the
    /// slope (finite difference) in every single-`∞` slot. Exercises the explicit (dim 1-3) AND the
    /// general (dim ≥ 4) expansion paths with the same defining property.
    #[test]
    fn expand_hypercube_embedding_and_slopes() {
        let mut rng = Rng(0xACE1);
        for dim in 1..=5usize {
            let input: Vec<F> = (0..(1usize << dim))
                .map(|_| F::from_u64(rng.next()))
                .collect();
            let out = expand(&input, dim);

            // (a) Boolean corners embed the input.
            for bits in all_digit_assignments(dim, 2) {
                assert_eq!(
                    out[base3_index(&bits)],
                    input[base2_index(&bits)],
                    "hypercube embedding dim={dim} corner={bits:?}"
                );
            }

            // (b) Single-∞ slot = (value with that coord = 1) − (value with that coord = 0).
            for p in 0..dim {
                for rest in all_digit_assignments(dim - 1, 2) {
                    let mk = |val: usize| {
                        let mut d = Vec::with_capacity(dim);
                        let mut it = rest.iter();
                        for q in 0..dim {
                            d.push(if q == p { val } else { *it.next().unwrap() });
                        }
                        d
                    };
                    let inf = out[base3_index(&mk(2))];
                    let hi = out[base3_index(&mk(1))];
                    let lo = out[base3_index(&mk(0))];
                    assert_eq!(inf, hi - lo, "slope dim={dim} axis={p} rest={rest:?}");
                }
            }
        }
    }

    /// `bind_first_variable(r)` evaluates the quadratic interpolant `f0(1−r) + f1·r + f∞·r(r−1)`.
    #[test]
    fn bind_is_quadratic_interpolation() {
        let mut rng = Rng(0xB1AD);
        // w = 1: 3 evals → 1.
        for _ in 0..8 {
            let q0 = F::from_u64(rng.next());
            let q1 = F::from_u64(rng.next());
            let qinf = F::from_u64(rng.next());
            let r = F::from_u64(rng.next());
            let mut poly = MultiquadraticPolynomial::<F>::new(1, vec![q0, q1, qinf]);
            poly.bind_first_variable(r);
            let one = F::from_u64(1);
            let expected = q0 * (one - r) + q1 * r + qinf * (r * (r - one));
            assert_eq!(poly.final_sumcheck_claim(), expected);
        }

        // w = 2: bind both variables, cross-check the iterated interpolation.
        let evals: Vec<F> = (0..9).map(|_| F::from_u64(rng.next())).collect();
        let r0 = F::from_u64(rng.next());
        let r1 = F::from_u64(rng.next());
        let one = F::from_u64(1);
        let interp = |t: &[F], r: F| t[0] * (one - r) + t[1] * r + t[2] * (r * (r - one));
        // First bind (z_0 = r0): three groups of 3.
        let after0: Vec<F> = (0..3)
            .map(|g| interp(&evals[g * 3..g * 3 + 3], r0))
            .collect();
        let expected = interp(&after0, r1);

        let mut poly = MultiquadraticPolynomial::<F>::new(2, evals);
        poly.bind(r0, BindingOrder::LowToHigh);
        poly.bind(r1, BindingOrder::LowToHigh);
        assert_eq!(poly.final_sumcheck_claim(), expected);
    }
}
