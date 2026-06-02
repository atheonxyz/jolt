//! Lagrange interpolation over a symmetric, zero-centered, consecutive-integer grid — the
//! high-degree univariate analogue of `EqPolynomial`. The univariate-skip foundation (Spartan
//! outer/product first round) and the inner Az·Bz evaluation rely on it.
//!
//! Vendored from jolt-core `poly/lagrange_poly.rs` (the parity oracle), retargeted to the lean
//! [`jolt_field::Field`]: `JoltField → Field`, and the challenge generic `C` collapses to `F`
//! (the `C = F = Fp3` convention). The maths is **field-agnostic** — only `from_i64`/`from_u64`/
//! `mul_u64`/`mul_i64`/`inverse`/`is_zero`/arithmetic — so it ports verbatim; [`LagrangeHelper`] is
//! pure const-fn integer arithmetic (no field). The grid is the symmetric integer window
//! `{start, …, start+N-1}` with `start = -⌊(N-1)/2⌋` — NOT a roots-of-unity / FFT domain, so it is
//! valid over Goldilocks/Fp3 identically to BN254.

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_poly::UnivariatePoly;

/// Lagrange polynomials over a zero-centered, symmetric, consecutive-integer domain, e.g.
/// `[-6, -5, …, 6, 7]`. Used in the univariate-skip optimization of Spartan's outer/product
/// sum-checks.
pub struct LagrangePolynomial<F: Field>(PhantomData<F>);

impl<F: Field> LagrangePolynomial<F> {
    /// Univariate Lagrange kernel on the symmetric integer grid: `K(x, y) = Σ_i L_i(x)·L_i(y)`,
    /// where `{L_i}` are the Lagrange basis polynomials for nodes `start..start+N-1` with
    /// `start = -⌊(N-1)/2⌋`. Kronecker delta at coinciding nodes; barycentric kernel otherwise.
    /// Constraint: `N <= 20`.
    #[expect(
        clippy::unwrap_used,
        reason = "barycentric denominators over distinct integer grid nodes are nonzero by construction"
    )]
    pub fn lagrange_kernel<const N: usize>(x: &F, y: &F) -> F {
        debug_assert!(N > 0, "N must be positive");
        debug_assert!(N <= 20, "lagrange_kernel intended for small N (<= 20)");
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);

        let mut dists_x = [F::zero(); N];
        let mut dists_y = [F::zero(); N];
        let mut base_x = *x - F::from_i64(start);
        let mut base_y = *y - F::from_i64(start);
        let one = F::one();
        let mut ix: Option<usize> = None;
        let mut it: Option<usize> = None;
        let mut i: usize = 0;
        while i < N {
            let dx = base_x;
            let dy = base_y;
            if dx.is_zero() {
                ix = Some(i);
            }
            if dy.is_zero() {
                it = Some(i);
            }
            dists_x[i] = dx;
            dists_y[i] = dy;
            base_x -= one;
            base_y -= one;
            i += 1;
        }

        if let (Some(ix), Some(it)) = (ix, it) {
            return if ix == it { F::one() } else { F::zero() };
        }

        let inv_denom = Self::inv_denom::<N>();

        if let Some(ix) = ix {
            let (terms_y, sum_y) = Self::bary_terms_from_dists::<N>(&dists_y, &inv_denom);
            return terms_y[ix] * sum_y.inverse().unwrap();
        }
        if let Some(it) = it {
            let (terms_x, sum_x) = Self::bary_terms_from_dists::<N>(&dists_x, &inv_denom);
            return terms_x[it] * sum_x.inverse().unwrap();
        }

        let (terms_x, s_x) = Self::bary_terms_from_dists::<N>(&dists_x, &inv_denom);
        let (terms_y, s_y) = Self::bary_terms_from_dists::<N>(&dists_y, &inv_denom);
        let mut num = F::zero();
        let mut i = 0usize;
        while i < N {
            num += terms_x[i] * terms_y[i];
            i += 1;
        }
        let inv_den = (s_x * s_y).inverse().unwrap();
        num * inv_den
    }

    /// Start of the symmetric integer grid: `start = -⌊(N-1)/2⌋`.
    #[inline]
    fn start_i64<const N: usize>() -> i64 {
        let d = N - 1;
        -((d / 2) as i64)
    }

    /// Distances `dᵢ = r - xᵢ` to grid nodes `xᵢ = start + i`; `hit = Some(i)` if `dᵢ = 0`.
    #[inline]
    fn distances<const N: usize>(r: &F) -> ([F; N], Option<usize>) {
        let start = Self::start_i64::<N>();
        let mut dists = [F::zero(); N];
        let mut base = *r - F::from_i64(start);
        let one = F::one();
        let mut hit: Option<usize> = None;
        let mut i: usize = 0;
        while i < N {
            let di = base;
            if di.is_zero() {
                hit = Some(i);
            }
            dists[i] = di;
            base -= one;
            i += 1;
        }
        (dists, hit)
    }

    /// Inverse barycentric denominators `wᵢ = (-1)^(N-1-i) / (i!·(N-1-i)!)` (one inversion).
    #[expect(
        clippy::unwrap_used,
        reason = "factorial product of distinct nodes is nonzero in a prime field"
    )]
    #[inline]
    fn inv_denom<const N: usize>() -> [F; N] {
        let den_i64 = LagrangeHelper::den_row_i64::<N>();
        let mut denom = [F::zero(); N];
        let mut i = 0usize;
        while i < N {
            denom[i] = F::from_i64(den_i64[i]);
            i += 1;
        }
        let mut left = [F::one(); N];
        i = 1;
        while i < N {
            left[i] = left[i - 1] * denom[i - 1];
            i += 1;
        }
        let inv_total = (left[N - 1] * denom[N - 1]).inverse().unwrap();
        let mut inv_denom = [F::zero(); N];
        let mut right = F::one();
        let mut t: isize = (N as isize) - 1;
        while t >= 0 {
            let u = t as usize;
            inv_denom[u] = left[u] * right * inv_total;
            right *= denom[u];
            t -= 1;
        }
        inv_denom
    }

    /// Unnormalized barycentric terms `termᵢ = wᵢ/dᵢ` and their sum `S = Σᵢ termᵢ`
    /// (prefix/suffix products + batch inversion).
    #[expect(
        clippy::unwrap_used,
        reason = "product of nonzero off-node distances is invertible"
    )]
    #[inline]
    fn bary_terms_from_dists<const N: usize>(dists: &[F; N], inv_denom: &[F; N]) -> ([F; N], F) {
        let mut prefix = [F::one(); N];
        let mut i = 1usize;
        while i < N {
            prefix[i] = prefix[i - 1] * dists[i - 1];
            i += 1;
        }
        let inv_prod = (prefix[N - 1] * dists[N - 1]).inverse().unwrap();

        let mut suffix = [F::one(); N];
        let mut j: isize = (N as isize) - 2;
        while j >= 0 {
            let u = j as usize;
            suffix[u] = suffix[u + 1] * dists[u + 1];
            j -= 1;
        }

        let mut terms = [F::zero(); N];
        let mut sum = F::zero();
        i = 0;
        while i < N {
            let inv_di = prefix[i] * suffix[i] * inv_prod;
            let term = inv_denom[i] * inv_di;
            terms[i] = term;
            sum += term;
            i += 1;
        }
        (terms, sum)
    }

    /// Evaluate `p(r)` from values on the symmetric grid via barycentric Lagrange. If `r = x_k`,
    /// returns `values[k]`.
    #[expect(
        clippy::unwrap_used,
        reason = "barycentric weight sum at an off-node point is nonzero"
    )]
    #[inline]
    pub fn evaluate<const N: usize>(values: &[F; N], r: &F) -> F {
        debug_assert!(N > 0, "N must be positive");
        debug_assert!(N <= 20, "evaluate intended for small N (<= 20)");
        let (dists, hit) = Self::distances::<N>(r);
        if let Some(i) = hit {
            return values[i];
        }
        let inv_denom = Self::inv_denom::<N>();
        let (terms, sum) = Self::bary_terms_from_dists::<N>(&dists, &inv_denom);
        let inv_sum = sum.inverse().unwrap();
        let mut num = F::zero();
        let mut i = 0usize;
        while i < N {
            num += values[i] * terms[i];
            i += 1;
        }
        num * inv_sum
    }

    /// All Lagrange basis values `[L_0(r), …, L_{N-1}(r)]` at `r`, such that
    /// `p(r) = Σᵢ Lᵢ(r)·p(xᵢ)`. Constraint: `N <= 20`.
    #[expect(
        clippy::unwrap_used,
        reason = "barycentric weight sum at an off-node point is nonzero"
    )]
    pub fn evals<const N: usize>(r: &F) -> [F; N] {
        debug_assert!(N <= 20, "N cannot exceed 20");
        debug_assert!(N > 0, "N must be positive");
        let (dists, hit) = Self::distances::<N>(r);
        if let Some(i) = hit {
            let mut out = [F::zero(); N];
            out[i] = F::one();
            return out;
        }
        let inv_denom = Self::inv_denom::<N>();
        let (terms, sum) = Self::bary_terms_from_dists::<N>(&dists, &inv_denom);
        let inv_sum = sum.inverse().unwrap();
        let mut out = [F::zero(); N];
        let mut i = 0usize;
        while i < N {
            out[i] = terms[i] * inv_sum;
            i += 1;
        }
        out
    }

    /// Interpolate monomial coefficients `[c_0, …, c_{N-1}]` (`p(x) = Σⱼ cⱼ·xʲ`) from values on the
    /// symmetric grid (`values[i] = p(start + i)`). Newton form via one batch inversion.
    #[expect(
        clippy::unwrap_used,
        reason = "factorial of the degree is nonzero in a prime field"
    )]
    #[inline]
    pub fn interpolate_coeffs<const N: usize>(values: &[F; N]) -> [F; N] {
        debug_assert!(N > 0, "N must be positive");
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);

        let mut smalls = [0u64; N];
        let mut pref = [F::one(); N];
        let mut m: usize = 1;
        while m <= d {
            smalls[m] = m as u64;
            pref[m] = pref[m - 1].mul_u64(smalls[m]);
            m += 1;
        }
        let inv_total = pref[d].inverse().unwrap();
        let mut right = F::one();
        let mut invs = [F::zero(); N];
        let mut i: isize = d as isize;
        while i >= 1 {
            let idx = i as usize;
            invs[idx] = pref[idx - 1] * right * inv_total;
            right = right.mul_u64(smalls[idx]);
            i -= 1;
        }

        let mut dd = *values;
        let mut newton = [F::zero(); N];
        newton[0] = dd[0];
        let mut order: usize = 1;
        while order <= d {
            let inv = invs[order];
            let mut i: usize = 0;
            while i + order < N {
                dd[i] = (dd[i + 1] - dd[i]) * inv;
                i += 1;
            }
            newton[order] = dd[0];
            order += 1;
        }

        let mut coeffs = [F::zero(); N];
        let mut basis = [F::zero(); N];
        basis[0] = F::one();
        let mut deg: usize = 0;
        let mut k: usize = 0;
        while k < N {
            let scale = newton[k];
            let mut j: usize = 0;
            while j <= deg {
                coeffs[j] += scale * basis[j];
                j += 1;
            }
            if k == d {
                break;
            }
            let a: i64 = start + (k as i64);
            let last = basis[deg];
            let mut t: isize = deg as isize;
            while t >= 1 {
                let idx = t as usize;
                let old = basis[idx];
                let term = old.mul_i64(a);
                basis[idx] = basis[idx - 1] - term;
                t -= 1;
            }
            basis[0] = -basis[0].mul_i64(a);
            deg += 1;
            basis[deg] = last;
            k += 1;
        }
        coeffs
    }
}

/// Field-agnostic const-fn helpers for Lagrange interpolation / extrapolation over integer domains
/// (binomials, factorials, shift coefficients, power sums). No field dependency.
pub struct LagrangeHelper;

impl LagrangeHelper {
    /// Binomial coefficient `C(n, k)`.
    #[inline]
    pub const fn binomial_coeff(n: usize, k: usize) -> u64 {
        let kk = if k <= n - k { k } else { n - k };
        let mut i = 0usize;
        let mut res: u128 = 1u128;
        while i < kk {
            let num = (n - i) as u128;
            let den = (i + 1) as u128;
            res = (res * num) / den;
            i += 1;
        }
        res as u64
    }

    /// `n!` for small `n` (valid up to 20).
    #[inline]
    pub const fn fact(n: usize) -> u64 {
        let mut acc: u64 = 1;
        let mut i: usize = 2;
        while i <= n {
            acc *= i as u64;
            i += 1;
        }
        acc
    }

    /// Precomputed `[0!, 1!, …, 20!]`.
    pub const FACT_U64_0_TO_20: [u64; 21] = {
        let mut out = [0u64; 21];
        let mut i: usize = 0;
        while i <= 20 {
            out[i] = Self::fact(i);
            i += 1;
        }
        out
    };

    /// `den[i] = (-1)^{N-1-i} · i! · (N-1-i)!` as i64. Constraint: `N <= 20`.
    #[inline]
    pub const fn den_row_i64<const N: usize>() -> [i64; N] {
        let mut out = [0i64; N];
        let mut i: usize = 0;
        while i < N {
            let a = Self::FACT_U64_0_TO_20[i] as i128;
            let b = Self::FACT_U64_0_TO_20[N - 1 - i] as i128;
            let mut v = a * b;
            if ((N - 1 - i) & 1) == 1 {
                v = -v;
            }
            out[i] = v as i64;
            i += 1;
        }
        out
    }

    /// Generalized binomial `C(t, k)` for integer `t` (negative `t` via `(-1)^k C(-t+k-1, k)`).
    #[inline]
    pub const fn generalized_binomial(t: i64, k: usize) -> i128 {
        if k == 0 {
            return 1;
        }
        if t >= 0 {
            let tt = t as i128;
            if (k as i128) > tt {
                return 0;
            }
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            let mut j: usize = 0;
            while j < k {
                num *= tt - (j as i128);
                den *= (j as i128) + 1;
                j += 1;
            }
            num / den
        } else {
            let sign = if (k & 1) == 1 { -1i128 } else { 1i128 };
            let tt = (-t) as i128 + (k as i128) - 1;
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            let mut j: usize = 0;
            while j < k {
                num *= tt - (j as i128);
                den *= (j as i128) + 1;
                j += 1;
            }
            sign * (num / den)
        }
    }

    /// Lagrange shift coefficients (i32): `p(shift) = Σ_i alpha[i]·p(i)` from base values `p(0..N-1)`.
    #[inline]
    pub const fn shift_coeffs_i32<const N: usize>(shift: i64) -> [i32; N] {
        let mut out = [0i32; N];
        let n_minus_1 = (N - 1) as i64;
        let mut i: usize = 0;
        while i < N {
            let s1 = Self::generalized_binomial(shift, i);
            let s2 = Self::generalized_binomial(shift - (i as i64) - 1, (N - 1) - i);
            let sign = if (((n_minus_1 as usize) - i) & 1) == 1 {
                -1i128
            } else {
                1i128
            };
            let val = sign * s1 * s2;
            out[i] = val as i32;
            i += 1;
        }
        out
    }

    /// Lagrange shift coefficients (i128, higher precision).
    #[inline]
    pub const fn shift_coeffs_i128<const N: usize>(shift: i64) -> [i128; N] {
        let mut out = [0i128; N];
        let n_minus_1 = (N - 1) as i64;
        let mut i = 0usize;
        while i < N {
            let s1 = Self::generalized_binomial(shift, i);
            let s2 = Self::generalized_binomial(shift - (i as i64) - 1, (N - 1) - i);
            let sign = if (((n_minus_1 as usize) - i) & 1) == 1 {
                -1i128
            } else {
                1i128
            };
            out[i] = sign * s1 * s2;
            i += 1;
        }
        out
    }

    /// Power sums `[S_0, …, S_{OUT_LEN-1}]`, `S_k = Σ_t t^k` over the symmetric integer window of
    /// `WINDOW_N` consecutive integers centered at 0.
    #[expect(
        clippy::panic,
        reason = "power-sum overflow on an oversized window is a compile-time misuse, caught in const eval"
    )]
    #[inline]
    pub const fn power_sums<const WINDOW_N: usize, const OUT_LEN: usize>() -> [i128; OUT_LEN] {
        let mut sums = [0i128; OUT_LEN];
        if OUT_LEN == 0 {
            return sums;
        }
        let d = WINDOW_N - 1;
        let start: i64 = -((d / 2) as i64);
        let mut j: usize = 0;
        while j < WINDOW_N {
            let t = (start + (j as i64)) as i128;
            sums[0] += 1;
            let mut pow = t;
            let mut k: usize = 1;
            while k < OUT_LEN {
                sums[k] += pow;
                pow = match pow.checked_mul(t) {
                    Some(v) => v,
                    None => panic!("power_sums overflow"),
                };
                k += 1;
            }
            j += 1;
        }
        sums
    }
}

/// Check that a univariate `poly` (ascending-coefficient form) sums to `claim` over the symmetric
/// `N`-point integer window: `Σ_t poly(t) = Σ_j coeff[j]·S_j == claim`, where `S_j` are the window
/// power sums. The uni-skip first-round sum check (degree `N` coeffs ⇒ `OUT_LEN = poly_len`).
///
/// Vendored from jolt-core `UniPoly::check_sum_evals` (`poly/unipoly.rs`).
pub fn check_sum_evals<F: Field, const N: usize, const OUT_LEN: usize>(
    poly: &UnivariatePoly<F>,
    claim: F,
) -> bool {
    let power_sums = LagrangeHelper::power_sums::<N, OUT_LEN>();
    let mut sum = F::zero();
    for (j, coeff) in poly.coefficients().iter().enumerate() {
        debug_assert!(j < OUT_LEN, "poly has more coeffs than power sums");
        sum += coeff.mul_i128(power_sums[j]);
    }
    sum == claim
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    fn grid_nodes<const N: usize>() -> [F; N] {
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);
        core::array::from_fn(|i| F::from_i64(start + i as i64))
    }

    fn pow_u64(mut base: F, mut exp: u64) -> F {
        let mut acc = F::from_u64(1);
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            base = base * base;
            exp >>= 1;
        }
        acc
    }

    fn eval_poly(coeffs: &[F], r: F) -> F {
        let mut acc = F::from_u64(0);
        for &c in coeffs.iter().rev() {
            acc = acc * r + c;
        }
        acc
    }

    /// Closed forms for N=1,2,3 plus on-node early exits.
    #[test]
    fn closed_forms() {
        // N=2 grid {0,1}: L0=1-r, L1=r.
        for k in 0..7u64 {
            let r = F::from_u64(k);
            let [l0, l1] = LagrangePolynomial::<F>::evals::<2>(&r);
            assert_eq!(l0, F::from_u64(1) - r);
            assert_eq!(l1, r);
        }
        // N=3 grid {-1,0,1}: L0=r(r-1)/2, L1=1-r^2, L2=r(r+1)/2.
        let two_inv = F::from_u64(2).inverse().unwrap();
        for k in 0..7u64 {
            let r = F::from_u64(k) - F::from_u64(1);
            let [l0, l1, l2] = LagrangePolynomial::<F>::evals::<3>(&r);
            assert_eq!(l0, (r * (r - F::from_u64(1))) * two_inv);
            assert_eq!(l1, F::from_u64(1) - r * r);
            assert_eq!(l2, (r * (r + F::from_u64(1))) * two_inv);
        }
        // On-node evaluate returns the stored value.
        let nodes = grid_nodes::<3>();
        let vals = [F::from_u64(11), F::from_u64(13), F::from_u64(17)];
        for i in 0..3 {
            assert_eq!(
                LagrangePolynomial::<F>::evaluate::<3>(&vals, &nodes[i]),
                vals[i]
            );
        }
    }

    /// Partition of unity, Kronecker delta at nodes, and monomial reproduction.
    #[test]
    fn basis_properties_and_monomials() {
        fn check<const N: usize>() {
            // Partition of unity at integer points.
            for t in -3..=3 {
                let basis = LagrangePolynomial::<F>::evals::<N>(&F::from_i64(t));
                let sum: F = basis.iter().copied().sum();
                assert_eq!(sum, F::from_u64(1), "partition of unity N={N} t={t}");
            }
            // Delta at nodes.
            let nodes = grid_nodes::<N>();
            for (i, &xi) in nodes.iter().enumerate() {
                let basis = LagrangePolynomial::<F>::evals::<N>(&xi);
                for (j, &bj) in basis.iter().enumerate() {
                    assert_eq!(
                        bj,
                        if i == j {
                            F::from_u64(1)
                        } else {
                            F::from_u64(0)
                        },
                        "delta N={N} i={i} j={j}"
                    );
                }
            }
            // Monomial reproduction: interpolating x^m and evaluating at t gives t^m.
            let d = N - 1;
            let start: i64 = -((d / 2) as i64);
            for m in 0..N {
                let vals: [F; N] =
                    core::array::from_fn(|i| pow_u64(F::from_i64(start + i as i64), m as u64));
                for t in -2..=2 {
                    let r = F::from_i64(t);
                    assert_eq!(
                        LagrangePolynomial::<F>::evaluate::<N>(&vals, &r),
                        pow_u64(r, m as u64),
                        "monomial N={N} m={m} t={t}"
                    );
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<5>();
        check::<8>();
        check::<11>();
    }

    /// `lagrange_kernel(x,y) == Σ_i L_i(x)·L_i(y)` and Kronecker delta at nodes; symmetric.
    #[test]
    fn kernel_matches_evals() {
        fn check<const N: usize>() {
            for r in -2..=2 {
                for s in -2..=2 {
                    let (rf, sf) = (F::from_i64(r), F::from_i64(s));
                    let k = LagrangePolynomial::<F>::lagrange_kernel::<N>(&rf, &sf);
                    let br = LagrangePolynomial::<F>::evals::<N>(&rf);
                    let bs = LagrangePolynomial::<F>::evals::<N>(&sf);
                    let dot: F = (0..N).map(|i| br[i] * bs[i]).sum();
                    assert_eq!(k, dot, "kernel==dot N={N}");
                    assert_eq!(
                        k,
                        LagrangePolynomial::<F>::lagrange_kernel::<N>(&sf, &rf),
                        "kernel symmetric"
                    );
                }
            }
        }
        check::<2>();
        check::<3>();
        check::<8>();
        check::<11>();
    }

    /// `interpolate_coeffs` round-trips coefficients and reproduces monomials.
    #[test]
    fn interpolate_roundtrip() {
        const N: usize = 9;
        let coeffs: [F; N] = core::array::from_fn(|i| F::from_u64((i + 1) as u64));
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);
        let values: [F; N] =
            core::array::from_fn(|i| eval_poly(&coeffs, F::from_i64(start + i as i64)));
        assert_eq!(
            LagrangePolynomial::<F>::interpolate_coeffs::<N>(&values),
            coeffs
        );
    }

    /// `shift_coeffs_i32` reproduces shifted evaluations of a cubic.
    #[test]
    fn shift_coeffs_match_shifted_eval() {
        const N: usize = 7;
        // p(x) = 2 - 3x + x^3.
        let coeffs = [
            F::from_u64(2),
            F::from_i64(-3),
            F::from_u64(0),
            F::from_u64(1),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
        ];
        let base: [F; N] = core::array::from_fn(|i| eval_poly(&coeffs, F::from_i64(i as i64)));
        for shift in -10..=10i64 {
            let cs = LagrangeHelper::shift_coeffs_i32::<N>(shift);
            let acc: F = (0..N).map(|i| base[i] * F::from_i64(cs[i] as i64)).sum();
            assert_eq!(acc, eval_poly(&coeffs, F::from_i64(shift)), "shift={shift}");
        }
    }

    /// `power_sums` matches a naive computation; odd power sums vanish on the symmetric window.
    #[test]
    fn power_sums_match_naive() {
        const WINDOW_N: usize = 7;
        const OUT_LEN: usize = 6;
        let sums = LagrangeHelper::power_sums::<WINDOW_N, OUT_LEN>();
        let start: i64 = -((WINDOW_N as i64 - 1) / 2);
        let mut naive = [0i128; OUT_LEN];
        for j in 0..WINDOW_N {
            let t = (start + j as i64) as i128;
            let mut pow = 1i128;
            for slot in &mut naive {
                *slot += pow;
                pow *= t;
            }
        }
        assert_eq!(sums, naive);
        for k in (1..OUT_LEN).step_by(2) {
            assert_eq!(sums[k], 0, "odd power sum vanishes");
        }
    }

    /// `check_sum_evals` accepts the correct window-sum claim and rejects a tampered one.
    #[test]
    fn check_sum_evals_accepts_and_rejects() {
        const N: usize = 5;
        // A degree-3 poly p(x) = 1 + 2x + 3x^2 + 4x^3 (4 coeffs, OUT_LEN = 4).
        let coeffs = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let poly = UnivariatePoly::new(coeffs.clone());
        // True sum over the symmetric 5-window {-2,-1,0,1,2}: Σ_t p(t).
        let start: i64 = -((N as i64 - 1) / 2);
        let claim: F = (0..N)
            .map(|j| eval_poly(&coeffs, F::from_i64(start + j as i64)))
            .sum();
        assert!(check_sum_evals::<F, N, 4>(&poly, claim));
        assert!(!check_sum_evals::<F, N, 4>(&poly, claim + F::from_u64(1)));
    }
}
