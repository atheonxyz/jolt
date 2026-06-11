//! BN254-optimized fractional GKR per-layer sumcheck.
//!
//! Reimplements `gkr.rs`'s `run_gkr` / `prove_family_gkr` /
//! `prove_single_instance_sumcheck` for the concrete BN254 field, using
//! Jolt-native primitives on the hot path:
//!
//! 1. **`jolt_field::WideAccumulator`** for the per-pair `α·(nl·dr + nr·dl) +
//!    β·dl·dr` and X²-coef accumulation. 9-limb deferred-Montgomery accumulator
//!    folds six `fmadd(a, b)` calls into one `reduce()`, saving ~5 of the 6
//!    Montgomery reductions in the inner loop.
//!
//! 2. **`jolt_poly::GruenSplitEqPolynomial<JoltFr>`** to derive the cubic
//!    round polynomial from `(q_constant, q_quadratic_coef, prev_claim)`.
//!    Same arithmetic as gkr.rs's hand-coded path; just centralized.
//!
//! ## Field-type story
//!
//! WHIR's `Field256` is now `ark_bn254::Fr` (via Path A in
//! `whir/src/algebra/fields.rs`). Jolt's `jolt_field::Field` is implemented
//! on `jolt_field::Fr`, which is a `#[repr(transparent)]` newtype around
//! `ark_bn254::Fr`. The two are bitwise-identical in memory but distinct
//! Rust types.
//!
//! This module accepts `FamilyState<ArkFr>` from the caller (matching the
//! generic API in `gkr.rs`), and converts to `JoltFr` for all arithmetic.
//! Conversions are zero-cost under `repr(transparent)` + `#[inline(always)]`
//! `From` impls.
//!
//! Prep phase (`prepare_pushforwards`) is reused unchanged from `gkr.rs` —
//! the BN254 `Field::from_u64` precomp-table win there is marginal.

use std::time::{Duration, Instant};

use ark_bn254::Fr as ArkFr;
use jolt_field::{AdditiveAccumulator, Fr as JoltFr, RingAccumulator, WideAccumulator};
use jolt_poly::BindingOrder;
use rayon::prelude::*;
use whir::transcript::{ProverState, VerifierMessage};

use crate::gkr::{build_circuit, lsb_eq_table, Circuit, FamilyState, Timing};
use crate::gruen::GruenSplitEqPolynomial;

#[tracing::instrument(skip_all, name = "gkr_bn254.run_gkr", fields(n_families = family_states.len()))]
pub(crate) fn run_gkr(
    prover_state: &mut ProverState,
    family_states: Vec<FamilyState<ArkFr>>,
    pre_commit_time: Duration,
) -> Timing {
    let max_a_depth = family_states
        .iter()
        .map(|f| f.log_t + f.log_d)
        .max()
        .unwrap_or(0);
    let mut per_layer_gkr = vec![Duration::ZERO; max_a_depth];
    let mut per_family_gkr = Vec::with_capacity(family_states.len());

    let t_gkr_start = Instant::now();

    for family in family_states {
        let t_family = Instant::now();
        prove_family_gkr(prover_state, family, &mut per_layer_gkr);
        per_family_gkr.push(t_family.elapsed());
    }

    let gkr_only_time = t_gkr_start.elapsed();

    Timing {
        gkr: pre_commit_time + gkr_only_time,
        per_family_gkr,
        per_layer_gkr,
    }
}

#[tracing::instrument(skip_all, name = "gkr_bn254.family", fields(name = family.name))]
fn prove_family_gkr(
    prover_state: &mut ProverState,
    family: FamilyState<ArkFr>,
    per_layer_times: &mut [Duration],
) {
    let log_size_a = family.log_t + family.log_d;
    let log_size_b = family.log_m;

    // Build circuits in ArkFr (gkr::build_circuit is generic over ArkField).
    let circuit_a = {
        let _span =
            tracing::info_span!("gkr_bn254.build_circuit_a", log_size = log_size_a).entered();
        build_circuit(family.eq_m_row, family.leaf_denom_a, log_size_a)
    };
    let circuit_b = {
        let _span =
            tracing::info_span!("gkr_bn254.build_circuit_b", log_size = log_size_b).entered();
        build_circuit(family.pushforward, family.leaf_denom_b, log_size_b)
    };

    let (na, da) = circuit_a.root();
    let (nb, db) = circuit_b.root();
    let lhs = na * db;
    let rhs = nb * da;
    assert!(
        lhs == rhs,
        "GKR histogram identity violated for family {}: \
         N_A·D_B = {lhs:?} ≠ N_B·D_A = {rhs:?}",
        family.name,
    );

    prove_single_instance_sumcheck(prover_state, &circuit_a, log_size_a, per_layer_times, "a");
    prove_single_instance_sumcheck(prover_state, &circuit_b, log_size_b, per_layer_times, "b");
}

/// AoS pack of the four split values per pair, in JoltFr (for accumulator use).
#[derive(Clone, Copy)]
struct PairVals {
    nl: JoltFr,
    nr: JoltFr,
    dl: JoltFr,
    dr: JoltFr,
}

fn prove_single_instance_sumcheck(
    prover_state: &mut ProverState,
    circuit: &Circuit<ArkFr>,
    log_size: usize,
    per_layer_times: &mut [Duration],
    side_tag: &'static str,
) {
    // -- Transcript I/O in ArkFr (WHIR's transcript Codec is on ArkFr) --
    let (n_root_ark, d_root_ark) = circuit.root();
    prover_state.prover_messages(&[n_root_ark, d_root_ark]);
    let cur_alpha_ark: ArkFr = prover_state.verifier_message();
    let cur_beta_ark: ArkFr = prover_state.verifier_message();

    // -- Internal arithmetic in JoltFr --
    let mut cur_alpha: JoltFr = cur_alpha_ark.into();
    let mut cur_beta: JoltFr = cur_beta_ark.into();
    let n_root: JoltFr = n_root_ark.into();
    let d_root: JoltFr = d_root_ark.into();
    let mut combined_claim: JoltFr = cur_alpha * n_root + cur_beta * d_root;
    let mut point: Vec<JoltFr> = Vec::new();

    let max_half = if log_size == 0 {
        1
    } else {
        1usize << (log_size - 1)
    };
    let mut vals: Vec<PairVals> = Vec::with_capacity(max_half);
    let mut vals_swap: Vec<PairVals> = Vec::with_capacity(max_half);
    let mut eq_unbound: Vec<JoltFr> = Vec::with_capacity(max_half);
    let mut eq_unbound_swap: Vec<JoltFr> = Vec::with_capacity(max_half);

    #[expect(clippy::needless_range_loop)]
    for k in 0..log_size {
        let layer_t0 = Instant::now();
        let _span = tracing::info_span!("gkr_bn254.layer", side = side_tag, k).entered();

        // Pack PairVals from circuit (ArkFr → JoltFr per element).
        let level_pairs = &circuit.levels[k + 1].pairs;
        let half = level_pairs.len() / 2;
        vals.clear();
        vals.spare_capacity_mut()[..half]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dst)| {
                let (nl, dl) = level_pairs[2 * i];
                let (nr, dr) = level_pairs[2 * i + 1];
                dst.write(PairVals {
                    nl: nl.into(),
                    nr: nr.into(),
                    dl: dl.into(),
                    dr: dr.into(),
                });
            });
        // SAFETY: the parallel loop initialized every slot in `0..half`.
        unsafe {
            vals.set_len(half);
        }

        // Initialize eq_unbound = eq table over point[1..] (LSB-first).
        // lsb_eq_table is generic over ArkField, so we convert point[1..]
        // back to ArkFr, call lsb_eq_table, and re-wrap the Vec to JoltFr
        // (zero-copy under repr(transparent)).
        let eq_unbound_init_len = 1usize << k.saturating_sub(1);
        eq_unbound.clear();
        if k <= 1 {
            eq_unbound.spare_capacity_mut()[0].write(JoltFr::from(1u64));
        } else {
            let point_rest_ark: Vec<ArkFr> = point[1..].iter().map(|&p| p.into()).collect();
            let tmp_ark = lsb_eq_table(&point_rest_ark);
            let tmp_jolt = vec_ark_to_jolt(tmp_ark);
            debug_assert_eq!(tmp_jolt.len(), eq_unbound_init_len);
            eq_unbound.spare_capacity_mut()[..eq_unbound_init_len]
                .par_iter_mut()
                .zip(tmp_jolt.par_iter())
                .for_each(|(dst, &src)| {
                    dst.write(src);
                });
        }
        // SAFETY: either the scalar case or the parallel copy initialized
        // every slot in `0..eq_unbound_init_len`.
        unsafe {
            eq_unbound.set_len(eq_unbound_init_len);
        }

        // Gruen state for THIS layer's sumcheck. point has k entries.
        // BindingOrder::HighToLow matches gkr.rs's `cur_w = point[round_idx]`.
        // The constructor asserts point.len() >= 1; skip for k=0 (no rounds).
        let mut gruen: Option<GruenSplitEqPolynomial<JoltFr>> = if k == 0 {
            None
        } else {
            Some(GruenSplitEqPolynomial::<JoltFr>::new(
                &point,
                BindingOrder::HighToLow,
            ))
        };

        let mut r_prime: Vec<JoltFr> = Vec::with_capacity(k);

        for _round_idx in 0..k {
            let gruen_ref = gruen.as_mut().expect("gruen present when k > 0");
            let cur_half = vals.len() / 2;
            debug_assert_eq!(cur_half, eq_unbound.len());

            // Inner pair-fold using WideAccumulator.
            let (q_constant, q_quadratic_coef) =
                compute_pair_q_values(&vals, &eq_unbound, cur_alpha, cur_beta, cur_half);

            // Round polynomial via Gruen.
            let round_poly =
                gruen_ref.gruen_poly_deg_3(q_constant, q_quadratic_coef, combined_claim);

            // Wire format: send h(0), h(2), h(3) in ArkFr. Verifier
            // recovers h(1) = combined_claim − h(0).
            let h0_j: JoltFr = round_poly.evaluate(JoltFr::from(0u64));
            let h2_j: JoltFr = round_poly.evaluate(JoltFr::from(2u64));
            let h3_j: JoltFr = round_poly.evaluate(JoltFr::from(3u64));
            let h0: ArkFr = h0_j.into();
            let h2: ArkFr = h2_j.into();
            let h3: ArkFr = h3_j.into();
            prover_state.prover_messages(&[h0, h2, h3]);

            // Squeeze in ArkFr, convert to JoltFr for arithmetic.
            let r_j_ark: ArkFr = prover_state.verifier_message();
            let r_j: JoltFr = r_j_ark.into();
            r_prime.push(r_j);

            // Fold PairVals via ping-pong buffer.
            vals_swap.clear();
            vals_swap.spare_capacity_mut()[..cur_half]
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, dst)| {
                    let v0 = vals[2 * i];
                    let v1 = vals[2 * i + 1];
                    dst.write(PairVals {
                        nl: v0.nl + r_j * (v1.nl - v0.nl),
                        nr: v0.nr + r_j * (v1.nr - v0.nr),
                        dl: v0.dl + r_j * (v1.dl - v0.dl),
                        dr: v0.dr + r_j * (v1.dr - v0.dr),
                    });
                });
            // SAFETY: the parallel loop initialized every slot in `0..cur_half`.
            unsafe {
                vals_swap.set_len(cur_half);
            }
            std::mem::swap(&mut vals, &mut vals_swap);

            // eq_unbound update by pair-sum (drops the leading point coord).
            if eq_unbound.len() > 1 {
                let new_eq_len = eq_unbound.len() / 2;
                eq_unbound_swap.clear();
                eq_unbound_swap.spare_capacity_mut()[..new_eq_len]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, dst)| {
                        dst.write(eq_unbound[2 * i] + eq_unbound[2 * i + 1]);
                    });
                // SAFETY: the parallel loop initialized every slot in `0..new_eq_len`.
                unsafe {
                    eq_unbound_swap.set_len(new_eq_len);
                }
                std::mem::swap(&mut eq_unbound, &mut eq_unbound_swap);
            }

            combined_claim = round_poly.evaluate(r_j);
            gruen_ref.bind(r_j);
        }

        debug_assert_eq!(vals.len(), 1);
        let n_l_val = vals[0].nl;
        let n_r_val = vals[0].nr;
        let d_l_val = vals[0].dl;
        let d_r_val = vals[0].dr;

        // Leaf claims to transcript in ArkFr.
        let n_l_ark: ArkFr = n_l_val.into();
        let n_r_ark: ArkFr = n_r_val.into();
        let d_l_ark: ArkFr = d_l_val.into();
        let d_r_ark: ArkFr = d_r_val.into();
        prover_state.prover_messages(&[n_l_ark, n_r_ark, d_l_ark, d_r_ark]);

        // Soundness identity check.
        let f_at_r =
            cur_alpha * (n_l_val * d_r_val + n_r_val * d_l_val) + cur_beta * d_l_val * d_r_val;
        // For k=0 (no rounds), no Gruen state was constructed; scalar is
        // F::one() because the sumcheck reduced nothing.
        let scalar = match &gruen {
            None => JoltFr::from(1u64),
            Some(g) => g.current_scalar(),
        };
        let expected = scalar * f_at_r;
        assert_eq!(
            combined_claim, expected,
            "single-instance sumcheck identity failed at layer k={k}"
        );

        // Merge to next layer.
        let t_ark: ArkFr = prover_state.verifier_message();
        let new_alpha_ark: ArkFr = prover_state.verifier_message();
        let new_beta_ark: ArkFr = prover_state.verifier_message();
        let t: JoltFr = t_ark.into();
        let new_alpha: JoltFr = new_alpha_ark.into();
        let new_beta: JoltFr = new_beta_ark.into();

        let mut new_point = Vec::with_capacity(k + 1);
        new_point.push(t);
        new_point.extend(r_prime);

        let n_combined = n_l_val + t * (n_r_val - n_l_val);
        let d_combined = d_l_val + t * (d_r_val - d_l_val);
        combined_claim = new_alpha * n_combined + new_beta * d_combined;

        cur_alpha = new_alpha;
        cur_beta = new_beta;
        point = new_point;

        per_layer_times[k] += layer_t0.elapsed();
    }

    let _ = (combined_claim, point, cur_alpha, cur_beta);
}

/// Per-round q-coefficient accumulation using `WideAccumulator`.
///
/// For each pair `(v0, v1)` at index `i`:
///   q_const += eq[i] · [α·(v0.nl·v0.dr + v0.nr·v0.dl) + β·v0.dl·v0.dr]
///   q_x2    += eq[i] · [α·(Δnl·Δdr + Δnr·Δdl) + β·Δdl·Δdr]
/// where Δ_ = v1._ − v0._.
fn compute_pair_q_values(
    vals: &[PairVals],
    eq_unbound: &[JoltFr],
    cur_alpha: JoltFr,
    cur_beta: JoltFr,
    cur_half: usize,
) -> (JoltFr, JoltFr) {
    type Acc = WideAccumulator;

    let (acc_q_const, acc_q_x2) = (0..cur_half)
        .into_par_iter()
        .fold(
            || (Acc::default(), Acc::default()),
            |(mut acc_q_const, mut acc_q_x2), i| {
                let e = eq_unbound[i];
                let v0 = vals[2 * i];
                let v1 = vals[2 * i + 1];

                let e_alpha = e * cur_alpha;
                let e_beta = e * cur_beta;

                // q_constant: F(X=0) weighted by eq[i].
                acc_q_const.fmadd(e_alpha * v0.nl, v0.dr);
                acc_q_const.fmadd(e_alpha * v0.nr, v0.dl);
                acc_q_const.fmadd(e_beta * v0.dl, v0.dr);

                // q_quadratic_coef: X² coef weighted by eq[i].
                let dnl = v1.nl - v0.nl;
                let dnr = v1.nr - v0.nr;
                let ddl = v1.dl - v0.dl;
                let ddr = v1.dr - v0.dr;
                acc_q_x2.fmadd(e_alpha * dnl, ddr);
                acc_q_x2.fmadd(e_alpha * dnr, ddl);
                acc_q_x2.fmadd(e_beta * ddl, ddr);

                (acc_q_const, acc_q_x2)
            },
        )
        .reduce(
            || (Acc::default(), Acc::default()),
            |(mut a1, mut a2), (b1, b2)| {
                a1.merge(b1);
                a2.merge(b2);
                (a1, a2)
            },
        );

    (acc_q_const.reduce(), acc_q_x2.reduce())
}

/// Reinterpret a `Vec<ArkFr>` as `Vec<JoltFr>`. Safe because `JoltFr` is
/// `#[repr(transparent)]` over `ArkFr` (see [crates/jolt-field/src/arkworks/bn254.rs:17]).
/// Both types have identical size, alignment, and layout; the
/// `Vec`'s `(ptr, len, cap)` triple is type-erased at runtime.
#[inline]
fn vec_ark_to_jolt(v: Vec<ArkFr>) -> Vec<JoltFr> {
    let mut v = std::mem::ManuallyDrop::new(v);
    let len = v.len();
    let cap = v.capacity();
    let ptr = v.as_mut_ptr() as *mut JoltFr;
    // SAFETY: ArkFr and JoltFr have identical layout via repr(transparent).
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}
