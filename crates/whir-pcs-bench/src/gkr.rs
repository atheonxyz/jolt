//! Paper-faithful LogUp\* commitment-scheme prover.
//!
//! Implements the protocol from:
//! - [`twist_shout_logup_star.pdf`](../../jolt/twist_shout_logup_star.pdf) §4
//!   (Figure 1 — commitment to a one-hot matrix via its dense encoding and
//!   the eq-weighted pushforward) and §4.1 (batching across the d chunks of
//!   a family).
//! - [`ProofsArgsAndZK.pdf`](../../jolt/ProofsArgsAndZK.pdf) §4.5.2 (reducing
//!   d evaluation claims of a multilinear to a single claim). For our use
//!   case the d input points share `(r_row, r_col)` so the canonical-curve
//!   construction collapses to an `eq`-weighted linear combination on the
//!   chunk dimension (verifier-side O(d), prover-side ≈ 0).
//!
//! Per Jolt family F (InstructionRa with d=32; BytecodeRa, RamRa with d=4):
//!
//! 1. `claim_eval`: prover squeezes a per-family `(r_row^F, r_col^F)` and
//!    computes the d MLE evaluations `M̃^(i)(r_row^F, r_col^F)` for i = 0..d.
//!    These are the §4.5.2 inputs — in a real protocol they come from each
//!    family's independent upstream sumcheck (instruction lookups / bytecode
//!    read-write / RAM read-write).
//! 2. `§4.5.2`: prover squeezes `r_chunk` and forms the combined claim
//!    `Σ_i M̃^(i)(r_row,r_col) · eq(bits(i), r_chunk)`. This equals
//!    `M̃^(*)(r_row, r_chunk, r_col)` where `M^(*)` is the row-concatenated
//!    matrix from §4.1.
//! 3. Build the eq-weighted pushforward
//!    `P^F[k] = Σ_{j : M^(*)[j] = k} eq(bits(j), r_M_row)`
//!    with `r_M_row = (r_row, r_chunk)`. Length `m_pad = 2^15` (padded for
//!    WHIR-ZK blinding).
//! 4. Commit `P^F` (handled by `main.rs` via the WHIR commit phase).
//! 5. Run the **eq-weighted fan-in-2 fractional GKR** of Figure 1:
//!    `Σ_j eq(bits(j),r_M_row)/(α − M^(*)[j]) = Σ_k P^F[k]/(α − k)`.
//!
//! Bit-ordering convention: this module uses **LSB-first** for polynomial
//! arrays (matching the LogUp\* paper's variable ordering). WHIR's
//! `eq_weights` places `point[0]` at the MSB — we reverse the point at the
//! call site to bridge the conventions.
//!
//! Sumcheck: the per-layer sumcheck uses the **Gruen / Dao-Thaler eq
//! factorization** (PAZK §4.6.5, Thaler 2013 eprint 2013/351). The bound eq
//! factor is tracked in a running scalar `current_eq_scalar`, and the
//! degree-3 round polynomial is derived from two per-pair quantities
//! (F(X=0) and the X² coefficient) plus the sumcheck constraint. Cuts the
//! per-pair multiplication count by ~45% vs the naive degree-3 eq×F path.
//! Wire format is unchanged (3 field elements per round).
//!
//! Fiat-Shamir continuity: this module absorbs the d input claims into the
//! transcript before squeezing `r_chunk`, and absorbs the combined claim
//! before squeezing `α`. Soundness of the §4.5.2 RLC and the LogUp Lemma 2
//! bound both require these absorbs.

use std::time::{Duration, Instant};

use ark_ff::Field as ArkField;
use ark_std::rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use whir::algebra::eq_weights;
use whir::transcript::{Codec, ProverState, VerifierMessage};

/// One LogUp\* family of d one-hot chunks, sharing the same column space.
///
/// All d chunks have `T = 2^log_t` rows and column domain `2^log_k_chunk`
/// (the unpadded `K_chunk`). Pushforwards are committed at `num_vars=log_m`
/// (the padded length).
pub(crate) struct Family<'a, F> {
    pub name: &'static str,
    pub d: usize,
    pub log_t: usize,
    pub log_d: usize,
    pub log_m: usize,
    /// d slices of length `T`, F-encoded, used for the commit phase and as
    /// the source of `α − M^(*)[j]` in the A-circuit leaf denoms.
    pub ra_dense_chunks_f: Vec<&'a [F]>,
    /// d slices of length `T`, raw u8 indices `∈ [0, K_chunk)`, used to
    /// scatter into the eq-weighted pushforward and to index the leaf-denom
    /// computation. Always `< 2^log_m` (padding region is zero).
    pub ra_dense_chunks_u8: Vec<&'a [u8]>,
}

/// Wall-clock timings emitted by the LogUp\* prover.
#[derive(Default, Clone, Debug)]
pub(crate) struct Timing {
    /// Total of §4.5.2 RLC + eq_M_row + P^F build + Figure-1 GKR sumchecks.
    /// `claim_eval` is tracked separately in [`PrepResult::claim_eval_time`].
    pub gkr: Duration,
    /// Per-family GKR cost (3 entries, sums of §4.5.2/P^F/GKR per family).
    pub per_family_gkr: Vec<Duration>,
    /// Per-layer cumulative GKR sumcheck cost across families.
    /// Length = max A-circuit depth (= max log_t + log_d, typically 24).
    pub per_layer_gkr: Vec<Duration>,
}

/// Per-family intermediate state computed during `prepare_pushforwards` and
/// consumed by `run_gkr`. Owned: the GKR phase mutates the pyramids in place.
pub(crate) struct FamilyState<F> {
    pub name: &'static str,
    pub log_t: usize,
    pub log_d: usize,
    pub log_m: usize,
    /// `eq_M_row[j]` for `j ∈ [0, T·d)`, LSB-first indexed; doubles as
    /// A-circuit leaf-num.
    pub eq_m_row: Vec<F>,
    /// `α − M^(*)[j]` for `j ∈ [0, T·d)`; A-circuit leaf-denom (α baked in).
    pub leaf_denom_a: Vec<F>,
    /// The eq-weighted pushforward P^F (length `2^log_m`); B-circuit leaf-num
    /// source. Mirrors what was handed to the commit phase.
    pub pushforward: Vec<F>,
    /// `α − k` for `k ∈ [0, 2^log_m)`; B-circuit leaf-denom (α baked in).
    pub leaf_denom_b: Vec<F>,
}

/// Result of the prep phase: pushforwards (to feed the WHIR commit) plus
/// per-family state for the GKR phase plus phase timings.
pub(crate) struct PrepResult<F> {
    pub pushforwards: Vec<Vec<F>>,
    pub family_states: Vec<FamilyState<F>>,
    pub claim_eval_time: Duration,
    pub gkr_pre_commit_time: Duration,
}

// =============================================================================
// Phase 1 + part of phase 2: claim_eval, §4.5.2 RLC, eq_M_row + P^F build.
// =============================================================================

#[tracing::instrument(skip_all, name = "gkr.prepare_pushforwards", fields(n_families = families.len()))]
pub(crate) fn prepare_pushforwards<F>(
    prover_state: &mut ProverState,
    families: &[Family<'_, F>],
) -> PrepResult<F>
where
    F: ArkField + Codec<[u8]> + Send + Sync,
    Standard: Distribution<F>,
{
    if families.is_empty() {
        return PrepResult {
            pushforwards: vec![],
            family_states: vec![],
            claim_eval_time: Duration::ZERO,
            gkr_pre_commit_time: Duration::ZERO,
        };
    }

    // All families share the same T and column space in this Jolt config
    // (`OneHotParams.log_k_chunk` is one field for all 3 families). The
    // (r_row, r_col) points themselves are NOT shared — each family squeezes
    // its own (r_row^F, r_col^F) inside the loop, simulating independent
    // upstream binding.
    let log_t = families[0].log_t;
    let log_m = families[0].log_m;
    for f in families {
        assert_eq!(f.log_t, log_t, "all families must share log_t");
        assert_eq!(f.log_m, log_m, "all families must share log_m");
    }
    let t = 1usize << log_t;
    let m_pad = 1usize << log_m;

    let mut pushforwards = Vec::with_capacity(families.len());
    let mut family_states = Vec::with_capacity(families.len());
    let mut claim_eval_total = Duration::ZERO;
    let mut gkr_pre_commit_total = Duration::ZERO;

    for family in families {
        let d = family.d;
        let log_d = family.log_d;
        debug_assert_eq!(1usize << log_d, d.next_power_of_two());

        // === Per-family claim_eval phase ===
        // Squeeze this family's (r_row^F, r_col^F) and compute its d MLE
        // evaluations. In an integrated prover these would come from this
        // family's upstream sumcheck (instruction-lookup / bytecode / RAM);
        // the bench simulates that by squeezing fresh challenges per family.
        let t_claim_eval = Instant::now();
        let _span_eval = tracing::info_span!("gkr.claim_eval", family = family.name).entered();

        let r_row: Vec<F> = prover_state.verifier_message_vec(log_t);
        let r_col: Vec<F> = prover_state.verifier_message_vec(log_m);

        // Build eq tables (LSB-first w.r.t. the polynomial-array convention).
        // WHIR's eq_weights is MSB-first; lsb_eq_table reverses the point.
        let eq_row = lsb_eq_table(&r_row); // length T
        let eq_col = lsb_eq_table(&r_col); // length m_pad

        // d MLE evals for this family:
        //   M̃^(i)(r_row^F, r_col^F) = Σ_j eq_row[j] · eq_col[ra_dense_indices[i][j]]
        let claims: Vec<F> = family
            .ra_dense_chunks_u8
            .par_iter()
            .map(|indices_i| {
                debug_assert_eq!(indices_i.len(), t);
                indices_i
                    .par_iter()
                    .enumerate()
                    .map(|(j, &k)| eq_row[j] * eq_col[k as usize])
                    .sum::<F>()
            })
            .collect();

        drop(_span_eval);
        claim_eval_total += t_claim_eval.elapsed();

        // === Per-family gkr_pre_commit phase ===
        let t_gkr_pre_commit = Instant::now();
        let _span_prep = tracing::info_span!("gkr.gkr_pre_commit", family = family.name).entered();

        // FS-continuity (Gap 1 fix): absorb the d claim values into the
        // transcript BEFORE squeezing r_chunk. Without this, an adversarial
        // prover could choose the d claims after seeing r_chunk and trivially
        // satisfy the §4.5.2 RLC.
        prover_state.prover_messages(&claims);

        // §4.5.2: squeeze r_chunk, form combined claim.
        let r_chunk: Vec<F> = prover_state.verifier_message_vec(log_d);
        let eq_chunk = lsb_eq_table(&r_chunk); // length 2^log_d
        let combined_claim: F = claims
            .iter()
            .enumerate()
            .map(|(i, &c)| c * eq_chunk[i])
            .sum();

        // FS-continuity (Gap 2 fix): absorb the combined claim before
        // squeezing α. The LogUp soundness bound (n+m)/|F| per Lemma 2 of
        // logUp.pdf §4 relies on α being chosen *after* the prover commits
        // to the claim.
        prover_state.prover_messages(&[combined_claim]);

        // Fiat-Shamir α for the GKR.
        let alpha: F = prover_state.verifier_message();

        // Build eq_M_row (length T·d, LSB-first):
        //   eq_M_row[j] = eq_row[j_cycle] · eq_chunk[i_chunk]
        // where j = i_chunk · T + j_cycle  (chunk in high bits, cycle in low bits
        // — matches the §4 paper concatenation M^(*) := M^(0) || M^(1) || …).
        let n_d = t * d;
        let eq_m_row: Vec<F> = (0..n_d)
            .into_par_iter()
            .map(|j| {
                let j_cycle = j & (t - 1);
                let i_chunk = j >> log_t;
                eq_row[j_cycle] * eq_chunk[i_chunk]
            })
            .collect();

        // Build P^F via scatter (sequential — m_pad is small, contention dominates).
        let mut p_f = vec![F::ZERO; m_pad];
        for i_chunk in 0..d {
            let indices_i = family.ra_dense_chunks_u8[i_chunk];
            let base = i_chunk * t;
            for j_cycle in 0..t {
                let k = indices_i[j_cycle] as usize;
                debug_assert!(k < m_pad, "ra_dense index out of pushforward range");
                p_f[k] += eq_m_row[base + j_cycle];
            }
        }

        // Build leaf_denom_a = α − M^(*)[j]
        // M^(*)[j] = ra_dense_chunks_f[i_chunk][j_cycle], so we read from the
        // F-encoded slices.
        let leaf_denom_a: Vec<F> = (0..n_d)
            .into_par_iter()
            .map(|j| {
                let j_cycle = j & (t - 1);
                let i_chunk = j >> log_t;
                alpha - family.ra_dense_chunks_f[i_chunk][j_cycle]
            })
            .collect();

        // Build leaf_denom_b = α − k
        let leaf_denom_b: Vec<F> = (0..m_pad)
            .into_par_iter()
            .map(|k| alpha - F::from(k as u64))
            .collect();

        // Internal soundness check: the §4.5.2 combined claim should equal
        // P̃^F(r_col^F) by the LogUp\* main identity (eq. 5 of the paper).
        // If this fails, the eq-weighted pushforward was built incorrectly.
        let p_eval = mle_eval_lsb(&p_f, &r_col);
        assert!(
            p_eval == combined_claim,
            "LogUp* main identity (M̃^(*)(r_row,r_chunk,r_col) = P̃^F(r_col)) failed for family {}: \
             combined_claim={:?} P̃^F(r_col)={:?}",
            family.name,
            combined_claim,
            p_eval,
        );

        pushforwards.push(p_f.clone());
        family_states.push(FamilyState {
            name: family.name,
            log_t: family.log_t,
            log_d: family.log_d,
            log_m: family.log_m,
            eq_m_row,
            leaf_denom_a,
            pushforward: p_f,
            leaf_denom_b,
        });

        drop(_span_prep);
        gkr_pre_commit_total += t_gkr_pre_commit.elapsed();
    }

    PrepResult {
        pushforwards,
        family_states,
        claim_eval_time: claim_eval_total,
        gkr_pre_commit_time: gkr_pre_commit_total,
    }
}

// =============================================================================
// Phase 3: per-family Figure-1 fractional GKR (eq-weighted).
// =============================================================================

// Generic (field-agnostic) fractional-GKR reference loop. The BN254-only
// benchmark ships `gkr_bn254::run_gkr`; this generic path is retained only as
// the correctness reference exercised by the unit tests below, so it is gated
// to test builds (otherwise it is dead code in the binary).
#[cfg(test)]
#[tracing::instrument(skip_all, name = "gkr.run_gkr", fields(n_families = family_states.len()))]
pub(crate) fn run_gkr<F>(
    prover_state: &mut ProverState,
    family_states: Vec<FamilyState<F>>,
    pre_commit_time: Duration,
) -> Timing
where
    F: ArkField + Codec<[u8]> + Send + Sync,
    Standard: Distribution<F>,
{
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

#[cfg(test)]
#[tracing::instrument(skip_all, name = "gkr.family", fields(name = family.name, log_size_a = family.log_t + family.log_d, log_size_b = family.log_m))]
fn prove_family_gkr<F>(
    prover_state: &mut ProverState,
    family: FamilyState<F>,
    per_layer_times: &mut [Duration],
) where
    F: ArkField + Codec<[u8]> + Send + Sync,
    Standard: Distribution<F>,
{
    let log_size_a = family.log_t + family.log_d;
    let log_size_b = family.log_m;

    // Build A and B circuits from leaves up to root. Single instance each.
    let circuit_a = {
        let _span = tracing::info_span!("gkr.build_circuit_a", log_size = log_size_a).entered();
        build_circuit(family.eq_m_row, family.leaf_denom_a, log_size_a)
    };
    let circuit_b = {
        let _span = tracing::info_span!("gkr.build_circuit_b", log_size = log_size_b).entered();
        build_circuit(family.pushforward, family.leaf_denom_b, log_size_b)
    };

    // Internal soundness assert: histogram identity at the root.
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

    // Top-down per-layer sumcheck on the A-circuit.
    prove_single_instance_sumcheck(
        prover_state,
        &circuit_a,
        log_size_a,
        per_layer_times,
        SumcheckSide::A,
    );

    // Top-down per-layer sumcheck on the B-circuit.
    prove_single_instance_sumcheck(
        prover_state,
        &circuit_b,
        log_size_b,
        per_layer_times,
        SumcheckSide::B,
    );
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum SumcheckSide {
    A,
    B,
}

// =============================================================================
// Circuit (fan-in-2 fractional add) — single instance.
// =============================================================================

/// One gate's (numerator, denominator) pair, packed contiguous so the
/// pyramid traversal reads 1 cache stream instead of 2.
pub(crate) struct Level<F> {
    pub(crate) pairs: Vec<(F, F)>,
}

pub(crate) struct Circuit<F> {
    /// `levels[k]` has `2^k` gates. `levels[0]` is the root (length 1).
    /// `levels[L]` are the leaves (length `2^L`).
    pub(crate) levels: Vec<Level<F>>,
}

impl<F: ArkField + Send + Sync> Circuit<F> {
    pub(crate) fn root(&self) -> (F, F) {
        self.levels[0].pairs[0]
    }
}

#[tracing::instrument(skip_all, name = "gkr.build_circuit", fields(log_size))]
pub(crate) fn build_circuit<F: ArkField + Send + Sync>(
    leaves_num: Vec<F>,
    leaves_denom: Vec<F>,
    log_size: usize,
) -> Circuit<F> {
    debug_assert_eq!(leaves_num.len(), 1 << log_size);
    debug_assert_eq!(leaves_denom.len(), 1 << log_size);
    let mut levels: Vec<Level<F>> = Vec::with_capacity(log_size + 1);
    // Zip leaf nums/denoms into (n, d) pair layout in parallel.
    let leaf_pairs: Vec<(F, F)> = leaves_num
        .into_par_iter()
        .zip(leaves_denom.into_par_iter())
        .collect();
    levels.push(Level { pairs: leaf_pairs });
    while levels.last().unwrap().pairs.len() > 1 {
        let prev = levels.last().unwrap();
        let n = prev.pairs.len() / 2;
        // Per-gate: read 2 contiguous (n, d) pairs (48 B for Fp3) → write
        // one new (n, d) pair. Single cache stream throughout.
        let pairs: Vec<(F, F)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (nl, dl) = prev.pairs[2 * i];
                let (nr, dr) = prev.pairs[2 * i + 1];
                (nl * dr + nr * dl, dl * dr)
            })
            .collect();
        levels.push(Level { pairs });
    }
    // We pushed bottom-up (leaves at index 0). Reverse so levels[0] = root.
    levels.reverse();
    Circuit { levels }
}

// =============================================================================
// Single-instance top-down sumcheck (Gruen / Dao-Thaler eq-absorbing).
//
// The sumcheck claim at iteration k reduces
//   combined_claim = Σ_x eq(point, x) · F_combined(x)
// to a layer-(k+1) claim. F_combined(y) = α·(NL·DR + NR·DL) + β·DL·DR is
// degree 2 in each variable; multiplied by eq, the per-round polynomial is
// nominally degree 3. The Gruen factorization avoids materializing the eq
// table over remaining variables and instead derives the degree-3 round
// polynomial from two per-pair values:
//   q_constant      = F(X=0)     summed over remaining pairs
//   q_quadratic_coef = X² coef    summed over remaining pairs
// The eq factor for already-bound variables accumulates into a scalar
// `current_eq_scalar`. Per-pair work drops from 18 muls to 10 muls (~45%)
// while the wire format (3 field elements per round) is unchanged.
//
// References:
//   - PAZK §4.6.5 (Thaler 2013, eprint 2013/351) — the underlying trick.
//   - jolt/crates/jolt-poly/src/split_eq.rs::gruen_poly_deg_3 — the
//     equivalent formula in Jolt-core. We can't import it from this
//     workspace due to the digest 0.10/0.11 conflict, so we replicate the
//     algorithm locally.
// =============================================================================

/// AoS pack of the four split values at one logical pair-position. Replaces
/// four separate `Vec<F>` arrays (n_l, n_r, d_l, d_r) with one
/// `Vec<PairVals<F>>` — collapsing 4 prefetch streams into 1 and giving the
/// hot inner loop contiguous 4F-byte loads per pair.
#[cfg(test)]
#[derive(Clone, Copy)]
struct PairVals<F: Copy> {
    nl: F,
    nr: F,
    dl: F,
    dr: F,
}

#[cfg(test)]
fn prove_single_instance_sumcheck<F>(
    prover_state: &mut ProverState,
    circuit: &Circuit<F>,
    log_size: usize,
    per_layer_times: &mut [Duration],
    side: SumcheckSide,
) where
    F: ArkField + Codec<[u8]> + Send + Sync,
    Standard: Distribution<F>,
{
    // Send root fractions (4 scalars) and squeeze (α_combine, β_combine).
    let (n_root, d_root) = circuit.root();
    prover_state.prover_messages(&[n_root, d_root]);
    let mut cur_alpha: F = prover_state.verifier_message();
    let mut cur_beta: F = prover_state.verifier_message();

    let mut combined_claim: F = cur_alpha * n_root + cur_beta * d_root;
    let mut point: Vec<F> = Vec::new();

    // Pre-allocate scratch buffers once. Each round halves the working set,
    // so capacity = max layer half-size suffices for all rounds in all layers.
    // The buffers ping-pong via `mem::swap` after each fold.
    //
    // AoS layout: `PairVals { nl, nr, dl, dr }` packs all four values for
    // one logical pair-position into 4×F bytes contiguous. The hot inner
    // loop reads `vals[2i]` and `vals[2i+1]` — two sequential 4F chunks —
    // collapsing the previous 4 separate cache streams (n_l, n_r, d_l, d_r)
    // into one. For Fp3 (F = 24 B), each PairVals is 96 B; per-pair access
    // reads 192 B contiguously, which the M2 prefetcher handles ideally.
    let max_half = if log_size == 0 {
        1
    } else {
        1usize << (log_size - 1)
    };
    let mut vals: Vec<PairVals<F>> = Vec::with_capacity(max_half);
    let mut vals_swap: Vec<PairVals<F>> = Vec::with_capacity(max_half);
    let mut eq_unbound: Vec<F> = Vec::with_capacity(max_half);
    let mut eq_unbound_swap: Vec<F> = Vec::with_capacity(max_half);

    // Layer transitions: claim at levels[k] → claim at levels[k+1]. Layer
    // k's sumcheck has k rounds.
    #[allow(clippy::needless_range_loop)]
    for k in 0..log_size {
        let layer_t0 = Instant::now();
        let _span = match side {
            SumcheckSide::A => tracing::info_span!("gkr.layer.a", k).entered(),
            SumcheckSide::B => tracing::info_span!("gkr.layer.b", k).entered(),
        };

        // Pack the four split values (nl, nr, dl, dr) at each pair-position i
        // into one contiguous PairVals entry. Reads 2 contiguous (n, d) pairs
        // from the level — single cache stream.
        let level_pairs = &circuit.levels[k + 1].pairs;
        let half = level_pairs.len() / 2;
        vals.clear();
        vals.spare_capacity_mut()[..half]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dst)| {
                let (nl, dl) = level_pairs[2 * i];
                let (nr, dr) = level_pairs[2 * i + 1];
                dst.write(PairVals { nl, nr, dl, dr });
            });
        // SAFETY: the parallel loop initialized every slot in `0..half`.
        unsafe {
            vals.set_len(half);
        }

        // Gruen state:
        //   `current_eq_scalar` — bound eq factor `Π_{i<round} eq(point[i], r'_i)`.
        //   `eq_unbound` — eq table for FUTURE variables only, length
        //     2^{k-1-round}. At round j, eq_unbound[i] = eq(point[j+1..],
        //     bits(i)). Updated by pair-sum each round: when we move from
        //     round j to round j+1, the new unbound point coordinates drop
        //     point[j+1], and `eq(point[j+2..], i) = eq_unbound[2i] +
        //     eq_unbound[2i+1]` because `(1−point[j+1]) + point[j+1] = 1`.
        let mut current_eq_scalar: F = F::ONE;
        let eq_unbound_init_len = 1usize << k.saturating_sub(1);
        eq_unbound.clear();
        if k <= 1 {
            eq_unbound.spare_capacity_mut()[0].write(F::ONE);
        } else {
            // eq(point[1..], bits) — covers variables x_1..x_{k-1}. We can't
            // build directly into our scratch buffer without duplicating
            // eq_weights' logic, so allocate temp and copy. Cost is small at
            // k<=24 (worst case 2^22 elements written twice).
            let tmp = lsb_eq_table(&point[1..]);
            debug_assert_eq!(tmp.len(), eq_unbound_init_len);
            eq_unbound.spare_capacity_mut()[..eq_unbound_init_len]
                .par_iter_mut()
                .zip(tmp.par_iter())
                .for_each(|(dst, &src)| {
                    dst.write(src);
                });
        }
        // SAFETY: either the scalar case or the parallel copy initialized
        // every slot in `0..eq_unbound_init_len`.
        unsafe {
            eq_unbound.set_len(eq_unbound_init_len);
        }
        let mut r_prime: Vec<F> = Vec::with_capacity(k);

        // k sumcheck rounds. For k=0 (root → level-1) this loop is empty.
        for round_idx in 0..k {
            let cur_w = point[round_idx];
            let cur_half = vals.len() / 2;
            debug_assert_eq!(cur_half, eq_unbound.len());

            // Per pair, compute eq-weighted contributions:
            //   q_constant      += eq_unbound[i] · F(X=0, pair_i)
            //   q_quadratic_coef += eq_unbound[i] · (X² coef of F at pair_i)
            //
            // F_combined(X) = α·(NL(X)·DR(X) + NR(X)·DL(X)) + β·DL(X)·DR(X)
            // NL(X) = nl0 + X·Δnl, etc.
            //   F(X=0)  = α·(nl0·dr0 + nr0·dl0) + β·dl0·dr0
            //   X² coef = α·(Δnl·Δdr + Δnr·Δdl) + β·Δdl·Δdr
            //
            // The weighting by eq_unbound[i] folds in the eq factor for
            // the future (still-unbound) variables; the current variable's
            // eq factor and the bound eq factor go through Gruen below.
            //
            // AoS access: `vals[2i]` provides (nl0, nr0, dl0, dr0) and
            // `vals[2i+1]` provides (nl1, nr1, dl1, dr1) — two contiguous
            // 4F-byte loads instead of 8 stride-2 loads from 4 arrays.
            let (q_constant, q_quadratic_coef) = (0..cur_half)
                .into_par_iter()
                .map(|i| {
                    let e = eq_unbound[i];
                    let v0 = vals[2 * i];
                    let v1 = vals[2 * i + 1];

                    let f0 = cur_alpha * (v0.nl * v0.dr + v0.nr * v0.dl) + cur_beta * v0.dl * v0.dr;
                    let dnl = v1.nl - v0.nl;
                    let dnr = v1.nr - v0.nr;
                    let ddl = v1.dl - v0.dl;
                    let ddr = v1.dr - v0.dr;
                    let f_x2_coef = cur_alpha * (dnl * ddr + dnr * ddl) + cur_beta * ddl * ddr;

                    (e * f0, e * f_x2_coef)
                })
                .reduce(|| (F::ZERO, F::ZERO), |a, b| (a.0 + b.0, a.1 + b.1));

            // Gruen formula: derive h(0), h(2), h(3) of the cubic round
            // polynomial from (q_constant, q_quadratic_coef, combined_claim).
            //
            // Let Q(X) := q_constant + X·g1 + X²·q_quadratic_coef be the
            // (eq-weighted-over-unbound) degree-2 polynomial. Then
            //   h(X) = current_eq_scalar · eq(cur_w, X) · Q(X)
            // is cubic in X. We use the sumcheck constraint h(0) + h(1) =
            // combined_claim to recover g1 without summing over the
            // hypercube a second time.
            //
            //   eq_eval_0 = current_eq_scalar · (1 − cur_w)
            //   eq_eval_1 = current_eq_scalar · cur_w
            //   h(0) = eq_eval_0 · q_constant
            //   h(1) = combined_claim − h(0)
            //   Q(1) = h(1) / eq_eval_1
            //   Q(2) = 2·Q(1) − q_constant + 2·q_quadratic_coef
            //   Q(3) = Q(2) + Q(1) − q_constant + 4·q_quadratic_coef
            //   h(2) = eq_eval_2 · Q(2),  h(3) = eq_eval_3 · Q(3)
            let eq_eval_1 = current_eq_scalar * cur_w;
            let eq_eval_0 = current_eq_scalar - eq_eval_1;
            let eq_slope = eq_eval_1 - eq_eval_0;
            let eq_eval_2 = eq_eval_1 + eq_slope;
            let eq_eval_3 = eq_eval_2 + eq_slope;

            let h0 = eq_eval_0 * q_constant;
            let h1 = combined_claim - h0;
            // SAFETY: eq_eval_1 = current_eq_scalar · cur_w. Both factors
            // are derived from Fiat-Shamir challenges over a 192/256-bit
            // field; vanishing happens with probability ≤ 1/|F|, which is
            // negligible. An honest prover hits this division.
            let q1 = h1 / eq_eval_1;
            let two_q2 = q_quadratic_coef + q_quadratic_coef;
            let q2 = q1 + q1 - q_constant + two_q2;
            let q3 = q2 + q1 - q_constant + two_q2 + two_q2;
            let h2 = eq_eval_2 * q2;
            let h3 = eq_eval_3 * q3;

            // Send h(0), h(2), h(3). Verifier derives h(1) = current − h(0).
            // Wire format identical to the previous degree-3 path; only the
            // prover's computation changed.
            prover_state.prover_messages(&[h0, h2, h3]);

            // Squeeze round challenge and fold value arrays.
            let r_j: F = prover_state.verifier_message();
            r_prime.push(r_j);

            // Fold via scratch-buffer ping-pong: write the folded result to
            // vals_swap (sized to cur_half), then mem::swap. Single Vec
            // write pass replaces the previous 4 separate fold passes.
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

            // eq_unbound update by pair-sum (NOT a fold by r_j): drops the
            // leading point coordinate point[round_idx+1] from the unbound
            // eq factor. The pair sum equals the eq table over the remaining
            // future variables.
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

            // Update running claim to h(r_j) and accumulate the bound eq
            // factor into current_eq_scalar.
            combined_claim = interpolate_degree3(h0, h1, h2, h3, r_j);
            current_eq_scalar = eq_eval_0 + r_j * (eq_eval_1 - eq_eval_0);
        }

        // After k rounds the array is length 1.
        debug_assert_eq!(vals.len(), 1);
        let n_l_val = vals[0].nl;
        let n_r_val = vals[0].nr;
        let d_l_val = vals[0].dl;
        let d_r_val = vals[0].dr;

        prover_state.prover_messages(&[n_l_val, n_r_val, d_l_val, d_r_val]);

        // Internal sumcheck identity check (Gruen form).
        // combined_claim should equal current_eq_scalar · F(NL, NR, DL, DR)
        // — the fully-bound eq factor times the gate evaluation at the
        // bound point. Equivalent to the previous `eq_table[0] · f_at_r`.
        let f_at_r =
            cur_alpha * (n_l_val * d_r_val + n_r_val * d_l_val) + cur_beta * d_l_val * d_r_val;
        let expected = current_eq_scalar * f_at_r;
        assert_eq!(
            combined_claim, expected,
            "single-instance sumcheck identity failed at layer k={k}"
        );

        // Squeeze merge challenge t and fresh (α', β') for the next layer.
        let t: F = prover_state.verifier_message();
        let new_alpha: F = prover_state.verifier_message();
        let new_beta: F = prover_state.verifier_message();

        // New claim point at level k+1, LSB-first:
        //   (X_k = t, X_{k-1} = r'_0, ..., X_0 = r'_{k-1})
        let mut new_point = Vec::with_capacity(k + 1);
        new_point.push(t);
        new_point.extend(r_prime);

        // New combined claim via low-bit merge of (NL, NR) and (DL, DR).
        let n_combined = n_l_val + t * (n_r_val - n_l_val);
        let d_combined = d_l_val + t * (d_r_val - d_l_val);
        combined_claim = new_alpha * n_combined + new_beta * d_combined;

        cur_alpha = new_alpha;
        cur_beta = new_beta;
        point = new_point;

        per_layer_times[k] += layer_t0.elapsed();
    }

    // Leaf claims sit at level `log_size`. In a real protocol they feed PCS
    // opens on P̃^F / M̃^(*); the bench stops here.
    let _ = (combined_claim, point, cur_alpha, cur_beta);
}

// =============================================================================
// Small helpers (LSB-first eq table + MLE eval, degree-3 interpolation, fold).
// =============================================================================

/// Build `eq[i] = Π_b eq(point[b], i_b)` indexed in LSB-first ordering.
/// WHIR's `eq_weights` is MSB-first; we reverse the point at the call site.
pub(crate) fn lsb_eq_table<F: ArkField + Send + Sync>(point: &[F]) -> Vec<F> {
    if point.is_empty() {
        return vec![F::ONE];
    }
    let reversed: Vec<F> = point.iter().rev().copied().collect();
    eq_weights(&reversed)
}

/// Evaluate the multilinear extension of `evals` at `point` (LSB-first).
/// `evals.len() == 1 << point.len()` (no zero-padding handled).
fn mle_eval_lsb<F: ArkField + Send + Sync>(evals: &[F], point: &[F]) -> F {
    debug_assert_eq!(evals.len(), 1usize << point.len());
    let eq = lsb_eq_table(point);
    evals
        .par_iter()
        .zip(eq.par_iter())
        .map(|(&v, &w)| v * w)
        .sum()
}

/// Lagrange-interpolate a degree-3 polynomial through (0,h0),(1,h1),(2,h2),(3,h3)
/// at point `x`.
#[cfg(test)]
#[inline]
fn interpolate_degree3<F: ArkField>(h0: F, h1: F, h2: F, h3: F, x: F) -> F {
    let one = F::ONE;
    let two = F::from(2u64);
    let three = F::from(3u64);
    let six = F::from(6u64);
    let inv2 = two.inverse().expect("2 invertible");
    let inv6 = six.inverse().expect("6 invertible");

    let xm1 = x - one;
    let xm2 = x - two;
    let xm3 = x - three;

    let l0 = -inv6 * xm1 * xm2 * xm3;
    let l1 = inv2 * x * xm2 * xm3;
    let l2 = -inv2 * x * xm1 * xm3;
    let l3 = inv6 * x * xm1 * xm2;

    h0 * l0 + h1 * l1 + h2 * l2 + h3 * l3
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::rand::rngs::StdRng;
    use ark_std::rand::SeedableRng;
    use whir::algebra::fields::Field64_3;
    use whir::transcript::codecs::Empty;
    use whir::transcript::DomainSeparator;

    type F = Field64_3;

    fn synth_family(d: usize, log_t: usize, log_m: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(seed);
        use ark_std::rand::RngCore;
        let t = 1usize << log_t;
        let k_logical = 1usize << log_m.min(4); // bound indices to a small range
        (0..d)
            .map(|_| {
                (0..t)
                    .map(|_| (rng.next_u32() % k_logical as u32) as u8)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn small_family_round_trips() {
        let log_t = 6;
        let log_d = 2;
        let d = 1usize << log_d;
        let log_m = 4;

        // Build 2 synthetic families of (d, T, m).
        let chunks_indices_a: Vec<Vec<u8>> = synth_family(d, log_t, log_m, 0x1111);
        let chunks_indices_b: Vec<Vec<u8>> = synth_family(d, log_t, log_m, 0x2222);

        // Encode to F.
        let chunks_f_a: Vec<Vec<F>> = chunks_indices_a
            .iter()
            .map(|c| c.iter().map(|&k| F::from(u64::from(k))).collect())
            .collect();
        let chunks_f_b: Vec<Vec<F>> = chunks_indices_b
            .iter()
            .map(|c| c.iter().map(|&k| F::from(u64::from(k))).collect())
            .collect();

        // Borrow into Family slices.
        let ra_dense_chunks_f_a: Vec<&[F]> = chunks_f_a.iter().map(|v| v.as_slice()).collect();
        let ra_dense_chunks_u8_a: Vec<&[u8]> =
            chunks_indices_a.iter().map(|v| v.as_slice()).collect();
        let ra_dense_chunks_f_b: Vec<&[F]> = chunks_f_b.iter().map(|v| v.as_slice()).collect();
        let ra_dense_chunks_u8_b: Vec<&[u8]> =
            chunks_indices_b.iter().map(|v| v.as_slice()).collect();

        let families = vec![
            Family {
                name: "synthA",
                d,
                log_t,
                log_d,
                log_m,
                ra_dense_chunks_f: ra_dense_chunks_f_a,
                ra_dense_chunks_u8: ra_dense_chunks_u8_a,
            },
            Family {
                name: "synthB",
                d,
                log_t,
                log_d,
                log_m,
                ra_dense_chunks_f: ra_dense_chunks_f_b,
                ra_dense_chunks_u8: ra_dense_chunks_u8_b,
            },
        ];

        let ds = DomainSeparator::protocol(&"gkr-test")
            .session(&"round-trip")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let prep = prepare_pushforwards::<F>(&mut ps, &families);
        assert_eq!(prep.pushforwards.len(), 2);
        for pf in &prep.pushforwards {
            assert_eq!(pf.len(), 1usize << log_m);
        }
        let timing = run_gkr::<F>(&mut ps, prep.family_states, prep.gkr_pre_commit_time);
        assert_eq!(timing.per_family_gkr.len(), 2);
        assert_eq!(timing.per_layer_gkr.len(), log_t + log_d);
    }

    #[test]
    #[should_panic(expected = "GKR histogram identity violated")]
    fn bad_pushforward_panics() {
        // If we corrupt P^F after prep but before GKR, the root cross-check
        // inside `prove_family_gkr` must fire. (Corrupting the F-encoded
        // ra_dense values without changing u8 indices would only fail in
        // GKR — the prep's internal LogUp* main-identity check uses u8
        // indices for both sides.)
        let log_t = 5;
        let log_d = 1;
        let d = 1usize << log_d;
        let log_m = 4;

        let chunks_indices: Vec<Vec<u8>> = synth_family(d, log_t, log_m, 0xBEEF);
        let chunks_f: Vec<Vec<F>> = chunks_indices
            .iter()
            .map(|c| c.iter().map(|&k| F::from(u64::from(k))).collect())
            .collect();
        let ra_dense_chunks_f: Vec<&[F]> = chunks_f.iter().map(|v| v.as_slice()).collect();
        let ra_dense_chunks_u8: Vec<&[u8]> = chunks_indices.iter().map(|v| v.as_slice()).collect();

        let families = vec![Family {
            name: "bogus",
            d,
            log_t,
            log_d,
            log_m,
            ra_dense_chunks_f,
            ra_dense_chunks_u8,
        }];

        let ds = DomainSeparator::protocol(&"gkr-test")
            .session(&"bad-pushforward")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let mut prep = prepare_pushforwards::<F>(&mut ps, &families);

        // Tamper with the pushforward in family-state (and the pushforwards list).
        prep.family_states[0].pushforward[0] += F::ONE;

        let mut gkr_ps = ProverState::new_std(&ds);
        let _ = run_gkr::<F>(&mut gkr_ps, prep.family_states, prep.gkr_pre_commit_time);
    }
}
