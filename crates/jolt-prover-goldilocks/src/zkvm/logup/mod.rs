//! Twist/Shout-via-LogUp\* commitment — the per-family pushforward-GKR opening.
//!
//! Ported from the prover-only prototype [`crates/whir-pcs-bench/src/gkr.rs`] (Wiese 2025,
//! `twist_shout_logup_star.pdf` §4 Figure 1 + §4.1 batching + §4.5.2 claim reduction;
//! `gkr4logup.pdf` / `logUp.pdf` for the fan-in-2 fractional GKR; `ProofsArgsAndZK.pdf` §4.5.2).
//! Retargeted onto [`crate::framework`] over the lean `Field` (`C = F = Fp3`), expressed as
//! framework [`SumcheckInstance`](crate::framework::sumcheck::SumcheckInstance)s on the one shared
//! transcript, and given a real prover→verifier round-trip (the prototype is prover-only — its three
//! internal `assert!`s become fallible [`GkrError`] checks here).
//!
//! Per committed-witness family `F` (`InstructionRa`/`BytecodeRa`/`RamRa`), to open `M̃(r_row,r_col)`
//! for the one-hot read matrix without committing the one-hot `ra`, the prover commits the
//! eq-weighted pushforward `P^F[k] = Σ_{j: M*[j]=k} eq(bits(j), r_M_row)` of the dense `ra_dense`
//! (the d address-chunk index columns row-concatenated into `M* ∈ Fp^{T·d}`, §4.1) and proves it is
//! the genuine pushforward via a fan-in-2 fractional-add GKR:
//! `Σ_j eq(bits(j),r_M_row)/(α − M*[j]) == Σ_k P^F[k]/(α − k)` (degree-3 per layer, Gruen
//! eq-factorization; A-side depth `log_t+log_d`, B-side depth `log_m`). At the leaves the protocol
//! produces two openings — on `ra_dense` (`M̃*`) and `P^F` (`P̃`) — plus the §4.5.2 reduction's
//! `P̃^F(r_col)` claim; all feed the [`Openings`](crate::framework::accumulator::Openings)
//! accumulator for the stage-8 `WhirScheme` batch open (M8).
//!
//! **Bit-ordering:** the module is **LSB-first** (the paper's variable order). The framework's
//! `EqPolynomial::evals` is MSB-first, so [`lsb_eq_table`] reverses the point; each GKR layer's
//! [`GruenSplitEqPolynomial`](jolt_poly::GruenSplitEqPolynomial) is built from the **reversed** layer
//! point so its high-to-low binding matches the LSB-first eq factor (mirrors `output_check.rs`).
//!
//! **Decoupled from the trace / read-raf** (the M5 convention): the prep takes the materialized
//! `ra_dense` chunk columns + the shared `(r_row, r_col)` directly. In the integrated prover the
//! `(r_row, r_col)` and the d input claims come from the read-raf sumcheck's cached `ra_i` openings
//! (the `m7-logupstar-readraf-relationship` note); the `driver` module wires that hand-off. The §4.1
//! streaming-pyramid memory optimization for `K = 2³²` (design §13-Q4) is deferred (TODO in `gkr`).

use jolt_field::Field;
use jolt_poly::EqPolynomial;

pub mod pushforward;

/// LogUp\*-GKR verification / preparation error. The three prototype `assert!`s
/// (`gkr.rs:278` main identity eq. 5, `gkr.rs:377` root histogram, `gkr.rs:744` per-layer
/// consistency) become these fallible checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GkrError {
    /// The eq. 5 main identity `M̃*(r_row,r_chunk,r_col) = P̃^F(r_col)` failed during prep — the
    /// eq-weighted pushforward was built inconsistently with `ra_dense`.
    MainIdentity,
    /// The GKR root histogram `N_A·D_B ≠ N_B·D_A` — the two fractional sums disagree, so the
    /// committed `P^F` is not the genuine pushforward of `ra_dense`.
    RootHistogram,
    /// A per-layer sumcheck's reduced claim did not equal `eq(point, r')·F(NL,NR,DL,DR)`.
    LayerConsistency { circuit: char, layer: usize },
    /// A GKR-leaf numerator/denominator disagreed with its public structural MLE (`eq_m_row` on the
    /// A-side, the index polynomial `α − k` on the B-side).
    LeafStructural { circuit: char },
    /// The underlying framework sumcheck verifier rejected a layer proof.
    Sumcheck,
}

impl std::fmt::Display for GkrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MainIdentity => write!(f, "LogUp* main identity (eq. 5) failed"),
            Self::RootHistogram => write!(f, "GKR root histogram N_A·D_B ≠ N_B·D_A"),
            Self::LayerConsistency { circuit, layer } => {
                write!(
                    f,
                    "per-layer sumcheck consistency failed (circuit {circuit}, layer {layer})"
                )
            }
            Self::LeafStructural { circuit } => {
                write!(f, "GKR leaf structural check failed (circuit {circuit})")
            }
            Self::Sumcheck => write!(f, "framework sumcheck verifier rejected a GKR layer"),
        }
    }
}

impl std::error::Error for GkrError {}

/// Build `eq[i] = Π_b eq(point[b], i_b)` in **LSB-first** index order (`point[0]` ↔ bit 0 of `i`).
///
/// The framework's [`EqPolynomial::evals`] is MSB-first (`r[0]` ↔ the MSB); reversing the point
/// bridges to the LogUp\* paper's LSB-first variable order (mirrors the prototype's `lsb_eq_table`).
pub fn lsb_eq_table<F: Field>(point: &[F]) -> Vec<F> {
    if point.is_empty() {
        return vec![F::one()];
    }
    let reversed: Vec<F> = point.iter().rev().copied().collect();
    EqPolynomial::<F>::evals(&reversed, None)
}

/// Evaluate the multilinear extension of a dense `evals` table at `point` (LSB-first).
/// `evals.len()` must equal `1 << point.len()`.
pub fn mle_eval_lsb<F: Field>(evals: &[F], point: &[F]) -> F {
    debug_assert_eq!(evals.len(), 1usize << point.len());
    let eq = lsb_eq_table(point);
    evals
        .iter()
        .zip(eq.iter())
        .fold(F::zero(), |acc, (&v, &e)| acc + v * e)
}

/// MLE of the index polynomial `idx(k) = k` at `point` (LSB-first): `Σ_b 2^b · point[b]`.
///
/// `leaf_denom_b[k] = α − k` so the B-circuit's denominator leaf must equal `α − idx_mle(point)`.
pub fn idx_mle_lsb<F: Field>(point: &[F]) -> F {
    let mut acc = F::zero();
    let mut weight = F::one();
    let two = F::from_u64(2);
    for &p in point {
        acc += weight * p;
        weight *= two;
    }
    acc
}
