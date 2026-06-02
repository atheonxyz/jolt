//! Wide-limb RANGE CHECK stage (piece R) — proves every committed limb cell lies in `[0, 2^log_m)`
//! via a LogUp\* set-membership argument, reusing the M7 pushforward-GKR stack verbatim.
//!
//! ## Why the M7 family proves range MEMBERSHIP (not just pushforward consistency)
//!
//! A range check is the LogUp membership lookup `{col[j]} ⊆ {0..2^k−1}`. The M7 per-family GKR
//! ([`prove_family`]) proves the fractional identity
//! `Σ_j eq(j, r_row)/(α − M*[j]) == Σ_k P^F[k]/(α − k)` and (a) pins the B-side denominators to the
//! public table `α − k`, `k ∈ [0, 2^log_m)` via the B-leaf structural check, and (b) ties the two
//! sums via the root histogram `N_A·D_B == N_B·D_A`. At the Fiat-Shamir `α` the histogram is a
//! rational-function identity (Schwartz–Zippel / LogUp Lemma 2), so the A-side poles `{M*[j]}` — the
//! committed limb values — MUST be a subset of the B-side poles `{0..2^log_m−1}`. That is exactly
//! range membership. The eq-weighting only shapes the residues (`P^F[k]`), never the pole structure,
//! so it does not affect the membership guarantee. So R needs no new GKR: it is one `log_d > 0`
//! family per limb-width group (all columns of a width row-concatenated into `M*`, sharing one
//! multiplicity table `P^F`).
//!
//! ## Openings / stage-8
//!
//! Identical to the M7 families: per group `prove_family` appends `RaDense(base_index)` (the limb
//! cells `M̃*`) and `Pushforward(base_index)` (the multiplicity `P^F`) openings to the accumulator,
//! discharged by the deferred stage-8 WHIR open (P8/P9). `base_index` keys the group after the M7
//! families (Instruction/Bytecode/Ram) so the global chunk indices do not collide — the full
//! `prove()` driver assigns the next free index.
//!
//! ## Out-of-range inputs
//!
//! Rejected up front with [`RangeCheckError::OutOfRange`]: the underlying pushforward scatter would
//! otherwise index past the `2^log_m` table. An honest prover's witness is always in range, so this
//! is a prover-side guard; the verifier-side membership soundness rests on the GKR root histogram
//! above (a malicious commitment is caught there, never reaching this code path).

use jolt_field::Field;

use crate::framework::accumulator::Openings;
use crate::framework::transcript::{ProverFs, VerifierFs};
use crate::zkvm::logup::driver::{prove_family, verify_family, FamilyVerifierParams};
use crate::zkvm::logup::gkr::GkrProof;
use crate::zkvm::logup::pushforward::{claim_eval, Family};
use crate::zkvm::logup::GkrError;

const RANGE_CHECK_NAME: &str = "RangeCheck";

/// Range-check stage failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeCheckError {
    /// A supplied limb value was `≥ 2^log_m` (does not fit the range table).
    OutOfRange,
    /// The underlying LogUp\*-GKR membership proof was rejected.
    Gkr(GkrError),
}

/// The range-check proof for one limb-width group: the (data-less marker) family GKR proof plus the
/// `d` input membership claims `M̃^(i)(r_row, r_col)` the verifier replays (it cannot recompute them
/// without the committed columns). `r_row`/`r_col` are squeezed on both sides, so they are not carried.
#[derive(Clone, Debug)]
pub struct RangeCheckProof<F: Field> {
    pub gkr: GkrProof<F>,
    pub claims: Vec<F>,
}

#[inline]
fn log2_ceil(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

/// Prove every entry of every `column` is in `[0, 2^log_m)`. Columns have length `T = 2^log_t` and
/// share one `[0, 2^log_m)` table (one multiplicity `P^F`). Returns [`RangeCheckError::OutOfRange`]
/// if any entry is `≥ 2^log_m`. The GKR-leaf + multiplicity openings land in `accumulator` under
/// `RaDense(base_index)` / `Pushforward(base_index)` for the stage-8 open.
pub fn prove_range_check<F, T>(
    columns: &[Vec<u32>],
    log_t: usize,
    log_m: usize,
    base_index: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<RangeCheckProof<F>, RangeCheckError>
where
    F: Field,
    T: ProverFs<F>,
{
    let t = 1usize << log_t;
    let m_pad = 1usize << log_m;
    assert!(!columns.is_empty(), "range check needs at least one column");
    for col in columns {
        assert_eq!(col.len(), t, "range-check column must have length 2^log_t");
        if col.iter().any(|&v| (v as usize) >= m_pad) {
            return Err(RangeCheckError::OutOfRange);
        }
    }

    // Row-concatenate the columns into one family (padded to a power of two with all-zero columns,
    // which are trivially in range) so all columns share one multiplicity table `P^F`.
    let log_d = log2_ceil(columns.len());
    let mut indices: Vec<Vec<u32>> = columns.to_vec();
    indices.resize(1usize << log_d, vec![0u32; t]);

    let r_row = transcript.challenge_vector(log_t);
    let r_col = transcript.challenge_vector(log_m);
    let family = Family {
        name: RANGE_CHECK_NAME,
        log_t,
        log_d,
        log_m,
        r_row,
        r_col,
        indices,
    };
    let claims = claim_eval(&family);
    let gkr = prove_family(&family, &claims, base_index, accumulator, transcript)
        .map_err(RangeCheckError::Gkr)?;
    Ok(RangeCheckProof { gkr, claims })
}

/// Verify a range-check proof (mirror of [`prove_range_check`]). `num_columns` must match the prover's
/// column count (it determines `log_d`).
pub fn verify_range_check<F, T>(
    proof: &RangeCheckProof<F>,
    log_t: usize,
    log_m: usize,
    num_columns: usize,
    base_index: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), RangeCheckError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let log_d = log2_ceil(num_columns);
    let r_row = transcript.challenge_vector(log_t);
    let r_col = transcript.challenge_vector(log_m);
    let params = FamilyVerifierParams {
        log_t,
        log_d,
        log_m,
        r_row,
        r_col,
    };
    verify_family(
        &params,
        &proof.claims,
        &proof.gkr,
        base_index,
        accumulator,
        transcript,
    )
    .map_err(RangeCheckError::Gkr)
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::accumulator::{CommittedPolynomial, OpeningAccumulator, SumcheckId};
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

    fn in_range_columns(rng: &mut Rng, n_cols: usize, t: usize, m_pad: u32) -> Vec<Vec<u32>> {
        (0..n_cols)
            .map(|_| (0..t).map(|_| (rng.next() as u32) % m_pad).collect())
            .collect()
    }

    fn round_trip(seed: u64, log_t: usize, log_m: usize, n_cols: usize) {
        let t = 1usize << log_t;
        let m_pad = 1u32 << log_m;
        let mut rng = Rng(seed);
        let columns = in_range_columns(&mut rng, n_cols, t, m_pad);

        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = ProverTranscript::new("range-check");
        let proof =
            prove_range_check::<F, _>(&columns, log_t, log_m, 0, &mut prover_acc, &mut prover_t)
                .expect("in-range columns must prove");
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = VerifierTranscript::new("range-check", &narg);
        verify_range_check::<F, _>(
            &proof,
            log_t,
            log_m,
            n_cols,
            0,
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("range check must verify");

        // The committed limb (RaDense) + multiplicity (Pushforward) openings agree across sides.
        for (poly, sc) in [
            (CommittedPolynomial::RaDense(0), SumcheckId::PushforwardGkr),
            (
                CommittedPolynomial::Pushforward(0),
                SumcheckId::PushforwardGkr,
            ),
            (
                CommittedPolynomial::Pushforward(0),
                SumcheckId::PushforwardReduction,
            ),
        ] {
            let (pp, pc) = prover_acc.get_committed_polynomial_opening(poly, sc);
            let (vp, vc) = verifier_acc.get_committed_polynomial_opening(poly, sc);
            assert_eq!(pp, vp, "opening point agrees for {poly:?}/{sc:?}");
            assert_eq!(pc, vc, "opening claim agrees for {poly:?}/{sc:?}");
        }
    }

    #[test]
    fn range_check_round_trip() {
        round_trip(0x9001, 4, 4, 1); // single column, log_d = 0
        round_trip(0x9002, 5, 4, 3); // 3 columns, padded to 4
        round_trip(0x9003, 4, 5, 2);
        round_trip(0x9004, 6, 3, 4);
    }

    /// An out-of-range entry is rejected up front (before the pushforward scatter could panic).
    #[test]
    fn out_of_range_rejected() {
        let (log_t, log_m) = (4usize, 4usize);
        let t = 1usize << log_t;
        let mut rng = Rng(0x90FE);
        let mut columns = in_range_columns(&mut rng, 1, t, 1u32 << log_m);
        columns[0][3] = (1u32 << log_m) + 5; // ≥ 2^log_m
        let mut acc = Openings::<F>::new(log_t);
        let mut transcript = ProverTranscript::new("range-check");
        assert!(matches!(
            prove_range_check::<F, _>(&columns, log_t, log_m, 0, &mut acc, &mut transcript),
            Err(RangeCheckError::OutOfRange),
        ));
    }

    /// A tampered NARG (corrupted membership-claim / GKR byte) is rejected by the verifier.
    #[test]
    fn tampered_proof_rejected() {
        let (log_t, log_m, n_cols) = (4usize, 4usize, 2usize);
        let t = 1usize << log_t;
        let mut rng = Rng(0x90AB);
        let columns = in_range_columns(&mut rng, n_cols, t, 1u32 << log_m);
        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = ProverTranscript::new("range-check");
        let proof =
            prove_range_check::<F, _>(&columns, log_t, log_m, 0, &mut prover_acc, &mut prover_t)
                .expect("prove");
        let mut narg = prover_t.into_proof();
        narg.narg_string[0] ^= 0x01;

        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = VerifierTranscript::new("range-check", &narg);
        assert!(
            verify_range_check::<F, _>(
                &proof,
                log_t,
                log_m,
                n_cols,
                0,
                &mut verifier_acc,
                &mut verifier_t,
            )
            .is_err(),
            "tampered range-check proof must be rejected"
        );
    }
}
