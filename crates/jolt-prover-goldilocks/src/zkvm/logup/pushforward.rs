//! Pushforward preparation: the §4.5.2 claim reduction + the eq-weighted pushforward `P^F` build.
//!
//! Ported from `crates/whir-pcs-bench/src/gkr.rs::prepare_pushforwards` (the `claim_eval` + §4.5.2
//! RLC + `eq_M_row`/`P^F` build phases). Per family, with the shared `(r_row, r_col)` and the d
//! `ra_dense` address-chunk index columns:
//!
//! 1. **`claim_eval`** — the d MLE evaluations `M̃^(i)(r_row, r_col) = Σ_j eq(j,r_row)·eq(idx_i[j],r_col)`
//!    (in the integrated prover these are the read-raf sumcheck's cached `ra_i` openings; the
//!    decoupled prep recomputes them from the index columns — see [`super`] / `driver`).
//! 2. **§4.5.2** — absorb the d claims, squeeze `r_chunk` (len `log_d`), form the combined claim
//!    `combined = Σ_i M̃^(i)(r_row,r_col)·eq(i, r_chunk) = M̃*(r_row, r_chunk, r_col)` (the d points
//!    share `(r_row,r_col)`, so §4.5.2's canonical-curve construction collapses to this eq-weighted
//!    RLC on the chunk dimension); absorb `combined`, then squeeze the GKR challenge `α`.
//! 3. Build the A-circuit leaves `eq_m_row[j] = eq(j, r_M_row)` (`r_M_row = r_row ++ r_chunk`) and
//!    `leaf_denom_a[j] = α − M*[j]`, the eq-weighted pushforward `P^F` (scatter), and the B-circuit
//!    leaves `(P^F, leaf_denom_b[k] = α − k)`.
//! 4. **eq. 5 main identity** (prototype `assert!` → fallible): `combined == P̃^F(r_col)`. A failure
//!    means `P^F` is not the genuine eq-weighted pushforward of `ra_dense`.
//!
//! The Fiat-Shamir absorbs (d claims before `r_chunk`; `combined` before `α`) are soundness-load-
//! bearing — the §4.5.2 RLC and the LogUp Lemma-2 bound require the challenges follow the claims.

use jolt_field::Field;
use jolt_transcript::Transcript;

use super::{lsb_eq_table, mle_eval_lsb, GkrError};

/// One LogUp\* family's decoupled inputs: the materialized `ra_dense` index columns + the shared
/// `(r_row, r_col)` opening point (cycle / column) inherited from the upstream read-raf sumcheck.
#[derive(Clone, Debug)]
pub struct Family<F: Field> {
    pub name: &'static str,
    pub log_t: usize,
    /// `d = 1 << log_d` address chunks per family (InstructionRa: 5; BytecodeRa/RamRa: 2).
    pub log_d: usize,
    /// Padded column width: `P^F` has length `2^log_m` and indices are `< 2^log_m`.
    pub log_m: usize,
    /// Shared cycle point `r_row ∈ F^{log_t}` (the read-raf `r_cycle`).
    pub r_row: Vec<F>,
    /// Shared column point `r_col ∈ F^{log_m}` (the read-raf address point).
    pub r_col: Vec<F>,
    /// `d` index columns, each length `T = 2^log_t`, entries `< 2^log_m`. `M*[i·T + j] = idx[i][j]`.
    pub indices: Vec<Vec<u32>>,
}

impl<F: Field> Family<F> {
    /// `d = 1 << log_d` (the chunk count; row-concatenated into `M* ∈ F^{T·d}`).
    #[inline]
    pub fn d(&self) -> usize {
        1usize << self.log_d
    }
}

/// Prover-side prepared state for one family: the four GKR-circuit leaf arrays plus the metadata
/// the verifier needs to recompute structural leaves and run the histogram check.
#[derive(Clone, Debug)]
pub struct PushforwardData<F: Field> {
    pub name: &'static str,
    pub log_t: usize,
    pub log_d: usize,
    pub log_m: usize,
    /// `r_M_row = r_row ++ r_chunk` (len `log_t + log_d`); the A-circuit numerator is `eq(·, r_M_row)`.
    pub r_m_row: Vec<F>,
    /// Shared column point (len `log_m`); the §4.5.2 `P^F` opening is at this point.
    pub r_col: Vec<F>,
    /// GKR challenge `α` (leaf denominators are `α − value`).
    pub alpha: F,
    /// `M̃*(r_row, r_chunk, r_col) = P̃^F(r_col)` — the upstream claim, reduced onto `P^F`.
    pub combined_claim: F,
    /// A-circuit numerator leaves `eq_m_row[j] = eq(j, r_M_row)`, length `T·d`.
    pub eq_m_row: Vec<F>,
    /// A-circuit denominator leaves `α − M*[j]`, length `T·d`.
    pub leaf_denom_a: Vec<F>,
    /// B-circuit numerator leaves — the eq-weighted pushforward `P^F`, length `2^log_m`.
    pub pushforward: Vec<F>,
    /// B-circuit denominator leaves `α − k`, length `2^log_m`.
    pub leaf_denom_b: Vec<F>,
}

impl<F: Field> PushforwardData<F> {
    #[inline]
    pub fn log_size_a(&self) -> usize {
        self.log_t + self.log_d
    }

    #[inline]
    pub fn log_size_b(&self) -> usize {
        self.log_m
    }
}

/// `claim_eval`: the d MLE evaluations `M̃^(i)(r_row, r_col)` from the index columns.
///
/// `M̃^(i)(r_row, r_col) = Σ_j eq(j, r_row) · eq(idx_i[j], r_col)` (one-hot: row `j` selects column
/// `idx_i[j]`). These are the §4.5.2 inputs; in the integrated prover they are the read-raf cached
/// `ra_i` openings.
pub fn claim_eval<F: Field>(family: &Family<F>) -> Vec<F> {
    let t = 1usize << family.log_t;
    let eq_row = lsb_eq_table(&family.r_row);
    let eq_col = lsb_eq_table(&family.r_col);
    debug_assert_eq!(eq_row.len(), t);
    family
        .indices
        .iter()
        .map(|idx_i| {
            debug_assert_eq!(idx_i.len(), t);
            idx_i.iter().enumerate().fold(F::zero(), |acc, (j, &k)| {
                acc + eq_row[j] * eq_col[k as usize]
            })
        })
        .collect()
}

/// Run the §4.5.2 reduction + the pushforward build for one family, threading the shared transcript.
///
/// `claims` are the d `M̃^(i)(r_row, r_col)` evaluations ([`claim_eval`], or the upstream read-raf
/// openings). Returns the prover-side [`PushforwardData`] or [`GkrError::MainIdentity`] if eq. 5
/// fails.
pub fn prepare_family<F, T>(
    family: &Family<F>,
    claims: &[F],
    transcript: &mut T,
) -> Result<PushforwardData<F>, GkrError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let d = family.d();
    assert_eq!(claims.len(), d, "expected d = 2^log_d claims");
    assert_eq!(family.indices.len(), d, "expected d index columns");
    let t = 1usize << family.log_t;
    let m_pad = 1usize << family.log_m;

    // §4.5.2 Fiat-Shamir continuity: absorb the d claims BEFORE squeezing r_chunk, so an adversary
    // cannot pick the claims after seeing r_chunk and trivially satisfy the RLC.
    for c in claims {
        transcript.append(c);
    }
    let r_chunk = transcript.challenge_vector(family.log_d);
    let eq_chunk = lsb_eq_table(&r_chunk);
    debug_assert_eq!(eq_chunk.len(), d);

    let combined_claim = claims
        .iter()
        .zip(eq_chunk.iter())
        .fold(F::zero(), |acc, (&c, &e)| acc + c * e);

    // Absorb the combined claim before squeezing α (the LogUp Lemma-2 (n+m)/|F| bound requires α
    // after the prover commits to the claim).
    transcript.append(&combined_claim);
    let alpha: F = transcript.challenge();

    // r_M_row = r_row ++ r_chunk (cycle in the low log_t bits, chunk in the high log_d bits — matches
    // the M* concatenation M*[i·T + j] = idx_i[j]).
    let mut r_m_row = family.r_row.clone();
    r_m_row.extend_from_slice(&r_chunk);

    let eq_m_row = lsb_eq_table(&r_m_row); // length T·d, eq_m_row[j] = eq(j, r_M_row)
    debug_assert_eq!(eq_m_row.len(), t * d);

    let mut leaf_denom_a = Vec::with_capacity(t * d);
    let mut pushforward = vec![F::zero(); m_pad];
    for (i_chunk, idx_col) in family.indices.iter().enumerate() {
        let base = i_chunk * t;
        for (j_cycle, &k) in idx_col.iter().enumerate() {
            debug_assert!(
                (k as usize) < m_pad,
                "ra_dense index out of pushforward range"
            );
            leaf_denom_a.push(alpha - F::from_u64(u64::from(k)));
            pushforward[k as usize] += eq_m_row[base + j_cycle];
        }
    }

    let leaf_denom_b: Vec<F> = (0..m_pad).map(|k| alpha - F::from_u64(k as u64)).collect();

    // eq. 5 main identity: M̃*(r_row, r_chunk, r_col) == P̃^F(r_col).
    if mle_eval_lsb(&pushforward, &family.r_col) != combined_claim {
        return Err(GkrError::MainIdentity);
    }

    Ok(PushforwardData {
        name: family.name,
        log_t: family.log_t,
        log_d: family.log_d,
        log_m: family.log_m,
        r_m_row,
        r_col: family.r_col.clone(),
        alpha,
        combined_claim,
        eq_m_row,
        leaf_denom_a,
        pushforward,
        leaf_denom_b,
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_transcript::{Blake2bTranscript, Transcript};

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

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<F> {
        (0..n).map(|_| F::from_u64(rng.next())).collect()
    }

    fn synth_family(rng: &mut Rng, log_t: usize, log_d: usize, log_m: usize) -> Family<F> {
        let t = 1usize << log_t;
        let d = 1usize << log_d;
        // Keep indices in a small logical range (≤ 2^min(log_m,4)) so multiple rows collide into the
        // same pushforward bucket (exercises the scatter accumulation).
        let k_logical = 1u32 << log_m.min(4);
        let indices: Vec<Vec<u32>> = (0..d)
            .map(|_| (0..t).map(|_| (rng.next() as u32) % k_logical).collect())
            .collect();
        Family {
            name: "synth",
            log_t,
            log_d,
            log_m,
            r_row: rand_vec(rng, log_t),
            r_col: rand_vec(rng, log_m),
            indices,
        }
    }

    /// The eq. 5 main identity holds for honest data: `combined == P̃^F(r_col)`, and the leaf arrays
    /// have the expected shapes. Also checks the §4.5.2 RLC against a direct recomputation.
    fn main_identity_holds(seed: u64, log_t: usize, log_d: usize, log_m: usize) {
        let mut rng = Rng(seed);
        let family = synth_family(&mut rng, log_t, log_d, log_m);
        let claims = claim_eval(&family);
        assert_eq!(claims.len(), 1 << log_d);

        let mut transcript = Blake2bTranscript::<F>::new(b"logup-prep");
        let data = prepare_family(&family, &claims, &mut transcript).unwrap();

        assert_eq!(data.eq_m_row.len(), (1 << log_t) * (1 << log_d));
        assert_eq!(data.leaf_denom_a.len(), (1 << log_t) * (1 << log_d));
        assert_eq!(data.pushforward.len(), 1 << log_m);
        assert_eq!(data.leaf_denom_b.len(), 1 << log_m);
        assert_eq!(data.r_m_row.len(), log_t + log_d);

        // combined == P̃^F(r_col) (the prep would have errored otherwise; assert it explicitly).
        assert_eq!(
            data.combined_claim,
            mle_eval_lsb(&data.pushforward, &family.r_col)
        );

        // P^F is the genuine eq-weighted pushforward: Σ_k P^F[k] == Σ_j eq_m_row[j].
        let sum_pf: F = data.pushforward.iter().copied().sum();
        let sum_eq: F = data.eq_m_row.iter().copied().sum();
        assert_eq!(sum_pf, sum_eq, "pushforward total mass = Σ eq_m_row");
    }

    #[test]
    fn pushforward_main_identity() {
        main_identity_holds(0xF001, 6, 2, 4);
        main_identity_holds(0xF002, 5, 1, 4);
        main_identity_holds(0xF003, 4, 3, 5);
        main_identity_holds(0xF004, 7, 2, 3);
    }

    /// A `P^F` inconsistent with `ra_dense` (forced by perturbing one claim so the §4.5.2 combined
    /// claim no longer matches the genuine pushforward) trips the eq. 5 fallible check.
    #[test]
    fn corrupted_claim_trips_main_identity() {
        let mut rng = Rng(0xFEED);
        let family = synth_family(&mut rng, 5, 2, 4);
        let mut claims = claim_eval(&family);
        // Perturb a single input claim: combined = Σ claims·eq_chunk no longer equals P̃^F(r_col),
        // which is built from the (unchanged) genuine index columns.
        claims[0] += F::from_u64(1);
        let mut transcript = Blake2bTranscript::<F>::new(b"logup-prep");
        assert!(matches!(
            prepare_family(&family, &claims, &mut transcript),
            Err(GkrError::MainIdentity),
        ));
    }
}
