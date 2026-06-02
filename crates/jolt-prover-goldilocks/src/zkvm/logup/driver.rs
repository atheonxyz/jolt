//! Per-family driver: source the pushforward-GKR's §4.5.2 input claims from the upstream read-raf
//! sumcheck, then run [`prove_family_gkr`](super::gkr::prove_family_gkr) /
//! [`verify_family_gkr`](super::gkr::verify_family_gkr).
//!
//! This is the M7 entry point the M8 stage driver calls per committed-witness family. The read-raf
//! sumcheck ([`shout_read_raf`](crate::zkvm::shout_read_raf)) is **unchanged** by M7 — only the
//! commitment/opening of `ra` changes (one-hot → `ra_dense` + pushforward-GKR). Its cached per-chunk
//! `ra_i(r_k_i, r_cycle)` openings ARE the one-hot evaluations `M̃^(i)` the §4.5.2 reduction
//! consumes (the `m7-logupstar-readraf-relationship` note; demonstrated by the integration test).
//!
//! **M8 read-raf reconciliation.** Two gaps between the read-raf's outputs and the §4.5.2 inputs:
//! 1. *Bit-ordering.* The read-raf builds its eq tables MSB-first (`EqPolynomial::evals`) and caches
//!    openings at `reverse(challenges)`; this module is LSB-first. The bridge is a point reversal:
//!    the read-raf's `ra_i` equals [`claim_eval`](super::pushforward::claim_eval) of chunk `i` at
//!    `(reverse(r_cycle), reverse(r_k_i))`.
//! 2. *Distinct column points — RESOLVED via Option C ([`prove_family_per_chunk`]).* The read-raf
//!    opens the `d` chunks at **distinct** column points `r_k_i`, but the §4.1 row-concatenated
//!    pushforward (design §1A, prototype) assumes a **shared** `(r_row, r_col)`. M8 discharges each
//!    chunk with its *own* pushforward GKR at its own `r_k_i` — the `log_d = 0` special case of
//!    [`prove_family`], base identity `M̃^(i)(r_cycle, r_k_i) = P̃^F_i(r_k_i)` — rather than
//!    row-concatenating + reducing to a shared point. This defers the §4.1 single-`P^F` optimization
//!    (and its shared-point reduction) to OPT-D / full-`d`; it adds no new soundness-critical math
//!    (reuses the M7 GKR verbatim) and is faithful to what the read-raf produces. See the
//!    `m7-readraf-shared-point-gap` memory for the full rationale.
//!
//! [`prove_family`] (and the `log_d > 0` row-concatenated form) is retained for that future OPT-D
//! path; the M8 stage driver calls [`prove_family_per_chunk`].

use jolt_field::Field;

use crate::framework::accumulator::Openings;
use crate::framework::transcript::{ProverFs, VerifierFs};

use super::gkr::{prove_family_gkr, verify_family_gkr, GkrProof};
use super::pushforward::{prepare_family, prepare_family_verifier, Family};
use super::GkrError;

/// Prove one family's pushforward GKR from the upstream read-raf input claims. `input_claims` are
/// the `d` `M̃^(i)(r_row, r_col)` evaluations (the read-raf's cached `ra_i` openings, aligned onto
/// the shared column point). Errors only via the eq. 5 prover check.
pub fn prove_family<F, T>(
    family: &Family<F>,
    input_claims: &[F],
    family_index: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<GkrProof<F>, GkrError>
where
    F: Field,
    T: ProverFs<F>,
{
    let data = prepare_family(family, input_claims, transcript)?;
    Ok(prove_family_gkr(
        &data,
        family_index,
        accumulator,
        transcript,
    ))
}

/// Metadata the verifier needs to discharge a family's pushforward GKR (no index columns — the
/// verifier never sees `ra_dense`).
#[derive(Clone, Debug)]
pub struct FamilyVerifierParams<F: Field> {
    pub log_t: usize,
    pub log_d: usize,
    pub log_m: usize,
    pub r_row: Vec<F>,
    pub r_col: Vec<F>,
}

/// Verify one family's pushforward GKR against the same `input_claims` the prover used.
pub fn verify_family<F, T>(
    params: &FamilyVerifierParams<F>,
    input_claims: &[F],
    proof: &GkrProof<F>,
    family_index: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), GkrError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let view = prepare_family_verifier(
        params.log_t,
        params.log_d,
        params.log_m,
        &params.r_row,
        &params.r_col,
        input_claims,
        transcript,
    );
    verify_family_gkr(&view, proof, family_index, accumulator, transcript)
}

#[inline]
fn rev<F: Field>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

/// One chunk's **Option C** input: the read-raf's cached `ra_i` opening at the chunk's *own* column
/// point `r_col` (BIG_ENDIAN, as cached) + the chunk's `ra_dense` index column. The shared row point
/// `r_cycle` is passed once to [`prove_family_per_chunk`].
#[derive(Clone, Debug)]
pub struct ChunkPushforward<F: Field> {
    /// Chunk column width: `P^F` has length `2^log_m`, indices `< 2^log_m`.
    pub log_m: usize,
    /// The chunk's distinct column point `r_k_i` (BIG_ENDIAN, the read-raf address slice).
    pub r_col: Vec<F>,
    /// The chunk's `ra_dense` index column (`idx_i[j] < 2^log_m`), length `T = 2^log_t`.
    pub indices: Vec<u32>,
    /// The read-raf cached opening `ra_i = M̃^(i)(r_cycle, r_k_i)` — this chunk's §4.5.2 input claim.
    pub claim: F,
}

/// **Option C** (M8 read-raf ↔ §4.5.2 reconciliation, see `m7-readraf-shared-point-gap`): discharge
/// each of a family's `d` read-raf chunk openings with its *own* per-chunk pushforward GKR at the
/// chunk's distinct column point — no §4.1 row-concatenation, no shared-column reduction. Each chunk
/// is the `log_d = 0` special case of [`prove_family`]: a single index column, the base LogUp\* main
/// identity `M̃^(i)(r_cycle, r_k_i) = P̃^F_i(r_k_i)`. The `d` GKR-leaf openings key under
/// `RaDense(base_index + i)` / `Pushforward(base_index + i)` (a global chunk index across families).
///
/// The read-raf caches points MSB-first (BIG_ENDIAN); this module is LSB-first, so the shared
/// `r_cycle` and each chunk's `r_col` are reversed into the [`Family`] (the bit-ordering bridge).
/// `r_cycle` is shared across the `d` chunks — the read-raf binds one cycle point per family.
pub fn prove_family_per_chunk<F, T>(
    name: &'static str,
    log_t: usize,
    base_index: usize,
    r_cycle: &[F],
    chunks: &[ChunkPushforward<F>],
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<Vec<GkrProof<F>>, GkrError>
where
    F: Field,
    T: ProverFs<F>,
{
    let r_row = rev(r_cycle);
    let mut proofs = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let family = Family {
            name,
            log_t,
            log_d: 0,
            log_m: chunk.log_m,
            r_row: r_row.clone(),
            r_col: rev(&chunk.r_col),
            indices: vec![chunk.indices.clone()],
        };
        proofs.push(prove_family(
            &family,
            std::slice::from_ref(&chunk.claim),
            base_index + i,
            accumulator,
            transcript,
        )?);
    }
    Ok(proofs)
}

/// Verifier-side per-chunk input: the chunk's width, its distinct column point, and the read-raf
/// claim (no index column — the verifier never sees `ra_dense`).
#[derive(Clone, Debug)]
pub struct ChunkVerifierInput<F: Field> {
    pub log_m: usize,
    pub r_col: Vec<F>,
    pub claim: F,
}

/// Verify the Option C per-chunk pushforward GKRs (mirror of [`prove_family_per_chunk`]); the
/// transcript interaction order must match the prover's chunk-by-chunk loop.
pub fn verify_family_per_chunk<F, T>(
    log_t: usize,
    base_index: usize,
    r_cycle: &[F],
    chunks: &[ChunkVerifierInput<F>],
    proofs: &[GkrProof<F>],
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), GkrError>
where
    F: Field,
    T: VerifierFs<F>,
{
    if proofs.len() != chunks.len() {
        return Err(GkrError::Sumcheck);
    }
    let r_row = rev(r_cycle);
    for (i, (chunk, proof)) in chunks.iter().zip(proofs.iter()).enumerate() {
        let params = FamilyVerifierParams {
            log_t,
            log_d: 0,
            log_m: chunk.log_m,
            r_row: r_row.clone(),
            r_col: rev(&chunk.r_col),
        };
        verify_family(
            &params,
            std::slice::from_ref(&chunk.claim),
            proof,
            base_index + i,
            accumulator,
            transcript,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::accumulator::{
        CommittedPolynomial, OpeningAccumulator, OpeningPoint, SumcheckId, VirtualPolynomial,
    };
    use crate::framework::sumcheck::prove as sumcheck_prove;
    use crate::zkvm::logup::pushforward::claim_eval;
    use crate::zkvm::shout_read_raf::{OneHotReadRaf, OneHotReadRafParams, ReadRafStage};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_poly::EqPolynomial;

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

    fn reversed(p: &[F]) -> Vec<F> {
        p.iter().rev().copied().collect()
    }

    /// End-to-end hand-off: build genuine one-hot read columns, run the read-raf sumcheck, then
    /// (1) confirm its cached `ra_i` openings are exactly the one-hot evals the §4.5.2 reduction
    /// consumes (bridged by point reversal), and (2) drive the per-family pushforward GKR from those
    /// upstream claims, round-tripping prover→verifier.
    #[test]
    fn readraf_handoff_round_trip() {
        let mut rng = Rng(0x7E51);
        let (log_k0, log_k1, log_t) = (2usize, 2usize, 4usize);
        let k0 = 1usize << log_k0;
        let k1 = 1usize << log_k1;
        let t = 1usize << log_t;

        // Index columns (the ra_dense), entries < 2^log_k_i.
        let idx0: Vec<u32> = (0..t).map(|_| (rng.next() as u32) % (k0 as u32)).collect();
        let idx1: Vec<u32> = (0..t).map(|_| (rng.next() as u32) % (k1 as u32)).collect();

        // Genuine one-hot read columns over (chunk_i, cycle): ra_i[k·T + j] = [idx_i[j] == k].
        let one_hot = |idx: &[u32], k_dim: usize| -> Vec<F> {
            let mut col = vec![F::from_u64(0); k_dim * t];
            for (j, &k) in idx.iter().enumerate() {
                col[(k as usize) * t + j] = F::from_u64(1);
            }
            col
        };
        let ra0 = one_hot(&idx0, k0);
        let ra1 = one_hot(&idx1, k1);

        // Single read-raf stage with a shared cycle point and a random address-value column.
        let r_cycle = rand_vec(&mut rng, log_t);
        let val_addr = rand_vec(&mut rng, k0 * k1);
        let rv_key = (
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );
        let stages = vec![ReadRafStage {
            r_cycle: r_cycle.clone(),
            val_addr: val_addr.clone(),
            rv_key,
        }];

        // Seed the read value rv = Σ_j eq(j) · val_addr(idx0[j]·k1 + idx1[j]).
        let eq_cycle = EqPolynomial::<F>::evals(&r_cycle, None);
        let mut rv = F::from_u64(0);
        for j in 0..t {
            let kk = (idx0[j] as usize) * k1 + (idx1[j] as usize);
            rv += eq_cycle[j] * val_addr[kk];
        }
        let seed = |acc: &mut Openings<F>| {
            acc.append_virtual(rv_key.0, rv_key.1, OpeningPoint::new(r_cycle.clone()), rv);
        };

        // Run the read-raf sumcheck (the upstream is UNCHANGED by M7).
        let mut rr_acc = Openings::<F>::new(log_t);
        seed(&mut rr_acc);
        let mut rr_t = ProverTranscript::new("readraf");
        let params = OneHotReadRafParams::new(
            CommittedPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            [log_k0, log_k1],
            log_t,
            stages,
            &mut rr_t,
        );
        let mut prover = OneHotReadRaf::new_prover(params, [ra0, ra1]);
        let _ = sumcheck_prove(&mut prover, &mut rr_acc, &mut rr_t);

        // Extract the cached per-chunk ra_i openings (point = (r_k_i, r_cycle), value = M̃^(i)).
        let (pt0, ra0_claim) = rr_acc.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
        );
        let (pt1, ra1_claim) = rr_acc.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(1),
            SumcheckId::InstructionReadRaf,
        );
        let (r_k0, r_cyc0) = pt0.r.split_at(log_k0);
        let (r_k1, r_cyc1) = pt1.r.split_at(log_k1);
        assert_eq!(r_cyc0, r_cyc1, "both chunks share the cycle point");

        // (1) Hand-off identity: the read-raf's ra_i IS the §4.5.2 input claim M̃^(i), bridged by
        // reversing the MSB-first read-raf points into this module's LSB-first claim_eval.
        let single_family = |idx: Vec<u32>, log_k: usize, r_k: &[F], r_cyc: &[F]| {
            let fam = Family::<F> {
                name: "chunk",
                log_t,
                log_d: 0,
                log_m: log_k,
                r_row: reversed(r_cyc),
                r_col: reversed(r_k),
                indices: vec![idx],
            };
            claim_eval(&fam)[0]
        };
        assert_eq!(
            ra0_claim,
            single_family(idx0.clone(), log_k0, r_k0, r_cyc0),
            "read-raf ra_0 == M̃^(0)(reverse(r_cycle), reverse(r_k0))",
        );
        assert_eq!(
            ra1_claim,
            single_family(idx1.clone(), log_k1, r_k1, r_cyc1),
            "read-raf ra_1 == M̃^(1)(reverse(r_cycle), reverse(r_k1))",
        );

        // (2) Drive the per-family GKR from the upstream claims. The d=2 chunks share the column
        // width (log_k0 == log_k1) and (per the M8-alignment note) the column point r_col; here we
        // use chunk 0's aligned point, so the chunk-0 input claim IS the read-raf's ra_0.
        let log_d = 1; // d = 2
        let log_m = log_k0;
        let r_row = reversed(r_cyc0);
        let r_col = reversed(r_k0);
        let family = Family::<F> {
            name: "InstructionRa",
            log_t,
            log_d,
            log_m,
            r_row: r_row.clone(),
            r_col: r_col.clone(),
            indices: vec![idx0, idx1],
        };
        let input_claims = claim_eval(&family);
        assert_eq!(
            input_claims[0], ra0_claim,
            "chunk-0 GKR input claim is the read-raf ra_0 opening",
        );

        // Prover
        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = ProverTranscript::new("logup-driver");
        let proof =
            prove_family(&family, &input_claims, 0, &mut prover_acc, &mut prover_t).expect("prove");
        let narg = prover_t.into_proof();

        // Verifier
        let vparams = FamilyVerifierParams {
            log_t,
            log_d,
            log_m,
            r_row,
            r_col,
        };
        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = VerifierTranscript::new("logup-driver", &narg);
        verify_family(
            &vparams,
            &input_claims,
            &proof,
            0,
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("verify");

        // The GKR leaf + reduction openings agree between prover and verifier.
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
            let (_, pc) = prover_acc.get_committed_polynomial_opening(poly, sc);
            let (_, vc) = verifier_acc.get_committed_polynomial_opening(poly, sc);
            assert_eq!(pc, vc, "opening agrees for {poly:?}/{sc:?}");
        }
    }

    /// One chunk extracted from a genuine read-raf run: index column, width, the chunk's distinct
    /// column point + the shared cycle point (both BIG_ENDIAN, as cached), and the `ra_i` claim.
    struct ChunkOut {
        idx: Vec<u32>,
        log_k: usize,
        r_col: Vec<F>,
        r_cycle: Vec<F>,
        claim: F,
    }

    /// Run a genuine `d = 2` read-raf sumcheck and return the two cached chunk openings — the inputs
    /// the Option C per-chunk pushforward consumes. (Shares structure with `readraf_handoff_round_trip`.)
    fn run_read_raf(seed: u64, log_k0: usize, log_k1: usize, log_t: usize) -> [ChunkOut; 2] {
        let mut rng = Rng(seed);
        let k0 = 1usize << log_k0;
        let k1 = 1usize << log_k1;
        let t = 1usize << log_t;

        let idx0: Vec<u32> = (0..t).map(|_| (rng.next() as u32) % (k0 as u32)).collect();
        let idx1: Vec<u32> = (0..t).map(|_| (rng.next() as u32) % (k1 as u32)).collect();

        let one_hot = |idx: &[u32], k_dim: usize| -> Vec<F> {
            let mut col = vec![F::from_u64(0); k_dim * t];
            for (j, &k) in idx.iter().enumerate() {
                col[(k as usize) * t + j] = F::from_u64(1);
            }
            col
        };
        let ra0 = one_hot(&idx0, k0);
        let ra1 = one_hot(&idx1, k1);

        let r_cycle = rand_vec(&mut rng, log_t);
        let val_addr = rand_vec(&mut rng, k0 * k1);
        let rv_key = (
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );
        let stages = vec![ReadRafStage {
            r_cycle: r_cycle.clone(),
            val_addr: val_addr.clone(),
            rv_key,
        }];

        let eq_cycle = EqPolynomial::<F>::evals(&r_cycle, None);
        let mut rv = F::from_u64(0);
        for j in 0..t {
            let kk = (idx0[j] as usize) * k1 + (idx1[j] as usize);
            rv += eq_cycle[j] * val_addr[kk];
        }

        let mut rr_acc = Openings::<F>::new(log_t);
        rr_acc.append_virtual(rv_key.0, rv_key.1, OpeningPoint::new(r_cycle.clone()), rv);
        let mut rr_t = ProverTranscript::new("readraf");
        let params = OneHotReadRafParams::new(
            CommittedPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            [log_k0, log_k1],
            log_t,
            stages,
            &mut rr_t,
        );
        let mut prover = OneHotReadRaf::new_prover(params, [ra0, ra1]);
        let _ = sumcheck_prove(&mut prover, &mut rr_acc, &mut rr_t);

        let (pt0, c0) = rr_acc.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
        );
        let (pt1, c1) = rr_acc.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(1),
            SumcheckId::InstructionReadRaf,
        );
        let (r_k0, r_cyc0) = pt0.split_at(log_k0);
        let (r_k1, r_cyc1) = pt1.split_at(log_k1);

        [
            ChunkOut {
                idx: idx0,
                log_k: log_k0,
                r_col: r_k0.r,
                r_cycle: r_cyc0.r,
                claim: c0,
            },
            ChunkOut {
                idx: idx1,
                log_k: log_k1,
                r_col: r_k1.r,
                r_cycle: r_cyc1.r,
                claim: c1,
            },
        ]
    }

    /// **Option C end-to-end:** run the real read-raf, then discharge *both* distinct-column-point
    /// chunk openings via per-chunk pushforward GKRs (`log_d = 0` each), prover→verifier. This is the
    /// M8 read-raf ↔ §4.5.2 hand-off — both chunks, at their own `r_k_i`, no shared-point reduction.
    #[test]
    fn readraf_per_chunk_option_c_round_trip() {
        let log_t = 4usize;
        let out = run_read_raf(0xC0DE, 2, 2, log_t);
        assert_eq!(
            out[0].r_cycle, out[1].r_cycle,
            "the d chunks share the read-raf cycle point",
        );
        assert_ne!(
            out[0].r_col, out[1].r_col,
            "the d chunks open at distinct column points (the M8 fork)",
        );
        let r_cycle = out[0].r_cycle.clone();

        let chunks: Vec<ChunkPushforward<F>> = out
            .iter()
            .map(|c| ChunkPushforward {
                log_m: c.log_k,
                r_col: c.r_col.clone(),
                indices: c.idx.clone(),
                claim: c.claim,
            })
            .collect();

        let mut prover_acc = Openings::<F>::new(log_t);
        let mut prover_t = ProverTranscript::new("option-c");
        let proofs = prove_family_per_chunk(
            "InstructionRa",
            log_t,
            0,
            &r_cycle,
            &chunks,
            &mut prover_acc,
            &mut prover_t,
        )
        .expect("per-chunk prove");
        assert_eq!(proofs.len(), 2);
        let narg = prover_t.into_proof();

        let vchunks: Vec<ChunkVerifierInput<F>> = out
            .iter()
            .map(|c| ChunkVerifierInput {
                log_m: c.log_k,
                r_col: c.r_col.clone(),
                claim: c.claim,
            })
            .collect();
        let mut verifier_acc = Openings::<F>::new(log_t);
        let mut verifier_t = VerifierTranscript::new("option-c", &narg);
        verify_family_per_chunk(
            log_t,
            0,
            &r_cycle,
            &vchunks,
            &proofs,
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("per-chunk verify");

        for idx in 0..2usize {
            for (poly, sc) in [
                (
                    CommittedPolynomial::RaDense(idx),
                    SumcheckId::PushforwardGkr,
                ),
                (
                    CommittedPolynomial::Pushforward(idx),
                    SumcheckId::PushforwardGkr,
                ),
                (
                    CommittedPolynomial::Pushforward(idx),
                    SumcheckId::PushforwardReduction,
                ),
            ] {
                let (pp, pc) = prover_acc.get_committed_polynomial_opening(poly, sc);
                let (vp, vc) = verifier_acc.get_committed_polynomial_opening(poly, sc);
                assert_eq!(pp, vp, "opening point agrees for {poly:?}/{sc:?}");
                assert_eq!(pc, vc, "opening claim agrees for {poly:?}/{sc:?}");
            }
        }
    }

    /// A chunk input claim inconsistent with its genuine `ra_dense` (perturbed) trips the per-chunk
    /// eq. 5 main identity inside `prepare_family` (`combined == claim` at `log_d = 0`).
    #[test]
    fn corrupted_chunk_claim_trips_main_identity() {
        let log_t = 4usize;
        let out = run_read_raf(0x0BAD_C0DE, 2, 2, log_t);
        let r_cycle = out[0].r_cycle.clone();
        let mut chunks: Vec<ChunkPushforward<F>> = out
            .iter()
            .map(|c| ChunkPushforward {
                log_m: c.log_k,
                r_col: c.r_col.clone(),
                indices: c.idx.clone(),
                claim: c.claim,
            })
            .collect();
        chunks[1].claim += F::from_u64(1);

        let mut acc = Openings::<F>::new(log_t);
        let mut transcript = ProverTranscript::new("option-c");
        assert!(matches!(
            prove_family_per_chunk(
                "InstructionRa",
                log_t,
                0,
                &r_cycle,
                &chunks,
                &mut acc,
                &mut transcript,
            ),
            Err(GkrError::MainIdentity),
        ));
    }
}
