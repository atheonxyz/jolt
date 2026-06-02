//! Stage-8 WHIR batched open (P9) — the final PCS step that discharges every committed
//! base-Goldilocks column against the claims the sumcheck stages accumulated. Drives
//! [`WhirScheme`](crate::field::WhirScheme) commit/open on the **shared** spongefish
//! [`ProverTranscript`](crate::field::ProverTranscript) (the same NARG the framework sumchecks
//! write), consuming the transcript-free [`Stage8Inventory`](crate::framework::stage8) (P8) for the
//! per-column opening points + claims.
//!
//! ## v1: per-column open (`M = 1`)
//!
//! `WhirScheme::open_batch` opens `N` same-size columns at `M` points as a dense `M×N` form-major
//! eval matrix — which requires the verifier to know every *cross* eval `columns[v](points[f≠v])`.
//! At stage 8 the verifier has no columns, so it cannot reproduce the off-diagonal evals. v1
//! therefore opens **each column at its own point** (`M = 1`) via the single-column
//! [`WhirScheme::open`]/[`WhirScheme::verify`], looped on the shared transcript — verifier-symmetric
//! (each side needs only the per-column claim). True intra-size-class RLC batching (one `open_batch`
//! per class) is a later optimization gated on a shared-point claim reduction.
//!
//! **Canonical order** = [`Stage8Inventory::by_size_class`] iteration (size class ascending; within
//! a class, the inventory's insertion order). Prover commits every column in that order, then opens
//! each; the verifier receives every commitment in the identical order, then verifies each — so the
//! shared sponge stays in lockstep.
//!
//! Built incrementally (per the P9 plan): **S1** is the core commit → open → verify round-trip over
//! a [`Stage8Columns`] map + a [`Stage8Inventory`] ([`prove_stage8`]/[`verify_stage8`]). **S2** adds
//! the per-limb Inc open + linear reconstruct ([`prove_inc_open`]/[`verify_inc_open`]):
//! `Inc(ρ) = lo + 2³²·hi` checked against the memory stage's recomposed `RamInc`/`RdInc` claims, run
//! after the inventory open on the same transcript (the combined round-trip is tested). **S3a** adds
//! the `Pushforward` `P^F` Fp3 → 3-base-limb open + β-reconstruct
//! ([`prove_pushforward_open`]/[`verify_pushforward_open`]): `P^F(r) = c0(r) + β·c1(r) + β²·c2(r)`
//! checked against the GKR/reduction claim. Remaining (S3b): `build_committed_columns` to
//! materialize all base columns from the real `CommittedWitness`/`CommitmentTraceSources`/pushforward
//! outputs, and appending the Inc-limb / `Pushforward`-limb / range-check-half opens to
//! [`canonical_requests`](crate::framework::stage8::canonical_requests).

use std::collections::HashMap;

use jolt_field::Field;

use crate::field::{
    Base, ProverTranscript, VerifierTranscript, WhirCommitment, WhirConfig, WhirError, WhirHint,
    WhirScheme, F,
};
use crate::framework::accumulator::CommittedPolynomial;
use crate::framework::stage8::Stage8Inventory;

/// The committed base-Goldilocks columns, keyed by [`CommittedPolynomial`], that the stage-8 open
/// commits + opens. Each column has length `2^{committed_num_vars}` for its size class. (Limb-split
/// columns — Inc `lo`/`hi`, `Pushforward` `P^F` limbs — get their own keys when those sub-commits
/// land; v1 keys whatever the inventory references.)
#[derive(Clone, Debug, Default)]
pub struct Stage8Columns {
    pub columns: HashMap<CommittedPolynomial, Vec<Base>>,
}

impl Stage8Columns {
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
        }
    }

    /// Insert a committed column (overwriting any existing entry for `poly`).
    pub fn insert(&mut self, poly: CommittedPolynomial, column: Vec<Base>) {
        let _ = self.columns.insert(poly, column);
    }

    #[inline]
    fn get(&self, poly: CommittedPolynomial) -> Option<&[Base]> {
        self.columns.get(&poly).map(Vec::as_slice)
    }
}

/// One staged prover opening: `(committed_num_vars, hint, column, reduced point, claim)`.
type StagedOpen<'a> = (usize, WhirHint, &'a [Base], Vec<F>, F);
/// One staged verifier opening: `(committed_num_vars, commitment, reduced point, claim)`.
type StagedVerify = (usize, WhirCommitment, Vec<F>, F);

/// Stage-8 open failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage8OpenError {
    /// An inventory opening referenced a column not present in [`Stage8Columns`].
    MissingColumn(CommittedPolynomial),
    /// A WHIR commit/open/verify rejected.
    Whir(WhirError),
    /// A per-limb Inc reconstruct `lo + 2³²·hi` did not match the recomposed claim.
    IncReconstruct,
}

/// The Inc committed columns: each of `RdInc`/`RamInc` is committed as its two signed base-Goldilocks
/// limbs `lo`/`hi` (Fork 3), not the recomposed value. Length `2^log_t` each.
#[derive(Clone, Debug)]
pub struct IncLimbColumns {
    pub rd_inc_lo: Vec<Base>,
    pub rd_inc_hi: Vec<Base>,
    pub ram_inc_lo: Vec<Base>,
    pub ram_inc_hi: Vec<Base>,
}

/// The Inc-limb opening evals carried in the proof (WHIR-proven at the recomposed-claim points):
/// `[rd_lo(ρ_rd), rd_hi(ρ_rd), ram_lo(ρ_ram), ram_hi(ρ_ram)]`. The verifier reconstructs
/// `lo + 2³²·hi` from these and checks against the memory stage's recomposed `RamInc`/`RdInc` claims.
///
/// `present[i] = false` marks a limb whose committed column is identically zero (WHIR cannot open a
/// zero polynomial): it is neither committed nor opened, its eval is forced to `0`, and the
/// recompose check binds it (a falsely-skipped non-zero limb fails `lo + 2³²·hi == claim`).
#[derive(Clone, Debug)]
pub struct Stage8IncProof<F2> {
    pub evals: [F2; 4],
    pub present: [bool; 4],
}

/// `2³²` over `F` — the linear limb-recomposition weight (`signed_limbs_recompose`).
#[inline]
fn limb_weight() -> F {
    F::from_u64(1u64 << 32)
}

/// **Prover (Inc limbs).** Commit the four signed Inc limb columns on the shared transcript and open
/// each at the recomposed-claim point for its family (`rd` limbs at `rd_point`, `ram` limbs at
/// `ram_point`). Returns the four limb evals for the proof. Call AFTER [`prove_stage8`] (same
/// transcript) so the commit order is fixed.
pub fn prove_inc_open(
    transcript: &mut ProverTranscript,
    inc: &IncLimbColumns,
    rd_point: &[F],
    ram_point: &[F],
) -> Stage8IncProof<F> {
    let rd_cfg = WhirScheme::config(inc.rd_inc_lo.len());
    let ram_cfg = WhirScheme::config(inc.ram_inc_lo.len());
    let cols: [&[Base]; 4] = [
        &inc.rd_inc_lo,
        &inc.rd_inc_hi,
        &inc.ram_inc_lo,
        &inc.ram_inc_hi,
    ];
    let cfgs: [&WhirConfig; 4] = [&rd_cfg, &rd_cfg, &ram_cfg, &ram_cfg];
    let points: [&[F]; 4] = [rd_point, rd_point, ram_point, ram_point];

    // Two passes (commit all, then open all), skipping all-zero limbs: WHIR cannot open a zero
    // polynomial, and a program may never set a limb (e.g. the high 32 bits of small increments).
    let zero = Base::from_u64(0);
    let mut present = [false; 4];
    let mut evals = [F::from_u64(0); 4];
    let mut staged: Vec<(WhirHint, usize)> = Vec::new();
    for (i, col) in cols.iter().enumerate() {
        if col.iter().all(|&x| x == zero) {
            continue;
        }
        present[i] = true;
        staged.push((WhirScheme::commit(transcript, cfgs[i], col), i));
    }
    for (hint, i) in staged {
        let eval = WhirScheme::evaluate(cfgs[i], cols[i], points[i]);
        WhirScheme::open(transcript, cfgs[i], cols[i], hint, points[i], eval);
        evals[i] = eval;
    }

    Stage8IncProof { evals, present }
}

/// One Fp3 committed column (the eq-weighted pushforward `P^F`) decomposed into its three base-
/// Goldilocks coefficient limbs `c0`/`c1`/`c2` — the WHIR commit alphabet is base-Goldilocks, so the
/// Fp3 `P^F` is committed coefficient-wise and reconstructed at open. Each limb has length `2^log_m`.
#[derive(Clone, Debug)]
pub struct Fp3LimbColumns {
    pub c0: Vec<Base>,
    pub c1: Vec<Base>,
    pub c2: Vec<Base>,
}

impl Fp3LimbColumns {
    /// Decompose an Fp3 MLE column into its three base-Goldilocks coefficient columns.
    pub fn from_fp3(column: &[F]) -> Self {
        let mut c0 = Vec::with_capacity(column.len());
        let mut c1 = Vec::with_capacity(column.len());
        let mut c2 = Vec::with_capacity(column.len());
        for x in column {
            let c = x.coeffs();
            c0.push(c[0]);
            c1.push(c[1]);
            c2.push(c[2]);
        }
        Self { c0, c1, c2 }
    }
}

/// Linear reconstruct `P^F(r) = c0(r) + β·c1(r) + β²·c2(r)` from the three limb evals (β is the Fp3
/// generator `[0,1,0]`). Equals the Fp3 MLE of `P^F` at `r` by linearity of the WHIR evaluation over
/// the coefficient columns.
#[inline]
fn fp3_reconstruct(evals: [F; 3]) -> F {
    let beta = F::new(Base::from_u64(0), Base::from_u64(1), Base::from_u64(0));
    let beta2 = F::new(Base::from_u64(0), Base::from_u64(0), Base::from_u64(1));
    evals[0] + beta * evals[1] + beta2 * evals[2]
}

/// Per-chunk pushforward `P^F` open: the `[c0, c1, c2]` limb evals + a `present` flag per limb. As in
/// [`Stage8IncProof`], `present[i][limb] = false` marks an all-zero limb (skipped, eval forced 0) —
/// e.g. the c1/c2 of a zero-index chunk's `P^F = [1,0,…]`; the β-reconstruct check binds it.
#[derive(Clone, Debug)]
pub struct Stage8PushforwardProof<F2> {
    pub evals: Vec<[F2; 3]>,
    pub present: Vec<[bool; 3]>,
}

/// **Prover (Pushforward limbs).** For each pushforward chunk, commit its three Fp3-coefficient base
/// columns on the shared transcript and open each at the chunk's point. Returns the per-chunk
/// `[c0, c1, c2]` evals (+ present flags) for the proof. Call after [`prove_stage8`]/[`prove_inc_open`]
/// (same transcript) so the commit order is fixed.
pub fn prove_pushforward_open(
    transcript: &mut ProverTranscript,
    chunks: &[Fp3LimbColumns],
    points: &[Vec<F>],
) -> Stage8PushforwardProof<F> {
    debug_assert_eq!(
        chunks.len(),
        points.len(),
        "one point per pushforward chunk"
    );
    // Commit present (non-zero) limbs (chunk order, c0/c1/c2), staging the opens. WHIR cannot open a
    // zero polynomial, and a zero-index chunk has `P^F = [1,0,…]` → its c1/c2 limbs are all-zero, so
    // skip them (eval forced to 0); the β-reconstruct check binds it.
    let zero = Base::from_u64(0);
    let mut present = vec![[false; 3]; chunks.len()];
    let mut evals = vec![[F::from_u64(0); 3]; chunks.len()];
    let mut staged: Vec<(WhirHint, usize, usize)> = Vec::new(); // (hint, chunk, limb)
    for (i, chunk) in chunks.iter().enumerate() {
        let cfg = WhirScheme::config(chunk.c0.len());
        for (limb, col) in [&chunk.c0, &chunk.c1, &chunk.c2].into_iter().enumerate() {
            if col.iter().all(|&x| x == zero) {
                continue;
            }
            present[i][limb] = true;
            staged.push((WhirScheme::commit(transcript, &cfg, col), i, limb));
        }
    }
    for (hint, chunk, limb) in staged {
        let col = [&chunks[chunk].c0, &chunks[chunk].c1, &chunks[chunk].c2][limb];
        let cfg = WhirScheme::config(col.len());
        let eval = WhirScheme::evaluate(&cfg, col, &points[chunk]);
        WhirScheme::open(transcript, &cfg, col, hint, &points[chunk], eval);
        evals[chunk][limb] = eval;
    }
    Stage8PushforwardProof { evals, present }
}

/// **Verifier (Pushforward limbs)** (mirror of [`prove_pushforward_open`]). Receive each chunk's
/// three limb commitments, verify the opens against the proof-carried evals, then reconstruct
/// `c0 + β·c1 + β²·c2` and check it against the chunk's claimed `P^F(r)` (`claims[i]`).
pub fn verify_pushforward_open(
    transcript: &mut VerifierTranscript,
    points: &[Vec<F>],
    proof: &Stage8PushforwardProof<F>,
    claims: &[F],
) -> Result<(), Stage8OpenError> {
    if proof.evals.len() != points.len()
        || proof.present.len() != points.len()
        || claims.len() != points.len()
    {
        return Err(Stage8OpenError::IncReconstruct);
    }
    // Receive commitments for present limbs (chunk order, c0/c1/c2), then verify the opens.
    let mut comms: Vec<(WhirCommitment, usize, usize)> = Vec::new();
    for (i, point) in points.iter().enumerate() {
        let cfg = WhirScheme::config(1usize << point.len());
        for limb in 0..3 {
            if proof.present[i][limb] {
                let c = WhirScheme::receive_commitment(transcript, &cfg)
                    .map_err(Stage8OpenError::Whir)?;
                comms.push((c, i, limb));
            }
        }
    }
    for (c, chunk, limb) in &comms {
        let cfg = WhirScheme::config(1usize << points[*chunk].len());
        WhirScheme::verify(
            transcript,
            &cfg,
            c,
            &points[*chunk],
            proof.evals[*chunk][*limb],
        )
        .map_err(Stage8OpenError::Whir)?;
    }
    // Reconstruct with eval 0 for skipped (all-zero) limbs.
    for (i, claim) in claims.iter().enumerate() {
        let e = |limb: usize| {
            if proof.present[i][limb] {
                proof.evals[i][limb]
            } else {
                F::from_u64(0)
            }
        };
        if fp3_reconstruct([e(0), e(1), e(2)]) != *claim {
            return Err(Stage8OpenError::IncReconstruct);
        }
    }
    Ok(())
}

/// **Verifier (Inc limbs)** (mirror of [`prove_inc_open`]). Receive the four limb commitments, verify
/// each open against the proof-carried evals, then check the linear reconstruct
/// `lo + 2³²·hi == recomposed claim` for each family (`rd_claim`/`ram_claim` from the memory stage's
/// `IncClaimReduction`).
pub fn verify_inc_open(
    transcript: &mut VerifierTranscript,
    rd_point: &[F],
    ram_point: &[F],
    proof: &Stage8IncProof<F>,
    rd_claim: F,
    ram_claim: F,
) -> Result<(), Stage8OpenError> {
    let rd_cfg = WhirScheme::config(1usize << rd_point.len());
    let ram_cfg = WhirScheme::config(1usize << ram_point.len());
    let cfgs: [&WhirConfig; 4] = [&rd_cfg, &rd_cfg, &ram_cfg, &ram_cfg];
    let points: [&[F]; 4] = [rd_point, rd_point, ram_point, ram_point];

    // Mirror the prover: receive commitments for present limbs (in order), then verify the opens.
    let mut comms: Vec<(WhirCommitment, usize)> = Vec::new();
    for (i, &is_present) in proof.present.iter().enumerate() {
        if is_present {
            let c = WhirScheme::receive_commitment(transcript, cfgs[i])
                .map_err(Stage8OpenError::Whir)?;
            comms.push((c, i));
        }
    }
    for (c, i) in &comms {
        WhirScheme::verify(transcript, cfgs[*i], c, points[*i], proof.evals[*i])
            .map_err(Stage8OpenError::Whir)?;
    }

    // Skipped limbs contribute eval 0 (forced — never trust the proof's eval for an unopened limb).
    let eval = |i: usize| {
        if proof.present[i] {
            proof.evals[i]
        } else {
            F::from_u64(0)
        }
    };
    let w = limb_weight();
    if eval(0) + w * eval(1) != rd_claim || eval(2) + w * eval(3) != ram_claim {
        return Err(Stage8OpenError::IncReconstruct);
    }
    Ok(())
}

/// **Prover.** Commit every inventory column on the shared transcript (size class ascending), then
/// open each at its own point with its claimed eval. The opening bytes live in the NARG; nothing
/// else is emitted.
pub fn prove_stage8(
    transcript: &mut ProverTranscript,
    columns: &Stage8Columns,
    inventory: &Stage8Inventory<F>,
) -> Result<(), Stage8OpenError> {
    let by_class = inventory.by_size_class();

    // PASS 1 — commit every column (class asc, intra-class order), staging the per-column opens.
    let mut staged: Vec<StagedOpen> = Vec::new();
    for (&cv, entries) in &by_class {
        let config = WhirScheme::config(1usize << cv);
        for entry in entries {
            let col = columns
                .get(entry.poly)
                .ok_or(Stage8OpenError::MissingColumn(entry.poly))?;
            let hint = WhirScheme::commit(transcript, &config, col);
            staged.push((cv, hint, col, entry.point.r.clone(), entry.claim));
        }
    }

    // PASS 2 — open each column at its own point (M = 1).
    for (cv, hint, col, point, claim) in staged {
        let config = WhirScheme::config(1usize << cv);
        WhirScheme::open(transcript, &config, col, hint, &point, claim);
    }
    Ok(())
}

/// **Verifier** (mirror of [`prove_stage8`]). Receive every commitment in the identical order, then
/// verify each opening against the inventory's per-column point + claim.
pub fn verify_stage8(
    transcript: &mut VerifierTranscript,
    inventory: &Stage8Inventory<F>,
) -> Result<(), Stage8OpenError> {
    let by_class = inventory.by_size_class();

    // PASS 1 — receive every commitment, identical order.
    let mut staged: Vec<StagedVerify> = Vec::new();
    for (&cv, entries) in &by_class {
        let config = WhirScheme::config(1usize << cv);
        for entry in entries {
            let commitment = WhirScheme::receive_commitment(transcript, &config)
                .map_err(Stage8OpenError::Whir)?;
            staged.push((cv, commitment, entry.point.r.clone(), entry.claim));
        }
    }

    // PASS 2 — verify each opening.
    for (cv, commitment, point, claim) in &staged {
        let config = WhirScheme::config(1usize << *cv);
        WhirScheme::verify(transcript, &config, commitment, point, *claim)
            .map_err(Stage8OpenError::Whir)?;
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::accumulator::{OpeningPoint, BIG_ENDIAN};
    use jolt_field::goldilocks::decompose::{i128_to_signed_limbs, signed_limbs_recompose};
    use jolt_field::Field;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(0x5851_F42D_4C95_7F2D)
                .wrapping_add(0x1442_6950_4088_8963);
            self.0 ^ (self.0 >> 29)
        }
    }

    /// A non-degenerate base-Goldilocks column (spread across all bits, never all-zero — WHIR's open
    /// divides by the polynomial's evaluation, which would be 0 for the zero polynomial).
    fn col(size: usize, seed: u64) -> Vec<Base> {
        let mut rng = Rng(seed | 1);
        (0..size).map(|_| Base::from_u64(rng.next() | 1)).collect()
    }

    fn pt(num_vars: usize, seed: u64) -> Vec<F> {
        let mut rng = Rng(seed | 1);
        (0..num_vars).map(|_| F::from_u64(rng.next())).collect()
    }

    /// Build a `(Stage8Columns, Stage8Inventory)` over `specs = [(poly, committed_num_vars, seed)]`,
    /// each opened at its own native point with the honest WHIR eval as the claim.
    fn build(specs: &[(CommittedPolynomial, usize, u64)]) -> (Stage8Columns, Stage8Inventory<F>) {
        let mut columns = Stage8Columns::new();
        let mut inventory = Stage8Inventory::<F>::new();
        for (poly, cv, seed) in specs {
            let column = col(1usize << cv, *seed);
            let point = pt(*cv, seed.wrapping_add(0xABCD));
            let config = WhirScheme::config(1usize << cv);
            let claim = WhirScheme::evaluate(&config, &column, &point);
            columns.insert(*poly, column);
            let added = inventory.insert_or_alias(
                *poly,
                OpeningPoint::<BIG_ENDIAN, F>::new(point),
                claim,
                *cv,
            );
            assert!(added, "distinct (poly, point)");
        }
        (columns, inventory)
    }

    fn round_trip(specs: &[(CommittedPolynomial, usize, u64)]) {
        let (columns, inventory) = build(specs);
        let mut prover_t = ProverTranscript::new("stage8-open");
        prove_stage8(&mut prover_t, &columns, &inventory).expect("prove stage8");
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("stage8-open", &narg);
        verify_stage8(&mut verifier_t, &inventory).expect("verify stage8");
    }

    #[test]
    fn single_size_class_round_trip() {
        round_trip(&[
            (CommittedPolynomial::RaDense(0), 6, 0x10),
            (CommittedPolynomial::RaDense(1), 6, 0x11),
            (CommittedPolynomial::R1csAux(0), 6, 0x12),
        ]);
    }

    #[test]
    fn two_size_classes_round_trip() {
        // log_t = 6 class (RaDense/R1csAux) + a log_m = 4 class (Pushforward).
        round_trip(&[
            (CommittedPolynomial::RaDense(0), 6, 0x20),
            (CommittedPolynomial::Pushforward(0), 4, 0x21),
            (CommittedPolynomial::R1csAux(0), 6, 0x22),
            (CommittedPolynomial::Pushforward(1), 4, 0x23),
        ]);
    }

    #[test]
    fn missing_column_errors() {
        let (mut columns, inventory) = build(&[(CommittedPolynomial::RaDense(0), 5, 0x30)]);
        columns.columns.clear();
        let mut prover_t = ProverTranscript::new("stage8-open");
        assert_eq!(
            prove_stage8(&mut prover_t, &columns, &inventory),
            Err(Stage8OpenError::MissingColumn(
                CommittedPolynomial::RaDense(0)
            ))
        );
    }

    /// Inc limb columns + their two recomposed-claim points + the independent recomposed claims
    /// (the Inc MLE = `signed_limbs_recompose` per cell, evaluated at each point). Increments have
    /// magnitude > 2³² (so both lo + hi columns are non-degenerate) and mixed sign.
    fn inc_setup(log_t: usize) -> (IncLimbColumns, Vec<F>, Vec<F>, F, F) {
        let n = 1usize << log_t;
        let incs: Vec<i128> = (0..n as i128)
            .map(|j| {
                let base = (j + 1) * 0x1_2345_6789;
                if j % 2 == 0 {
                    base
                } else {
                    -base
                }
            })
            .collect();
        let limbs: Vec<[Base; 2]> = incs.iter().map(|&v| i128_to_signed_limbs(v)).collect();
        let inc = IncLimbColumns {
            rd_inc_lo: limbs.iter().map(|l| l[0]).collect(),
            rd_inc_hi: limbs.iter().map(|l| l[1]).collect(),
            ram_inc_lo: limbs.iter().map(|l| l[0]).collect(),
            ram_inc_hi: limbs.iter().map(|l| l[1]).collect(),
        };
        let rd_point = pt(log_t, 0x500);
        let ram_point = pt(log_t, 0x501);
        let recomposed: Vec<Base> = limbs.iter().map(|l| signed_limbs_recompose(*l)).collect();
        let cfg = WhirScheme::config(n);
        let rd_claim = WhirScheme::evaluate(&cfg, &recomposed, &rd_point);
        let ram_claim = WhirScheme::evaluate(&cfg, &recomposed, &ram_point);
        (inc, rd_point, ram_point, rd_claim, ram_claim)
    }

    #[test]
    fn inc_reconstruct_round_trip() {
        let (inc, rd_point, ram_point, rd_claim, ram_claim) = inc_setup(5);
        let mut prover_t = ProverTranscript::new("inc-open");
        let proof = prove_inc_open(&mut prover_t, &inc, &rd_point, &ram_point);
        // lo + 2^32·hi reconstructs the recomposed claim (linearity).
        assert_eq!(proof.evals[0] + limb_weight() * proof.evals[1], rd_claim);
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("inc-open", &narg);
        verify_inc_open(
            &mut verifier_t,
            &rd_point,
            &ram_point,
            &proof,
            rd_claim,
            ram_claim,
        )
        .expect("verify inc");
    }

    #[test]
    fn inc_reconstruct_tampered_claim_rejected() {
        let (inc, rd_point, ram_point, rd_claim, ram_claim) = inc_setup(5);
        let mut prover_t = ProverTranscript::new("inc-open");
        let proof = prove_inc_open(&mut prover_t, &inc, &rd_point, &ram_point);
        let narg = prover_t.into_proof();

        // Corrupt the recomposed RdInc claim → the WHIR opens still pass but the linear reconstruct
        // lo + 2^32·hi != rd_claim fails.
        let mut verifier_t = VerifierTranscript::new("inc-open", &narg);
        assert_eq!(
            verify_inc_open(
                &mut verifier_t,
                &rd_point,
                &ram_point,
                &proof,
                rd_claim + F::from_u64(1),
                ram_claim,
            ),
            Err(Stage8OpenError::IncReconstruct)
        );
    }

    fn fp3_col(size: usize, seed: u64) -> Vec<F> {
        let mut rng = Rng(seed | 1);
        (0..size)
            .map(|_| {
                F::new(
                    Base::from_u64(rng.next() | 1),
                    Base::from_u64(rng.next() | 1),
                    Base::from_u64(rng.next() | 1),
                )
            })
            .collect()
    }

    /// Fp3 MLE of `column` at `point` via the eq dot product (jolt-poly convention).
    fn fp3_mle(column: &[F], point: &[F]) -> F {
        let eq = jolt_poly::EqPolynomial::<F>::evals(point, None);
        column
            .iter()
            .zip(eq.iter())
            .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
    }

    #[test]
    fn pushforward_decompose_identity() {
        // Pure-math: an Fp3 column's MLE equals the β-reconstruct of its three base-limb MLEs.
        let point = pt(4, 0x80);
        let column = fp3_col(1 << 4, 0x81);
        let limbs = Fp3LimbColumns::from_fp3(&column);
        let lift = |c: &[Base]| {
            fp3_mle(
                &c.iter().map(|&b| F::from_base(b)).collect::<Vec<_>>(),
                &point,
            )
        };
        let reconstructed = fp3_reconstruct([lift(&limbs.c0), lift(&limbs.c1), lift(&limbs.c2)]);
        assert_eq!(fp3_mle(&column, &point), reconstructed);
    }

    #[test]
    fn pushforward_reconstruct_round_trip() {
        let log_m = 4;
        let n = 1usize << log_m;
        let cols = [fp3_col(n, 0x90), fp3_col(n, 0x91)];
        let limbs: Vec<Fp3LimbColumns> = cols.iter().map(|c| Fp3LimbColumns::from_fp3(c)).collect();
        let points = vec![pt(log_m, 0x92), pt(log_m, 0x93)];

        let mut prover_t = ProverTranscript::new("pf-open");
        let proof = prove_pushforward_open(&mut prover_t, &limbs, &points);
        // The reconstruct of the WHIR limb evals is the (WHIR-convention) P^F eval — use it as the
        // claim (these random columns are non-degenerate, so every limb is present).
        let claims: Vec<F> = proof.evals.iter().map(|e| fp3_reconstruct(*e)).collect();
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("pf-open", &narg);
        verify_pushforward_open(&mut verifier_t, &points, &proof, &claims)
            .expect("verify pushforward");
    }

    #[test]
    fn pushforward_tampered_claim_rejected() {
        let log_m = 4;
        let cols = [fp3_col(1 << log_m, 0xA0)];
        let limbs: Vec<Fp3LimbColumns> = cols.iter().map(|c| Fp3LimbColumns::from_fp3(c)).collect();
        let points = vec![pt(log_m, 0xA1)];
        let mut prover_t = ProverTranscript::new("pf-open");
        let proof = prove_pushforward_open(&mut prover_t, &limbs, &points);
        let narg = prover_t.into_proof();

        // Corrupt the claimed P^F eval → the limb opens still verify but the β-reconstruct mismatches.
        let claims = vec![fp3_reconstruct(proof.evals[0]) + F::from_u64(1)];
        let mut verifier_t = VerifierTranscript::new("pf-open", &narg);
        assert_eq!(
            verify_pushforward_open(&mut verifier_t, &points, &proof, &claims),
            Err(Stage8OpenError::IncReconstruct)
        );
    }

    /// Combined: inventory open (S1) then Inc limb open (S2) on ONE transcript — validates the
    /// interleaved commit/open ordering (inventory committed+opened, then Inc) round-trips.
    #[test]
    fn inventory_then_inc_round_trip() {
        let (columns, inventory) = build(&[
            (CommittedPolynomial::RaDense(0), 5, 0x70),
            (CommittedPolynomial::R1csAux(0), 5, 0x71),
        ]);
        let (inc, rd_point, ram_point, rd_claim, ram_claim) = inc_setup(5);

        let mut prover_t = ProverTranscript::new("combined");
        prove_stage8(&mut prover_t, &columns, &inventory).expect("prove inventory");
        let proof = prove_inc_open(&mut prover_t, &inc, &rd_point, &ram_point);
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("combined", &narg);
        verify_stage8(&mut verifier_t, &inventory).expect("verify inventory");
        verify_inc_open(
            &mut verifier_t,
            &rd_point,
            &ram_point,
            &proof,
            rd_claim,
            ram_claim,
        )
        .expect("verify inc");
    }

    #[test]
    fn tampered_claim_rejected() {
        let (columns, mut inventory) = build(&[
            (CommittedPolynomial::RaDense(0), 5, 0x40),
            (CommittedPolynomial::RaDense(1), 5, 0x41),
        ]);
        let mut prover_t = ProverTranscript::new("stage8-open");
        prove_stage8(&mut prover_t, &columns, &inventory).expect("prove stage8");
        let narg = prover_t.into_proof();

        // Re-build the verifier inventory with a corrupted claim for one column.
        let mut bad = Stage8Inventory::<F>::new();
        for (i, op) in inventory.unique().iter().enumerate() {
            let claim = if i == 0 {
                op.claim + F::from_u64(1)
            } else {
                op.claim
            };
            let _ = bad.insert_or_alias(op.poly, op.point.clone(), claim, op.committed_num_vars);
        }
        inventory = bad;

        let mut verifier_t = VerifierTranscript::new("stage8-open", &narg);
        assert!(
            matches!(
                verify_stage8(&mut verifier_t, &inventory),
                Err(Stage8OpenError::Whir(_))
            ),
            "tampered claim must be rejected by WHIR verify"
        );
    }
}
