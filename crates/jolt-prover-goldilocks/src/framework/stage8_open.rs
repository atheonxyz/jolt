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
//! Built incrementally (per the P9 plan): **S1 (this commit)** is the core commit → open → verify
//! round-trip over a [`Stage8Columns`] map + a [`Stage8Inventory`]. The per-limb Inc reconstruct
//! (`Inc(ρ) = lo + 2³²·hi` vs the recomposed `RamInc`/`RdInc` claims), the `Pushforward` `P^F`
//! base-limb decomposition, and the `build_committed_columns` materialization from the real witness
//! land in the following sub-commits.

use std::collections::HashMap;

use crate::field::{
    Base, ProverTranscript, VerifierTranscript, WhirCommitment, WhirError, WhirHint, WhirScheme, F,
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
