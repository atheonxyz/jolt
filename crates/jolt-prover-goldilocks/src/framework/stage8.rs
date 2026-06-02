//! Stage-8 opening inventory — the transcript-free bookkeeping the WHIR batched PCS open (P9)
//! consumes. Vendored from jolt-core's `poly/opening_proof.rs` dedup/alias
//! (`find_existing_opening_at_point` / `insert_or_alias_opening`) + `zkvm/mod.rs::stage8_opening_ids`,
//! retargeted to the Goldilocks committed set and the WHIR **per-size-class** batch open (jolt-core
//! instead packs everything into one Dory matrix with Lagrange embedding).
//!
//! Three pieces:
//! 1. **Dedup/alias.** Many sumchecks open the same committed column at the same point (e.g. `RdInc`
//!    cached by both read-write-checking and val-evaluation). [`Stage8Inventory::insert_or_alias`]
//!    keeps one canonical opening per `(poly, point)` and aliases the rest (asserting the claims
//!    agree) — so each committed column is PCS-opened once per distinct point.
//! 2. **Size-class grouping.** [`WhirScheme::open_batch`](crate::field::WhirScheme) batches columns
//!    of one committed length. [`Stage8Inventory::by_size_class`] groups the unique openings by their
//!    committed dimension (`committed_num_vars`): `RaDense`/`R1csAux`/Inc limbs/range-check halves are
//!    the `log_t` class; each family's `Pushforward` (`P^F`, length `2^log_m`) is its own small class.
//! 3. **`zero_selector` embedding.** A cycle-only column (Inc) opened at a wider `(r_addr, r_cycle)`
//!    point evaluates to `col(r_cycle) · ∏(1−r_addr_i)` — the length-`T` column is the address-0 row
//!    of the wider hypercube. The grouping strips the leading address vars into the reduced (native)
//!    point + folds `∏(1−r_addr_i)` into [`Stage8Entry::scaling_factor`] (jolt-core's
//!    `EqPolynomial::zero_selector(r_address)` embedding), leaving `claim` = the native column eval
//!    WHIR opens.
//!
//! Wiring (exactly which committed columns at which points/sumchecks, the per-limb Inc split, and the
//! range-check halves) is finalized with the WHIR commit + open at P9/P10; this module is the
//! field-agnostic, unit-tested bookkeeping it builds on.

use std::collections::{BTreeMap, HashMap};

use jolt_field::Field;
use jolt_poly::EqPolynomial;

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN,
};

/// One canonical (deduplicated) committed-column opening: the column, the **full** opening point as
/// cached by its sumcheck, the column's committed dimension, and the claimed evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniqueOpening<F: Field> {
    pub poly: CommittedPolynomial,
    pub point: OpeningPoint<BIG_ENDIAN, F>,
    /// `log2` of the committed column length (its native dimension).
    pub committed_num_vars: usize,
    pub claim: F,
}

/// A size-class-grouped opening ready for the WHIR batch open: the column, the **reduced** (native,
/// committed-dimension) point WHIR opens at, the native column eval, and the `zero_selector` factor
/// embedding it into the wider opening point (`F::one()` when the point is already native).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8Entry<F: Field> {
    pub poly: CommittedPolynomial,
    pub point: OpeningPoint<BIG_ENDIAN, F>,
    pub claim: F,
    pub scaling_factor: F,
}

impl<F: Field> Stage8Entry<F> {
    /// The embedded value `claim · scaling_factor` — the column's contribution at the wider
    /// (pre-reduction) opening point, the term a joint RLC over the size class would sum.
    #[inline]
    pub fn embedded_claim(&self) -> F {
        self.claim * self.scaling_factor
    }
}

/// One committed-column opening to pull from the accumulator for the stage-8 batch open: which
/// `(poly, sumcheck)` opening, and the column's committed dimension (its size class).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8Request {
    pub poly: CommittedPolynomial,
    pub sumcheck: SumcheckId,
    pub committed_num_vars: usize,
}

/// The deduplicated, ordered stage-8 opening inventory. Insertion order is preserved (the canonical
/// PCS-open order); duplicates at the same `(poly, point)` are aliased away.
#[derive(Clone, Debug, Default)]
pub struct Stage8Inventory<F: Field> {
    unique: Vec<UniqueOpening<F>>,
    by_poly: HashMap<CommittedPolynomial, Vec<usize>>,
    aliased: usize,
}

impl<F: Field> Stage8Inventory<F> {
    pub fn new() -> Self {
        Self {
            unique: Vec::new(),
            by_poly: HashMap::new(),
            aliased: 0,
        }
    }

    /// The canonical (deduplicated) openings in insertion order.
    #[inline]
    pub fn unique(&self) -> &[UniqueOpening<F>] {
        &self.unique
    }

    /// Number of insertions that aliased an existing `(poly, point)` opening (deduped away).
    #[inline]
    pub fn num_aliased(&self) -> usize {
        self.aliased
    }

    /// Existing canonical opening of `poly` at `point` (the dedup probe), if any.
    fn find_existing_opening_at_point(
        &self,
        poly: CommittedPolynomial,
        point: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Option<&UniqueOpening<F>> {
        self.by_poly.get(&poly).and_then(|idxs| {
            idxs.iter()
                .map(|&i| &self.unique[i])
                .find(|existing| &existing.point == point)
        })
    }

    /// Add a committed opening, deduplicating by `(poly, point)`. Returns `true` if it is a new
    /// canonical opening, `false` if it aliased an existing one (claims must then agree — a mismatch
    /// is a prover/verifier wiring bug).
    pub fn insert_or_alias(
        &mut self,
        poly: CommittedPolynomial,
        point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
        committed_num_vars: usize,
    ) -> bool {
        if let Some(existing_claim) = self
            .find_existing_opening_at_point(poly, &point)
            .map(|existing| existing.claim)
        {
            assert_eq!(
                existing_claim, claim,
                "stage-8 duplicate opening claim mismatch for {poly:?} at the same point"
            );
            self.aliased += 1;
            return false;
        }
        let idx = self.unique.len();
        self.by_poly.entry(poly).or_default().push(idx);
        self.unique.push(UniqueOpening {
            poly,
            point,
            committed_num_vars,
            claim,
        });
        true
    }

    /// Build the inventory by pulling each requested committed opening from the accumulator and
    /// deduplicating. Mirrors jolt-core's `stage8_opening_ids` → `get_committed_polynomial_opening`
    /// loop; the `requests` order is the canonical PCS-open order (see [`canonical_requests`]).
    pub fn from_accumulator(
        accumulator: &dyn OpeningAccumulator<F>,
        requests: &[Stage8Request],
    ) -> Self {
        let mut inventory = Self::new();
        for req in requests {
            let (point, claim) =
                accumulator.get_committed_polynomial_opening(req.poly, req.sumcheck);
            let _ = inventory.insert_or_alias(req.poly, point, claim, req.committed_num_vars);
        }
        inventory
    }

    /// Group the unique openings by size class (`committed_num_vars`), reducing each wider opening
    /// point to its native dimension and folding the stripped leading address vars into the
    /// `zero_selector` scaling factor (BIG_ENDIAN: the address vars are the leading bits). Within a
    /// class, openings keep their insertion order.
    pub fn by_size_class(&self) -> BTreeMap<usize, Vec<Stage8Entry<F>>> {
        let mut classes: BTreeMap<usize, Vec<Stage8Entry<F>>> = BTreeMap::new();
        for opening in &self.unique {
            let full = &opening.point.r;
            let cv = opening.committed_num_vars;
            assert!(
                full.len() >= cv,
                "stage-8 opening point ({} vars) shorter than the committed dimension ({cv}) for {:?}",
                full.len(),
                opening.poly
            );
            let extra = full.len() - cv;
            let scaling_factor = if extra == 0 {
                F::one()
            } else {
                EqPolynomial::<F>::zero_selector(&full[..extra])
            };
            let point = OpeningPoint::<BIG_ENDIAN, F>::new(full[extra..].to_vec());
            classes.entry(cv).or_default().push(Stage8Entry {
                poly: opening.poly,
                point,
                claim: opening.claim,
                scaling_factor,
            });
        }
        classes
    }
}

/// Geometry of the Goldilocks committed set needed to enumerate the canonical stage-8 open order.
#[derive(Clone, Debug)]
pub struct Stage8Geometry {
    /// `log2` of the committed (padded) cycle count — the `log_t` size class.
    pub log_t: usize,
    /// One entry `(ra_family(i), pushforward_log_m)` per committed `RaDense`/`Pushforward` chunk, in
    /// global-chunk-index order (instruction chunks, then bytecode, then ram).
    pub ra_chunks: Vec<RaChunkGeometry>,
    /// Number of `R1csAux` committed columns (each the `log_t` class).
    pub num_r1cs_aux: usize,
}

/// One committed RA chunk's geometry: which read-raf RA key it was opened under (for the `RaDense`
/// leaf the pushforward GKR caches) and its pushforward column width `log_m`.
#[derive(Clone, Copy, Debug)]
pub struct RaChunkGeometry {
    /// Global chunk index — the `RaDense(global_index)` / `Pushforward(global_index)` key.
    pub global_index: usize,
    /// Pushforward `P^F` width: the `Pushforward` column is the `log_m` size class.
    pub log_m: usize,
}

/// The canonical ordered stage-8 open requests over the Goldilocks committed set: per chunk the
/// `RaDense` leaf (length `T`, opened by the pushforward GKR) and the `Pushforward` `P^F` (length
/// `2^log_m`), then the `R1csAux` columns (length `T`, opened by booleanity). The per-limb Inc
/// columns and the range-check halves are appended once their commit layout lands at P9.
pub fn canonical_requests(geom: &Stage8Geometry) -> Vec<Stage8Request> {
    let mut requests = Vec::new();
    for chunk in &geom.ra_chunks {
        requests.push(Stage8Request {
            poly: CommittedPolynomial::RaDense(chunk.global_index),
            sumcheck: SumcheckId::PushforwardGkr,
            committed_num_vars: geom.log_t,
        });
        requests.push(Stage8Request {
            poly: CommittedPolynomial::Pushforward(chunk.global_index),
            sumcheck: SumcheckId::PushforwardGkr,
            committed_num_vars: chunk.log_m,
        });
    }
    for i in 0..geom.num_r1cs_aux {
        requests.push(Stage8Request {
            poly: CommittedPolynomial::R1csAux(i),
            sumcheck: SumcheckId::Booleanity,
            committed_num_vars: geom.log_t,
        });
    }
    requests
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::accumulator::Openings;
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    fn pt(vals: &[u64]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(vals.iter().map(|&v| F::from_u64(v)).collect())
    }

    #[test]
    fn dedup_aliases_same_poly_and_point() {
        let mut inv = Stage8Inventory::<F>::new();
        assert!(inv.insert_or_alias(
            CommittedPolynomial::RaDense(0),
            pt(&[1, 2, 3]),
            F::from_u64(7),
            3
        ));
        // Same poly + same point + same claim → aliased, not re-added.
        assert!(!inv.insert_or_alias(
            CommittedPolynomial::RaDense(0),
            pt(&[1, 2, 3]),
            F::from_u64(7),
            3
        ));
        assert_eq!(inv.unique().len(), 1, "duplicate is deduped");
        assert_eq!(inv.num_aliased(), 1);
    }

    #[test]
    fn distinct_points_kept_separate() {
        let mut inv = Stage8Inventory::<F>::new();
        assert!(inv.insert_or_alias(
            CommittedPolynomial::RaDense(0),
            pt(&[1, 2, 3]),
            F::from_u64(7),
            3
        ));
        assert!(inv.insert_or_alias(
            CommittedPolynomial::RaDense(0),
            pt(&[9, 9, 9]),
            F::from_u64(8),
            3
        ));
        // Different poly, same point as the first.
        assert!(inv.insert_or_alias(
            CommittedPolynomial::Pushforward(0),
            pt(&[1, 2, 3]),
            F::from_u64(5),
            3
        ));
        assert_eq!(inv.unique().len(), 3);
        assert_eq!(inv.num_aliased(), 0);
    }

    #[test]
    #[should_panic(expected = "claim mismatch")]
    fn duplicate_claim_mismatch_panics() {
        let mut inv = Stage8Inventory::<F>::new();
        assert!(inv.insert_or_alias(
            CommittedPolynomial::RaDense(0),
            pt(&[1, 2]),
            F::from_u64(7),
            2
        ));
        let _ = inv.insert_or_alias(
            CommittedPolynomial::RaDense(0),
            pt(&[1, 2]),
            F::from_u64(8),
            2,
        );
    }

    #[test]
    fn size_class_grouping_and_native_points() {
        let mut inv = Stage8Inventory::<F>::new();
        // log_t = 3 class: RaDense + R1csAux opened at their native cycle points (no address vars).
        assert!(inv.insert_or_alias(
            CommittedPolynomial::RaDense(0),
            pt(&[1, 2, 3]),
            F::from_u64(10),
            3
        ));
        assert!(inv.insert_or_alias(
            CommittedPolynomial::R1csAux(0),
            pt(&[4, 5, 6]),
            F::from_u64(11),
            3
        ));
        // log_m = 1 class: a Pushforward column.
        assert!(inv.insert_or_alias(
            CommittedPolynomial::Pushforward(0),
            pt(&[2]),
            F::from_u64(12),
            1
        ));

        let classes = inv.by_size_class();
        assert_eq!(classes.len(), 2, "two size classes (log_t=3, log_m=1)");
        let c3 = &classes[&3];
        assert_eq!(c3.len(), 2);
        for entry in c3 {
            assert_eq!(
                entry.scaling_factor,
                F::from_u64(1),
                "native point ⇒ scaling 1"
            );
            assert_eq!(entry.point.r.len(), 3, "native dimension preserved");
            assert_eq!(entry.embedded_claim(), entry.claim);
        }
        assert_eq!(classes[&1].len(), 1);
    }

    #[test]
    fn zero_selector_embeds_cycle_only_at_wider_point() {
        // A length-T (log_t = 2) Inc column opened at a wider (r_addr ‖ r_cycle) point: the leading
        // 2 address vars strip into ∏(1−r_addr_i), the trailing 2 are the native point.
        let mut inv = Stage8Inventory::<F>::new();
        let full = pt(&[3, 5, 7, 9]); // [addr0, addr1, cyc0, cyc1]
        assert!(inv.insert_or_alias(CommittedPolynomial::RdInc, full, F::from_u64(20), 2));

        let classes = inv.by_size_class();
        let entry = &classes[&2][0];
        assert_eq!(
            entry.point.r,
            pt(&[7, 9]).r,
            "native point = trailing cycle vars"
        );
        let expected = (F::from_u64(1) - F::from_u64(3)) * (F::from_u64(1) - F::from_u64(5));
        assert_eq!(entry.scaling_factor, expected, "∏(1 − r_addr_i)");
        assert_eq!(
            entry.claim,
            F::from_u64(20),
            "claim is the native column eval"
        );
        assert_eq!(entry.embedded_claim(), F::from_u64(20) * expected);
    }

    #[test]
    fn from_accumulator_pulls_and_dedups() {
        let mut acc = Openings::<F>::new(3);
        // RaDense(0) opened by the pushforward GKR at a cycle point.
        acc.append_dense(
            CommittedPolynomial::RaDense(0),
            SumcheckId::PushforwardGkr,
            pt(&[1, 2, 3]),
            F::from_u64(10),
        );
        acc.append_dense(
            CommittedPolynomial::Pushforward(0),
            SumcheckId::PushforwardGkr,
            pt(&[4]),
            F::from_u64(11),
        );
        acc.append_dense(
            CommittedPolynomial::R1csAux(0),
            SumcheckId::Booleanity,
            pt(&[5, 6, 7]),
            F::from_u64(12),
        );

        let geom = Stage8Geometry {
            log_t: 3,
            ra_chunks: vec![RaChunkGeometry {
                global_index: 0,
                log_m: 1,
            }],
            num_r1cs_aux: 1,
        };
        let requests = canonical_requests(&geom);
        assert_eq!(requests.len(), 3, "RaDense + Pushforward + R1csAux");

        let inv = Stage8Inventory::from_accumulator(&acc, &requests);
        assert_eq!(inv.unique().len(), 3);
        let classes = inv.by_size_class();
        // log_t = 3 class: RaDense + R1csAux; log_m = 1 class: Pushforward.
        assert_eq!(classes[&3].len(), 2);
        assert_eq!(classes[&1].len(), 1);
    }

    /// A committed column opened by two different sumchecks at the *same* point is opened once.
    #[test]
    fn cross_sumcheck_same_point_dedups() {
        let mut acc = Openings::<F>::new(3);
        let point = pt(&[1, 2, 3]);
        acc.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            point.clone(),
            F::from_u64(42),
        );
        acc.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            point,
            F::from_u64(42),
        );
        let requests = vec![
            Stage8Request {
                poly: CommittedPolynomial::RdInc,
                sumcheck: SumcheckId::RegistersReadWriteChecking,
                committed_num_vars: 3,
            },
            Stage8Request {
                poly: CommittedPolynomial::RdInc,
                sumcheck: SumcheckId::RegistersValEvaluation,
                committed_num_vars: 3,
            },
        ];
        let inv = Stage8Inventory::from_accumulator(&acc, &requests);
        assert_eq!(inv.unique().len(), 1, "same column + point opened once");
        assert_eq!(inv.num_aliased(), 1);
    }
}
