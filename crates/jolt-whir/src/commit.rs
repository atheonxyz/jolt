//! Phase-1 WHIR base-field commit of the Goldilocks witness columns.
//!
//! Commits the base-Goldilocks limb columns (`ra_dense`, `Inc` limbs) over the
//! `Basefield<Field64_3>` embedding — i.e. the committed alphabet is base
//! `Field64` (8 B/elem), while folds/challenges live in the `Fp3` extension.
//! This is plain (sound, non-hiding) WHIR; hiding (`whir_zk` over `Basefield`)
//! is Phase 2.

use std::time::Instant;

use whir::algebra::embedding::Basefield;
use whir::algebra::fields::{Field64, Field64_3};
use whir::protocols::whir::Config;
use whir::transcript::codecs::Empty;
use whir::transcript::{DomainSeparator, ProverState};

use jolt_witness::goldilocks::GoldilocksWitnessColumns;

use crate::convert::column_to_field64;
use crate::params::whir_params;

/// Result of committing the Goldilocks witness columns.
#[derive(Clone, Copy, Debug)]
pub struct CommitReport {
    /// `log2` of each committed column's length.
    pub log_t: usize,
    /// Number of committed columns.
    pub num_columns: usize,
    /// Total committed base-field elements across all columns.
    pub total_base_elements: usize,
    /// Committed base-field volume in bytes (`8 B`/Goldilocks element).
    pub committed_base_bytes: usize,
    /// Wall-clock commit time (ms), excluding the base→`Field64` conversion.
    pub commit_ms: f64,
}

/// Commit every base-Goldilocks column via WHIR (one Merkle tree per column,
/// threaded through a single Fiat-Shamir transcript). Witnesses are dropped
/// after each commit so peak memory stays at one codeword.
pub fn commit_witness(cols: &GoldilocksWitnessColumns) -> CommitReport {
    let log_t = cols.log_t;
    let size = 1usize << log_t;
    let params = whir_params();
    let config = Config::<Basefield<Field64_3>>::new(size, &params);

    let ds = DomainSeparator::protocol(&config)
        .session(&"jolt-whir/goldilocks-commit")
        .instance(&Empty);
    let mut prover_state = ProverState::new_std(&ds);

    // Base→Field64 conversion is upstream of WHIR; exclude it from commit timing.
    let field_cols: Vec<Vec<Field64>> = cols
        .columns
        .iter()
        .map(|c| column_to_field64(&c.values))
        .collect();

    let t0 = Instant::now();
    for col in &field_cols {
        let _witness = config.commit(&mut prover_state, &[col.as_slice()]);
    }
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;

    let total_base_elements = cols.total_elements();
    CommitReport {
        log_t,
        num_columns: cols.columns.len(),
        total_base_elements,
        committed_base_bytes: total_base_elements * 8,
        commit_ms,
    }
}
