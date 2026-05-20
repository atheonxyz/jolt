//! Dory commitment timing.
//!
//! For each polynomial in the Jolt set:
//! - one-hot chunks: wrap in `AddressMajorOneHotPolynomial` and call
//!   `DoryScheme::commit(&poly, &setup)` (sparse path)
//! - dense polys (RdInc, RamInc): call
//!   `DoryScheme::commit_evaluations_with_row_len(data, row_len, &setup)`
//!
//! Mirrors `crates/jolt-prover/src/stages/commitment.rs:543-573`.

use std::time::{Duration, Instant};

use jolt_dory::{DoryProverSetup, DoryScheme};
use jolt_openings::CommitmentScheme;

use crate::jolt_polys::{AddressMajorOneHotPolynomial, JoltPolynomialSet};

// I.6: PerOracleTiming uses static name + optional chunk index instead of
// allocating a `String` per chunk per run.
#[derive(Clone, Debug)]
pub(crate) struct PerOracleTiming {
    pub family_name: &'static str,
    pub chunk_idx: Option<usize>,
    pub num_vars: usize,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct DoryRunResult {
    pub total_ms: f64,
    pub per_oracle: Vec<PerOracleTiming>,
}

#[derive(Clone, Debug)]
pub(crate) struct DoryBenchSummary {
    pub setup_ms: f64,
    pub setup_num_vars: usize,
    pub runs: Vec<DoryRunResult>,
}

impl DoryBenchSummary {
    pub(crate) fn total_times_ms(&self) -> Vec<f64> {
        self.runs.iter().map(|r| r.total_ms).collect()
    }
}

/// Max num_vars across all polynomials (drives `setup_prover`).
fn max_num_vars(polys: &JoltPolynomialSet) -> usize {
    let one_hot_max = polys
        .one_hot_families
        .iter()
        .flat_map(|fam| fam.chunks.iter())
        .map(|c| c.layout_num_vars)
        .max()
        .unwrap_or(0);
    let dense_max = polys.dense.iter().map(|d| d.num_vars).max().unwrap_or(0);
    one_hot_max.max(dense_max)
}

#[tracing::instrument(skip_all, name = "bench.dory_bench.run_once")]
fn run_once(polys: &JoltPolynomialSet, setup: &DoryProverSetup) -> DoryRunResult {
    let total_start = Instant::now();
    let mut per_oracle = Vec::new();

    for family in &polys.one_hot_families {
        for chunk in &family.chunks {
            let poly = AddressMajorOneHotPolynomial::from_chunk(chunk);
            let t0 = Instant::now();
            let (_commit, _hint) = DoryScheme::commit(&poly, setup);
            let elapsed = t0.elapsed();
            per_oracle.push(PerOracleTiming {
                family_name: family.name,
                chunk_idx: Some(chunk.chunk),
                num_vars: chunk.layout_num_vars,
                elapsed_ms: dur_ms(elapsed),
            });
        }
    }

    for dense in &polys.dense {
        let row_len = 1usize << dense.num_vars.div_ceil(2);
        let t0 = Instant::now();
        let (_commit, _hint) =
            DoryScheme::commit_evaluations_with_row_len(&dense.values, row_len, setup);
        let elapsed = t0.elapsed();
        per_oracle.push(PerOracleTiming {
            family_name: dense.name,
            chunk_idx: None,
            num_vars: dense.num_vars,
            elapsed_ms: dur_ms(elapsed),
        });
    }

    DoryRunResult {
        total_ms: dur_ms(total_start.elapsed()),
        per_oracle,
    }
}

pub(crate) fn bench_dory(
    polys: &JoltPolynomialSet,
    warmup: usize,
    runs: usize,
) -> DoryBenchSummary {
    let setup_num_vars = max_num_vars(polys);
    let setup_start = Instant::now();
    println!("[dory] setup_prover(num_vars={setup_num_vars}) — generating SRS...");
    let setup = DoryScheme::setup_prover(setup_num_vars);
    let setup_ms = dur_ms(setup_start.elapsed());
    println!("[dory] setup done in {setup_ms:.1}ms");

    for w in 0..warmup {
        let t0 = Instant::now();
        let _ = run_once(polys, &setup);
        println!(
            "[dory] warmup {}/{warmup}: {:.1}ms",
            w + 1,
            dur_ms(t0.elapsed())
        );
    }

    let mut measured = Vec::with_capacity(runs);
    for r in 0..runs {
        let result = run_once(polys, &setup);
        println!(
            "[dory] run {}/{runs}: total={:.1}ms",
            r + 1,
            result.total_ms
        );
        measured.push(result);
    }

    DoryBenchSummary {
        setup_ms,
        setup_num_vars,
        runs: measured,
    }
}

fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}
