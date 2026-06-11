//! Dory commitment timing.
//!
//! For each polynomial in the Jolt set, commit via `jolt_dory::DoryScheme`
//! (the PCS the in-development jolt-prover uses, through `jolt_openings`):
//! - one-hot chunks: wrap in `AddressMajorOneHotPolynomial` (column/address-major
//!   `MultilinearPoly`) and call `DoryScheme::commit` (sparse path)
//! - dense polys (RdInc, RamInc): `DoryScheme::commit(&values)` — `Vec<Fr>`
//!   implements `MultilinearPoly`; the matrix shape is derived internally.
//!
//! Mirrors production's polynomial-level parallel commitment scheduling.

use std::time::{Duration, Instant};

use jolt_dory::{DoryProverSetup, DoryScheme};
use jolt_openings::CommitmentScheme;
use rayon::prelude::*;

use crate::jolt_polys::{AddressMajorOneHotPolynomial, DensePoly, JoltPolynomialSet, OneHotChunk};

#[derive(Clone, Copy)]
enum DoryOracle<'a> {
    OneHot {
        family_name: &'static str,
        chunk: &'a OneHotChunk,
    },
    Dense(&'a DensePoly),
}

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
    let mut oracles = Vec::with_capacity(
        polys
            .one_hot_families
            .iter()
            .map(|family| family.chunks.len())
            .sum::<usize>()
            + polys.dense.len(),
    );

    for family in &polys.one_hot_families {
        for chunk in &family.chunks {
            oracles.push(DoryOracle::OneHot {
                family_name: family.name,
                chunk,
            });
        }
    }
    for dense in &polys.dense {
        oracles.push(DoryOracle::Dense(dense));
    }

    let per_oracle = oracles
        .into_par_iter()
        .map(|oracle| match oracle {
            DoryOracle::OneHot { family_name, chunk } => {
                let poly = AddressMajorOneHotPolynomial::from_chunk(chunk);
                let t0 = Instant::now();
                let (_commit, _hint) = DoryScheme::commit(&poly, setup);
                PerOracleTiming {
                    family_name,
                    chunk_idx: Some(chunk.chunk),
                    num_vars: chunk.layout_num_vars,
                    elapsed_ms: dur_ms(t0.elapsed()),
                }
            }
            DoryOracle::Dense(dense) => {
                // `commit` derives the matrix shape internally (sigma =
                // num_vars.div_ceil(2)); `Vec<Fr>` implements `MultilinearPoly<Fr>`.
                let t0 = Instant::now();
                let (_commit, _hint) = DoryScheme::commit(&dense.values, setup);
                PerOracleTiming {
                    family_name: dense.name,
                    chunk_idx: None,
                    num_vars: dense.num_vars,
                    elapsed_ms: dur_ms(t0.elapsed()),
                }
            }
        })
        .collect();

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
