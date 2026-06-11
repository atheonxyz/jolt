//! Enumerates Jolt's committed polynomial set and builds them in the same
//! shape Jolt-prover's `SparseCommitmentInputs::commit_oracle` would.
//!
//! This bench is read-only with respect to the prover crate, so we duplicate
//! the minimal `AddressMajorOneHotPolynomial` wrapper here (its upstream
//! counterpart lives at `crates/jolt-prover/src/stages/commitment.rs:214-318`
//! and is `pub(crate)`).

use jolt_core::zkvm::config::OneHotParams;
use jolt_field::Fr;
use jolt_poly::{EqPolynomial, MultilinearPoly};
use rayon::prelude::*;

use crate::sources::{dense_i128_column_to_field, one_hot_chunk_indices, CommitmentSources};

/// One source family ("instruction", "bytecode", "ram") of one-hot indices,
/// chunked across `d` factors.
#[derive(Clone, Debug)]
pub(crate) struct OneHotFamily {
    pub name: &'static str,
    /// Used by `verify_transformation` to dispatch the independent
    /// chunk-decomposition cross-check.
    pub source: OneHotSource,
    pub chunks: Vec<OneHotChunk>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OneHotSource {
    InstructionKeys,
    RamAddresses,
    BytecodeIndices,
}

#[derive(Clone, Debug)]
pub(crate) struct OneHotChunk {
    /// Per-cycle index into [0, k_chunk). `None` means no entry that cycle.
    pub indices: Vec<Option<u8>>,
    /// Position within the d-decomposition (0 == most-significant chunk).
    pub chunk: usize,
    /// Number of variables for the address-major Dory commitment layout
    /// (log_T + log_k_chunk).
    pub layout_num_vars: usize,
    /// k_chunk = 1 << log_k_chunk.
    pub chunk_domain: usize,
    /// Trace length T (already padded to power of 2).
    pub trace_len: usize,
}

/// 1D dense polynomial of length T. Either a signed-i128 transition
/// (Rd/RamInc, materialized into Fr) or already-Fr advice oracle.
#[derive(Clone, Debug)]
pub(crate) struct DensePoly {
    pub name: &'static str,
    pub num_vars: usize,
    pub values: Vec<Fr>,
}

/// The complete polynomial set Jolt's prover commits to for one ECDSA proof.
pub(crate) struct JoltPolynomialSet {
    pub one_hot_families: Vec<OneHotFamily>,
    pub dense: Vec<DensePoly>,
}

impl JoltPolynomialSet {
    pub(crate) fn total_field_elements(&self) -> usize {
        let one_hot: usize = self
            .one_hot_families
            .iter()
            .flat_map(|fam| fam.chunks.iter())
            .map(|c| c.trace_len * c.chunk_domain)
            .sum();
        let dense: usize = self.dense.iter().map(|d| d.values.len()).sum();
        one_hot + dense
    }
}

#[tracing::instrument(skip_all, name = "bench.build_polynomial_set")]
pub(crate) fn build_polynomial_set(
    sources: &CommitmentSources,
    params: &OneHotParams,
    trace_len: usize,
) -> JoltPolynomialSet {
    let log_t = trace_len.trailing_zeros() as usize;
    let layout_num_vars = log_t + params.log_k_chunk;
    let chunk_domain = params.k_chunk;

    let make_family = |name: &'static str,
                       source: OneHotSource,
                       values: &[Option<u128>],
                       num_chunks: usize,
                       padding: Option<u128>|
     -> OneHotFamily {
        let chunks: Vec<OneHotChunk> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk| {
                let indices = one_hot_chunk_indices(
                    values,
                    chunk,
                    num_chunks,
                    params.log_k_chunk,
                    trace_len,
                    padding,
                );
                OneHotChunk {
                    indices,
                    chunk,
                    layout_num_vars,
                    chunk_domain,
                    trace_len,
                }
            })
            .collect();
        OneHotFamily {
            name,
            source,
            chunks,
        }
    };

    let one_hot_families = vec![
        make_family(
            "InstructionRa",
            OneHotSource::InstructionKeys,
            &sources.instruction_keys,
            params.instruction_d,
            Some(0),
        ),
        make_family(
            "BytecodeRa",
            OneHotSource::BytecodeIndices,
            &sources.bytecode_indices,
            params.bytecode_d,
            Some(0),
        ),
        make_family(
            "RamRa",
            OneHotSource::RamAddresses,
            &sources.ram_addresses,
            params.ram_d,
            None,
        ),
    ];

    // 1D dense polys. RdInc/RamInc are dense i128 transitions; advice oracles
    // are absent in the ECDSA workload (no advice tape).
    let dense = vec![
        DensePoly {
            name: "RdInc",
            num_vars: log_t,
            values: dense_i128_column_to_field(&sources.rd_inc, trace_len),
        },
        DensePoly {
            name: "RamInc",
            num_vars: log_t,
            values: dense_i128_column_to_field(&sources.ram_inc, trace_len),
        },
    ];

    JoltPolynomialSet {
        one_hot_families,
        dense,
    }
}

/// Sparse one-hot polynomial in the address-major flat layout that Dory
/// commits to (mirrors `crates/jolt-prover/src/stages/commitment.rs:214-318`).
pub(crate) struct AddressMajorOneHotPolynomial<'a> {
    trace_len: usize,
    chunk_domain: usize,
    indices: &'a [Option<u8>],
    num_vars: usize,
}

impl<'a> AddressMajorOneHotPolynomial<'a> {
    pub(crate) fn from_chunk(chunk: &'a OneHotChunk) -> Self {
        let active_len = chunk.trace_len * chunk.chunk_domain;
        let target_len = 1usize << chunk.layout_num_vars;
        assert!(
            active_len <= target_len,
            "one-hot active_len {active_len} exceeds layout target {target_len}"
        );
        Self {
            trace_len: chunk.trace_len,
            chunk_domain: chunk.chunk_domain,
            indices: &chunk.indices,
            num_vars: chunk.layout_num_vars,
        }
    }

    fn nonzero_flat_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices
            .iter()
            .enumerate()
            .filter_map(|(cycle, &index)| {
                index.map(|index| {
                    let index = index as usize;
                    assert!(
                        index < self.chunk_domain,
                        "one-hot index {index} exceeds domain {}",
                        self.chunk_domain
                    );
                    index * self.trace_len + cycle
                })
            })
    }
}

impl MultilinearPoly<Fr> for AddressMajorOneHotPolynomial<'_> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        assert_eq!(point.len(), self.num_vars);
        let eq_evals = EqPolynomial::new(point.to_vec()).evaluations();
        self.nonzero_flat_indices()
            .fold(Fr::from(0u64), |acc, flat| acc + eq_evals[flat])
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[Fr])) {
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << (self.num_vars - sigma);
        let mut entries = Vec::with_capacity(self.indices.len());
        for flat in self.nonzero_flat_indices() {
            entries.push((flat / num_cols, flat % num_cols));
        }
        entries.sort_unstable_by_key(|(row, _)| *row);

        let mut cursor = 0;
        let mut row = vec![Fr::from(0u64); num_cols];
        for row_index in 0..num_rows {
            row.fill(Fr::from(0u64));
            while cursor < entries.len() && entries[cursor].0 == row_index {
                row[entries[cursor].1] = Fr::from(1u64);
                cursor += 1;
            }
            f(row_index, &row);
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << (self.num_vars - sigma);
        assert_eq!(left.len(), num_rows);
        let mut result = vec![Fr::from(0u64); num_cols];
        for flat in self.nonzero_flat_indices() {
            result[flat % num_cols] += left[flat / num_cols];
        }
        result
    }

    fn is_one_hot(&self) -> bool {
        true
    }

    fn for_each_one(&self, f: &mut dyn FnMut(usize)) {
        for flat in self.nonzero_flat_indices() {
            f(flat);
        }
    }
}
