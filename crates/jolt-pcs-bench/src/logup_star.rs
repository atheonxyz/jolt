//! LogUp* transformation per twist_shout_logup_star.pdf §5.1 (Shout) and §5.2 (Twist).
//!
//! Maps each one-hot Ra polynomial (sparse {0,1}^{K x T}) to:
//!
//!   ra_dense:    Fr^T              — argmax over k at each cycle (= the index itself)
//!   pushforward: Fr^K (padded)     — histogram P[k] = |{j : ra(k,j) = 1}|
//!
//! The argmax extraction is trivial here because Jolt already stores the
//! one-hot polynomial as a Vec<Option<u8>> of indices (the dense form §5.1
//! references). `None` cycles produce ra_dense[j]=0 and contribute nothing to
//! the histogram (matches Jolt's `padding = None` convention for RAM).
//!
//! Pushforward vectors are padded with zeros up to `WHIR_MIN_NUM_VARS` to
//! satisfy whir_zk's blinding-variable minimum (see plan risks #2/#3).

use jolt_field::{Field, Fr};
use rayon::prelude::*;

// `Field` brings num_traits::{Zero, One} into scope when called via the trait,
// but free use of `Fr::zero()` needs `num_traits::Zero` imported directly.
use num_traits::Zero;

use crate::jolt_polys::{JoltPolynomialSet, OneHotChunk, OneHotFamily};

/// Minimum num_vars for any vector committed via whir_zk.
///
/// `whir_zk::Config::new` asserts `num_blinding_variables < num_witness_variables`.
/// At 128-bit security with rate 1/2 and folding_factor=4, the blinding-side
/// upper bound works out to num_blinding ≈ 14, so num_witness must be ≥ 15.
/// (Empirically verified: 14 fails, 15 passes.) Padding overhead is
/// 80 * 2^15 = 2.6M zeros across 80 pushforward vectors — still ~25% of the
/// WHIR field-element budget but unavoidable for ZK at 128 bits.
pub const WHIR_MIN_NUM_VARS: usize = 15;

#[derive(Clone, Debug)]
pub struct DenseRa {
    /// length T — `ra_dense[j] = Fr::from_u64(index[j] as u64)` if Some, else 0
    pub values: Vec<Fr>,
}

#[derive(Clone, Debug)]
pub struct Pushforward {
    /// Length is `max(k_chunk, 1 << WHIR_MIN_NUM_VARS)`.
    pub values: Vec<Fr>,
}

#[derive(Clone, Debug)]
pub struct LogUpStarSet {
    pub ra_dense: Vec<DenseRa>,
    pub pushforwards: Vec<Pushforward>,
}

impl LogUpStarSet {
    #[allow(dead_code)] // historical metric — legacy dump shape
    pub fn total_field_elements(&self) -> usize {
        let dense: usize = self.ra_dense.iter().map(|d| d.values.len()).sum();
        let push: usize = self.pushforwards.iter().map(|p| p.values.len()).sum();
        dense + push
    }
}

fn transform_chunk(chunk: &OneHotChunk) -> (DenseRa, Pushforward) {
    let k_chunk = chunk.chunk_domain;

    // ra_dense[j] = Fr::from_u64(idx[j]) if Some(idx), else 0.
    let ra_values: Vec<Fr> = chunk
        .indices
        .par_iter()
        .map(|opt| match opt {
            Some(k) => Fr::from_u64(u64::from(*k)),
            None => Fr::zero(),
        })
        .collect();

    // Histogram P[k] = count of cycles with index == k. Sequential is fine
    // because k_chunk is small (≤256) and contention would dominate parallelism.
    let mut hist = vec![0u64; k_chunk];
    for k in chunk.indices.iter().flatten() {
        hist[*k as usize] += 1;
    }

    let padded_len = (1usize << WHIR_MIN_NUM_VARS).max(k_chunk.next_power_of_two());
    let mut pushforward = vec![Fr::zero(); padded_len];
    for (k, &count) in hist.iter().enumerate() {
        pushforward[k] = Fr::from_u64(count);
    }

    (
        DenseRa { values: ra_values },
        Pushforward { values: pushforward },
    )
}

#[tracing::instrument(skip_all, name = "bench.logup_star.transform")]
pub fn transform(set: &JoltPolynomialSet) -> LogUpStarSet {
    let mut ra_dense = Vec::new();
    let mut pushforwards = Vec::new();
    for family in &set.one_hot_families {
        let (dense, push): (Vec<DenseRa>, Vec<Pushforward>) =
            family.chunks.par_iter().map(transform_chunk).unzip();
        ra_dense.extend(dense);
        pushforwards.extend(push);
    }
    LogUpStarSet {
        ra_dense,
        pushforwards,
    }
}

#[allow(dead_code)] // exercised by verify.rs
pub fn family_total_nonzero(family: &OneHotFamily) -> usize {
    family
        .chunks
        .iter()
        .map(|c| c.indices.iter().filter(|i| i.is_some()).count())
        .sum()
}
