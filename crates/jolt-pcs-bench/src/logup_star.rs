//! LogUp* transformation per twist_shout_logup_star.pdf §5.1 (Shout) and §5.2 (Twist).
//!
//! Maps each one-hot Ra polynomial (sparse {0,1}^{K x T}) to a dense
//! `ra_dense: Fr^T` (argmax over k at each cycle = the index itself). The
//! eq-weighted pushforward `P^F` is built WHIR-side at runtime from raw u8
//! indices and Fiat-Shamir-squeezed randomness (see
//! `whir-pcs-bench/src/gkr.rs::prepare_pushforwards`); the Jolt side no
//! longer materializes any pushforward vector.
//!
//! The argmax extraction is trivial because Jolt already stores the one-hot
//! polynomial as a `Vec<Option<u8>>` of indices (the dense form §5.1
//! references). `None` cycles produce `ra_dense[j] = 0`.

use jolt_field::Fr;
use rayon::prelude::*;

use crate::jolt_polys::{JoltPolynomialSet, OneHotChunk};

/// Minimum num_vars for any vector committed via whir_zk.
///
/// `whir_zk::Config::new` asserts `num_blinding_variables < num_witness_variables`.
/// At 128-bit security with rate 1/2 and folding_factor=4, the blinding-side
/// upper bound works out to num_blinding ≈ 14, so num_witness must be ≥ 15.
/// (Empirically verified: 14 fails, 15 passes.)
pub(crate) const WHIR_MIN_NUM_VARS: usize = 15;

#[derive(Clone, Debug)]
pub(crate) struct DenseRa {
    /// length T — `ra_dense[j] = Fr::from_u64(index[j] as u64)` if Some, else 0
    pub values: Vec<Fr>,
}

#[derive(Clone, Debug)]
pub(crate) struct LogUpStarSet {
    pub ra_dense: Vec<DenseRa>,
}

fn transform_chunk(chunk: &OneHotChunk) -> DenseRa {
    let values: Vec<Fr> = chunk
        .indices
        .par_iter()
        .map(|opt| match opt {
            Some(k) => Fr::from(u64::from(*k)),
            None => Fr::from(0u64),
        })
        .collect();
    DenseRa { values }
}

#[tracing::instrument(skip_all, name = "bench.logup_star.transform")]
pub(crate) fn transform(set: &JoltPolynomialSet) -> LogUpStarSet {
    let ra_dense: Vec<DenseRa> = set
        .one_hot_families
        .iter()
        .flat_map(|family| {
            family
                .chunks
                .par_iter()
                .map(transform_chunk)
                .collect::<Vec<_>>()
        })
        .collect();
    LogUpStarSet { ra_dense }
}
