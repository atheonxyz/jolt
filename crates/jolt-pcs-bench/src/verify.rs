//! Invariant checks on the LogUp* transformation. Debug builds run them
//! automatically; release builds run them only with `--verify-only`.
//!
//! Each assertion mirrors a property the §5.1/§5.2 protocol relies on:
//!   (1) ra_dense lengths match T
//!   (2) ra_dense values equal Fr::from_u64(index[j])
//!   (3) chunk-decompose agrees with `OneHotParams::*_chunk` (independent
//!       cross-check against the upstream chunk-decomposition API)

use jolt_core::zkvm::config::OneHotParams;
use jolt_field::Fr;

use crate::jolt_polys::{JoltPolynomialSet, OneHotSource};
use crate::logup_star::LogUpStarSet;
use crate::workload::EcdsaWorkload;

#[tracing::instrument(skip_all, name = "bench.verify_transformation")]
pub(crate) fn verify_transformation(
    workload: &EcdsaWorkload,
    polys: &JoltPolynomialSet,
    logup: &LogUpStarSet,
) {
    let params = &workload.one_hot_params;
    let mut ra_iter = logup.ra_dense.iter();

    for family in &polys.one_hot_families {
        for chunk in &family.chunks {
            let ra = ra_iter.next().expect("ra_dense matches chunk count");

            // (1) Size.
            assert_eq!(
                ra.values.len(),
                chunk.trace_len,
                "[verify] {} chunk {}: ra_dense.len={} expected T={}",
                family.name,
                chunk.chunk,
                ra.values.len(),
                chunk.trace_len
            );

            // (2) Argmax — ra_dense[j] reflects indices[j].
            for (j, opt) in chunk.indices.iter().enumerate() {
                let expected = opt.map_or(Fr::from(0u64), |k| Fr::from(u64::from(k)));
                assert_eq!(
                    ra.values[j], expected,
                    "[verify] {} chunk {}: ra_dense[{j}] mismatch",
                    family.name, chunk.chunk
                );
            }

            // (3) Chunk reconstruction (random sample) — confirms our d-decomposition
            // is consistent with the source value's bit layout against the
            // independent `OneHotParams::*_chunk` upstream API.
            verify_chunk_reconstruction(family.source, family.name, chunk.chunk, params, workload);
        }
    }

    assert!(
        ra_iter.next().is_none(),
        "[verify] LogUp* set length mismatch"
    );
    println!("[verify] LogUp* transformation passes all invariants");
}

fn verify_chunk_reconstruction(
    source: OneHotSource,
    family_name: &'static str,
    chunk_idx: usize,
    params: &OneHotParams,
    workload: &EcdsaWorkload,
) {
    match source {
        OneHotSource::InstructionKeys => sample_chunks(
            &workload.sources.instruction_keys,
            params.instruction_d,
            |v, i| params.lookup_index_chunk(v, i),
            params,
            workload,
            chunk_idx,
            family_name,
        ),
        OneHotSource::BytecodeIndices => sample_chunks(
            &workload.sources.bytecode_indices,
            params.bytecode_d,
            |v, i| params.bytecode_pc_chunk(v as usize, i),
            params,
            workload,
            chunk_idx,
            family_name,
        ),
        OneHotSource::RamAddresses => sample_chunks(
            &workload.sources.ram_addresses,
            params.ram_d,
            |v, i| params.ram_address_chunk(v as u64, i),
            params,
            workload,
            chunk_idx,
            family_name,
        ),
    }
}

fn sample_chunks<F: Fn(u128, usize) -> u8>(
    values: &[Option<u128>],
    num_chunks: usize,
    decompose: F,
    params: &OneHotParams,
    workload: &EcdsaWorkload,
    chunk_idx: usize,
    family_name: &'static str,
) {
    let stride = (workload.trace_len / 1000).max(1);
    let mut checked = 0usize;
    for j in (0..workload.trace_len).step_by(stride) {
        if let Some(value) = values.get(j).copied().flatten() {
            let expected_chunk = decompose(value, chunk_idx);
            let shift = params.log_k_chunk * (num_chunks - 1 - chunk_idx);
            let mask = (params.k_chunk - 1) as u128;
            let actual = ((value >> shift) & mask) as u8;
            assert_eq!(
                expected_chunk, actual,
                "[verify] {family_name} chunk {chunk_idx} cycle {j}: \
                 OneHotParams::*_chunk={expected_chunk} vs shift-decompose={actual}"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0 || workload.trace_len == 0,
        "[verify] {family_name} chunk {chunk_idx}: no cycles sampled"
    );
}
