//! Invariant checks on the LogUp* transformation.
//!
//! Run on `--verify` (and at startup of a normal bench run as a sanity gate).
//! Each assertion mirrors a property the §5.1/§5.2 protocol relies on.

use jolt_core::zkvm::config::OneHotParams;
use jolt_field::{Field, Fr};
use num_traits::Zero;

use crate::jolt_polys::{JoltPolynomialSet, OneHotSource};
use crate::logup_star::{LogUpStarSet, WHIR_MIN_NUM_VARS};
use crate::workload::EcdsaWorkload;

pub fn verify_transformation(
    workload: &EcdsaWorkload,
    polys: &JoltPolynomialSet,
    logup: &LogUpStarSet,
) {
    let params = &workload.one_hot_params;
    let mut ra_iter = logup.ra_dense.iter();
    let mut pf_iter = logup.pushforwards.iter();

    for family in &polys.one_hot_families {
        for chunk in &family.chunks {
            let ra = ra_iter.next().expect("ra_dense matches chunk count");
            let pf = pf_iter.next().expect("pushforward matches chunk count");

            // (1) Sizes.
            assert_eq!(
                ra.values.len(),
                chunk.trace_len,
                "[verify] {} chunk {}: ra_dense.len={} expected T={}",
                family.name,
                chunk.chunk,
                ra.values.len(),
                chunk.trace_len
            );
            let expected_pf_len = (1usize << WHIR_MIN_NUM_VARS)
                .max(chunk.chunk_domain.next_power_of_two());
            assert_eq!(
                pf.values.len(),
                expected_pf_len,
                "[verify] {} chunk {}: pushforward.len={} expected {}",
                family.name,
                chunk.chunk,
                pf.values.len(),
                expected_pf_len
            );

            // (2) Argmax — ra_dense[j] reflects indices[j].
            for (j, opt) in chunk.indices.iter().enumerate() {
                let expected = opt.map_or(Fr::zero(), |k| Fr::from_u64(u64::from(k)));
                assert_eq!(
                    ra.values[j], expected,
                    "[verify] {} chunk {}: ra_dense[{j}] mismatch",
                    family.name, chunk.chunk
                );
            }

            // (3) Histogram — recompute and assert componentwise.
            let mut hist = vec![0u64; chunk.chunk_domain];
            for k in chunk.indices.iter().flatten() {
                hist[*k as usize] += 1;
            }
            for (k, &count) in hist.iter().enumerate() {
                assert_eq!(
                    pf.values[k],
                    Fr::from_u64(count),
                    "[verify] {} chunk {}: pushforward[{k}] mismatch",
                    family.name,
                    chunk.chunk
                );
            }
            // padding region after k_chunk should be zeros
            for k in chunk.chunk_domain..pf.values.len() {
                assert!(
                    pf.values[k].is_zero(),
                    "[verify] {} chunk {}: pushforward[{k}] (padding) not zero",
                    family.name,
                    chunk.chunk
                );
            }

            // (4) Sum invariant: Σ P[k] == nonzero_count(indices).
            let nonzero_count =
                chunk.indices.iter().filter(|i| i.is_some()).count() as u64;
            let hist_sum: u64 = hist.iter().sum();
            assert_eq!(
                hist_sum, nonzero_count,
                "[verify] {} chunk {}: Σ P[k] = {hist_sum}, expected {nonzero_count}",
                family.name, chunk.chunk
            );

            // (5) Chunk reconstruction (random sample) — confirms our d-decomposition
            // is consistent with the source value's bit layout.
            verify_chunk_reconstruction(family.source, family.name, chunk.chunk, params, workload);
        }
    }

    assert!(
        ra_iter.next().is_none() && pf_iter.next().is_none(),
        "[verify] LogUp* set length mismatch"
    );
    println!("[verify] LogUp* transformation passes all 5 invariants");
}

fn verify_chunk_reconstruction(
    source: OneHotSource,
    family_name: &'static str,
    chunk_idx: usize,
    params: &OneHotParams,
    workload: &EcdsaWorkload,
) {
    type Decomposer<'a> = Box<dyn Fn(u128, usize) -> u8 + 'a>;
    let (values, num_chunks, decompose): (&[Option<u128>], usize, Decomposer<'_>) = match source {
        OneHotSource::InstructionKeys => (
            &workload.sources.instruction_keys,
            params.instruction_d,
            Box::new(|v, i| params.lookup_index_chunk(v, i)),
        ),
        OneHotSource::BytecodeIndices => (
            &workload.sources.bytecode_indices,
            params.bytecode_d,
            Box::new(|v, i| params.bytecode_pc_chunk(v as usize, i)),
        ),
        OneHotSource::RamAddresses => (
            &workload.sources.ram_addresses,
            params.ram_d,
            Box::new(|v, i| params.ram_address_chunk(v as u64, i)),
        ),
    };

    // Sample 1000 evenly-spaced cycles.
    let stride = (workload.trace_len / 1000).max(1);
    let mut checked = 0usize;
    for j in (0..workload.trace_len).step_by(stride) {
        if let Some(value) = values.get(j).copied().flatten() {
            let expected_chunk = decompose(value, chunk_idx);
            // Re-derive what the indices vector should hold for this cycle.
            let _ = expected_chunk; // assertion below
            // Compute what one_hot_chunk_indices would produce here.
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
