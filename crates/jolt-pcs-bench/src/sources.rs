//! Native per-cycle commitment sources (seam over jolt-main's witness path).
//!
//! Replaces the Bolt-only `jolt-trace::extract_trace` + `jolt-witness`
//! (`CommitmentTraceSources`, `one_hot_chunk_indices`, `dense_i128_column_to_field`).
//! Everything here is derived directly from the RISC-V `trace` using the exact
//! per-cycle formulas jolt-main's `CommittedPolynomial::generate_witness` uses
//! (`jolt-core/src/zkvm/witness.rs`), so the committed oracle data matches the
//! production witness.
//!
//! ## Forward-compatibility seam
//!
//! All jolt-main-version-specific API calls (trace accessors, `OneHotParams`
//! chunk methods, the `host::Program` pipeline) are confined to this module and
//! `workload.rs`. When `jolt-prover` merges into jolt-main and the trace/witness
//! APIs move to `jolt-witness`/`jolt-program`, only this seam changes.

use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use jolt_core::zkvm::bytecode::{get_pc_for_cycle, BytecodePreprocessing};
use jolt_core::zkvm::instruction::LookupQuery;
use jolt_core::zkvm::ram::remap_address;
use jolt_field::Fr;
use tracer::instruction::{Cycle, RAMAccess};

/// Dense per-cycle index/transition columns the commitment phase consumes.
/// Vectors have length `trace.len()` (unpadded); downstream padding to the
/// committed `trace_len` uses the per-family policy (`Some(0)` for
/// instruction/bytecode, `None` for RAM) — matching jolt-main's padded NoOp
/// cycles and the Bolt `CycleInput::PADDING` semantics.
#[derive(Clone, Debug, Default)]
pub(crate) struct CommitmentSources {
    pub rd_inc: Vec<i128>,
    pub ram_inc: Vec<i128>,
    pub instruction_keys: Vec<Option<u128>>,
    pub ram_addresses: Vec<Option<u128>>,
    pub bytecode_indices: Vec<Option<u128>>,
}

/// Build the commitment sources from the trace, mirroring the per-variant
/// computations in `CommittedPolynomial::generate_witness`:
/// - `RdInc`   = `rd_write` post − pre
/// - `RamInc`  = `ram_access` write post − pre (else 0)
/// - instruction key = `LookupQuery::to_lookup_index`
/// - bytecode index  = `get_pc_for_cycle`
/// - ram address     = `remap_address(ram_access().address())` (None if no access)
#[tracing::instrument(skip_all, name = "bench.build_sources", fields(cycles = trace.len()))]
pub(crate) fn build_sources(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
) -> CommitmentSources {
    let n = trace.len();
    let mut sources = CommitmentSources {
        rd_inc: Vec::with_capacity(n),
        ram_inc: Vec::with_capacity(n),
        instruction_keys: Vec::with_capacity(n),
        ram_addresses: Vec::with_capacity(n),
        bytecode_indices: Vec::with_capacity(n),
    };

    for cycle in trace {
        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
        sources.rd_inc.push(post_value as i128 - pre_value as i128);

        sources.ram_inc.push(match cycle.ram_access() {
            RAMAccess::Write(write) => write.post_value as i128 - write.pre_value as i128,
            _ => 0,
        });

        let ram_address = remap_address(cycle.ram_access().address() as u64, memory_layout);
        sources.ram_addresses.push(ram_address.map(u128::from));

        sources
            .instruction_keys
            .push(Some(LookupQuery::<XLEN>::to_lookup_index(cycle)));

        sources
            .bytecode_indices
            .push(Some(get_pc_for_cycle(bytecode, cycle) as u128));
    }

    sources
}

/// Builds sparse per-cycle one-hot chunk indices (MSB-first; chunk `0` is the
/// most significant), padding to `trace_len` with `padding_value`. Pure integer
/// decomposition matching `OneHotParams::{lookup_index,bytecode_pc,ram_address}_chunk`
/// (cross-checked in `verify.rs`).
pub(crate) fn one_hot_chunk_indices(
    values: &[Option<u128>],
    chunk: usize,
    num_chunks: usize,
    chunk_bits: usize,
    trace_len: usize,
    padding_value: Option<u128>,
) -> Vec<Option<u8>> {
    assert!(
        values.len() <= trace_len,
        "one-hot source has {} values, trace length is {trace_len}",
        values.len()
    );
    assert!(
        chunk < num_chunks,
        "chunk index {chunk} out of bounds for {num_chunks} chunks"
    );
    assert!(
        chunk_bits <= u8::BITS as usize,
        "chunk_bits must fit in one byte"
    );
    assert!(
        chunk_bits * num_chunks <= u128::BITS as usize,
        "one-hot chunks must fit in u128 source values"
    );

    let chunk_domain = 1usize << chunk_bits;
    let shift = chunk_bits * (num_chunks - 1 - chunk);
    let mask = (chunk_domain - 1) as u128;

    (0..trace_len)
        .map(|cycle| {
            let value = values.get(cycle).copied().flatten().or(padding_value);
            value.map(|value| ((value >> shift) & mask) as u8)
        })
        .collect()
}

/// Converts an i128 transition column to BN254 field elements, padded to
/// `target_len` with zero. Negative values use signed modular reduction via
/// `From<i128> for Fr`.
pub(crate) fn dense_i128_column_to_field(values: &[i128], target_len: usize) -> Vec<Fr> {
    assert!(
        values.len() <= target_len,
        "dense trace column has {} values, target length is {target_len}",
        values.len()
    );
    let mut output: Vec<Fr> = values.iter().map(|&value| Fr::from(value)).collect();
    output.resize(target_len, Fr::from(0u64));
    output
}
