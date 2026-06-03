//! Per-cycle instruction-lookup trace extraction (P3b-1) — the goldilocks-crate mirror of
//! `jolt-kernels::trace::stage5_lookup_trace`, decoupled from jolt-kernels (the framework is
//! vendored; see `framework/mod.rs`).
//!
//! Produces the three per-cycle columns the [`super::read_raf_sumcheck::InstructionTrace`] consumes
//! beyond the committed chunk indices: the interleaved lookup index, the lookup table (via the
//! jolt-core-free [`jolt_lookup_tables::instruction_lookup_table_index`] dispatch, P3b-0), and the
//! interleaved-operands flag. The committed `D` chunk-index columns come from
//! `CommittedWitness.ra_dense[instruction_range]` separately.

use jolt_lookup_tables::instruction_lookup_table_index;
use jolt_trace::{Cycle, CycleRow, InterleavedBitsMarker};

/// The per-cycle instruction-lookup columns over a padded length `T = 1 << log_t`.
pub struct InstructionLookupColumns {
    /// The `2·XLEN`-bit interleaved/combined lookup index per cycle (`cycle.lookup_index()`).
    pub lookup_indices: Vec<u128>,
    /// The lookup table per cycle (`None` = no table read; load/store/system/no-op).
    pub lookup_table_indices: Vec<Option<usize>>,
    /// Whether the operands are interleaved (RAF = `γ·left + γ²·right`) vs identity.
    pub is_interleaved: Vec<bool>,
}

/// Extract the instruction-lookup columns over `size` cycles (`size` = the padded committed length
/// `1 << log_t`, so the columns align with the committed `ra_dense` chunk columns). Out-of-range
/// cycles pad to `(index 0, no table, not interleaved)` — the same convention as
/// `stage5_lookup_trace`'s `None` branch and the committed one-hot zero-padding.
pub fn instruction_lookup_columns<const XLEN: usize>(
    trace: &[Cycle],
    size: usize,
) -> InstructionLookupColumns {
    let mut lookup_indices = Vec::with_capacity(size);
    let mut lookup_table_indices = Vec::with_capacity(size);
    let mut is_interleaved = Vec::with_capacity(size);
    for index in 0..size {
        let cycle = trace.get(index);
        lookup_indices.push(cycle.map_or(0, CycleRow::lookup_index));
        lookup_table_indices
            .push(cycle.and_then(|c| instruction_lookup_table_index::<XLEN>(&c.instruction())));
        is_interleaved.push(cycle.is_some_and(|c| c.circuit_flags().is_interleaved_operands()));
    }
    InstructionLookupColumns {
        lookup_indices,
        lookup_table_indices,
        is_interleaved,
    }
}
