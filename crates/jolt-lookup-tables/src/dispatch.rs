//! `jolt-core`-free dispatch from a tracer `Instruction`/`Cycle` to its lookup table.
//!
//! The per-instruction [`InstructionLookupTable`] impls live on the typed ISA structs
//! (`Add<C>`, `Mul<C>`, …, declared via `impl_lookup_table!`), but the tracer `Instruction` enum
//! is opaque to this crate. We reach the typed struct via `jolt-trace`'s [`with_isa_struct!`] macro
//! (the same dispatch `instruction_circuit_flags` uses), so the variant → table mapping is the
//! single authoritative one and the compiler — not a hand-maintained 60-way match — keeps it in
//! sync with the ISA.
//!
//! This replaces jolt-core's `InstructionLookup::lookup_table(cycle).map(enum_index)`
//! (forbidden on `refactor/crates`); [`LookupTableKind::index`] is the `enum_index` analog
//! (`#[repr(u8)]` discriminant = `LookupTableKind::all()` order). A `jolt-equivalence` parity test
//! gates this against jolt-core on the real muldiv trace.

use jolt_trace::{with_isa_struct, Instruction};

use crate::tables::LookupTableKind;
use crate::traits::InstructionLookupTable;

/// The lookup table an instruction decomposes into, or `None` (loads/stores/system/no-op).
///
/// Opcode-static: depends only on the instruction kind, not on its runtime operands.
pub fn instruction_lookup_table<const XLEN: usize>(
    instr: &Instruction,
) -> Option<LookupTableKind<XLEN>> {
    with_isa_struct!(instr, |i| InstructionLookupTable::<XLEN>::lookup_table(&i), noop => None)
}

/// The `LookupTableKind::all()`-order index of an instruction's lookup table, or `None`.
///
/// This is the per-cycle `lookup_table_index` the instruction read-raf consumes (its
/// `expected_output_claim` iterates `LookupTableKind::all()` and indexes the table flags by exactly
/// this value).
pub fn instruction_lookup_table_index<const XLEN: usize>(instr: &Instruction) -> Option<usize> {
    instruction_lookup_table::<XLEN>(instr).map(|table| table.index())
}
