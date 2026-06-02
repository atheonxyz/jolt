//! Instruction-lookups read + RAF checking — instantiates the shared
//! [`crate::zkvm::shout_read_raf`] `OneHotReadRaf` with the `InstructionRa` committed family and
//! the [`SumcheckId::InstructionReadRaf`] id. Ported from jolt-core's
//! `zkvm/instruction_lookups/read_raf_checking.rs` (the parity oracle).
//!
//! The instruction-lookups identity (jolt-core) is
//! `rv + γ·left_op + γ²·right_op = Σ_{j,k} eq(j;r_reduction)·ra(k,j)·(Val_j(k) + γ·RafVal_j(k))`,
//! i.e. the shared batched read identity specialized to a **single shared cycle eq**
//! (`r_cycle = r_reduction` for every stage) with stages `{lookup-output value, left operand,
//! right operand}`. The wide-limb range checks (design §4.2) fold in as additional stages whose
//! `Val_s` are the `RangeCheck`/`LowerHalfWord`/`UpperWord` table-membership columns.
//!
//! Deferred (beyond the shared-module deferrals): jolt-core's prefix/suffix `Val_j(k)`
//! materialization (multi-table selection + operand prefixes giving the full `(k,j)` value), which
//! here is modeled as opaque address-only per-stage `Val_s(k)` columns — the M8 witness-gen layer.

use crate::framework::transcript::Challenge;
use jolt_field::Field;

use crate::framework::accumulator::{CommittedPolynomial, SumcheckId};

pub use crate::zkvm::shout_read_raf::{
    OneHotReadRaf, OneHotReadRafParams, ReadRafStage, NUM_CHUNKS,
};

/// Build the instruction-lookups read-raf params (`InstructionRa` family, `InstructionReadRaf` id).
/// Every stage should share `r_cycle = r_reduction` (the single instruction eq point).
pub fn instruction_read_raf_params<F: Field>(
    log_k_chunks: [usize; NUM_CHUNKS],
    log_t: usize,
    stages: Vec<ReadRafStage<F>>,
    transcript: &mut impl Challenge<F>,
) -> OneHotReadRafParams<F> {
    OneHotReadRafParams::new(
        CommittedPolynomial::InstructionRa,
        SumcheckId::InstructionReadRaf,
        log_k_chunks,
        log_t,
        stages,
        transcript,
    )
}
