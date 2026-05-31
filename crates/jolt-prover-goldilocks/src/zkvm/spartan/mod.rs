//! Spartan sumchecks — ported from jolt-core `zkvm/spartan/` onto [`crate::framework`].
//! jolt-core is the parity oracle.
//!
//! Decoupled / correctness-first (the M5 convention): the univariate-skip first round
//! (`outer`/`product`), the streaming sumcheck, the prefix-suffix `EqPlusOne` two-phase
//! (`shift`), and the `R1CSEval`/`ALL_R1CS_INPUTS` matrix→`z` reduction are perf optimizations
//! deferred to a later pass (OPT-E); each instance here binds all variables plainly.

pub mod instruction_input;
pub mod shift;

pub use instruction_input::{InstructionInput, InstructionInputParams};
pub use shift::{SpartanShift, SpartanShiftParams};
