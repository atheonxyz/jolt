//! Spartan sumchecks — ported from jolt-core `zkvm/spartan/` onto [`crate::framework`].
//! jolt-core is the parity oracle.
//!
//! Decoupled / correctness-first (the M5 convention): the univariate-skip first round
//! (`outer`/`product`), the streaming sumcheck, the prefix-suffix `EqPlusOne` two-phase
//! (`shift`), and the `R1CSEval`/`ALL_R1CS_INPUTS` matrix→`z` reduction are perf optimizations
//! deferred to a later pass (OPT-E); each instance here binds all variables plainly.
//!
//! ## M8 Spartan = **BINARY** (uni-skip DEFERRED — user-approved 2026-06-02)
//!
//! The M8 e2e uses a **binary** Spartan (this decoupled [`outer`] deg-3 zero-check + a binary inner
//! reduction built on the workspace `jolt_r1cs::R1csKey`), NOT jolt-core's univariate-skip Spartan.
//! Rationale: the workspace is binary-Spartan (its `R1csKey` + verifier are binary), so binary is the
//! reuse-the-workspace, correctness-first path that reaches the equivalence gate soonest; the gate is
//! witness-level and passes with binary Spartan. **The faithful jolt-core univariate-skip Spartan
//! (outer + R1CSEval grouping) is DEFERRED to a later perf pass (task #6 / memory
//! `m8-opt-e-faithful-port`).** Its foundations are already built + tested and waiting:
//! [`crate::framework::lagrange`], [`crate::framework::multiquadratic`],
//! [`crate::framework::univariate_skip`] (plus `prove_batched`). Uni-skip collapses the ≤6 limbed
//! constraint rounds → 2 (main win: memory/streaming at production scale; perf-irrelevant for the
//! gate's small traces).

pub mod inner;
pub mod instruction_input;
pub mod outer;
pub mod shift;
pub mod stage;

pub use inner::{SpartanInner, SpartanInnerParams};
pub use instruction_input::{InstructionInput, InstructionInputParams};
pub use outer::{SpartanOuter, SpartanOuterParams};
pub use shift::{SpartanShift, SpartanShiftParams};
pub use stage::{prove_spartan, verify_spartan, SpartanProof, SpartanStageError};
