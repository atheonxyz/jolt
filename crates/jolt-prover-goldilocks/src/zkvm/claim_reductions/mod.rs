//! Claim-reduction sumchecks — reduce multiple openings of a committed polynomial to a single
//! opening point. Ported from jolt-core `zkvm/claim_reductions/` onto [`crate::framework`].
//!
//! Ported (single-phase; jolt-core's prefix/suffix two-phase materialization is a deferred perf
//! optimization): `increments`, `registers`, `ram_ra`, `instruction_lookups`, `hamming_weight`.
//!
//! **Deferred — `advice` (jolt-core `claim_reductions/advice.rs`):** the trusted/untrusted advice
//! claim-reduction is the multi-phase (cycle + address `ReductionPhase`) reduction over the advice
//! columns. It is only exercised when advice polynomials are present; the M8 e2e gate programs
//! (`muldiv`, `fibonacci`) use no advice, so it is deferred until the advice e2e path is wired (it
//! must land before any advice-using guest is proved).

pub mod hamming_weight;
pub mod increments;
pub mod instruction_lookups;
pub mod ram_ra;
pub mod registers;

pub use hamming_weight::{
    FamilyCounts, HammingWeightClaimReduction, HammingWeightClaimReductionParams,
};
pub use increments::{IncClaimReduction, IncClaimReductionParams};
pub use instruction_lookups::{
    InstructionLookupsClaimReduction, InstructionLookupsClaimReductionParams,
};
pub use ram_ra::{RamRaClaimReduction, RamRaReductionParams};
pub use registers::{RegistersClaimReduction, RegistersClaimReductionParams};
