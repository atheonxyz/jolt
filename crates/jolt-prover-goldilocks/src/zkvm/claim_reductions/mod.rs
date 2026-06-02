//! Claim-reduction sumchecks — reduce multiple openings of a committed polynomial to a single
//! opening point. Ported from jolt-core `zkvm/claim_reductions/` onto [`crate::framework`].

pub mod increments;
pub mod ram_ra;
pub mod registers;

pub use increments::{IncClaimReduction, IncClaimReductionParams};
pub use ram_ra::{RamRaClaimReduction, RamRaReductionParams};
pub use registers::{RegistersClaimReduction, RegistersClaimReductionParams};
