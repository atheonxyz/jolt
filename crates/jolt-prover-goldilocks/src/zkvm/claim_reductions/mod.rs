//! Claim-reduction sumchecks — reduce multiple openings of a committed polynomial to a single
//! opening point. Ported from jolt-core `zkvm/claim_reductions/` onto [`crate::framework`].

pub mod increments;

pub use increments::{IncClaimReduction, IncClaimReductionParams};
