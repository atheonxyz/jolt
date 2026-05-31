//! RAM checking subprotocols — ported from jolt-core `zkvm/ram/` onto [`crate::framework`].
//! jolt-core is the parity oracle.

pub mod output_check;
pub mod val_check;

pub use output_check::{RamOutputCheck, RamOutputCheckParams};
pub use val_check::{RamValCheck, RamValCheckParams};
