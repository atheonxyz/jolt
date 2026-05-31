//! Limbed RV64 R1CS for the Goldilocks port.
//!
//! A new constraint set (the BN254 `crates/jolt-r1cs/src/constraints/rv64.rs` is
//! field-generic and shared, and every u64 value aliases mod the Goldilocks prime
//! so a single small-field element is unsound). See `../../LIMBED_R1CS.md` for the
//! pinned representation. [`mul`] is the 4-limb MUL schoolbook, [`signed_value`] the
//! degree-2 signed-value derivation (reserved for the negative-`Right` linear-use case),
//! and [`rv64_limbed`] assembles the full per-cycle constraint matrices limb-wise.

pub mod mul;
pub mod rv64_limbed;
pub mod signed_value;

pub use mul::{push_mul_constraints, MulVars, NUM_MUL_ROWS};
pub use rv64_limbed::{layout, rv64_limbed_constraints, Vars, NUM_LIMBED_ROWS};
pub use signed_value::{push_signed_value_derivation, SignedValueVars, NUM_SIGNED_VALUE_ROWS};
