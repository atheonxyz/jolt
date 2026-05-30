//! Limbed RV64 R1CS for the Goldilocks port.
//!
//! A new constraint set (the BN254 `crates/jolt-r1cs/src/constraints/rv64.rs` is
//! field-generic and shared, and every u64 value aliases mod the Goldilocks prime
//! so a single small-field element is unsound). See `../../LIMBED_R1CS.md` for the
//! pinned representation. Built incrementally: this step is the MUL schoolbook
//! ([`mul`]); the eq-conditional constraints + full variable layout follow.

pub mod mul;
pub mod signed_value;

pub use mul::{push_mul_constraints, MulVars, NUM_MUL_ROWS};
pub use signed_value::{push_signed_value_derivation, SignedValueVars, NUM_SIGNED_VALUE_ROWS};
