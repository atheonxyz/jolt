//! Handwritten Goldilocks + WHIR Jolt prover/verifier (Phase 2).
//!
//! Mirrors the structure of legacy `jolt-core/src/zkvm/` but over the **Goldilocks**
//! base field with **Fp3** challenges ([`jolt_field`]), the shared spongefish
//! transcript + WHIR PCS used *concretely* ([`jolt_whir`]), and base-field-limb
//! witness columns ([`jolt_witness`]). The BN254/Dory `jolt-core` prover is the
//! equivalence oracle and is **not** modified; this is a separate crate.
//!
//! Built incrementally (M5+): this commit wires the field/PCS/transcript namespace
//! ([`field`]); the limbed RV64 R1CS constraints, the ported leaf subprotocols, and
//! the stage driver land in subsequent steps.

#![cfg(feature = "goldilocks")]

pub mod field;

pub use field::{Base, F};
