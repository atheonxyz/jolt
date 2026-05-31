//! Hand-written Goldilocks prover subprotocols, mirroring `jolt-core/src/zkvm/` and built on the
//! prover [`crate::framework`]. Ported in dependency order; jolt-core is the parity oracle.

pub mod claim_reductions;
pub mod ram;
pub mod registers;
