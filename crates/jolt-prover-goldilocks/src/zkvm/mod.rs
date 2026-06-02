//! Hand-written Goldilocks prover subprotocols, mirroring `jolt-core/src/zkvm/` and built on the
//! prover [`crate::framework`]. Ported in dependency order; jolt-core is the parity oracle.

pub mod booleanity;
pub mod bytecode;
pub mod claim_reductions;
pub mod instruction_lookups;
pub mod logup;
pub mod r1cs_witness;
pub mod ram;
pub mod registers;
pub mod shout_read_raf;
pub mod spartan;
pub mod witness;
