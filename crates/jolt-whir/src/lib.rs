//! WHIR commitment scheme for Jolt — **Phase 1**: a plain (sound, non-hiding)
//! base-field commit of the Goldilocks limb witness over the
//! `Basefield<Field64_3>` embedding (commit base `Field64`, fold/challenge in
//! `Fp3`).
//!
//! Phase 1 surface: [`commit_witness`] (commit the base-Goldilocks columns) and
//! [`sanity_roundtrip`] (commit → open → verify a single column). The `#1521`
//! `CommitmentScheme` trait impl, hiding (`whir_zk` over `Basefield`), LogUp\*
//! pushforward GKR, and the shared spongefish transcript are Phase 2.

#![cfg(feature = "goldilocks")]

pub mod commit;
pub mod convert;
pub mod params;
pub mod sanity;

pub use commit::{commit_witness, CommitReport};
pub use params::whir_params;
pub use sanity::sanity_roundtrip;
