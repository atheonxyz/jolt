//! WHIR commitment scheme for Jolt — **Phase 1**: a plain (sound, non-hiding)
//! base-field commit of the Goldilocks limb witness over the
//! `Basefield<Field64_3>` embedding (commit base `Field64`, fold/challenge in
//! `Fp3`).
//!
//! Phase 1 surface: [`commit_witness`] (commit the base-Goldilocks columns) and
//! [`sanity_roundtrip`] (commit → open → verify a single column).
//!
//! Phase 2 (in progress): the shared spongefish [`transcript`] (used concretely —
//! see its module docs), the `CommitmentScheme` trait impl, LogUp\* pushforward
//! GKR. Hiding (`whir_zk` over `Basefield`) remains Phase 3.

#![cfg(feature = "goldilocks")]

pub mod commit;
pub mod convert;
pub mod params;
pub mod sanity;
pub mod scheme;
pub mod transcript;

pub use commit::{commit_witness, CommitReport};
pub use params::whir_params;
pub use sanity::sanity_roundtrip;
pub use scheme::{WhirCommitment, WhirConfig, WhirError, WhirHint, WhirScheme};
pub use transcript::{ProverTranscript, VerifierTranscript};
