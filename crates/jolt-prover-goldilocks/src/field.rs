//! Field / PCS / transcript wiring for the Goldilocks+WHIR prover.
//!
//! The prover scalar ([`F`]) is the cubic extension `Fp3`; challenges, sumcheck
//! round polynomials, and opening points/evals all live here. Witness columns are
//! committed as base-field [`Base`] (`Goldilocks`) limbs. The PCS and the shared
//! spongefish transcript are re-exported from [`jolt_whir`] and used concretely
//! (no `jolt_transcript::Transcript` impl — see `jolt_whir::transcript`).

pub use jolt_field::goldilocks::{Goldilocks, GoldilocksFp3};
pub use jolt_whir::{
    ProverTranscript, VerifierTranscript, WhirCommitment, WhirConfig, WhirError, WhirHint,
    WhirScheme,
};

/// The prover scalar field — the `Fp3` extension (~192-bit). Every Fiat-Shamir
/// challenge and sumcheck round polynomial is over `F`.
pub type F = GoldilocksFp3;

/// The committed base field — `Goldilocks` (`p = 2^64 − 2^32 + 1`). Witness limbs
/// (`ra_dense` indices, `Inc` limbs) are base elements; the WHIR commit alphabet.
pub type Base = Goldilocks;
