//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.

mod commitment_scheme;
mod dory_globals;
mod jolt_dory_routines;
#[cfg(feature = "webgpu-pairing")]
pub mod webgpu_g2;
#[cfg(feature = "webgpu-pairing")]
pub mod webgpu_msm;
#[cfg(feature = "webgpu-pairing")]
pub mod webgpu_onehot;
#[cfg(feature = "webgpu-pairing")]
pub mod webgpu_pairing;
#[cfg(feature = "webgpu-pairing")]
mod webgpu_utils;
mod wrappers;

#[cfg(test)]
mod tests;

#[cfg(feature = "zk")]
pub use commitment_scheme::bind_opening_inputs_zk;
pub use commitment_scheme::{bind_opening_inputs, DoryCommitmentScheme, DoryOpeningProofHint};
pub use dory_globals::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup,
    ArkworksVerifierSetup, JoltFieldWrapper, BN254,
};
