//! Prover-side sumcheck framework for the hand-written Goldilocks prover, vendored from
//! legacy `jolt-core` and retargeted to the lean [`jolt_field::Field`] (challenges are plain
//! `F`, the `C = F = Fp3` convention). The workspace crates expose only the *verifier* side
//! (`jolt-sumcheck`) and shared primitives; the prover-side sumcheck-instance traits, opening
//! accumulator, and `MultilinearPolynomial` enum live only in `jolt-core` (hand-written) or
//! `jolt-kernels` (Bolt-generated, BN254-specialized). Per `specs/jolt-prover-model-crate.md`
//! the prover is a hand-written, field-generic crate with jolt-core as parity oracle only — this
//! module is that framework, instantiated at `F = GoldilocksFp3`, `PCS = WhirScheme`.
//!
//! Built incrementally: [`poly`] (dense multilinear) + [`sumcheck`] (instance trait + driver,
//! bridged to the workspace verifier) land first; the opening accumulator and the committed/ZK
//! path follow.

pub mod accumulator;
pub mod lagrange;
pub mod multiquadratic;
pub mod poly;
pub mod sumcheck;
pub mod transcript;
pub mod univariate_skip;

pub use accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN, LITTLE_ENDIAN,
};
pub use poly::MultilinearPolynomial;
pub use sumcheck::{prove, verify, SumcheckInstance};
pub use transcript::{Challenge, ProverFs, VerifierFs};
