//! Prover-side sumcheck framework for the hand-written Goldilocks prover, vendored from
//! legacy `jolt-core` and retargeted to the lean [`jolt_field::Field`] (challenges are plain
//! `F`, the `C = F = Fp3` convention). The workspace crates expose only the *verifier* side
//! (`jolt-sumcheck`) and shared primitives; the prover-side sumcheck-instance traits, opening
//! accumulator, and `MultilinearPolynomial` enum live only in `jolt-core` (hand-written) or
//! `jolt-kernels` (Bolt-generated, BN254-specialized). Per `specs/jolt-prover-model-crate.md`
//! the prover is a hand-written, field-generic crate with jolt-core as parity oracle only — this
//! module is that framework, instantiated at `F = GoldilocksFp3`, `PCS = WhirScheme`.
//!
//! Built incrementally: [`poly`] (dense multilinear) lands first; the opening accumulator, the
//! sumcheck-instance traits, and the batched-sumcheck driver follow.

pub mod poly;

pub use poly::MultilinearPolynomial;
