//! Montgomery-free Goldilocks (`p = 2^64 − 2^32 + 1`) base field + cubic
//! extension `Fp3 = Fp[x]/(x³ − 2)` (feature `goldilocks`).
//!
//! - [`Goldilocks`] — the base field (RV64 witness limbs live here).
//! - [`GoldilocksFp3`] — the ~192-bit extension; the Phase-2 prover scalar.
//! - [`decompose`] — value ↔ base-field limb primitives for the witness.
//!
//! Arithmetic is hand-coded from the Plonky2 / lambda_vm algorithms; correctness
//! is guarded by `num-bigint` oracle tests in `tests`.

mod accumulator;
mod base;
pub mod decompose;
mod ext3;

#[cfg(test)]
mod tests;

pub use accumulator::{
    Fp3Accumulator, Fp3ScalarAccumulator, GoldilocksAccumulator, GoldilocksScalarAccumulator,
};
pub use base::Goldilocks;
pub use ext3::GoldilocksFp3;
