//! Register checking subprotocols — ported from jolt-core `zkvm/registers/` onto
//! [`crate::framework`]. jolt-core is the parity oracle.

pub mod val_evaluation;

pub use val_evaluation::{RegistersValEvaluation, RegistersValEvaluationParams};
