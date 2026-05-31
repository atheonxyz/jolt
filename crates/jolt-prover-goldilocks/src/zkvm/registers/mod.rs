//! Register checking subprotocols — ported from jolt-core `zkvm/registers/` onto
//! [`crate::framework`]. jolt-core is the parity oracle.

pub mod read_write_checking;
pub mod val_evaluation;

pub use read_write_checking::{RegistersReadWriteChecking, RegistersReadWriteCheckingParams};
pub use val_evaluation::{RegistersValEvaluation, RegistersValEvaluationParams};
