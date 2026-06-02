//! Register checking subprotocols — ported from jolt-core `zkvm/registers/` onto
//! [`crate::framework`]. jolt-core is the parity oracle.

pub mod read_write_checking;
pub mod stage;
pub mod val_evaluation;
pub mod witness;

pub use read_write_checking::{RegistersReadWriteChecking, RegistersReadWriteCheckingParams};
pub use stage::{prove_registers, verify_registers, RegistersStageError, RegistersStageProof};
pub use val_evaluation::{RegistersValEvaluation, RegistersValEvaluationParams};
pub use witness::{register_witness, RegisterWitness};
