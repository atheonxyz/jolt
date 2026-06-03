//! RAM checking subprotocols — ported from jolt-core `zkvm/ram/` onto [`crate::framework`].
//! jolt-core is the parity oracle.

pub mod output_check;
pub mod ra_virtual;
pub mod raf_evaluation;
pub mod read_write_checking;
pub mod stage;
pub mod val_check;
pub mod witness;

pub use output_check::{RamOutputCheck, RamOutputCheckParams};
pub use ra_virtual::{
    prove_ram_ra_virtualization, verify_ram_ra_virtualization, RamRaVirtualization,
    RamRaVirtualizationError, RamRaVirtualizationProof,
};
pub use raf_evaluation::{RamRafEvaluation, RamRafEvaluationParams};
pub use read_write_checking::{RamReadWriteChecking, RamReadWriteCheckingParams};
pub use stage::{prove_ram, verify_ram, RamStageError, RamStageProof};
pub use val_check::{RamValCheck, RamValCheckParams};
pub use witness::{ram_witness, RamWitness};
