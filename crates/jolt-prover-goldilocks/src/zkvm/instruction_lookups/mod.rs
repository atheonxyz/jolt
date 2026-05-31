//! Instruction-lookup checking subprotocols — ported from jolt-core `zkvm/instruction_lookups/`
//! onto [`crate::framework`]. jolt-core is the parity oracle.

pub mod read_raf_checking;

pub use read_raf_checking::{
    instruction_read_raf_params, OneHotReadRaf, OneHotReadRafParams, ReadRafStage,
};
