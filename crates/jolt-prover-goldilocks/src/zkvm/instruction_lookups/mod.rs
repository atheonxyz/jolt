//! Instruction-lookup checking subprotocols — ported from jolt-core `zkvm/instruction_lookups/`
//! onto [`crate::framework`]. jolt-core is the parity oracle.

pub mod operand_poly;
pub mod read_raf_checking;

pub use operand_poly::{OperandPolynomial, OperandSide};
pub use read_raf_checking::{
    instruction_read_raf_params, OneHotReadRaf, OneHotReadRafParams, ReadRafStage,
};
