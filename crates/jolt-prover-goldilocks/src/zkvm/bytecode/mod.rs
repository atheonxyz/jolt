//! Bytecode checking subprotocols — ported from jolt-core `zkvm/bytecode/` onto
//! [`crate::framework`]. jolt-core is the parity oracle.

pub mod read_raf_checking;

pub use read_raf_checking::{
    bytecode_read_raf_params, OneHotReadRaf, OneHotReadRafParams, ReadRafStage,
};
