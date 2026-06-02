//! Bytecode read + RAF checking — instantiates the shared
//! [`crate::zkvm::shout_read_raf`] `OneHotReadRaf` with the `BytecodeRa` committed family and the
//! [`SumcheckId::BytecodeReadRaf`] id. Ported from jolt-core's `zkvm/bytecode/read_raf_checking.rs`
//! (the parity oracle); see [`crate::zkvm::shout_read_raf`] for the shared identity and the M5
//! decoupling/deferral notes.

use crate::framework::transcript::Challenge;
use jolt_field::Field;

use crate::framework::accumulator::{CommittedPolynomial, SumcheckId};

pub use crate::zkvm::shout_read_raf::{
    OneHotReadRaf, OneHotReadRafParams, ReadRafStage, NUM_CHUNKS,
};

/// Build the bytecode read-raf params (`BytecodeRa` family, `BytecodeReadRaf` id).
pub fn bytecode_read_raf_params<F: Field>(
    log_k_chunks: [usize; NUM_CHUNKS],
    log_t: usize,
    stages: Vec<ReadRafStage<F>>,
    transcript: &mut impl Challenge<F>,
) -> OneHotReadRafParams<F> {
    OneHotReadRafParams::new(
        CommittedPolynomial::BytecodeRa,
        SumcheckId::BytecodeReadRaf,
        log_k_chunks,
        log_t,
        stages,
        transcript,
    )
}
