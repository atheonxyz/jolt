//! WHIR protocol parameters for the Jolt commit, matching the benchmark
//! (`crates/whir-pcs-bench`): 128-bit security, rate 1/2, folding factor 4,
//! 20 grinding bits, list decoding, Blake3 Merkle.

use whir::hash;
use whir::parameters::ProtocolParameters;

/// Production-aligned WHIR parameters for the Goldilocks base-commit.
pub fn whir_params() -> ProtocolParameters {
    ProtocolParameters {
        security_level: 128,
        pow_bits: 20,
        initial_folding_factor: 4,
        folding_factor: 4,
        unique_decoding: false,
        starting_log_inv_rate: 1,
        batch_size: 1,
        hash_id: hash::BLAKE3,
    }
}
