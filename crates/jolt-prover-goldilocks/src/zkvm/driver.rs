//! Top-level prove/verify driver (M8, growing toward the full 8-stage e2e).
//!
//! Wires the per-stage instances onto one shared transcript + opening accumulator, the way the full
//! `prove()`/`verify()` will. Composes, in Fiat-Shamir order on one accumulator:
//! **Spartan stage** (outer + inner reduction, [`crate::zkvm::spartan::stage`]) → **memory stage**
//! (RAM + registers read-write-checking/val-evaluation + the `RamRa`/`Inc` claim-reductions,
//! [`crate::zkvm::memory`]) → **booleanity stage** (the M6 carry/sign residual over `R1csAux`).
//!
//! Each stage is self-seeding under the interim binary-Spartan path (fork 2): the memory stage seeds
//! its own `SpartanOuter` register openings rather than consuming Spartan's (the binding arrives with
//! uni-skip Spartan, task #6). The stages share only the transcript stream + a non-conflicting set of
//! accumulator keys, so chaining them is sound up to that documented interim gap.
//!
//! Remaining toward the full e2e: the read-raf stage (P6 + the bytecode `Val_s`), the M7 per-chunk
//! pushforward (P7), and the stage-8 WHIR batched open (P9). The committed openings the verifier
//! checks against (`Az/Bz/Cz(r_x)`, `z(r_y)`, `R1csAux(i)(ρ)`, `RamInc`/`RdInc`(ρ)) are carried in
//! the proof and discharged by that stage-8 open.

use jolt_field::Field;
use jolt_r1cs::R1csKey;
use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, Openings, SumcheckId,
};
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
use crate::framework::transcript::{ProverFs, VerifierFs};
use crate::zkvm::booleanity::{Booleanity, BooleanityParams};
use crate::zkvm::memory::{prove_memory, verify_memory, MemoryStageError, MemoryStageProof};
use crate::zkvm::r1cs_witness::R1csWitness;
use crate::zkvm::ram::witness::RamWitness;
use crate::zkvm::registers::witness::RegisterWitness;
use crate::zkvm::spartan::stage::{prove_spartan, verify_spartan, SpartanProof, SpartanStageError};

const BOOLEANITY_DEGREE: usize = 3;

/// Public RAM columns (address `unmap`, the I/O value column, the I/O mask) the memory stage's
/// output-check consumes.
#[derive(Clone, Debug)]
pub struct RamPublicColumns<F: Field> {
    pub unmap: Vec<F>,
    pub val_io: Vec<F>,
    pub io_mask: Vec<F>,
}

/// Driver verification failure (per stage).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverError {
    Spartan(SpartanStageError),
    Memory(MemoryStageError),
}

/// The combined proof (Spartan + memory + booleanity; grows as the read-raf / M7 / stage-8 stages
/// are wired).
#[derive(Clone, Debug)]
pub struct BinaryProof<F: Field> {
    pub spartan: SpartanProof<F>,
    pub memory: MemoryStageProof<F>,
    /// `R1csAux(i)(ρ)` openings the booleanity reduction discharges against (PCS-opened at stage 8).
    pub aux_evals: Vec<F>,
}

/// Top-level prover: Fiat-Shamir-threaded Spartan → memory → booleanity, on a fresh accumulator.
pub fn prove_binary<F, T>(
    witness: &R1csWitness<F>,
    ram_w: &RamWitness<F>,
    reg_w: &RegisterWitness<F>,
    ram_public: &RamPublicColumns<F>,
    key: &R1csKey<F>,
    transcript: &mut T,
) -> BinaryProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let mut accumulator = Openings::<F>::new(witness.log_num_cycles);
    prove_binary_into(
        witness,
        ram_w,
        reg_w,
        ram_public,
        key,
        &mut accumulator,
        transcript,
    )
}

/// Prove the binary stages onto a caller-owned `accumulator`, leaving its cached openings available
/// for the stage-8 WHIR open (the full e2e in [`crate::zkvm::e2e`]). [`prove_binary`] is the wrapper
/// that creates the accumulator.
pub fn prove_binary_into<F, T>(
    witness: &R1csWitness<F>,
    ram_w: &RamWitness<F>,
    reg_w: &RegisterWitness<F>,
    ram_public: &RamPublicColumns<F>,
    key: &R1csKey<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> BinaryProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let spartan = prove_spartan(witness, key, accumulator, transcript);

    let memory = prove_memory(
        ram_w,
        reg_w,
        &ram_public.unmap,
        &ram_public.val_io,
        &ram_public.io_mask,
        accumulator,
        transcript,
    );

    let aux_cols = witness.boolean_aux_columns();
    let n_aux = aux_cols.len();
    let r_bool = transcript.challenge_vector(witness.log_num_cycles);
    let bparams = BooleanityParams::new(r_bool, n_aux, transcript);
    let mut booleanity = Booleanity::new_prover(bparams, aux_cols);
    let _ = prove(&mut booleanity, accumulator, transcript);

    let aux_evals = (0..n_aux)
        .map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::R1csAux(i),
                    SumcheckId::Booleanity,
                )
                .1
        })
        .collect();

    BinaryProof {
        spartan,
        memory,
        aux_evals,
    }
}

/// Top-level verifier (mirror of [`prove_binary`]).
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors prove_binary: proof + R1CS key + Spartan/memory geometry + RAM public columns + transcript"
)]
pub fn verify_binary<F, T>(
    proof: &BinaryProof<F>,
    key: &R1csKey<F>,
    num_row_vars: usize,
    log_num_cycles: usize,
    ram_log_k: usize,
    reg_log_k: usize,
    ram_public: &RamPublicColumns<F>,
    transcript: &mut T,
) -> Result<(), DriverError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let mut accumulator = Openings::<F>::new(log_num_cycles);
    verify_binary_into(
        proof,
        key,
        num_row_vars,
        log_num_cycles,
        ram_log_k,
        reg_log_k,
        ram_public,
        &mut accumulator,
        transcript,
    )
}

/// Verify the binary stages against a caller-owned `accumulator`, leaving its appended openings for
/// the stage-8 WHIR verify (the full e2e in [`crate::zkvm::e2e`]). [`verify_binary`] is the wrapper.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors prove_binary_into: proof + R1CS key + Spartan/memory geometry + RAM public columns + accumulator + transcript"
)]
pub fn verify_binary_into<F, T>(
    proof: &BinaryProof<F>,
    key: &R1csKey<F>,
    num_row_vars: usize,
    log_num_cycles: usize,
    ram_log_k: usize,
    reg_log_k: usize,
    ram_public: &RamPublicColumns<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), DriverError>
where
    F: Field,
    T: VerifierFs<F>,
{
    verify_spartan(&proof.spartan, key, num_row_vars, accumulator, transcript)
        .map_err(DriverError::Spartan)?;

    verify_memory(
        &proof.memory,
        log_num_cycles,
        ram_log_k,
        reg_log_k,
        &ram_public.unmap,
        &ram_public.val_io,
        &ram_public.io_mask,
        accumulator,
        transcript,
    )
    .map_err(DriverError::Memory)?;

    let n_aux = proof.aux_evals.len();
    let r_bool = transcript.challenge_vector(log_num_cycles);
    let bparams = BooleanityParams::new(r_bool, n_aux, transcript);
    let booleanity = Booleanity::new_verifier(bparams);
    let bclaim = SumcheckClaim {
        num_vars: log_num_cycles,
        degree: BOOLEANITY_DEGREE,
        claimed_sum: F::zero(),
    };
    let EvaluationClaim { point, value } = verify(&bclaim, transcript)
        .map_err(|_| DriverError::Spartan(SpartanStageError::Sumcheck))?;

    let rho = booleanity.normalize_opening_point(&point);
    for (i, &eval) in proof.aux_evals.iter().enumerate() {
        accumulator.append_dense(
            CommittedPolynomial::R1csAux(i),
            SumcheckId::Booleanity,
            rho.clone(),
            eval,
        );
    }
    if value != booleanity.expected_output_claim(&*accumulator, &point) {
        return Err(DriverError::Spartan(SpartanStageError::InnerClaim));
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::r1cs::rv64_limbed_constraints;
    use crate::zkvm::r1cs_witness::{build_limbed_z, tests_support::MockCycle};
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    use crate::zkvm::ram::witness::ram_witness;
    use crate::zkvm::registers::witness::register_witness;

    const MEM_K: usize = 8;

    fn witness_and_key(trace: &[MockCycle]) -> (R1csWitness<F>, R1csKey<F>) {
        let pcs = vec![0u64; trace.len()];
        let per_cycle = build_limbed_z::<MockCycle, F>(trace, &pcs);
        let witness = R1csWitness::<F>::materialize(&per_cycle);
        let cycles_pad = 1usize << witness.log_num_cycles;
        let key = R1csKey::new(rv64_limbed_constraints::<F>(), cycles_pad);
        (witness, key)
    }

    /// Memory witnesses + public columns for `trace` (RAM/register address space `MEM_K`). Public
    /// `unmap`/`val_io`/`io_mask` mirror the memory-stage test (zero I/O for the simple traces here).
    fn memory_inputs(
        trace: &[MockCycle],
    ) -> (RamWitness<F>, RegisterWitness<F>, RamPublicColumns<F>) {
        let ram_w = ram_witness::<MockCycle, F>(trace, MEM_K);
        let reg_w = register_witness::<MockCycle, F>(trace, MEM_K);
        let k = 1usize << ram_w.log_k;
        let public = RamPublicColumns {
            unmap: (0..k)
                .map(|i| F::from_u64(0x8000_0000 + i as u64))
                .collect(),
            val_io: vec![F::from_u64(0); k],
            io_mask: vec![F::from_u64(0); k],
        };
        (ram_w, reg_w, public)
    }

    /// The multi-stage driver round-trips on a real (mock) `CycleRow` trace: Spartan → memory →
    /// booleanity, one shared transcript + accumulator, prove → verify. The trace is R1CS-satisfied
    /// (plain adds); its register/RAM activity exercises the memory stage's degenerate path.
    #[test]
    fn binary_driver_round_trip() {
        let trace = [
            MockCycle::add(0, 7, 11),
            MockCycle::add(4, 0xFFFF_FFFF, 1),
            MockCycle::add(8, 100, 40),
            MockCycle::noop_at(12),
        ];
        let (witness, key) = witness_and_key(&trace);
        assert!(witness.is_satisfied(), "witness must satisfy the R1CS");
        let (ram_w, reg_w, public) = memory_inputs(&trace);

        let mut prover_t = ProverTranscript::new("binary-driver");
        let proof = prove_binary(&witness, &ram_w, &reg_w, &public, &key, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_t = VerifierTranscript::new("binary-driver", &narg);
        verify_binary(
            &proof,
            &key,
            witness.num_row_vars(),
            witness.log_num_cycles,
            ram_w.log_k,
            reg_w.log_k,
            &public,
            &mut verifier_t,
        )
        .expect("binary driver must verify");
    }

    /// Tampering a booleanity opening (a corrupted `R1csAux` eval) is rejected.
    #[test]
    fn tampered_aux_rejected() {
        let trace = [MockCycle::add(0, 3, 5), MockCycle::noop_at(4)];
        let (witness, key) = witness_and_key(&trace);
        let (ram_w, reg_w, public) = memory_inputs(&trace);
        let mut prover_t = ProverTranscript::new("binary-driver");
        let mut proof = prove_binary(&witness, &ram_w, &reg_w, &public, &key, &mut prover_t);
        let narg = prover_t.into_proof();
        // Tamper to a NON-boolean value (2) so `b² − b ≠ 0` — flipping to 1 would stay boolean.
        proof.aux_evals[0] += F::from_u64(2);

        let mut verifier_t = VerifierTranscript::new("binary-driver", &narg);
        assert!(
            verify_binary(
                &proof,
                &key,
                witness.num_row_vars(),
                witness.log_num_cycles,
                ram_w.log_k,
                reg_w.log_k,
                &public,
                &mut verifier_t,
            )
            .is_err(),
            "tampered R1csAux eval must be rejected",
        );
    }
}
