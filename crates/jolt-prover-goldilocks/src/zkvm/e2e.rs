//! Full `prove()`/`verify()` e2e orchestrator (P10): the binary driver (Spartan → memory →
//! booleanity) followed by the stage-8 WHIR open, on ONE shared spongefish transcript + `Openings`
//! accumulator. Monomorphic on the concrete Goldilocks [`F`] — the stage-8 open commits over the
//! base-Goldilocks alphabet.
//!
//! Wired incrementally:
//!   M2 — discharge the booleanity `R1csAux(i)` openings via [`prove_stage8`]/[`verify_stage8`].
//!
//! The read-raf / M7 pushforward families and the `Inc` limb open extend the inventory in later
//! milestones; the binary driver's other committed openings (`RamInc`/`RdInc`, the RA chunks) are
//! still carried as cached claims until then.

use crate::field::{ProverTranscript, VerifierTranscript, F};
use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, Openings, SumcheckId,
};
use crate::framework::stage8::{Stage8Inventory, Stage8Request};
use crate::framework::stage8_open::{prove_stage8, verify_stage8, Stage8OpenError};
use crate::zkvm::driver::{prove_binary_into, verify_binary_into, BinaryProof, DriverError};
use crate::zkvm::real_trace::RealWitness;
use crate::zkvm::stage8_columns::r1cs_aux_columns;
use jolt_field::Field;
use jolt_r1cs::R1csKey;

use super::driver::RamPublicColumns;

/// e2e verification failure: a binary-driver stage or the stage-8 WHIR open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum E2eError {
    Driver(DriverError),
    Stage8(Stage8OpenError),
}

/// The full proof: the binary-driver sub-proof plus whatever the stage-8 open contributes (today
/// the open's bytes live entirely in the shared NARG, so no extra fields).
#[derive(Clone, Debug)]
pub struct E2eProof {
    pub binary: BinaryProof<F>,
}

/// Verifier-side public parameters (geometry + the R1CS key + RAM public columns). Derived from the
/// witness for the gate; in production these come from preprocessing.
pub struct VerifierParams<'a> {
    pub key: &'a R1csKey<F>,
    pub ram_public: &'a RamPublicColumns<F>,
    pub num_row_vars: usize,
    pub log_num_cycles: usize,
    pub ram_log_k: usize,
    pub reg_log_k: usize,
    pub n_aux: usize,
}

impl<'a> VerifierParams<'a> {
    pub fn from_witness(real: &'a RealWitness<F>) -> Self {
        VerifierParams {
            key: &real.key,
            ram_public: &real.ram_public,
            num_row_vars: real.r1cs.num_row_vars(),
            log_num_cycles: real.r1cs.log_num_cycles,
            ram_log_k: real.ram.log_k,
            reg_log_k: real.registers.log_k,
            n_aux: real.r1cs.boolean_aux_columns().len(),
        }
    }
}

/// Build the stage-8 requests for the R1csAux columns with a NON-ZERO booleanity claim.
///
/// WHIR's open inverts the claimed evaluation, so it cannot open a column whose claim is `0` — which
/// happens exactly when the committed column is the zero polynomial (e.g. a carry/sign bit a given
/// program never sets, like `sub_c0` on a multiply-only trace). The claim lives in the shared
/// accumulator (Fiat-Shamir-derived), so prover and verifier independently see the same zero claims
/// and skip those columns in lockstep — no signalling, transcript stays in sync.
///
/// INTERIM SOUNDNESS NOTE: a skipped (zero-claim) column is NOT PCS-bound here. It is still
/// constrained by (a) the booleanity output check, which fixes `aux_evals[i]` (a prover cannot forge
/// a zero claim for a column whose true MLE at the random point is non-zero), and (b) Spartan's `z`
/// opening, which binds the aux columns as part of the witness vector. Discharging zero columns with
/// a dedicated structural zero-check is deferred. A non-zero polynomial's MLE at the random point is
/// non-zero with overwhelming probability, so honest non-zero columns are always opened.
fn nonzero_r1cs_aux_requests(
    accumulator: &Openings<F>,
    n_aux: usize,
    log_t: usize,
) -> Vec<Stage8Request> {
    let zero = F::from_u64(0);
    (0..n_aux)
        .filter(|&i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::R1csAux(i),
                    SumcheckId::Booleanity,
                )
                .1
                != zero
        })
        .map(|i| Stage8Request {
            poly: CommittedPolynomial::R1csAux(i),
            sumcheck: SumcheckId::Booleanity,
            committed_num_vars: log_t,
        })
        .collect()
}

/// Prove the full e2e on a real-trace witness: binary stages, then the stage-8 WHIR open of the
/// `R1csAux` columns, on one transcript.
pub fn prove_e2e(
    real: &RealWitness<F>,
    transcript: &mut ProverTranscript,
) -> Result<E2eProof, E2eError> {
    let log_t = real.r1cs.log_num_cycles;
    let mut accumulator = Openings::<F>::new(log_t);

    let binary = prove_binary_into(
        &real.r1cs,
        &real.ram,
        &real.registers,
        &real.ram_public,
        &real.key,
        &mut accumulator,
        transcript,
    );

    let aux = real.r1cs.boolean_aux_columns();
    let requests = nonzero_r1cs_aux_requests(&accumulator, aux.len(), log_t);
    let inventory = Stage8Inventory::from_accumulator(&accumulator, &requests);
    let columns = r1cs_aux_columns(&aux);
    prove_stage8(transcript, &columns, &inventory).map_err(E2eError::Stage8)?;

    Ok(E2eProof { binary })
}

/// Verify the full e2e (mirror of [`prove_e2e`]).
pub fn verify_e2e(
    proof: &E2eProof,
    params: &VerifierParams,
    transcript: &mut VerifierTranscript,
) -> Result<(), E2eError> {
    let mut accumulator = Openings::<F>::new(params.log_num_cycles);

    verify_binary_into(
        &proof.binary,
        params.key,
        params.num_row_vars,
        params.log_num_cycles,
        params.ram_log_k,
        params.reg_log_k,
        params.ram_public,
        &mut accumulator,
        transcript,
    )
    .map_err(E2eError::Driver)?;

    let requests = nonzero_r1cs_aux_requests(&accumulator, params.n_aux, params.log_num_cycles);
    let inventory = Stage8Inventory::from_accumulator(&accumulator, &requests);
    verify_stage8(transcript, &inventory).map_err(E2eError::Stage8)?;

    Ok(())
}
