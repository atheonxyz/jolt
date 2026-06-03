//! Full `prove()`/`verify()` e2e orchestrator (P10): the binary driver (Spartan → memory →
//! booleanity) followed by the stage-8 WHIR open, on ONE shared spongefish transcript + `Openings`
//! accumulator. Monomorphic on the concrete Goldilocks [`F`] — the stage-8 open commits over the
//! base-Goldilocks alphabet.
//!
//! Full pipeline: binary driver (Spartan → memory → booleanity) → bytecode read-raf + M7 pushforward
//! → instruction-lookup read-raf (prefix/suffix, production `LOG_K=128`) + M7 pushforward → stage-8
//! WHIR opens, on ONE transcript. Discharged via the stage-8 open:
//!   - `R1csAux(i)` (booleanity) + bytecode & instruction `RaDense(base+i)` (pushforward GKR) — the
//!     inventory open;
//!   - `RdInc`/`RamInc` limbs ([`prove_inc_open`]) — `lo + 2³²·hi` recompose;
//!   - the bytecode & instruction `Pushforward` `P^F` limbs ([`prove_pushforward_open`]) — β-reconstruct.
//!
//! Interim Fork-2: the instruction read-raf's reduction point is freshly squeezed (not yet bound to
//! the upstream `InstructionClaimReduction` — the deferred uni-skip Spartan closure). The RAM
//! family's dense `RamRa` and the uni-skip Spartan soundness binding remain interim gaps.

use crate::field::{Base, ProverTranscript, VerifierTranscript, F};
use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, Openings, SumcheckId,
};
use crate::framework::stage8::{Stage8Inventory, Stage8Request};
use crate::framework::stage8_open::{
    prove_inc_open, prove_pushforward_open, prove_stage8, verify_inc_open, verify_pushforward_open,
    verify_stage8, Fp3LimbColumns, Stage8IncProof, Stage8OpenError, Stage8PushforwardProof,
};
use crate::framework::transcript::Challenge;
use crate::zkvm::bytecode::read_raf_checking::{
    prove_bytecode_read_raf, verify_bytecode_read_raf, BytecodeReadRafProof,
};
use crate::zkvm::driver::{prove_binary_into, verify_binary_into, BinaryProof, DriverError};
use crate::zkvm::instruction_lookups::read_raf_sumcheck::{
    prove_instruction_read_raf, verify_instruction_read_raf, InstructionReadRafError,
    InstructionReadRafProof, InstructionTrace,
};
use crate::zkvm::logup::driver::{
    prove_read_raf_pushforward, verify_read_raf_pushforward, ReadRafPushforward,
};
use crate::zkvm::logup::gkr::GkrProof;
use crate::zkvm::logup::GkrError;
use crate::zkvm::real_trace::RealWitness;
use crate::zkvm::shout_read_raf::ReadRafStageError;
use crate::zkvm::stage8_columns::{inc_limb_columns, r1cs_aux_columns};
use jolt_field::Field;
use jolt_r1cs::R1csKey;
use jolt_trace::Instruction;

use super::driver::RamPublicColumns;

/// Instruction-lookup family geometry at production word size: `LOG_K = 2·XLEN = 128` interleaved
/// index bits, decomposed into `INSTRUCTION_D` committed chunks of `LOG_K/INSTRUCTION_D = log_k_chunk`
/// bits each (`log_k_chunk = 4` for muldiv ⇒ `D = 32`). `NE = D + 2` round-poly evaluation points.
/// Pinned (not a const generic) since the instruction family is independent of the bytecode `D`; the
/// e2e asserts the committed `instruction_range` spans exactly `INSTRUCTION_D` chunks.
const INSTRUCTION_XLEN: usize = 64;
const INSTRUCTION_D: usize = 32;
const INSTRUCTION_NE: usize = INSTRUCTION_D + 2;

/// e2e verification failure: a binary-driver stage, the bytecode read-raf, the M7 pushforward, or the
/// stage-8 WHIR open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum E2eError {
    Driver(DriverError),
    BytecodeReadRaf(ReadRafStageError),
    InstructionReadRaf(InstructionReadRafError),
    Pushforward(GkrError),
    Stage8(Stage8OpenError),
}

/// The full proof: the binary-driver sub-proof, the bytecode read-raf stage, the M7 per-chunk
/// pushforward GKRs, and the stage-8 limb opens (the WHIR opening bytes live in the shared NARG).
#[derive(Clone, Debug)]
pub struct E2eProof {
    pub binary: BinaryProof<F>,
    pub bytecode_read_raf: BytecodeReadRafProof<F>,
    pub bytecode_pushforward_gkr: Vec<GkrProof<F>>,
    pub instruction_read_raf: InstructionReadRafProof<F>,
    pub instruction_pushforward_gkr: Vec<GkrProof<F>>,
    pub inc: Stage8IncProof<F>,
    pub pushforward: Stage8PushforwardProof<F>,
    pub instruction_pushforward: Stage8PushforwardProof<F>,
}

/// Prover-side bytecode read-raf inputs: the padded bytecode table, the `D` committed chunk-index
/// columns (`ra_dense[bytecode_range]`), the chunk widths, the register-address bit width, and the
/// global RA-chunk base index (`bytecode_range.start`) for the `RaDense`/`Pushforward` keys.
pub struct BytecodeProverInputs<'a, const D: usize> {
    pub bytecode: &'a [Instruction],
    pub indices: [Vec<u32>; D],
    pub log_k_chunks: [usize; D],
    pub log_register: usize,
    pub base_index: usize,
}

/// Verifier-side bytecode read-raf inputs (no witness chunk indices — those are the prover's).
pub struct BytecodeVerifierInputs<'a, const D: usize> {
    pub bytecode: &'a [Instruction],
    pub log_k_chunks: [usize; D],
    pub log_register: usize,
    pub base_index: usize,
}

/// Prover-side instruction-lookup read-raf inputs: the three per-cycle lookup columns
/// (from [`crate::zkvm::instruction_lookups::trace::instruction_lookup_columns`]), the
/// `INSTRUCTION_D` committed chunk-index columns (`ra_dense[instruction_range]`), the chunk width,
/// and the global RA-chunk base index (`instruction_range.start`).
pub struct InstructionProverInputs {
    pub lookup_indices: Vec<u128>,
    pub lookup_table_indices: Vec<Option<usize>>,
    pub is_interleaved: Vec<bool>,
    pub indices: [Vec<u32>; INSTRUCTION_D],
    pub log_k_chunk: usize,
    pub base_index: usize,
}

/// Verifier-side instruction-lookup read-raf inputs (no witness columns — the verifier replays the
/// sumcheck against the proof-carried claim and re-seeds the proof openings).
pub struct InstructionVerifierInputs {
    pub log_k_chunk: usize,
    pub base_index: usize,
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

/// The bytecode family's per-chunk pushforward bridge (`BytecodeRa` read-raf → `RaDense`/`Pushforward`
/// GKR), keyed at the global RA-chunk `base_index`.
fn bytecode_pushforward_family<const D: usize>(
    log_t: usize,
    base_index: usize,
    log_k_chunks: &[usize; D],
) -> ReadRafPushforward {
    ReadRafPushforward {
        name: "BytecodeRa",
        log_t,
        base_index,
        log_m_chunks: log_k_chunks.to_vec(),
        ra_family: CommittedPolynomial::BytecodeRa,
        read_raf_id: SumcheckId::BytecodeReadRaf,
    }
}

/// The instruction-lookup family's per-chunk pushforward bridge (`InstructionRa` read-raf →
/// `RaDense`/`Pushforward` GKR), keyed at the global RA-chunk `base_index` (`instruction_range.start`).
/// Consumes the `INSTRUCTION_D` `InstructionRa(i)@InstructionReadRaf` openings the read-raf cached.
fn instruction_pushforward_family(
    log_t: usize,
    base_index: usize,
    log_k_chunk: usize,
) -> ReadRafPushforward {
    ReadRafPushforward {
        name: "InstructionRa",
        log_t,
        base_index,
        log_m_chunks: vec![log_k_chunk; INSTRUCTION_D],
        ra_family: CommittedPolynomial::InstructionRa,
        read_raf_id: SumcheckId::InstructionReadRaf,
    }
}

/// Append the bytecode `RaDense(base+i)@PushforwardGkr` requests for chunks with a NON-ZERO claim (a
/// zero-index chunk's column is all-zero — WHIR can't open it; skipped in lockstep, see
/// [`nonzero_r1cs_aux_requests`]).
fn push_radense_requests(
    requests: &mut Vec<Stage8Request>,
    accumulator: &Openings<F>,
    base_index: usize,
    d: usize,
    log_t: usize,
) {
    let zero = F::from_u64(0);
    for i in 0..d {
        let global = base_index + i;
        let claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RaDense(global),
                SumcheckId::PushforwardGkr,
            )
            .1;
        if claim != zero {
            requests.push(Stage8Request {
                poly: CommittedPolynomial::RaDense(global),
                sumcheck: SumcheckId::PushforwardGkr,
                committed_num_vars: log_t,
            });
        }
    }
}

/// The per-chunk `Pushforward(base+i)@PushforwardReduction` open points (`r_col`) + claimed `P^F(r_col)`.
fn pushforward_points_claims(
    accumulator: &Openings<F>,
    base_index: usize,
    d: usize,
) -> (Vec<Vec<F>>, Vec<F>) {
    let mut points = Vec::with_capacity(d);
    let mut claims = Vec::with_capacity(d);
    for i in 0..d {
        let (pt, claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::Pushforward(base_index + i),
            SumcheckId::PushforwardReduction,
        );
        points.push(pt.r);
        claims.push(claim);
    }
    (points, claims)
}

/// Prove the full e2e on a real-trace witness: binary stages → bytecode read-raf + M7 pushforward →
/// instruction-lookup read-raf + M7 pushforward → the stage-8 WHIR opens (`R1csAux` + bytecode +
/// instruction `RaDense` inventory, `Inc` limbs, both families' `Pushforward` limbs), on one
/// transcript. `D`/`NE = D+2` are the bytecode chunk count; the instruction family is pinned at
/// `INSTRUCTION_{XLEN,D,NE}`.
pub fn prove_e2e<const D: usize, const NE: usize>(
    real: &RealWitness<F>,
    bc: &BytecodeProverInputs<D>,
    instr: &InstructionProverInputs,
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

    // Bytecode read-raf: caches the D BytecodeRa(i) openings.
    let bytecode_read_raf = prove_bytecode_read_raf::<F, ProverTranscript, D, NE>(
        bc.bytecode,
        bc.indices.clone(),
        bc.log_k_chunks,
        log_t,
        bc.log_register,
        &mut accumulator,
        transcript,
    );

    // M7 per-chunk pushforward: discharge the bytecode BytecodeRa(i) openings into
    // RaDense(bc_base+i)@PushforwardGkr + Pushforward(bc_base+i)@{PushforwardGkr,PushforwardReduction}.
    let bc_fam = bytecode_pushforward_family(log_t, bc.base_index, &bc.log_k_chunks);
    let (bytecode_pushforward_gkr, bc_pf_columns) =
        prove_read_raf_pushforward(&bc_fam, &bc.indices, &mut accumulator, transcript)
            .map_err(E2eError::Pushforward)?;

    // Instruction-lookup read-raf (prefix/suffix address phase + one-hot cycle phase). Interim
    // Fork-2: the reduction point is freshly squeezed from the shared transcript (binding it to the
    // upstream InstructionClaimReduction is the deferred uni-skip Spartan closure). Caches the
    // INSTRUCTION_D InstructionRa(i) openings + the per-table / raf flags.
    let r_reduction = transcript.challenge_vector(log_t);
    let instruction_read_raf = prove_instruction_read_raf::<
        F,
        ProverTranscript,
        INSTRUCTION_XLEN,
        INSTRUCTION_D,
        INSTRUCTION_NE,
    >(
        InstructionTrace {
            lookup_indices: &instr.lookup_indices,
            lookup_table_indices: &instr.lookup_table_indices,
            is_interleaved: &instr.is_interleaved,
            indices: &instr.indices,
        },
        r_reduction,
        &mut accumulator,
        transcript,
    );

    // M7 per-chunk pushforward for the instruction family (keyed at instr.base_index, distinct from
    // the bytecode chunk range so the RaDense/Pushforward keys never collide).
    let instr_fam = instruction_pushforward_family(log_t, instr.base_index, instr.log_k_chunk);
    let (instruction_pushforward_gkr, instr_pf_columns) =
        prove_read_raf_pushforward(&instr_fam, &instr.indices, &mut accumulator, transcript)
            .map_err(E2eError::Pushforward)?;

    // Stage-8 inventory open: R1csAux + bytecode RaDense + instruction RaDense chunk columns (one
    // shared WHIR open; zero-claim chunks skipped in lockstep, see nonzero_r1cs_aux_requests).
    let aux = real.r1cs.boolean_aux_columns();
    let mut requests = nonzero_r1cs_aux_requests(&accumulator, aux.len(), log_t);
    push_radense_requests(&mut requests, &accumulator, bc.base_index, D, log_t);
    push_radense_requests(
        &mut requests,
        &accumulator,
        instr.base_index,
        INSTRUCTION_D,
        log_t,
    );
    let mut columns = r1cs_aux_columns(&aux);
    for (i, idx) in bc.indices.iter().enumerate() {
        let base: Vec<Base> = idx.iter().map(|&k| Base::from_u64(u64::from(k))).collect();
        columns.insert(CommittedPolynomial::RaDense(bc.base_index + i), base);
    }
    for (i, idx) in instr.indices.iter().enumerate() {
        let base: Vec<Base> = idx.iter().map(|&k| Base::from_u64(u64::from(k))).collect();
        columns.insert(CommittedPolynomial::RaDense(instr.base_index + i), base);
    }
    let inventory = Stage8Inventory::from_accumulator(&accumulator, &requests);
    prove_stage8(transcript, &columns, &inventory).map_err(E2eError::Stage8)?;

    // Inc limb open: the committed limbs decompose the memory stage's zero-init RdInc/RamInc (the
    // polynomials the IncClaimReduction claim is about), so they recompose to that claim.
    let committed_len = 1usize << log_t;
    let inc = inc_limb_columns(&real.registers.inc_i128, &real.ram.inc_i128, committed_len);
    let (rd_point, _) = accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RdInc,
        SumcheckId::IncClaimReduction,
    );
    let (ram_point, _) = accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RamInc,
        SumcheckId::IncClaimReduction,
    );
    let inc_proof = prove_inc_open(transcript, &inc, &rd_point.r, &ram_point.r);

    // Pushforward limb opens: bytecode then instruction (same order on the verify side). Commit each
    // family's surfaced P^F columns and open at each chunk's r_col.
    let (bc_pf_points, _) = pushforward_points_claims(&accumulator, bc.base_index, D);
    let bc_pf_chunks: Vec<Fp3LimbColumns> = bc_pf_columns
        .iter()
        .map(|c| Fp3LimbColumns::from_fp3(c))
        .collect();
    let pushforward = prove_pushforward_open(transcript, &bc_pf_chunks, &bc_pf_points);

    let (instr_pf_points, _) =
        pushforward_points_claims(&accumulator, instr.base_index, INSTRUCTION_D);
    let instr_pf_chunks: Vec<Fp3LimbColumns> = instr_pf_columns
        .iter()
        .map(|c| Fp3LimbColumns::from_fp3(c))
        .collect();
    let instruction_pushforward =
        prove_pushforward_open(transcript, &instr_pf_chunks, &instr_pf_points);

    Ok(E2eProof {
        binary,
        bytecode_read_raf,
        bytecode_pushforward_gkr,
        instruction_read_raf,
        instruction_pushforward_gkr,
        inc: inc_proof,
        pushforward,
        instruction_pushforward,
    })
}

/// Verify the full e2e (mirror of [`prove_e2e`]).
pub fn verify_e2e<const D: usize, const NE: usize>(
    proof: &E2eProof,
    params: &VerifierParams,
    bc: &BytecodeVerifierInputs<D>,
    instr: &InstructionVerifierInputs,
    transcript: &mut VerifierTranscript,
) -> Result<(), E2eError> {
    let log_t = params.log_num_cycles;
    let mut accumulator = Openings::<F>::new(log_t);

    verify_binary_into(
        &proof.binary,
        params.key,
        params.num_row_vars,
        log_t,
        params.ram_log_k,
        params.reg_log_k,
        params.ram_public,
        &mut accumulator,
        transcript,
    )
    .map_err(E2eError::Driver)?;

    verify_bytecode_read_raf::<F, VerifierTranscript, D, NE>(
        &proof.bytecode_read_raf,
        bc.bytecode,
        bc.log_k_chunks,
        log_t,
        bc.log_register,
        &mut accumulator,
        transcript,
    )
    .map_err(E2eError::BytecodeReadRaf)?;

    let bc_fam = bytecode_pushforward_family(log_t, bc.base_index, &bc.log_k_chunks);
    verify_read_raf_pushforward(
        &bc_fam,
        &proof.bytecode_pushforward_gkr,
        &mut accumulator,
        transcript,
    )
    .map_err(E2eError::Pushforward)?;

    // Instruction read-raf (mirror): draw r_reduction in lockstep, replay against the proof claim.
    let r_reduction = transcript.challenge_vector(log_t);
    verify_instruction_read_raf::<
        F,
        VerifierTranscript,
        INSTRUCTION_XLEN,
        INSTRUCTION_D,
        INSTRUCTION_NE,
    >(
        &proof.instruction_read_raf,
        r_reduction,
        &mut accumulator,
        transcript,
    )
    .map_err(E2eError::InstructionReadRaf)?;

    let instr_fam = instruction_pushforward_family(log_t, instr.base_index, instr.log_k_chunk);
    verify_read_raf_pushforward(
        &instr_fam,
        &proof.instruction_pushforward_gkr,
        &mut accumulator,
        transcript,
    )
    .map_err(E2eError::Pushforward)?;

    let mut requests = nonzero_r1cs_aux_requests(&accumulator, params.n_aux, log_t);
    push_radense_requests(&mut requests, &accumulator, bc.base_index, D, log_t);
    push_radense_requests(
        &mut requests,
        &accumulator,
        instr.base_index,
        INSTRUCTION_D,
        log_t,
    );
    let inventory = Stage8Inventory::from_accumulator(&accumulator, &requests);
    verify_stage8(transcript, &inventory).map_err(E2eError::Stage8)?;

    let (rd_point, rd_claim) = accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RdInc,
        SumcheckId::IncClaimReduction,
    );
    let (ram_point, ram_claim) = accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RamInc,
        SumcheckId::IncClaimReduction,
    );
    verify_inc_open(
        transcript,
        &rd_point.r,
        &ram_point.r,
        &proof.inc,
        rd_claim,
        ram_claim,
    )
    .map_err(E2eError::Stage8)?;

    // Pushforward limb opens: bytecode then instruction (same order as prove_e2e). Points + claimed
    // P^F(r_col) come from the accumulator (each read-raf pushforward registered
    // Pushforward(base+i)@PushforwardReduction).
    let (bc_pf_points, bc_pf_claims) = pushforward_points_claims(&accumulator, bc.base_index, D);
    verify_pushforward_open(transcript, &bc_pf_points, &proof.pushforward, &bc_pf_claims)
        .map_err(E2eError::Stage8)?;

    let (instr_pf_points, instr_pf_claims) =
        pushforward_points_claims(&accumulator, instr.base_index, INSTRUCTION_D);
    verify_pushforward_open(
        transcript,
        &instr_pf_points,
        &proof.instruction_pushforward,
        &instr_pf_claims,
    )
    .map_err(E2eError::Stage8)?;

    Ok(())
}
