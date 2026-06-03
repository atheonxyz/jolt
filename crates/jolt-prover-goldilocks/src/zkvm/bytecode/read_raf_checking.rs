//! Bytecode read + RAF checking — instantiates the shared
//! [`crate::zkvm::shout_read_raf`] `OneHotReadRaf` with the `BytecodeRa` committed family and the
//! [`SumcheckId::BytecodeReadRaf`] id. Ported from jolt-core's `zkvm/bytecode/read_raf_checking.rs`
//! (the parity oracle); see [`crate::zkvm::shout_read_raf`] for the shared identity and the M5
//! decoupling/deferral notes.

use crate::framework::transcript::{Challenge, ProverFs, VerifierFs};
use jolt_field::Field;
use jolt_lookup_tables::{instruction_lookup_table_index, LookupTableKind};
use jolt_poly::EqPolynomial;
use jolt_riscv::{CircuitFlags, InstructionFlags, InterleavedBitsMarker, NUM_CIRCUIT_FLAGS};
use jolt_trace::{instruction_circuit_flags, instruction_instruction_flags, Instruction};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
};

pub use crate::zkvm::shout_read_raf::{
    prove_read_raf, verify_read_raf, OneHotReadRaf, OneHotReadRafParams, ReadRafInputs,
    ReadRafStage, ReadRafStageError, ReadRafStageProof,
};

/// Bytecode address decomposition uses `D = 2` chunks (`NE = D + 2 = 4`).
pub const BYTECODE_D: usize = 2;

/// Number of bytecode read-raf stages (stages 1–5 of jolt-core's `compute_val_polys`).
///
/// Stage 5 (registers val-evaluation + instruction-lookup membership) virtualizes the claims output
/// by the registers val-evaluation + instruction read-raf sumchecks:
/// `Val_5(k) = eq(rd(k), r_register_5) + γ·[¬interleaved(k)] + Σ_t γ^{2+t}·[lookup_table(k) == t]`.
/// Its `lookup_table` term uses the jolt-core-free [`instruction_lookup_table_index`] dispatch
/// (P3b-0; the historical `LookupTableKind`-bridge gap is closed). Like stages 1–4 its `rv` seed is
/// carried in the proof (interim fork-2), bound by the read-raf sumcheck.
pub const N_BYTECODE_STAGES: usize = 5;

/// Number of lookup tables (`LookupTableKind::all()` length) — the per-table membership flags in the
/// stage-5 `Val_5` column.
const NUM_LOOKUP_TABLES: usize = 40;

/// Build the bytecode read-raf `Val_s(k)` columns (length `bytecode.len()`) — a field-generic port
/// of jolt-core's `compute_val_polys` (stages 1–4). Each `Val_s(k)` is a per-stage γ-power-weighted
/// combination of bytecode row `k`'s decoded fields/flags, read via the field-generic
/// `jolt_trace::instruction_{circuit,instruction}_flags` bridge (`tracer::Instruction` → jolt-riscv
/// flag sets):
/// - **Stage 1** (Spartan outer): `addr + γ¹·imm + Σ_t γ^{2+t}·circuit_flag_t`.
/// - **Stage 2** (product virtualization): `γ⁰·jump + γ¹·branch + γ²·write_lookup_to_rd
///   + γ³·virtual_instruction`.
/// - **Stage 3** (shift): `imm + γ¹·addr + γ²·L_is_rs1 + γ³·L_is_pc + γ⁴·R_is_rs2 + γ⁵·R_is_imm
///   + γ⁶·is_noop + γ⁷·virtual_instruction + γ⁸·is_first_in_sequence`.
/// - **Stage 4** (registers read-write): `γ⁰·eq(rd,r) + γ¹·eq(rs1,r) + γ²·eq(rs2,r)`,
///   `eq(x,r) = eq_r_register[x]` (`None` register → 0).
/// - **Stage 5** (registers val-eval + instruction-lookup membership):
///   `eq(rd, r_register_5) + γ¹·[¬is_interleaved] + Σ_t γ^{2+t}·[lookup_table == t]`. The membership
///   term routes through the [`instruction_lookup_table_index`] dispatch (`XLEN=64`).
///
/// `stage_gammas[s]` holds the within-stage γ powers; `eq_r_register`/`eq_r_register_5 =
/// EqPolynomial::evals(r_register{,_5})` (length the register-address space) are the stage-4 / stage-5
/// register points (DISTINCT: jolt-core binds stage 5 to `RdWa@RegistersValEvaluation`, stage 4 to
/// `RdWa@RegistersReadWriteChecking`). The columns feed `OneHotReadRaf` as the per-stage
/// [`ReadRafStage::val_addr`] (bytecode-row-indexed, the address-only `Val_s`).
pub fn bytecode_val_polys<F: Field>(
    bytecode: &[Instruction],
    stage_gammas: &[Vec<F>; N_BYTECODE_STAGES],
    eq_r_register: &[F],
    eq_r_register_5: &[F],
) -> [Vec<F>; N_BYTECODE_STAGES] {
    debug_assert_eq!(
        LookupTableKind::<64>::all().len(),
        NUM_LOOKUP_TABLES,
        "stage-5 γ vector sized for NUM_LOOKUP_TABLES membership flags"
    );
    let k = bytecode.len();
    let mut vals: [Vec<F>; N_BYTECODE_STAGES] = std::array::from_fn(|_| vec![F::zero(); k]);
    for (idx, instruction) in bytecode.iter().enumerate() {
        let instr = instruction.normalize();
        let cf = instruction_circuit_flags(instruction);
        let iflags = instruction_instruction_flags(instruction);
        let addr = F::from_u64(instr.address as u64);
        let imm = F::from_i128(instr.operands.imm);

        // Stage 1: addr + γ¹·imm + Σ_t γ^{2+t}·circuit_flag_t.
        {
            let g = &stage_gammas[0];
            let mut lc = addr + g[1] * imm;
            for t in 0..NUM_CIRCUIT_FLAGS {
                if (cf.bits() >> t) & 1 == 1 {
                    lc += g[2 + t];
                }
            }
            vals[0][idx] = lc;
        }
        // Stage 2: γ⁰·jump + γ¹·branch + γ²·write_lookup_to_rd + γ³·virtual_instruction.
        {
            let g = &stage_gammas[1];
            let mut lc = F::zero();
            if cf.get(CircuitFlags::Jump) {
                lc += g[0];
            }
            if iflags.get(InstructionFlags::Branch) {
                lc += g[1];
            }
            if cf.get(CircuitFlags::WriteLookupOutputToRD) {
                lc += g[2];
            }
            if cf.get(CircuitFlags::VirtualInstruction) {
                lc += g[3];
            }
            vals[1][idx] = lc;
        }
        // Stage 3: imm + γ¹·addr + operand-source / noop / virtual / first-in-seq flags.
        {
            let g = &stage_gammas[2];
            let mut lc = imm + g[1] * addr;
            if iflags.get(InstructionFlags::LeftOperandIsRs1Value) {
                lc += g[2];
            }
            if iflags.get(InstructionFlags::LeftOperandIsPC) {
                lc += g[3];
            }
            if iflags.get(InstructionFlags::RightOperandIsRs2Value) {
                lc += g[4];
            }
            if iflags.get(InstructionFlags::RightOperandIsImm) {
                lc += g[5];
            }
            if iflags.get(InstructionFlags::IsNoop) {
                lc += g[6];
            }
            if cf.get(CircuitFlags::VirtualInstruction) {
                lc += g[7];
            }
            if cf.get(CircuitFlags::IsFirstInSequence) {
                lc += g[8];
            }
            vals[2][idx] = lc;
        }
        // Stage 4: γ⁰·eq(rd,r) + γ¹·eq(rs1,r) + γ²·eq(rs2,r).
        {
            let g = &stage_gammas[3];
            let reg = |r: Option<u8>| r.map_or(F::zero(), |r| eq_r_register[r as usize]);
            vals[3][idx] = reg(instr.operands.rd) * g[0]
                + reg(instr.operands.rs1) * g[1]
                + reg(instr.operands.rs2) * g[2];
        }
        // Stage 5: eq(rd, r_register_5) + γ¹·[¬is_interleaved] + Σ_t γ^{2+t}·[lookup_table == t].
        // rd_eq is unweighted (jolt-core's implicit γ⁰ = 1). The lookup-table index uses the
        // jolt-core-free dispatch at XLEN = 64 (RV64; matches the instruction read-raf's table order).
        {
            let g = &stage_gammas[4];
            let mut lc = instr
                .operands
                .rd
                .map_or(F::zero(), |r| eq_r_register_5[r as usize]);
            if !cf.is_interleaved_operands() {
                lc += g[1];
            }
            if let Some(table) = instruction_lookup_table_index::<64>(instruction) {
                lc += g[2 + table];
            }
            vals[4][idx] = lc;
        }
    }
    vals
}

/// The bytecode read-raf stage proof: the carried interim `rv_s` seeds (one per `Val_s` stage) plus
/// the underlying read-raf proof. As with the memory stage's `spartan_seeds` (fork 2), the verifier
/// cannot recompute `rv_s` (it lacks the witness chunk indices), so the prover carries them; the
/// read-raf sumcheck binds them (a wrong seed fails the round-0 / output-claim check).
#[derive(Clone, Debug)]
pub struct BytecodeReadRafProof<F: Field> {
    pub rv_seeds: [F; N_BYTECODE_STAGES],
    pub read_raf: ReadRafStageProof<F>,
}

/// Five DISTINCT interim `rv_key`s for the bytecode `Val_s` stages, each free in the binary driver:
/// the shift/product-virtualization sumchecks aren't run, and `UnexpandedPC`/`PC`/`RdWa` aren't
/// seeded at these `(poly, sumcheck)` pairs by any stage (the driver seeds only
/// `Ram*`/`Rd*`/`Rs*`/`Spartan{Az,Bz,Cz}` at `SpartanOuter`, and `PC` at
/// `SpartanProductVirtualization`). They are labels for the interim seed slots; the sound binding to
/// the real upstream openings (stage 5's `RdWa@RegistersValEvaluation` + `*Flag@InstructionReadRaf`)
/// is the deferred uni-skip Spartan (fork 2).
fn bytecode_rv_keys() -> [(VirtualPolynomial, SumcheckId); N_BYTECODE_STAGES] {
    [
        (VirtualPolynomial::UnexpandedPC, SumcheckId::SpartanOuter),
        (
            VirtualPolynomial::PC,
            SumcheckId::SpartanProductVirtualization,
        ),
        (VirtualPolynomial::Imm, SumcheckId::SpartanShift),
        (VirtualPolynomial::PC, SumcheckId::SpartanOuter),
        (
            VirtualPolynomial::RdWa,
            SumcheckId::SpartanProductVirtualization,
        ),
    ]
}

/// γ-power vector length per `Val_s` stage (the highest `g[·]` index used in `bytecode_val_polys`).
/// Stage 5 uses `g[0..2+NUM_LOOKUP_TABLES]` (`g[0]` rd-eq weight = 1, `g[1]` ¬interleaved, `g[2+t]`
/// per-table membership).
fn stage_gamma_lens() -> [usize; N_BYTECODE_STAGES] {
    [2 + NUM_CIRCUIT_FLAGS, 4, 9, 3, 2 + NUM_LOOKUP_TABLES]
}

#[inline]
fn gamma_powers<F: Field>(g: F, len: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(len);
    let mut p = F::from_u64(1);
    for _ in 0..len {
        v.push(p);
        p *= g;
    }
    v
}

/// Per-cycle combined bytecode index `Σ_i idx_i[j]·suffix[i]` (chunk 0 most significant) — equals the
/// bytecode-row index, so `val_addr[combined[j]] = Val_s(row[j])`.
fn combined_indices<const D: usize>(
    indices: &[Vec<u32>; D],
    log_k_chunks: [usize; D],
    t: usize,
) -> Vec<usize> {
    let k_dims: [usize; D] = std::array::from_fn(|i| 1usize << log_k_chunks[i]);
    let mut suffix = [0usize; D];
    let mut acc = 1usize;
    for i in (0..D).rev() {
        suffix[i] = acc;
        acc *= k_dims[i];
    }
    (0..t)
        .map(|j| (0..D).fold(0usize, |a, i| a + (indices[i][j] as usize) * suffix[i]))
        .collect()
}

/// Draw the per-stage γ powers + the two register points (stage 4 + stage 5) in lockstep, build the
/// five `Val_s` columns (padded to `K_total = ∏ 2^{log_k_chunks}`), draw each stage's cycle point,
/// and assemble the `ReadRafStage`s (WITHOUT the `rv_s` seed — that is computed by the prover / read
/// from the proof by the verifier). Deterministic given the transcript, so prover and verifier agree.
fn bytecode_read_raf_setup<F: Field, T: Challenge<F>, const D: usize>(
    bytecode: &[Instruction],
    log_k_chunks: [usize; D],
    log_t: usize,
    log_register: usize,
    transcript: &mut T,
) -> Vec<ReadRafStage<F>> {
    let k_total: usize = log_k_chunks.iter().map(|&w| 1usize << w).product();
    let lens = stage_gamma_lens();
    let stage_gammas: [Vec<F>; N_BYTECODE_STAGES] =
        std::array::from_fn(|s| gamma_powers(transcript.challenge(), lens[s]));
    let r_register = transcript.challenge_vector(log_register);
    let eq_r_register = EqPolynomial::<F>::evals(&r_register, None);
    // Stage 5's register point is DISTINCT from stage 4's (jolt-core binds it to
    // RdWa@RegistersValEvaluation vs stage 4's RdWa@RegistersReadWriteChecking). Drawn fresh in the
    // interim fork-2 model (carried-seed); the upstream binding is the deferred uni-skip Spartan.
    let r_register_5 = transcript.challenge_vector(log_register);
    let eq_r_register_5 = EqPolynomial::<F>::evals(&r_register_5, None);
    let mut vals =
        bytecode_val_polys::<F>(bytecode, &stage_gammas, &eq_r_register, &eq_r_register_5);
    let keys = bytecode_rv_keys();
    (0..N_BYTECODE_STAGES)
        .map(|s| {
            let r_cycle = transcript.challenge_vector(log_t);
            let mut val_addr = std::mem::take(&mut vals[s]);
            val_addr.resize(k_total, F::zero());
            ReadRafStage {
                r_cycle,
                val_addr,
                rv_key: keys[s],
            }
        })
        .collect()
}

fn bytecode_read_raf_inputs<F: Field, const D: usize>(
    log_k_chunks: [usize; D],
    log_t: usize,
    stages: Vec<ReadRafStage<F>>,
) -> ReadRafInputs<F, D> {
    ReadRafInputs {
        ra_family: CommittedPolynomial::BytecodeRa,
        sumcheck_id: SumcheckId::BytecodeReadRaf,
        log_k_chunks,
        log_t,
        stages,
    }
}

/// **Bytecode read-raf stage (prover).** Build the four interim-seeded `Val_s` stages, seed each
/// stage's `rv_s = Σ_j eq(r_cycle_s, j)·Val_s(combined_index[j])` on the accumulator, then run the
/// sparse read-raf over the `D` bytecode chunk-index columns. Returns the carried `rv_s` seeds + the
/// read-raf proof (which caches the `D` `BytecodeRa(i)` openings for the M7 pushforward).
pub fn prove_bytecode_read_raf<F, T, const D: usize, const NE: usize>(
    bytecode: &[Instruction],
    indices: [Vec<u32>; D],
    log_k_chunks: [usize; D],
    log_t: usize,
    log_register: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> BytecodeReadRafProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let stages =
        bytecode_read_raf_setup::<F, T, D>(bytecode, log_k_chunks, log_t, log_register, transcript);
    let t = 1usize << log_t;
    let combined = combined_indices(&indices, log_k_chunks, t);
    let mut rv_seeds: [F; N_BYTECODE_STAGES] = std::array::from_fn(|_| F::zero());
    for (s, stage) in stages.iter().enumerate() {
        let eq = EqPolynomial::<F>::evals(&stage.r_cycle, None);
        let rv = (0..t).fold(F::zero(), |a, j| a + eq[j] * stage.val_addr[combined[j]]);
        rv_seeds[s] = rv;
        accumulator.append_virtual(
            stage.rv_key.0,
            stage.rv_key.1,
            OpeningPoint::new(stage.r_cycle.clone()),
            rv,
        );
    }
    let inputs = bytecode_read_raf_inputs(log_k_chunks, log_t, stages);
    let read_raf = prove_read_raf::<F, T, D, NE>(indices, inputs, accumulator, transcript);
    BytecodeReadRafProof { rv_seeds, read_raf }
}

/// **Bytecode read-raf stage (verifier)** (mirror of [`prove_bytecode_read_raf`]). Rebuild the same
/// stages, re-seed each `rv_s` from the proof (the verifier lacks the chunk indices), then replay the
/// read-raf sumcheck.
pub fn verify_bytecode_read_raf<F, T, const D: usize, const NE: usize>(
    proof: &BytecodeReadRafProof<F>,
    bytecode: &[Instruction],
    log_k_chunks: [usize; D],
    log_t: usize,
    log_register: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), ReadRafStageError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let stages =
        bytecode_read_raf_setup::<F, T, D>(bytecode, log_k_chunks, log_t, log_register, transcript);
    for (s, stage) in stages.iter().enumerate() {
        accumulator.append_virtual(
            stage.rv_key.0,
            stage.rv_key.1,
            OpeningPoint::new(stage.r_cycle.clone()),
            proof.rv_seeds[s],
        );
    }
    let inputs = bytecode_read_raf_inputs(log_k_chunks, log_t, stages);
    verify_read_raf::<F, T, D, NE>(&proof.read_raf, inputs, accumulator, transcript)
}

/// Build the bytecode read-raf params (`BytecodeRa` family, `BytecodeReadRaf` id).
pub fn bytecode_read_raf_params<F: Field>(
    log_k_chunks: [usize; BYTECODE_D],
    log_t: usize,
    stages: Vec<ReadRafStage<F>>,
    transcript: &mut impl Challenge<F>,
) -> OneHotReadRafParams<F, BYTECODE_D> {
    OneHotReadRafParams::new(
        CommittedPolynomial::BytecodeRa,
        SumcheckId::BytecodeReadRaf,
        log_k_chunks,
        log_t,
        stages,
        transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    /// `Val_s` for a bytecode of `NoOp`s: jolt-trace's flag bridge reports exactly
    /// `{DoNotUpdateUnexpandedPC}` (circuit) + `{IsNoop}` (instruction) for `NoOp`, so each stage's
    /// column is the formula with only those flag terms active.
    #[test]
    fn bytecode_val_polys_noop() {
        let bytecode: Vec<Instruction> = (0..3).map(|_| Instruction::NoOp).collect();
        let stage_gammas: [Vec<F>; N_BYTECODE_STAGES] = [
            (0..(2 + NUM_CIRCUIT_FLAGS) as u64)
                .map(|i| F::from_u64(100 + i))
                .collect(),
            (0..4).map(|i| F::from_u64(200 + i)).collect(),
            (0..9).map(|i| F::from_u64(300 + i)).collect(),
            (0..3).map(|i| F::from_u64(400 + i)).collect(),
            (0..(2 + NUM_LOOKUP_TABLES) as u64)
                .map(|i| F::from_u64(500 + i))
                .collect(),
        ];
        let eq_r = vec![F::from_u64(7); 32];
        let eq_r5 = vec![F::from_u64(11); 32];
        let v = bytecode_val_polys::<F>(&bytecode, &stage_gammas, &eq_r, &eq_r5);

        let n = Instruction::NoOp.normalize();
        let addr = F::from_u64(n.address as u64);
        let imm = F::from_i128(n.operands.imm);
        let dnu = CircuitFlags::DoNotUpdateUnexpandedPC as usize;

        for col in &v {
            assert_eq!(col.len(), 3, "one entry per bytecode row");
        }
        // Stage 1: addr + γ¹·imm + γ^{2+DoNotUpdateUnexpandedPC} (NoOp's only circuit flag).
        assert_eq!(
            v[0][0],
            addr + stage_gammas[0][1] * imm + stage_gammas[0][2 + dnu]
        );
        // Stage 2: no jump/branch/write_lookup/virtual ⇒ 0.
        assert_eq!(v[1][0], F::from_u64(0));
        // Stage 3: imm + γ¹·addr + γ⁶·IsNoop (NoOp's only instruction flag).
        assert_eq!(
            v[2][0],
            imm + stage_gammas[2][1] * addr + stage_gammas[2][6]
        );
        // Stage 4: NoOp has no rd/rs1/rs2 register ⇒ 0.
        let reg = |r: Option<u8>| r.map_or(F::from_u64(0), |r| eq_r[r as usize]);
        assert_eq!(
            v[3][0],
            reg(n.operands.rd) * stage_gammas[3][0]
                + reg(n.operands.rs1) * stage_gammas[3][1]
                + reg(n.operands.rs2) * stage_gammas[3][2]
        );
        // Stage 5: NoOp has no rd (⇒ 0), IS interleaved (none of Add/Sub/Mul/Advice ⇒ ¬interleaved
        // is false, no γ¹ term), and no lookup table (⇒ no membership term) ⇒ Val_5 = 0.
        assert!(
            instruction_circuit_flags(&Instruction::NoOp).is_interleaved_operands(),
            "NoOp is interleaved (no arithmetic/advice flag)"
        );
        assert_eq!(
            instruction_lookup_table_index::<64>(&Instruction::NoOp),
            None,
            "NoOp has no lookup table"
        );
        assert_eq!(v[4][0], F::from_u64(0), "Val_5(NoOp) = 0");
        // All rows identical (same instruction).
        assert_eq!(v[0][0], v[0][2]);
    }
}
