//! Bytecode read + RAF checking — instantiates the shared
//! [`crate::zkvm::shout_read_raf`] `OneHotReadRaf` with the `BytecodeRa` committed family and the
//! [`SumcheckId::BytecodeReadRaf`] id. Ported from jolt-core's `zkvm/bytecode/read_raf_checking.rs`
//! (the parity oracle); see [`crate::zkvm::shout_read_raf`] for the shared identity and the M5
//! decoupling/deferral notes.

use crate::framework::transcript::Challenge;
use jolt_field::Field;
use jolt_riscv::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS};
use jolt_trace::{instruction_circuit_flags, instruction_instruction_flags, Instruction};

use crate::framework::accumulator::{CommittedPolynomial, SumcheckId};

pub use crate::zkvm::shout_read_raf::{OneHotReadRaf, OneHotReadRafParams, ReadRafStage};

/// Bytecode address decomposition uses `D = 2` chunks (`NE = D + 2 = 4`).
pub const BYTECODE_D: usize = 2;

/// Number of bytecode read-raf stages ported (stages 1–4 of jolt-core's `compute_val_polys`).
///
/// Stage 5 (registers val-evaluation + instruction-lookup membership) is **deferred**: its
/// `Σ_i γ^{2+i}·table_i` term needs the `LookupTableKind` bridge for `tracer::Instruction` (only
/// jolt-core has `InstructionLookup for Instruction` today; jolt-trace exposes the flag bridge but
/// not a lookup-table one). The register `eq(rd,r)` + `¬interleaved` parts of stage 5 land with it.
pub const N_BYTECODE_STAGES: usize = 4;

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
/// - **Stage 4** (registers): `γ⁰·eq(rd,r) + γ¹·eq(rs1,r) + γ²·eq(rs2,r)`,
///   `eq(x,r) = eq_r_register[x]` (`None` register → 0).
///
/// `stage_gammas[s]` holds the within-stage γ powers; `eq_r_register = EqPolynomial::evals(r_register)`
/// (length the register-address space) is for stage 4. The columns feed `OneHotReadRaf` as the
/// per-stage [`ReadRafStage::val_addr`] (bytecode-row-indexed, the address-only `Val_s`).
pub fn bytecode_val_polys<F: Field>(
    bytecode: &[Instruction],
    stage_gammas: &[Vec<F>; N_BYTECODE_STAGES],
    eq_r_register: &[F],
) -> [Vec<F>; N_BYTECODE_STAGES] {
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
    }
    vals
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
        ];
        let eq_r = vec![F::from_u64(7); 32];
        let v = bytecode_val_polys::<F>(&bytecode, &stage_gammas, &eq_r);

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
        // All rows identical (same instruction).
        assert_eq!(v[0][0], v[0][2]);
    }
}
