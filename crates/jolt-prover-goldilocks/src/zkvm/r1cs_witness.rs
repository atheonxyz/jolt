//! Multi-cycle limbed R1CS witness materialization: per-cycle limbed `z`-assignments → the
//! cycle-major witness `z` + the `Az/Bz/Cz` columns over the `(cycle, constraint)` hypercube. This
//! is the bridge between the per-cycle limbed witness (the `r1cs/rv64_limbed.rs` `Vars` layout, one
//! assignment per trace cycle) and the binary Spartan **outer** (consumes `Az/Bz/Cz`) + **inner**
//! reduction (consumes `z` via [`jolt_r1cs::R1csKey`]).
//!
//! Layout (matching `R1csKey`'s uniform structure): `z[cycle·num_vars_padded + var]`,
//! `Az/Bz/Cz[cycle·num_cons_padded + constraint]` (cycle in the high bits, so the row/col MLE point
//! splits as `(r_cycle ‖ r_constraint)` / `(r_cycle ‖ r_var)` exactly as `R1csKey` expects). Vars
//! `70 → 128` padded, constraints `53 → 64` padded, cycles padded to a power of two.
//!
//! **Decoupled from the trace** (the M5 convention): takes the per-cycle limbed `z`-assignments
//! (`Vec<F>` of length `num_vars`). The trace `Cycle` → limbed-`z` extraction (per the
//! `r1cs/rv64_limbed.rs` op builders + the `Cycle` API) is the M8 stage-driver witness-gen.

use jolt_field::Field;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_trace::CycleRow;

use crate::r1cs::{layout, rv64_limbed_constraints, Vars};

const LIMB_MASK: u64 = 0xFFFF_FFFF;

/// Set a 2-limb unsigned column `[lo, hi]` from a `u64`.
#[inline]
fn set_u64_limbs<F: Field>(w: &mut [F], col: [usize; 2], val: u64) {
    w[col[0]] = F::from_u64(val & LIMB_MASK);
    w[col[1]] = F::from_u64(val >> 32);
}

/// Set a signed 2-limb column `[lo, hi]` from an `i128` (`v = lo + 2³²·hi`, `lo ∈ [0,2³²)`, `hi`
/// signed). Mirrors `jolt_field::goldilocks::decompose::i128_to_signed_limbs` but over generic `F`.
#[inline]
fn set_signed_limbs<F: Field>(w: &mut [F], col: [usize; 2], v: i128) {
    let lo = v.rem_euclid(1i128 << 32) as u64;
    let hi = (v - i128::from(lo)) >> 32;
    w[col[0]] = F::from_u64(lo);
    w[col[1]] = F::from_i128(hi);
}

/// The always-active MUL schoolbook product witness (constraint 19 + `Left.sign = 0`) for
/// `Left × Right` with `Left` unsigned and `Right` the magnitude. Ported from `rv64_limbed.rs`'s
/// `fill_product` test helper, generic over `F`.
pub(crate) fn fill_product<F: Field>(
    w: &mut [F],
    v: &Vars,
    left: u64,
    right_mag: u64,
    right_sign: bool,
) {
    let (llo, lhi) = (left & LIMB_MASK, left >> 32);
    let (rlo, rhi) = (right_mag & LIMB_MASK, right_mag >> 32);
    let q0 = u128::from(llo) * u128::from(rlo);
    let q1 = u128::from(llo) * u128::from(rhi);
    let q2 = u128::from(lhi) * u128::from(rlo);
    let q3 = u128::from(lhi) * u128::from(rhi);
    let (p0, c0) = (q0 & u128::from(LIMB_MASK), q0 >> 32);
    let s1 = q1 + q2 + c0;
    let (p1, c1) = (s1 & u128::from(LIMB_MASK), s1 >> 32);
    let s2 = q3 + c1;
    let (p2, c2) = (s2 & u128::from(LIMB_MASK), s2 >> 32);
    let p3 = c2;
    let u = F::from_u64;
    let um = F::from_u128;
    w[v.left[0]] = u(llo);
    w[v.left[1]] = u(lhi);
    w[v.left_sign] = u(0);
    w[v.right_mag[0]] = u(rlo);
    w[v.right_mag[1]] = u(rhi);
    w[v.right_sign] = u(u64::from(right_sign));
    w[v.product[0]] = um(p0);
    w[v.product[1]] = um(p1);
    w[v.product[2]] = um(p2);
    w[v.product[3]] = um(p3);
    w[v.product_sign] = u(u64::from(right_sign));
    w[v.q[0]] = um(q0);
    w[v.q[1]] = um(q1);
    w[v.q[2]] = um(q2);
    w[v.q[3]] = um(q3);
    w[v.mul_c0] = um(c0);
    w[v.mul_c1] = um(c1);
    w[v.mul_c2] = um(c2);
    w[v.sign_prod] = u(0);
}

/// `RightLookupOperand = Left + Right` (65-bit ADD), 4 limbs + the low-limb carry `add_c0`.
fn fill_add_rlo<F: Field>(w: &mut [F], v: &Vars, left: u64, right: u64) {
    let sum = u128::from(left) + u128::from(right);
    let c0 = (u128::from(left & LIMB_MASK) + u128::from(right & LIMB_MASK)) >> 32;
    w[v.add_c0] = F::from_u128(c0);
    w[v.rlo[0]] = F::from_u128(sum & u128::from(LIMB_MASK));
    w[v.rlo[1]] = F::from_u128((sum >> 32) & u128::from(LIMB_MASK));
    w[v.rlo[2]] = F::from_u128((sum >> 64) & u128::from(LIMB_MASK));
    w[v.rlo[3]] = F::from_u128(sum >> 96);
}

/// `RightLookupOperand = Left − Right + 2⁶⁴` (SUB), 4 limbs + carries `sub_c0`/`sub_c1` from
/// `RLO + Right = Left + 2⁶⁴`.
fn fill_sub_rlo<F: Field>(w: &mut [F], v: &Vars, left: u64, right: u64) {
    let rlo_val = u128::from(left) + (1u128 << 64) - u128::from(right);
    let r0 = (rlo_val & u128::from(LIMB_MASK)) as u64;
    let r1 = ((rlo_val >> 32) & u128::from(LIMB_MASK)) as u64;
    let r2 = ((rlo_val >> 64) & u128::from(LIMB_MASK)) as u64;
    let r3 = (rlo_val >> 96) as u64;
    w[v.rlo[0]] = F::from_u64(r0);
    w[v.rlo[1]] = F::from_u64(r1);
    w[v.rlo[2]] = F::from_u64(r2);
    w[v.rlo[3]] = F::from_u64(r3);
    let c0 = (u128::from(r0) + u128::from(right & LIMB_MASK)) >> 32;
    w[v.sub_c0] = F::from_u128(c0);
    let c1 = (u128::from(r1) + u128::from(right >> 32) + c0) >> 32;
    w[v.sub_c1] = F::from_u128(c1);
}

/// Map one trace cycle to its limbed `z`-assignment (length `num_vars`), mirroring jolt-trace's
/// `r1cs_cycle_witness` (the workspace BN254 extraction) into the limbed `Vars` layout with
/// per-value limbing + the MUL schoolbook / add-sub carries.
///
/// **Coverage:** no-op, the arithmetic ops (ADD/SUB/MUL via the lookup-operand cases), the default
/// lookup case, loads/stores (RAM address `Rs1 + Imm`), and the flag/PC/should-branch/should-jump
/// witness. Advice / virtual-sequence specifics are exercised + validated by the M8 e2e gate against
/// a real trace.
///
/// `pcs[t]` is the expanded (bytecode) PC for cycle `t` (precomputed by the caller via
/// `BytecodePreprocessing::get_cycle_pc`); the expanded PC feeds the bytecode instance, not the R1CS
/// satisfaction, so it is threaded in rather than coupling this to the bytecode.
pub fn cycle_to_z<C: CycleRow, F: Field>(trace: &[C], t: usize, pcs: &[u64]) -> Vec<F> {
    let (v, n) = layout();
    let mut w = vec![F::from_u64(0); n];
    w[v.const_one] = F::from_u64(1);
    let cycle = &trace[t];
    let next = trace.get(t + 1);
    let next_pc = pcs.get(t + 1).copied();

    if cycle.is_noop() {
        w[v.f_do_not_update_pc] = F::from_u64(1);
        w[v.next_is_noop] = F::from_u64(u64::from(next.is_none_or(|c| c.is_noop())));
        fill_next(&mut w, &v, next, next_pc);
        fill_product(&mut w, &v, 0, 0, false);
        return w;
    }

    let cflags = cycle.circuit_flags();
    let iflags = cycle.instruction_flags();

    let left_input = if iflags[InstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if iflags[InstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0, |(_, val)| val)
    } else {
        0
    };
    let right_input: i128 = if iflags[InstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if iflags[InstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0, |(_, val)| i128::from(val))
    } else {
        0
    };
    let right_mag = right_input.unsigned_abs();
    debug_assert!(
        right_mag <= u128::from(u64::MAX),
        "RightInstructionInput magnitude > 2^64"
    );
    let right_sign = right_input < 0;

    // Always-active product witness (Left × |Right|).
    fill_product(&mut w, &v, left_input, right_mag as u64, right_sign);

    let lookup_output = cycle.lookup_output();
    set_u64_limbs(&mut w, v.lookup_output, lookup_output);

    // RightLookupOperand (rlo) + LeftLookupOperand, per the constraint cases (mirrors
    // jolt-trace `lookup_operands`).
    if cflags[CircuitFlags::AddOperands] {
        fill_add_rlo(&mut w, &v, left_input, right_input as u64);
    } else if cflags[CircuitFlags::SubtractOperands] {
        fill_sub_rlo(&mut w, &v, left_input, right_input as u64);
    } else if cflags[CircuitFlags::MultiplyOperands] {
        for i in 0..4 {
            w[v.rlo[i]] = w[v.product[i]];
        }
    } else if cflags[CircuitFlags::Advice] {
        w[v.rlo[0]] = F::from_u64(lookup_output & LIMB_MASK);
        w[v.rlo[1]] = F::from_u64(lookup_output >> 32);
    } else {
        // Default: LeftLookupOperand = left_input, RightLookupOperand = right_input.
        set_u64_limbs(&mut w, v.left_lookup, left_input);
        let r = right_input;
        w[v.rlo[0]] = F::from_u64((r.rem_euclid(1i128 << 32)) as u64);
        w[v.rlo[1]] = F::from_u64(((r >> 32).rem_euclid(1i128 << 32)) as u64);
        w[v.rlo[2]] = F::from_u64(((r >> 64).rem_euclid(1i128 << 32)) as u64);
        w[v.rlo[3]] = F::from_u64((r >> 96) as u64);
    }

    set_u64_limbs(&mut w, v.rs1, cycle.rs1_read().map_or(0, |(_, val)| val));
    set_u64_limbs(&mut w, v.rs2, cycle.rs2_read().map_or(0, |(_, val)| val));
    set_u64_limbs(
        &mut w,
        v.rd_write,
        cycle.rd_write().map_or(0, |(_, _, post)| post),
    );
    set_u64_limbs(&mut w, v.ram_read, cycle.ram_read_value().unwrap_or(0));
    set_u64_limbs(&mut w, v.ram_write, cycle.ram_write_value().unwrap_or(0));

    // RAM address = Rs1 + Imm (load/store); limb-wise with the low-limb carry.
    if cflags[CircuitFlags::Load] || cflags[CircuitFlags::Store] {
        let rs1 = cycle.rs1_read().map_or(0, |(_, val)| val);
        let imm = cycle.imm();
        let addr = (i128::from(rs1) + imm) as u128;
        set_u64_limbs(&mut w, v.ram_address, (addr & u128::from(u64::MAX)) as u64);
        let imm_lo = imm.rem_euclid(1i128 << 32) as u64;
        let c0 = (u128::from(rs1 & LIMB_MASK) + u128::from(imm_lo)) >> 32;
        w[v.ram_addr_c0] = F::from_u128(c0);
    } else {
        set_u64_limbs(
            &mut w,
            v.ram_address,
            cycle.ram_access_address().unwrap_or(0),
        );
    }

    w[v.pc] = F::from_u64(pcs.get(t).copied().unwrap_or(0));
    w[v.unexpanded_pc] = F::from_u64(cycle.unexpanded_pc());
    set_signed_limbs(&mut w, v.imm, cycle.imm());

    let flag = |f: CircuitFlags| F::from_u64(u64::from(cflags[f]));
    w[v.f_add] = flag(CircuitFlags::AddOperands);
    w[v.f_sub] = flag(CircuitFlags::SubtractOperands);
    w[v.f_mul] = flag(CircuitFlags::MultiplyOperands);
    w[v.f_load] = flag(CircuitFlags::Load);
    w[v.f_store] = flag(CircuitFlags::Store);
    w[v.f_jump] = flag(CircuitFlags::Jump);
    w[v.f_write_lookup_to_rd] = flag(CircuitFlags::WriteLookupOutputToRD);
    w[v.f_virtual] = flag(CircuitFlags::VirtualInstruction);
    w[v.f_assert] = flag(CircuitFlags::Assert);
    w[v.f_do_not_update_pc] = flag(CircuitFlags::DoNotUpdateUnexpandedPC);
    w[v.f_advice] = flag(CircuitFlags::Advice);
    w[v.f_is_compressed] = flag(CircuitFlags::IsCompressed);
    w[v.f_is_first_in_sequence] = flag(CircuitFlags::IsFirstInSequence);
    w[v.f_is_last_in_sequence] = flag(CircuitFlags::IsLastInSequence);

    w[v.branch] = F::from_u64(u64::from(iflags[InstructionFlags::Branch]));
    // ShouldBranch = recompose(LookupOutput)·Branch.
    w[v.should_branch] = F::from_u64(lookup_output) * w[v.branch];

    fill_next(&mut w, &v, next, next_pc);
    let next_is_noop = next.is_none_or(|c| c.is_noop());
    w[v.next_is_noop] = F::from_u64(u64::from(next_is_noop));
    // ShouldJump = Jump·(1 − NextIsNoop).
    w[v.should_jump] = w[v.f_jump] * (F::from_u64(1) - w[v.next_is_noop]);

    w
}

fn fill_next<C: CycleRow, F: Field>(w: &mut [F], v: &Vars, next: Option<&C>, next_pc: Option<u64>) {
    if let Some(nc) = next {
        w[v.next_pc] = F::from_u64(next_pc.unwrap_or(0));
        w[v.next_unexpanded_pc] = F::from_u64(nc.unexpanded_pc());
        let nf = nc.circuit_flags();
        w[v.next_is_virtual] = F::from_u64(u64::from(nf[CircuitFlags::VirtualInstruction]));
        w[v.next_is_first_in_sequence] =
            F::from_u64(u64::from(nf[CircuitFlags::IsFirstInSequence]));
    }
}

/// Build the per-cycle limbed `z`-assignments for the whole trace (feeds [`R1csWitness::materialize`]).
/// `pcs[t]` is the expanded PC for cycle `t` (from `BytecodePreprocessing::get_cycle_pc`).
pub fn build_limbed_z<C: CycleRow, F: Field>(trace: &[C], pcs: &[u64]) -> Vec<Vec<F>> {
    (0..trace.len())
        .map(|t| cycle_to_z::<C, F>(trace, t, pcs))
        .collect()
}

/// The materialized R1CS witness for the binary Spartan stages.
#[derive(Clone, Debug)]
pub struct R1csWitness<F: Field> {
    /// `log2` of the padded cycle count (the cycle half of the row/col variables).
    pub log_num_cycles: usize,
    /// Padded per-cycle variable count (`70 → 128`).
    pub num_vars_padded: usize,
    /// Padded per-cycle constraint count (`53 → 64`).
    pub num_cons_padded: usize,
    /// Witness, cycle-major: `z[cycle·num_vars_padded + var]`, length `2^log_num_cycles·num_vars_padded`.
    pub z: Vec<F>,
    /// `Az[cycle·num_cons_padded + constraint]`, length `2^log_num_cycles·num_cons_padded`.
    pub az: Vec<F>,
    pub bz: Vec<F>,
    pub cz: Vec<F>,
}

impl<F: Field> R1csWitness<F> {
    /// Materialize from per-cycle limbed `z`-assignments (each length `num_vars`, `z[0] = const_one`).
    /// Applies the limbed RV64 constraint matrices row-wise per cycle to get `Az/Bz/Cz`.
    pub fn materialize(per_cycle_z: &[Vec<F>]) -> Self {
        let m = rv64_limbed_constraints::<F>();
        let num_vars = m.num_vars;
        let num_cons = m.num_constraints;
        let v_pad = num_vars.next_power_of_two();
        let k_pad = num_cons.next_power_of_two();
        let cycles_pad = per_cycle_z.len().max(1).next_power_of_two();
        let log_num_cycles = cycles_pad.trailing_zeros() as usize;

        let mut z = vec![F::zero(); cycles_pad * v_pad];
        let mut az = vec![F::zero(); cycles_pad * k_pad];
        let mut bz = vec![F::zero(); cycles_pad * k_pad];
        let mut cz = vec![F::zero(); cycles_pad * k_pad];

        let dot = |row: &[(usize, F)], zc: &[F]| {
            row.iter()
                .fold(F::zero(), |acc, &(idx, coeff)| acc + coeff * zc[idx])
        };

        for (cycle, zc) in per_cycle_z.iter().enumerate() {
            debug_assert_eq!(zc.len(), num_vars, "per-cycle z must have num_vars entries");
            let zoff = cycle * v_pad;
            z[zoff..zoff + num_vars].copy_from_slice(zc);
            let coff = cycle * k_pad;
            for con in 0..num_cons {
                az[coff + con] = dot(&m.a[con], zc);
                bz[coff + con] = dot(&m.b[con], zc);
                cz[coff + con] = dot(&m.c[con], zc);
            }
        }

        Self {
            log_num_cycles,
            num_vars_padded: v_pad,
            num_cons_padded: k_pad,
            z,
            az,
            bz,
            cz,
        }
    }

    /// `Cz == Az ∘ Bz` on the full `(cycle, constraint)` hypercube — R1CS satisfaction (honest witness).
    pub fn is_satisfied(&self) -> bool {
        self.az
            .iter()
            .zip(self.bz.iter())
            .zip(self.cz.iter())
            .all(|((a, b), c)| *a * *b == *c)
    }

    /// Number of row variables of the outer sumcheck `(cycle ‖ constraint)`.
    #[inline]
    pub fn num_row_vars(&self) -> usize {
        self.log_num_cycles + self.num_cons_padded.trailing_zeros() as usize
    }

    /// The boolean carry/sign columns (the M6 booleanity residual — the only booleanity surviving
    /// the LogUp\*-GKR design), one length-`2^log_num_cycles` column per var, extracted from the
    /// cycle-major `z`. These feed the booleanity (`x²−x = 0`) zero-check under `R1csAux`. The
    /// signs (`left/right/product/sign_prod`) and the `{0,1}` ADD/SUB/RAM-address carries are
    /// boolean; the wide MUL carries / limbs are range-checked separately (deferred).
    pub fn boolean_aux_columns(&self) -> Vec<Vec<F>> {
        let (v, _) = layout();
        let vars = [
            v.left_sign,
            v.right_sign,
            v.product_sign,
            v.sign_prod,
            v.add_c0,
            v.sub_c0,
            v.sub_c1,
            v.ram_addr_c0,
        ];
        let cycles_pad = 1usize << self.log_num_cycles;
        let vp = self.num_vars_padded;
        vars.iter()
            .map(|&var| (0..cycles_pad).map(|c| self.z[c * vp + var]).collect())
            .collect()
    }
}

/// Minimal `CycleRow` mock shared by the witness-gen + Spartan-stage tests.
#[cfg(test)]
pub(crate) mod tests_support {
    use jolt_riscv::{CircuitFlagSet, CircuitFlags, InstructionFlagSet, InstructionFlags};
    use jolt_trace::CycleRow;

    /// A minimal `CycleRow` mock for the trace→`z` tests (no-op / ADD).
    #[derive(Clone, Copy)]
    pub(crate) struct MockCycle {
        is_noop: bool,
        unexpanded_pc: u64,
        rs1: Option<(u8, u64)>,
        rs2: Option<(u8, u64)>,
        rd: Option<(u8, u64, u64)>,
        ram: Option<(u64, u64, u64)>,
        imm: i128,
        lookup_output: u64,
        cflags: CircuitFlagSet,
        iflags: InstructionFlagSet,
    }

    impl MockCycle {
        pub(crate) fn noop_at(unexpanded_pc: u64) -> Self {
            Self {
                is_noop: true,
                unexpanded_pc,
                rs1: None,
                rs2: None,
                rd: None,
                ram: None,
                imm: 0,
                lookup_output: 0,
                cflags: CircuitFlagSet::default(),
                iflags: InstructionFlagSet::default(),
            }
        }
        pub(crate) fn add(unexpanded_pc: u64, left: u64, right: u64) -> Self {
            Self {
                is_noop: false,
                unexpanded_pc,
                rs1: Some((1, left)),
                rs2: Some((2, right)),
                rd: None,
                ram: None,
                imm: 0,
                lookup_output: left.wrapping_add(right),
                cflags: CircuitFlagSet::default().set(CircuitFlags::AddOperands),
                iflags: InstructionFlagSet::default()
                    .set(InstructionFlags::LeftOperandIsRs1Value)
                    .set(InstructionFlags::RightOperandIsRs2Value),
            }
        }
        /// Set the register-write operand `(rd, pre_value, post_value)`. Used by the register
        /// read-write witness-gen tests.
        pub(crate) fn with_rd(mut self, rd: u8, pre: u64, post: u64) -> Self {
            self.rd = Some((rd, pre, post));
            self
        }
        /// Set a RAM access `(dense_address, read_value, write_value)`. The RAM witness-gen derives
        /// the pre-value from tracked state and uses only the `write − read` delta as the increment.
        pub(crate) fn with_ram(mut self, address: u64, read: u64, write: u64) -> Self {
            self.ram = Some((address, read, write));
            self
        }
        /// Set the read operands `(rs1_reg, rs2_reg)` (values are derived from register state by the
        /// register witness-gen, so the reported read values here are placeholders).
        pub(crate) fn with_reads(mut self, rs1: Option<u8>, rs2: Option<u8>) -> Self {
            self.rs1 = rs1.map(|r| (r, 0));
            self.rs2 = rs2.map(|r| (r, 0));
            self
        }
    }

    impl CycleRow for MockCycle {
        fn noop() -> Self {
            Self::noop_at(0)
        }
        fn is_noop(&self) -> bool {
            self.is_noop
        }
        fn unexpanded_pc(&self) -> u64 {
            self.unexpanded_pc
        }
        fn virtual_sequence_remaining(&self) -> Option<u16> {
            None
        }
        fn is_first_in_sequence(&self) -> bool {
            false
        }
        fn is_virtual(&self) -> bool {
            false
        }
        fn rs1_read(&self) -> Option<(u8, u64)> {
            self.rs1
        }
        fn rs2_read(&self) -> Option<(u8, u64)> {
            self.rs2
        }
        fn rd_write(&self) -> Option<(u8, u64, u64)> {
            self.rd
        }
        fn rd_operand(&self) -> Option<u8> {
            self.rd.map(|(reg, _, _)| reg)
        }
        fn ram_access_address(&self) -> Option<u64> {
            self.ram.map(|(addr, _, _)| addr)
        }
        fn ram_read_value(&self) -> Option<u64> {
            self.ram.map(|(_, read, _)| read)
        }
        fn ram_write_value(&self) -> Option<u64> {
            self.ram.map(|(_, _, write)| write)
        }
        fn imm(&self) -> i128 {
            self.imm
        }
        fn circuit_flags(&self) -> CircuitFlagSet {
            self.cflags
        }
        fn instruction_flags(&self) -> InstructionFlagSet {
            self.iflags
        }
        fn lookup_index(&self) -> u128 {
            0
        }
        fn lookup_output(&self) -> u64 {
            self.lookup_output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::MockCycle;
    use super::*;
    use crate::r1cs::layout;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_poly::EqPolynomial;
    use jolt_r1cs::R1csKey;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Honest no-op cycle: const=1, NextUnexpPC = UnexpPC + 4 (uses the module-level `fill_product`).
    fn noop_cycle() -> Vec<F> {
        let (v, n) = layout();
        let mut w = vec![F::from_u64(0); n];
        w[v.const_one] = F::from_u64(1);
        w[v.next_unexpanded_pc] = F::from_u64(4);
        fill_product(&mut w, &v, 0, 0, false);
        w
    }

    /// Honest ADD cycle: f_add=1, product witness + RightLookupOperand add-limbs.
    fn add_cycle(left: u64, right: u64) -> Vec<F> {
        let (v, _) = layout();
        let mut w = noop_cycle();
        w[v.f_add] = F::from_u64(1);
        fill_product(&mut w, &v, left, right, false);
        fill_add_rlo(&mut w, &v, left, right);
        w
    }

    /// A multi-cycle honest witness satisfies the limbed R1CS (`Cz = Az ∘ Bz` everywhere), and the
    /// materialized `Az/Bz/Cz` cycle-major layout matches `R1csKey`'s uniform factorization (so the
    /// outer's `Az(r_x)` lines up with the inner reduction's `evaluate_sparse_matvec`).
    #[test]
    fn honest_multicycle_witness_satisfies_and_matches_r1cskey() {
        let per_cycle = vec![
            noop_cycle(),
            add_cycle(7, 11),
            add_cycle(0xFFFF_FFFF, 1), // exercises the low-limb carry
            noop_cycle(),
        ];
        let w = R1csWitness::<F>::materialize(&per_cycle);

        assert!(w.is_satisfied(), "honest witness must satisfy Cz = Az ∘ Bz");
        let cycles_pad = 1usize << w.log_num_cycles;
        assert_eq!(w.z.len(), cycles_pad * w.num_vars_padded);
        assert_eq!(w.az.len(), cycles_pad * w.num_cons_padded);

        // Cross-check the cycle-major layout against R1csKey's uniform factorization: the Az/Bz/Cz
        // column MLEs at a random row point r_x = (r_cycle ‖ r_constraint) equal
        // evaluate_sparse_matvec(r_constraint, z_at_r_cycle).
        let key = R1csKey::new(rv64_limbed_constraints::<F>(), cycles_pad);
        let mut rng = Rng(0x0005_1A7E);
        let r_x: Vec<F> = (0..key.num_row_vars())
            .map(|_| F::from_u64(rng.next()))
            .collect();
        let cv = key.num_cycle_vars();
        let (rx_cycle, rx_con) = r_x.split_at(cv);

        let eq_x = EqPolynomial::new(r_x.clone()).evaluations();
        let az_mle: F = w.az.iter().zip(eq_x.iter()).map(|(&a, &e)| a * e).sum();

        // z_at_r_cycle[var] = Σ_cycle eq(rx_cycle)[cycle]·z[cycle·v_pad + var].
        let eq_cycle = EqPolynomial::new(rx_cycle.to_vec()).evaluations();
        let v_pad = w.num_vars_padded;
        let z_at_rx_cycle: Vec<F> = (0..v_pad)
            .map(|var| {
                (0..cycles_pad).fold(F::from_u64(0), |acc, cycle| {
                    acc + eq_cycle[cycle] * w.z[cycle * v_pad + var]
                })
            })
            .collect();
        let (a_eval, _, _) = key.evaluate_sparse_matvec(rx_con, &z_at_rx_cycle);
        assert_eq!(
            az_mle, a_eval,
            "Az column MLE matches R1csKey uniform factorization"
        );
    }

    /// `cycle_to_z` produces a satisfying limbed witness from a real `CycleRow` trace (ADD then a
    /// no-op; the ADD's `NextUnexpPC = UnexpPC + 4` is met by the no-op at PC 4).
    #[test]
    fn cycle_to_z_add_and_noop_satisfy() {
        let trace = [MockCycle::add(0, 7, 11), MockCycle::noop_at(4)];
        let pcs = [0u64, 0u64];
        let per_cycle = build_limbed_z::<MockCycle, F>(&trace, &pcs);
        let w = R1csWitness::<F>::materialize(&per_cycle);
        assert!(
            w.is_satisfied(),
            "trace→z (ADD + no-op) must satisfy the limbed R1CS"
        );
    }

    /// The extracted carry/sign columns are boolean for an honest witness (the ADD with a
    /// `0xFFFF_FFFF + 1` low-limb overflow exercises a `1` carry).
    #[test]
    fn boolean_aux_columns_are_boolean() {
        let trace = [
            MockCycle::add(0, 0xFFFF_FFFF, 1),
            MockCycle::add(4, 7, 11),
            MockCycle::noop_at(8),
        ];
        let pcs = vec![0u64; trace.len()];
        let per_cycle = build_limbed_z::<MockCycle, F>(&trace, &pcs);
        let w = R1csWitness::<F>::materialize(&per_cycle);
        let (zero, one) = (F::from_u64(0), F::from_u64(1));
        for (i, col) in w.boolean_aux_columns().iter().enumerate() {
            for (c, &val) in col.iter().enumerate() {
                assert!(
                    val == zero || val == one,
                    "aux col {i} cycle {c} not boolean"
                );
            }
        }
    }
}
