//! Real-trace witness assembly for the Goldilocks prover e2e.
//!
//! The in-crate witness builders ([`build_limbed_z`], [`ram_witness`],
//! [`register_witness`]) are generic over [`CycleRow`], so a real `tracer::Cycle`
//! drives them directly — with one exception. [`ram_witness`] indexes its
//! address-major one-hot by `ram_access_address() as usize`, but a real cycle
//! reports the *physical* byte address (≥ the RAM base), which would blow up the
//! address space. [`RemappedCycle`] wraps a real cycle and rewrites only
//! `ram_access_address` to the dense remapped index `(addr − ram_lowest) / 8`
//! (mirroring jolt-core's `remap_address`); every other method delegates, so the
//! R1CS witness — which wants the raw `rs1 + imm` address — is built from the
//! *unwrapped* trace and the memory-checking witness from the wrapped one.

use jolt_field::Field;
use jolt_r1cs::R1csKey;
use jolt_riscv::{CircuitFlagSet, InstructionFlagSet};
use jolt_trace::{BytecodePreprocessing, Cycle, CycleRow};

use crate::r1cs::rv64_limbed_constraints;
use crate::zkvm::driver::RamPublicColumns;
use crate::zkvm::r1cs_witness::{build_limbed_z, R1csWitness};
use crate::zkvm::ram::witness::{ram_witness, RamWitness};
use crate::zkvm::registers::witness::{register_witness, RegisterWitness};

/// Dense remap of a physical RAM byte address: `(addr − lowest) / 8`. Mirrors
/// jolt-core `remap_address` for the in-range case (real RAM accesses are always
/// `≥ lowest`; address `0` / non-access is filtered by the `Option` upstream).
#[inline]
pub(crate) fn remap_index(addr: u64, ram_lowest: u64) -> u64 {
    (addr - ram_lowest) / 8
}

/// A real `tracer::Cycle` with `ram_access_address` remapped to the dense memory
/// index space (see module docs). All other [`CycleRow`] methods delegate.
#[derive(Clone, Copy)]
pub struct RemappedCycle {
    inner: Cycle,
    ram_lowest: u64,
}

/// Wrap a real trace so RAM addresses are dense remapped indices, ready for
/// [`ram_witness`]. The raw trace is left untouched for the R1CS witness.
pub fn remap_trace(trace: &[Cycle], ram_lowest: u64) -> Vec<RemappedCycle> {
    trace
        .iter()
        .map(|&inner| RemappedCycle { inner, ram_lowest })
        .collect()
}

impl CycleRow for RemappedCycle {
    fn noop() -> Self {
        RemappedCycle {
            inner: <Cycle as CycleRow>::noop(),
            ram_lowest: 0,
        }
    }
    fn is_noop(&self) -> bool {
        self.inner.is_noop()
    }
    fn unexpanded_pc(&self) -> u64 {
        self.inner.unexpanded_pc()
    }
    fn virtual_sequence_remaining(&self) -> Option<u16> {
        self.inner.virtual_sequence_remaining()
    }
    fn is_first_in_sequence(&self) -> bool {
        self.inner.is_first_in_sequence()
    }
    fn is_virtual(&self) -> bool {
        self.inner.is_virtual()
    }
    fn rs1_read(&self) -> Option<(u8, u64)> {
        self.inner.rs1_read()
    }
    fn rs2_read(&self) -> Option<(u8, u64)> {
        self.inner.rs2_read()
    }
    fn rd_write(&self) -> Option<(u8, u64, u64)> {
        self.inner.rd_write()
    }
    fn rd_operand(&self) -> Option<u8> {
        self.inner.rd_operand()
    }
    fn ram_access_address(&self) -> Option<u64> {
        self.inner
            .ram_access_address()
            .map(|a| remap_index(a, self.ram_lowest))
    }
    fn ram_read_value(&self) -> Option<u64> {
        self.inner.ram_read_value()
    }
    fn ram_write_value(&self) -> Option<u64> {
        self.inner.ram_write_value()
    }
    fn imm(&self) -> i128 {
        self.inner.imm()
    }
    fn circuit_flags(&self) -> CircuitFlagSet {
        self.inner.circuit_flags()
    }
    fn instruction_flags(&self) -> InstructionFlagSet {
        self.inner.instruction_flags()
    }
    fn lookup_index(&self) -> u128 {
        self.inner.lookup_index()
    }
    fn lookup_output(&self) -> u64 {
        self.inner.lookup_output()
    }
}

/// All witnesses the binary driver consumes, assembled from one real trace.
pub struct RealWitness<F: Field> {
    pub r1cs: R1csWitness<F>,
    pub ram: RamWitness<F>,
    pub registers: RegisterWitness<F>,
    pub ram_public: RamPublicColumns<F>,
    pub key: R1csKey<F>,
}

/// Build the binary-driver witnesses from a real `tracer::Cycle` trace.
///
/// `ram_lowest` is `memory_layout.get_lowest_address()`; `ram_k` the (power-of-two)
/// remapped RAM address-space size; `register_count` the (power-of-two) register
/// file size (`REGISTER_COUNT = 128`: 32 real + 96 virtual). The `RamPublicColumns`
/// use the faithful affine inverse `unmap[k] = ram_lowest + 8·k` and an empty I/O
/// region (`val_io = io_mask = 0` → the output-check is an honest trivial zero-check;
/// real program-output binding is deferred with the uni-skip Spartan pass).
pub fn assemble_real_witness<F: Field>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    ram_lowest: u64,
    ram_k: usize,
    register_count: usize,
) -> RealWitness<F> {
    let pcs: Vec<u64> = trace
        .iter()
        .map(|c| bytecode.get_cycle_pc(c) as u64)
        .collect();
    let per_cycle = build_limbed_z::<Cycle, F>(trace, &pcs);
    let r1cs = R1csWitness::<F>::materialize(&per_cycle);
    let cycles_pad = 1usize << r1cs.log_num_cycles;
    let key = R1csKey::new(rv64_limbed_constraints::<F>(), cycles_pad);

    let remapped = remap_trace(trace, ram_lowest);
    let ram = ram_witness::<RemappedCycle, F>(&remapped, ram_k);
    let registers = register_witness::<Cycle, F>(trace, register_count);
    let ram_public = ram_public_columns::<F>(ram.log_k, ram_lowest);

    RealWitness {
        r1cs,
        ram,
        registers,
        ram_public,
        key,
    }
}

/// Public RAM columns: faithful affine unmap inverse + empty I/O region (see
/// [`assemble_real_witness`]).
fn ram_public_columns<F: Field>(log_k: usize, ram_lowest: u64) -> RamPublicColumns<F> {
    let k = 1usize << log_k;
    let unmap: Vec<F> = (0..k)
        .map(|i| F::from_u64(ram_lowest + 8 * i as u64))
        .collect();
    let zero = vec![F::from_u64(0); k];
    RamPublicColumns {
        unmap,
        val_io: zero.clone(),
        io_mask: zero,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_index_matches_remap_address_formula() {
        // jolt-core remap_address: (addr - lowest) / 8, 8-byte word stride.
        let lowest = 0x8000_0000u64;
        assert_eq!(remap_index(lowest, lowest), 0);
        assert_eq!(remap_index(lowest + 8, lowest), 1);
        assert_eq!(remap_index(lowest + 80, lowest), 10);
        // Inverse of the public unmap column: unmap[k] = lowest + 8·k.
        for k in 0u64..16 {
            assert_eq!(remap_index(lowest + 8 * k, lowest), k);
        }
    }
}
