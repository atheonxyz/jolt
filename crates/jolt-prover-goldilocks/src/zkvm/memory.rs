//! The combined **memory stage** — composes the RAM stage + registers stage + the two cross-cutting
//! claim-reductions (`RamRaClaimReduction`, `IncClaimReduction`) onto one shared transcript +
//! opening accumulator. This completes the M8 memory-checking portion (P3 + P4).
//!
//! Ordering (all on one accumulator/transcript):
//! 1. [`prove_ram`](crate::zkvm::ram::stage::prove_ram) — batched `RamReadWriteChecking +
//!    RamRafEvaluation + RamOutputCheck` (aligned `r_address`) then `RamValCheck` → seeds
//!    `RamInc@{RW,ValCheck}`, the three `RamRa@{RW,RAF,ValCheck}` (sharing `r_address`), `RamVal`,
//!    `RamValFinal`.
//! 2. [`prove_registers`](crate::zkvm::registers::stage::prove_registers) → seeds
//!    `RdInc@{RegistersRW,RegistersValEvaluation}` (+ the register virtuals).
//! 3. [`RamRaClaimReduction`] — consolidates the three `RamRa` openings (their shared `r_address` is
//!    exactly the alignment the batched RAM stage produces) into one `RamRa(r_address ‖ ρ)`.
//! 4. [`IncClaimReduction`] — batches the four `Inc` openings (`RamInc@{RW,ValCheck}` +
//!    `RdInc@{RegistersRW,RegistersValEvaluation}`) into one `(RamInc, RdInc)(ρ)`.
//!
//! Steps 3–4 require steps 1–2 to have run on the **same** accumulator (so their input openings are
//! present) — which is the whole point of composing them here. The committed `RamInc`/`RdInc`
//! openings the reductions produce are the ones the stage-8 WHIR open (P9) discharges.
//!
//! Interim binary-Spartan seeding (fork 2) is inherited from the two sub-stages; see their docs and
//! `PHASE3_REVIEW_GUIDE.md` §7. The two sub-stages each draw their own `r_spartan` (independent
//! interim seed points); a single shared `r_spartan` is the uni-skip-Spartan concern (task #6).

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::SumcheckClaim;

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
};
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
use crate::framework::transcript::{ProverFs, VerifierFs};
use crate::zkvm::claim_reductions::{
    IncClaimReduction, IncClaimReductionParams, RamRaClaimReduction, RamRaReductionParams,
};
use crate::zkvm::ram::stage::{prove_ram, verify_ram, RamStageError, RamStageProof};
use crate::zkvm::ram::witness::RamWitness;
use crate::zkvm::registers::stage::{
    prove_registers, verify_registers, RegistersStageError, RegistersStageProof,
};
use crate::zkvm::registers::witness::RegisterWitness;

const RAM_RA_REDUCTION_DEGREE: usize = 2;
const INC_REDUCTION_DEGREE: usize = 2;

/// Memory-stage verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryStageError {
    Ram(RamStageError),
    Registers(RegistersStageError),
    Sumcheck,
    RamRaClaim,
    IncClaim,
}

/// The combined memory-stage proof.
#[derive(Clone, Debug)]
pub struct MemoryStageProof<F: Field> {
    pub ram: RamStageProof<F>,
    pub registers: RegistersStageProof<F>,
    /// `RamRa` @ RamRaClaimReduction (the consolidated RAM-access opening).
    pub ram_ra_opening: F,
    /// `(RamInc, RdInc)` @ IncClaimReduction (the consolidated increment openings).
    pub inc_openings: [F; 2],
}

/// `ra(r_address, j) = Σ_k eq(r_address, k)·ra[k·T + j]` — the RAM-access column at the aligned
/// address point, which the RAM-RA reduction sums over the cycle.
fn ra_at_address<F: Field>(ra: &[F], r_address: &[F], log_t: usize, log_k: usize) -> Vec<F> {
    let t = 1usize << log_t;
    let k = 1usize << log_k;
    let eq = EqPolynomial::<F>::evals(r_address, None);
    (0..t)
        .map(|j| (0..k).fold(F::zero(), |acc, kk| acc + eq[kk] * ra[kk * t + j]))
        .collect()
}

/// Prove the combined memory stage. `unmap`/`val_io`/`io_mask` are the RAM public columns.
pub fn prove_memory<F, T>(
    ram_w: &RamWitness<F>,
    reg_w: &RegisterWitness<F>,
    unmap: &[F],
    val_io: &[F],
    io_mask: &[F],
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> MemoryStageProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    assert_eq!(
        ram_w.log_t, reg_w.log_t,
        "RAM and register witnesses must share the cycle count"
    );
    let log_t = ram_w.log_t;
    let ram_log_k = ram_w.log_k;

    let ram = prove_ram(ram_w, unmap, val_io, io_mask, accumulator, transcript);
    let registers = prove_registers(reg_w, accumulator, transcript);

    // Consolidate the three RamRa openings (shared r_address by the RAM-stage alignment).
    let ramra_params = RamRaReductionParams::new(log_t, ram_log_k, accumulator, transcript);
    let ra_col = ra_at_address(&ram_w.ra, &ramra_params.r_address, log_t, ram_log_k);
    let mut ramra = RamRaClaimReduction::new_prover(ramra_params, ra_col);
    let _ = prove(&mut ramra, accumulator, transcript);
    let ram_ra_opening = accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRaClaimReduction)
        .1;

    // Batch the four Inc openings (RamInc@{RW,ValCheck} + RdInc@{RegistersRW,RegistersValEval}).
    let inc_params = IncClaimReductionParams::new(log_t, accumulator, transcript);
    let mut inc = IncClaimReduction::new_prover(inc_params, ram_w.inc.clone(), reg_w.inc.clone());
    let _ = prove(&mut inc, accumulator, transcript);
    let inc_openings = [
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncClaimReduction,
            )
            .1,
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::IncClaimReduction,
            )
            .1,
    ];

    MemoryStageProof {
        ram,
        registers,
        ram_ra_opening,
        inc_openings,
    }
}

/// Verify the combined memory stage (mirror of [`prove_memory`]). The public columns must match.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors prove_memory: proof + geometry (log_t/ram_log_k/reg_log_k) + RAM public columns + acc/transcript"
)]
pub fn verify_memory<F, T>(
    proof: &MemoryStageProof<F>,
    log_t: usize,
    ram_log_k: usize,
    reg_log_k: usize,
    unmap: &[F],
    val_io: &[F],
    io_mask: &[F],
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), MemoryStageError>
where
    F: Field,
    T: VerifierFs<F>,
{
    verify_ram(
        &proof.ram,
        log_t,
        ram_log_k,
        unmap,
        val_io,
        io_mask,
        accumulator,
        transcript,
    )
    .map_err(MemoryStageError::Ram)?;
    verify_registers(&proof.registers, log_t, reg_log_k, accumulator, transcript)
        .map_err(MemoryStageError::Registers)?;

    // RamRaClaimReduction.
    let ramra_params = RamRaReductionParams::new(log_t, ram_log_k, accumulator, transcript);
    let r_address = ramra_params.r_address.clone();
    let ramra = RamRaClaimReduction::new_verifier(ramra_params);
    let ramra_input = ramra.input_claim(accumulator);
    let ramra_eval = verify(
        &SumcheckClaim {
            num_vars: log_t,
            degree: RAM_RA_REDUCTION_DEGREE,
            claimed_sum: ramra_input,
        },
        transcript,
    )
    .map_err(|_| MemoryStageError::Sumcheck)?;
    let ramra_cycle: Vec<F> = ramra_eval.point.iter().rev().copied().collect();
    let ramra_point = OpeningPoint::new([r_address.as_slice(), ramra_cycle.as_slice()].concat());
    accumulator.append_virtual(
        VirtualPolynomial::RamRa,
        SumcheckId::RamRaClaimReduction,
        ramra_point,
        proof.ram_ra_opening,
    );
    if ramra_eval.value != ramra.expected_output_claim(accumulator, &ramra_eval.point) {
        return Err(MemoryStageError::RamRaClaim);
    }

    // IncClaimReduction.
    let inc_params = IncClaimReductionParams::new(log_t, accumulator, transcript);
    let inc = IncClaimReduction::new_verifier(inc_params);
    let inc_input = inc.input_claim(accumulator);
    let inc_eval = verify(
        &SumcheckClaim {
            num_vars: log_t,
            degree: INC_REDUCTION_DEGREE,
            claimed_sum: inc_input,
        },
        transcript,
    )
    .map_err(|_| MemoryStageError::Sumcheck)?;
    let inc_rho: Vec<F> = inc_eval.point.iter().rev().copied().collect();
    let inc_point = OpeningPoint::new(inc_rho);
    accumulator.append_dense(
        CommittedPolynomial::RamInc,
        SumcheckId::IncClaimReduction,
        inc_point.clone(),
        proof.inc_openings[0],
    );
    accumulator.append_dense(
        CommittedPolynomial::RdInc,
        SumcheckId::IncClaimReduction,
        inc_point,
        proof.inc_openings[1],
    );
    if inc_eval.value != inc.expected_output_claim(accumulator, &inc_eval.point) {
        return Err(MemoryStageError::IncClaim);
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::zkvm::r1cs_witness::tests_support::MockCycle;
    use crate::zkvm::ram::witness::ram_witness;
    use crate::zkvm::registers::witness::register_witness;
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    /// A trace exercising both register writes/reads AND RAM loads/stores in the same cycles.
    fn sample_trace() -> Vec<MockCycle> {
        vec![
            MockCycle::add(0, 0, 0).with_rd(3, 0, 5).with_ram(2, 0, 5),
            MockCycle::add(4, 0, 0)
                .with_reads(Some(3), Some(0))
                .with_rd(4, 0, 9)
                .with_ram(2, 5, 5),
            MockCycle::add(8, 0, 0)
                .with_reads(Some(3), Some(4))
                .with_rd(3, 5, 14)
                .with_ram(5, 0, 9),
            MockCycle::noop_at(12),
        ]
    }

    fn public_columns(log_k: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
        let k = 1usize << log_k;
        let unmap: Vec<F> = (0..k)
            .map(|i| F::from_u64(0x8000_0000 + i as u64))
            .collect();
        let zero = vec![F::from_u64(0); k];
        (unmap, zero.clone(), zero)
    }

    #[test]
    fn memory_stage_round_trip() {
        let trace = sample_trace();
        let ram_w = ram_witness::<MockCycle, F>(&trace, 8);
        let reg_w = register_witness::<MockCycle, F>(&trace, 8);
        let (unmap, val_io, io_mask) = public_columns(ram_w.log_k);

        let mut prover_acc = Openings::<F>::new(ram_w.log_t);
        let mut prover_t = ProverTranscript::new("memory-stage");
        let proof = prove_memory(
            &ram_w,
            &reg_w,
            &unmap,
            &val_io,
            &io_mask,
            &mut prover_acc,
            &mut prover_t,
        );
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(ram_w.log_t);
        let mut verifier_t = VerifierTranscript::new("memory-stage", &narg);
        verify_memory(
            &proof,
            ram_w.log_t,
            ram_w.log_k,
            reg_w.log_k,
            &unmap,
            &val_io,
            &io_mask,
            &mut verifier_acc,
            &mut verifier_t,
        )
        .expect("memory stage must verify");
    }

    #[test]
    fn tampered_inc_reduction_rejected() {
        let trace = sample_trace();
        let ram_w = ram_witness::<MockCycle, F>(&trace, 8);
        let reg_w = register_witness::<MockCycle, F>(&trace, 8);
        let (unmap, val_io, io_mask) = public_columns(ram_w.log_k);

        let mut prover_acc = Openings::<F>::new(ram_w.log_t);
        let mut prover_t = ProverTranscript::new("memory-stage");
        let mut proof = prove_memory(
            &ram_w,
            &reg_w,
            &unmap,
            &val_io,
            &io_mask,
            &mut prover_acc,
            &mut prover_t,
        );
        let narg = prover_t.into_proof();
        // Corrupt the consolidated RdInc(ρ) opening → IncClaimReduction output-claim check fails.
        proof.inc_openings[1] += F::from_u64(1);

        let mut verifier_acc = Openings::<F>::new(ram_w.log_t);
        let mut verifier_t = VerifierTranscript::new("memory-stage", &narg);
        assert!(
            verify_memory(
                &proof,
                ram_w.log_t,
                ram_w.log_k,
                reg_w.log_k,
                &unmap,
                &val_io,
                &io_mask,
                &mut verifier_acc,
                &mut verifier_t,
            )
            .is_err(),
            "tampered IncClaimReduction opening must be rejected"
        );
    }
}
