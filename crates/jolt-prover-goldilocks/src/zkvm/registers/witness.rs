//! Register-file witness materialization: a `CycleRow` trace → the dense `K·T` matrices
//! (`ra1`/`ra2`/`wa`/`val`) + cycle columns (`inc`, `rd_write_value`/`rs1_value`/`rs2_value`) that
//! the register [`read_write_checking`](super::read_write_checking) +
//! [`val_evaluation`](super::val_evaluation) stages consume.
//!
//! This is a faithful **simulation** of the register file: the value columns and increment are
//! derived from the tracked register state, while the trace supplies only the read/write *addresses*
//! and the `rd` post-value. So the materialized witness satisfies the read-write-checking relation
//! ```text
//! Σ_{k,j} eq(r_cycle,j)·[ (γ·ra1 + γ²·ra2)·Val + wa·(Val + inc) ] = rd_wv + γ·rs1 + γ²·rs2
//! ```
//! by construction (`rs_value(j) = Val(rs, j)`, `rd_write_value(j) = Val(rd,j) + inc(j)`), which is
//! the soundness link between the committed register columns and the value claims.
//!
//! Layout matches the stages: `ra*/wa/val` are address-major (`k·T + j`), `inc` and the value
//! columns are cycle-only (length `T`). `T` is padded to a power of two; padding cycles carry the
//! final register state in `val` and zero everywhere else (so they contribute nothing to the
//! eq-weighted claims). `register_count` is the (power-of-two) register address-space size.

use jolt_field::Field;
use jolt_trace::CycleRow;

/// Materialized register-file witness columns for the read-write-checking + val-evaluation stages.
#[derive(Clone, Debug)]
pub struct RegisterWitness<F: Field> {
    /// `log2(register_count)` — the register-address bit width.
    pub log_k: usize,
    /// `log2` of the padded cycle count.
    pub log_t: usize,
    /// Read-address-1 one-hot, address-major (`k·T + j`).
    pub ra1: Vec<F>,
    /// Read-address-2 one-hot, address-major.
    pub ra2: Vec<F>,
    /// Write-address one-hot, address-major.
    pub wa: Vec<F>,
    /// Register value just before each cycle, address-major (`Val(k,j)`).
    pub val: Vec<F>,
    /// Write increment per cycle (`post − Val(rd,j)`), length `T`.
    pub inc: Vec<F>,
    /// `rd` post-value per cycle (`0` if no write), length `T`.
    pub rd_write_value: Vec<F>,
    /// `rs1` read value per cycle (`Val(rs1,j)`, `0` if no read), length `T`.
    pub rs1_value: Vec<F>,
    /// `rs2` read value per cycle, length `T`.
    pub rs2_value: Vec<F>,
}

/// Simulate the register file over `trace`, materializing the dense witness columns.
///
/// `register_count` must be a power of two and exceed every register index touched by the trace.
pub fn register_witness<C: CycleRow, F: Field>(
    trace: &[C],
    register_count: usize,
) -> RegisterWitness<F> {
    assert!(
        register_count.is_power_of_two(),
        "register_count must be a power of two"
    );
    let k = register_count;
    let log_k = k.trailing_zeros() as usize;
    let t = trace.len().max(1).next_power_of_two();
    let log_t = t.trailing_zeros() as usize;

    let mut ra1 = vec![F::zero(); k * t];
    let mut ra2 = vec![F::zero(); k * t];
    let mut wa = vec![F::zero(); k * t];
    let mut val = vec![F::zero(); k * t];
    let mut inc = vec![F::zero(); t];
    let mut rd_write_value = vec![F::zero(); t];
    let mut rs1_value = vec![F::zero(); t];
    let mut rs2_value = vec![F::zero(); t];

    // Tracked register state (value just before the current cycle).
    let mut state = vec![0u64; k];

    for (j, cycle) in trace.iter().enumerate() {
        // Snapshot Val(·, j) before applying this cycle's write.
        for (kk, &s) in state.iter().enumerate() {
            val[kk * t + j] = F::from_u64(s);
        }

        if let Some((rs1, _)) = cycle.rs1_read() {
            let rs1 = rs1 as usize;
            ra1[rs1 * t + j] = F::from_u64(1);
            rs1_value[j] = F::from_u64(state[rs1]);
        }
        if let Some((rs2, _)) = cycle.rs2_read() {
            let rs2 = rs2 as usize;
            ra2[rs2 * t + j] = F::from_u64(1);
            rs2_value[j] = F::from_u64(state[rs2]);
        }
        // Read-before-write: the increment uses the pre-write Val(rd, j).
        if let Some((rd, _pre, post)) = cycle.rd_write() {
            let rd = rd as usize;
            wa[rd * t + j] = F::from_u64(1);
            rd_write_value[j] = F::from_u64(post);
            inc[j] = F::from_i128(i128::from(post) - i128::from(state[rd]));
            state[rd] = post;
        }
    }

    // Padding cycles carry the final state in `val`; all indicators/values stay zero.
    for j in trace.len()..t {
        for (kk, &s) in state.iter().enumerate() {
            val[kk * t + j] = F::from_u64(s);
        }
    }

    RegisterWitness {
        log_k,
        log_t,
        ra1,
        ra2,
        wa,
        val,
        inc,
        rd_write_value,
        rs1_value,
        rs2_value,
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::accumulator::{
        OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    };
    use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
    use crate::zkvm::r1cs_witness::tests_support::MockCycle;
    use crate::zkvm::registers::read_write_checking::{
        RegistersReadWriteChecking, RegistersReadWriteCheckingParams,
    };
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

    const DEGREE: usize = 3;

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

    fn dot(col: &[F], eq: &[F]) -> F {
        col.iter()
            .zip(eq.iter())
            .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
    }

    /// A small consistent trace: r3 := 5, then r4 := r3+r1 with reads, etc. Built so reads observe
    /// prior writes (validating the read-before-write `Val` snapshot).
    fn sample_trace() -> Vec<MockCycle> {
        vec![
            // write r3 = 5 (no reads)
            MockCycle::add(0, 0, 0).with_rd(3, 0, 5),
            // read r3 (=5) and r0 (=0), write r4 = 9
            MockCycle::add(4, 0, 0)
                .with_reads(Some(3), Some(0))
                .with_rd(4, 0, 9),
            // read r3 (=5) and r4 (=9), write r3 = 14 (overwrites; read-before-write)
            MockCycle::add(8, 0, 0)
                .with_reads(Some(3), Some(4))
                .with_rd(3, 5, 14),
            MockCycle::noop_at(12),
        ]
    }

    /// The materialized matrices satisfy the read-write-checking relation: feeding them into
    /// `RegistersReadWriteChecking` with component claims computed from the columns round-trips.
    #[test]
    fn register_witness_satisfies_read_write_checking() {
        let trace = sample_trace();
        let w = register_witness::<MockCycle, F>(&trace, 8);
        assert_eq!(w.log_k, 3);
        assert_eq!(w.log_t, 2); // 4 cycles → 2^2

        let mut rng = Rng(0xA17E);
        let r_cycle: Vec<F> = (0..w.log_t).map(|_| F::from_u64(rng.next())).collect();
        let eq = EqPolynomial::<F>::evals(&r_cycle, None);

        // Component claims from the materialized cycle columns (what RegistersClaimReduction emits).
        let rd_wv = dot(&w.rd_write_value, &eq);
        let rs1 = dot(&w.rs1_value, &eq);
        let rs2 = dot(&w.rs2_value, &eq);

        let seed = |acc: &mut Openings<F>| {
            let pt = OpeningPoint::new(r_cycle.clone());
            acc.append_virtual(
                VirtualPolynomial::RdWriteValue,
                SumcheckId::RegistersClaimReduction,
                pt.clone(),
                rd_wv,
            );
            acc.append_virtual(
                VirtualPolynomial::Rs1Value,
                SumcheckId::RegistersClaimReduction,
                pt.clone(),
                rs1,
            );
            acc.append_virtual(
                VirtualPolynomial::Rs2Value,
                SumcheckId::RegistersClaimReduction,
                pt,
                rs2,
            );
        };

        let mut prover_acc = Openings::<F>::new(w.log_t);
        seed(&mut prover_acc);
        let mut prover_t = ProverTranscript::new("reg-witness-rw");
        let params = RegistersReadWriteCheckingParams::new(&prover_acc, w.log_k, &mut prover_t);
        // input_claim = rd_wv + γ·rs1 + γ²·rs2 (the private params method is module-local).
        let input_claim = rd_wv + params.gamma * (rs1 + params.gamma * rs2);
        let mut prover = RegistersReadWriteChecking::new_prover(
            params,
            w.ra1.clone(),
            w.ra2.clone(),
            w.wa.clone(),
            w.val.clone(),
            w.inc.clone(),
        );
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(w.log_t);
        seed(&mut verifier_acc);
        let mut verifier_t = VerifierTranscript::new("reg-witness-rw", &narg);
        let vparams =
            RegistersReadWriteCheckingParams::new(&verifier_acc, w.log_k, &mut verifier_t);
        let verifier = RegistersReadWriteChecking::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: w.log_k + w.log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } = verify(&claim, &mut verifier_t)
            .expect("materialized register witness must satisfy read-write-checking");
        assert_eq!(point, challenges);

        // Discharge the reduced openings to confirm the output claim closes.
        for poly in [
            VirtualPolynomial::RegistersVal,
            VirtualPolynomial::Rs1Ra,
            VirtualPolynomial::Rs2Ra,
            VirtualPolynomial::RdWa,
        ] {
            let (pt, c) = prover_acc
                .get_virtual_polynomial_opening(poly, SumcheckId::RegistersReadWriteChecking);
            verifier_acc.append_virtual(poly, SumcheckId::RegistersReadWriteChecking, pt, c);
        }
        let (inc_pt, inc_c) = prover_acc.get_committed_polynomial_opening(
            crate::framework::accumulator::CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        verifier_acc.append_dense(
            crate::framework::accumulator::CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            inc_pt,
            inc_c,
        );
        assert_eq!(
            value,
            verifier.expected_output_claim(&verifier_acc, &challenges),
            "reduced claim must close"
        );
    }

    /// Read-before-write: cycle 2 reads r3 = 5 (its pre-write value), and the cycle's `Val(r3,·)`
    /// snapshot precedes the same cycle's r3 := 14 write.
    #[test]
    fn read_before_write_snapshot() {
        let trace = sample_trace();
        let w = register_witness::<MockCycle, F>(&trace, 8);
        let t = 1usize << w.log_t;
        // cycle 2 reads r3: rs1_value[2] should be 5 (value before the r3:=14 write at cycle 2).
        assert_eq!(w.rs1_value[2], F::from_u64(5), "rs1 reads pre-write r3 = 5");
        // Val(r3, cycle 2) snapshot = 5 (before the write); Val(r3, cycle 3) = 14 (after).
        assert_eq!(w.val[3 * t + 2], F::from_u64(5), "Val(r3, 2) = 5 pre-write");
        assert_eq!(
            w.val[3 * t + 3],
            F::from_u64(14),
            "Val(r3, 3) = 14 post-write"
        );
        // inc at cycle 2 = 14 - 5 = 9.
        assert_eq!(w.inc[2], F::from_u64(9), "inc(2) = post - pre = 9");
    }
}
