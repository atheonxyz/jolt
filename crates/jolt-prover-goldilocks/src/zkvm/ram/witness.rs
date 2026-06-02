//! RAM witness materialization: a `CycleRow` trace → the dense `K·T` matrices (`ra`/`val`) + cycle
//! columns (`inc`, `ram_read_value`/`ram_write_value`) that the RAM
//! [`read_write_checking`](super::read_write_checking) + [`val_check`](super::val_check) +
//! [`raf_evaluation`](super::raf_evaluation) + [`output_check`](super::output_check) stages consume.
//!
//! Like [`register_witness`](crate::zkvm::registers::witness::register_witness) this is a faithful
//! **simulation** of RAM: the value matrix + increment are derived from tracked memory state, while
//! the trace supplies only the access *address* and the post-value. So the materialized witness
//! satisfies the read-write-checking relation
//! ```text
//! Σ_{k,j} eq(r_cycle,j)·ra(k,j)·(Val + γ·(inc + Val)) = rv + γ·wv
//! ```
//! by construction (`read(j) = Val(addr,j)`, `write(j) = Val(addr,j) + inc(j)`), which is the
//! soundness link between the committed RAM columns and the read/write value claims.
//!
//! Layout matches the stages: `ra/val` are address-major (`k·T + j`), `inc` and the value columns
//! are cycle-only (length `T`). `T` is padded to a power of two; padding/no-access cycles carry the
//! current memory state in `val` and zero `ra`/`inc` (so they contribute nothing to the eq-weighted
//! claims). `ram_k` is the (power-of-two) **remapped** RAM address-space size.
//!
//! **Address remap deferred:** the dense index is taken directly from `ram_access_address()` (the
//! caller/mock supplies dense indices). The real-trace remap `(addr − memory_layout.lowest)/8`
//! (jolt-core `remap_address`) is applied by the e2e driver before calling this — matching how
//! `extract_trace` already remaps for the committed RA columns.

use jolt_field::Field;
use jolt_trace::CycleRow;

/// Materialized RAM witness columns for the RAM read-write-checking + downstream stages.
#[derive(Clone, Debug)]
pub struct RamWitness<F: Field> {
    /// `log2(ram_k)` — the RAM-address bit width.
    pub log_k: usize,
    /// `log2` of the padded cycle count.
    pub log_t: usize,
    /// Access one-hot, address-major (`k·T + j`); at most one `k` set per accessing cycle.
    pub ra: Vec<F>,
    /// RAM value just before each cycle, address-major (`Val(k,j)`).
    pub val: Vec<F>,
    /// Write increment per cycle (`post − Val(addr,j)`, `0` for non-access/loads), length `T`.
    pub inc: Vec<F>,
    /// `inc` as a signed integer (the value `inc[j] = F::from_i128(inc_i128[j])`). The stage-8 Inc
    /// limb open decomposes THIS (the zero-init `RamInc` the memory stage claims), not the
    /// `extract_trace` real-init increments, so the committed limbs recompose to the stage claim.
    pub inc_i128: Vec<i128>,
    /// Read value per cycle (`Val(addr,j)`, `0` if no access), length `T`.
    pub ram_read_value: Vec<F>,
    /// Write (post) value per cycle (`0` if no access), length `T`.
    pub ram_write_value: Vec<F>,
    /// Final memory state after all cycles, address-major (length `K`). Feeds the output-check.
    pub val_final: Vec<F>,
}

/// Simulate RAM over `trace`, materializing the dense witness columns.
///
/// `ram_k` must be a power of two and exceed every (remapped) RAM index touched by the trace.
pub fn ram_witness<C: CycleRow, F: Field>(trace: &[C], ram_k: usize) -> RamWitness<F> {
    assert!(ram_k.is_power_of_two(), "ram_k must be a power of two");
    let k = ram_k;
    let log_k = k.trailing_zeros() as usize;
    let t = trace.len().max(1).next_power_of_two();
    let log_t = t.trailing_zeros() as usize;

    let mut ra = vec![F::zero(); k * t];
    let mut val = vec![F::zero(); k * t];
    let mut inc = vec![F::zero(); t];
    let mut inc_i128 = vec![0i128; t];
    let mut ram_read_value = vec![F::zero(); t];
    let mut ram_write_value = vec![F::zero(); t];

    // Tracked memory state (value just before the current cycle).
    let mut state = vec![0u64; k];

    for (j, cycle) in trace.iter().enumerate() {
        for (kk, &s) in state.iter().enumerate() {
            val[kk * t + j] = F::from_u64(s);
        }

        if let Some(addr) = cycle.ram_access_address() {
            let kk = addr as usize;
            ra[kk * t + j] = F::from_u64(1);
            // Derive the pre-value from state; the post-value is the trace's write.
            let pre = state[kk];
            let post = cycle.ram_write_value().unwrap_or(pre);
            ram_read_value[j] = F::from_u64(pre);
            ram_write_value[j] = F::from_u64(post);
            let delta = i128::from(post) - i128::from(pre);
            inc[j] = F::from_i128(delta);
            inc_i128[j] = delta;
            state[kk] = post;
        }
    }

    for j in trace.len()..t {
        for (kk, &s) in state.iter().enumerate() {
            val[kk * t + j] = F::from_u64(s);
        }
    }

    let val_final = state.iter().map(|&s| F::from_u64(s)).collect();

    RamWitness {
        log_k,
        log_t,
        ra,
        val,
        inc,
        inc_i128,
        ram_read_value,
        ram_write_value,
        val_final,
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::accumulator::{
        CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId,
        VirtualPolynomial,
    };
    use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
    use crate::zkvm::r1cs_witness::tests_support::MockCycle;
    use crate::zkvm::ram::read_write_checking::{RamReadWriteChecking, RamReadWriteCheckingParams};
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

    /// store r[2]=5, load r[2] (=5), store r[5]=9, store r[2]=14 (read-before-write), noop.
    fn sample_trace() -> Vec<MockCycle> {
        vec![
            MockCycle::add(0, 0, 0).with_ram(2, 0, 5), // write addr 2 = 5
            MockCycle::add(4, 0, 0).with_ram(2, 5, 5), // load addr 2 (delta 0)
            MockCycle::add(8, 0, 0).with_ram(5, 0, 9), // write addr 5 = 9
            MockCycle::add(12, 0, 0).with_ram(2, 5, 14), // write addr 2 = 14
            MockCycle::noop_at(16),
        ]
    }

    /// The materialized matrices satisfy the read-write-checking relation: feeding them into
    /// `RamReadWriteChecking` with component claims computed from the columns round-trips.
    #[test]
    fn ram_witness_satisfies_read_write_checking() {
        let trace = sample_trace();
        let w = ram_witness::<MockCycle, F>(&trace, 8);
        assert_eq!(w.log_k, 3);
        assert_eq!(w.log_t, 3); // 5 cycles → 2^3

        let mut rng = Rng(0x00C0_FFEE);
        let r_cycle: Vec<F> = (0..w.log_t).map(|_| F::from_u64(rng.next())).collect();
        let eq = EqPolynomial::<F>::evals(&r_cycle, None);
        // rv = Σ_j eq·read, wv = Σ_j eq·write (the Spartan-outer component claims).
        let dot = |c: &[F]| {
            c.iter()
                .zip(eq.iter())
                .fold(F::from_u64(0), |a, (x, e)| a + *x * *e)
        };
        let rv = dot(&w.ram_read_value);
        let wv = dot(&w.ram_write_value);

        let seed = |acc: &mut Openings<F>| {
            let pt = OpeningPoint::new(r_cycle.clone());
            acc.append_virtual(
                VirtualPolynomial::RamReadValue,
                SumcheckId::SpartanOuter,
                pt.clone(),
                rv,
            );
            acc.append_virtual(
                VirtualPolynomial::RamWriteValue,
                SumcheckId::SpartanOuter,
                pt,
                wv,
            );
        };

        let mut prover_acc = Openings::<F>::new(w.log_t);
        seed(&mut prover_acc);
        let mut prover_t = ProverTranscript::new("ram-witness-rw");
        let params = RamReadWriteCheckingParams::new(&prover_acc, w.log_k, &mut prover_t);
        // input_claim = rv + γ·wv (the private params method is module-local).
        let input_claim = rv + params.gamma * wv;
        let mut prover =
            RamReadWriteChecking::new_prover(params, w.ra.clone(), w.val.clone(), w.inc.clone());
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(w.log_t);
        seed(&mut verifier_acc);
        let mut verifier_t = VerifierTranscript::new("ram-witness-rw", &narg);
        let vparams = RamReadWriteCheckingParams::new(&verifier_acc, w.log_k, &mut verifier_t);
        let verifier = RamReadWriteChecking::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: w.log_k + w.log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } = verify(&claim, &mut verifier_t)
            .expect("materialized RAM witness must satisfy read-write-checking");
        assert_eq!(point, challenges);

        for poly in [VirtualPolynomial::RamVal, VirtualPolynomial::RamRa] {
            let (pt, c) =
                prover_acc.get_virtual_polynomial_opening(poly, SumcheckId::RamReadWriteChecking);
            verifier_acc.append_virtual(poly, SumcheckId::RamReadWriteChecking, pt, c);
        }
        let (inc_pt, inc_c) = prover_acc.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        verifier_acc.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            inc_pt,
            inc_c,
        );
        assert_eq!(
            value,
            verifier.expected_output_claim(&verifier_acc, &challenges),
            "reduced claim must close"
        );
    }

    /// Read-before-write: cycle 3 writes addr 2 := 14, and `Val(2, cycle 3)` snapshot precedes it.
    #[test]
    fn read_before_write_snapshot() {
        let trace = sample_trace();
        let w = ram_witness::<MockCycle, F>(&trace, 8);
        let t = 1usize << w.log_t;
        // addr 2: written 5 at cycle 0, so Val(2, j) = 5 for j∈{1,2,3}, then 14 for j≥4.
        assert_eq!(w.val[2 * t + 1], F::from_u64(5), "Val(2,1) = 5");
        assert_eq!(w.val[2 * t + 3], F::from_u64(5), "Val(2,3) = 5 pre-write");
        assert_eq!(
            w.val[2 * t + 4],
            F::from_u64(14),
            "Val(2,4) = 14 post-write"
        );
        // cycle 1 is a load (delta 0): inc 0, read = write = 5.
        assert_eq!(w.inc[1], F::from_u64(0), "load inc = 0");
        assert_eq!(w.ram_read_value[1], F::from_u64(5), "load reads 5");
        // cycle 3 write: inc = 14 - 5 = 9.
        assert_eq!(w.inc[3], F::from_u64(9), "store inc = post - pre = 9");
    }
}
