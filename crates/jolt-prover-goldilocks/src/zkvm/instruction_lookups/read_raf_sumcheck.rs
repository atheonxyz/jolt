//! The instruction-lookup read-raf [`SumcheckInstance`] (P2): the prefix/suffix address phase
//! ([`super::address_phase`], P1) + the one-hot cycle phase, over `num_rounds = LOG_K + log_t`.
//!
//! This is the prefix/suffix replacement for the dense-`Val_s` [`crate::zkvm::shout_read_raf`]
//! `OneHotReadRaf` — necessary at the production `LOG_K = 2·XLEN = 128`, where a dense length-`2^128`
//! `Val` table is infeasible. It proves the instruction-lookup read-raf identity
//!
//! ```text
//! Σ_j eq(r_reduction, j) · (Val_{table[j]}(idx[j]) + raf(idx[j])) = input_claim,
//! ```
//!
//! with `Val_t` the table MLE (prefix/suffix decomposition) and `raf` the operand/identity RAF.
//!
//! ## Two phases (mirrors `OneHotReadRaf`, address bound first, all `LowToHigh` at the framework level)
//! - **Address phase** (`round < LOG_K`): the P1 [`InstructionAddressPhase`] prefix/suffix sumcheck
//!   over the 128 interleaved-index bits (degree 2, padded to `NE = D+2` coeffs). It reduces the claim
//!   to `Σ_g u·eq(r_addr, idx_g)·[Val_{t_g}(r_addr) + raf(r_addr)]` (field MLEs at `r_addr`).
//! - **Hand-off** (after the last address bit): split `r_addr` into the `D` committed chunk points,
//!   lift the sparse `ra_i(r_k_i, ·)` columns (length `T`), and materialize the per-cycle column
//!   `g[j] = eq(r_reduction, j)·(Val_{table[j]}(r_addr) + raf(r_addr, j))`. The cycle-phase input claim
//!   `Σ_j (∏_i ra_i(r_addr, j))·g[j]` equals the address-phase output by construction (∏ ra_i = eq(r_addr,
//!   idx[j]); grouping by distinct index gives back the address reduction), so the framework's
//!   running-claim tracking carries the hand-off with no seam.
//! - **Cycle phase** (`round ≥ LOG_K`): bind the `D` `ra_i` + the combined `g` column; the message is
//!   `(∏_i ra_i)·g` (degree `D+1`).
//!
//! The `D` cached `ra_i(r_k_i, r_cycle)` openings are the M7 pushforward (P7) inputs (unchanged from
//! the dense path). The verifier reconstructs the read+raf via per-table `LookupTableFlag` openings
//! (double-eq weighted) + the operand/identity MLEs at `r_addr` (the [`super::address_phase`]
//! `operand_polynomial_eval`/`identity_polynomial_eval`, stage5 convention).

use jolt_field::Field;
use jolt_lookup_tables::LookupTableKind;
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
use crate::framework::transcript::{ProverFs, VerifierFs};
use jolt_sumcheck::SumcheckClaim;

use super::address_phase::{
    identity_polynomial_eval, lookup_groups_from_trace, operand_polynomial_eval,
    InstructionAddressPhase,
};

/// Per-cycle instruction-lookup trace columns the read-raf consumes (the decoupled, M5-style input).
pub struct InstructionTrace<'a, const D: usize> {
    /// The `2·XLEN`-bit interleaved lookup index per cycle (length `T`).
    pub lookup_indices: &'a [u128],
    /// The lookup table per cycle (`None` = no table read, identity raf only).
    pub lookup_table_indices: &'a [Option<usize>],
    /// Whether the operands are interleaved (RAF = `γ·left + γ²·right`) vs identity (`γ²·index`).
    pub is_interleaved: &'a [bool],
    /// The `D` committed chunk-index columns `idx_i[j] < 2^{LOG_K/D}` (chunk 0 most significant).
    pub indices: &'a [Vec<u32>; D],
}

/// The instruction-lookup read-raf sumcheck instance. `LOG_K = 2·XLEN`, `D` committed RA chunks of
/// `LOG_K/D` bits each, `NE = D + 2` round-poly evaluation points.
pub struct InstructionReadRaf<F: Field, const XLEN: usize, const D: usize, const NE: usize> {
    address: InstructionAddressPhase<F, XLEN>,
    gamma: F,
    log_t: usize,
    r_reduction: Vec<F>,
    lookup_indices: Vec<u128>,
    lookup_table_indices: Vec<Option<usize>>,
    is_interleaved: Vec<bool>,
    indices: [Vec<u32>; D],
    /// Cycle phase (materialized at the hand-off): the `D` `ra_i(r_k_i, ·)` columns.
    ra: Vec<MultilinearPolynomial<F>>,
    /// Cycle phase: the combined `g[j] = eq(r_reduction,j)·(read+raf at r_addr)` column.
    g: Option<MultilinearPolynomial<F>>,
}

impl<F: Field, const XLEN: usize, const D: usize, const NE: usize>
    InstructionReadRaf<F, XLEN, D, NE>
{
    const LOG_K: usize = 2 * XLEN;
    const RA_CHUNK_BITS: usize = (2 * XLEN) / D;

    /// Build the prover instance from the trace, the shared reduction point `r_reduction` (length
    /// `log_t`), and the batching challenge `gamma`.
    pub fn new_prover(trace: InstructionTrace<'_, D>, r_reduction: Vec<F>, gamma: F) -> Self {
        debug_assert_eq!(NE, D + 2, "NE must equal D + 2");
        debug_assert_eq!(Self::LOG_K % D, 0, "D must divide LOG_K = 2*XLEN");
        let log_t = r_reduction.len();
        let u_evals = EqPolynomial::<F>::evals(&r_reduction, None);
        let (groups, _by_cycle, groups_by_table) = lookup_groups_from_trace::<XLEN, F>(
            trace.lookup_indices,
            trace.lookup_table_indices,
            trace.is_interleaved,
            &u_evals,
        );
        let address =
            InstructionAddressPhase::<F, XLEN>::new(groups, groups_by_table, gamma, F::from_u64(1));
        Self {
            address,
            gamma,
            log_t,
            r_reduction,
            lookup_indices: trace.lookup_indices.to_vec(),
            lookup_table_indices: trace.lookup_table_indices.to_vec(),
            is_interleaved: trace.is_interleaved.to_vec(),
            indices: trace.indices.clone(),
            ra: Vec::new(),
            g: None,
        }
    }

    /// Build a verifier instance (no trace): only `gamma`, `r_reduction`, and `log_t` are needed by
    /// [`SumcheckInstance::expected_output_claim`] (which reads ra/flag openings from the accumulator
    /// and computes the table/operand MLEs at `r_addr` from the public tables).
    pub fn new_verifier(r_reduction: Vec<F>, gamma: F) -> Self {
        let log_t = r_reduction.len();
        let address =
            InstructionAddressPhase::<F, XLEN>::new(Vec::new(), Vec::new(), gamma, F::from_u64(1));
        Self {
            address,
            gamma,
            log_t,
            r_reduction,
            lookup_indices: Vec::new(),
            lookup_table_indices: Vec::new(),
            is_interleaved: Vec::new(),
            indices: std::array::from_fn(|_| Vec::new()),
            ra: Vec::new(),
            g: None,
        }
    }

    /// The MSB-first address point after the address phase completes. The address phase binds
    /// HighToLow (round 0 = MSB), so `address_challenges` is already MSB-first — NO reverse (unlike
    /// the LowToHigh `OneHotReadRaf`).
    fn r_addr(&self) -> Vec<F> {
        self.address.address_challenges().to_vec()
    }

    /// Read+raf value MLE at `r_addr` for one cycle's table/interleaved flags.
    fn cycle_value(&self, table_vals: &[F], r_addr_raf: (F, F), cycle: usize) -> F {
        let (raf_interleaved, raf_identity) = r_addr_raf;
        let read =
            self.lookup_table_indices[cycle].map_or_else(|| F::from_u64(0), |t| table_vals[t]);
        let raf = if self.is_interleaved[cycle] {
            raf_interleaved
        } else {
            raf_identity
        };
        read + raf
    }

    /// Per-table `Val_t(r_addr)` for every table (0 for tables no cycle reads).
    fn table_values_at(&self, r_addr: &[F]) -> Vec<F> {
        let tables = LookupTableKind::<XLEN>::all();
        let mut used = vec![false; tables.len()];
        for &t in self.lookup_table_indices.iter().flatten() {
            used[t] = true;
        }
        tables
            .iter()
            .enumerate()
            .map(|(t, table)| {
                if used[t] {
                    table.evaluate_mle::<F, F>(r_addr)
                } else {
                    F::from_u64(0)
                }
            })
            .collect()
    }

    /// `(γ·op_left + γ²·op_right, γ²·identity)` at `r_addr`.
    fn raf_values_at(&self, r_addr: &[F]) -> (F, F) {
        let g2 = self.gamma * self.gamma;
        let interleaved = self.gamma * operand_polynomial_eval(r_addr, true)
            + g2 * operand_polynomial_eval(r_addr, false);
        let identity = g2 * identity_polynomial_eval(r_addr);
        (interleaved, identity)
    }

    /// Address→cycle hand-off: lift the `D` `ra_i` columns and materialize the combined `g` column.
    fn materialize_cycle(&mut self) {
        let t = 1usize << self.log_t;
        let r_addr = self.r_addr();
        let chunk_bits = Self::RA_CHUNK_BITS;

        self.ra = (0..D)
            .map(|i| {
                let eq_addr =
                    EqPolynomial::<F>::evals(&r_addr[i * chunk_bits..(i + 1) * chunk_bits], None);
                let col: Vec<F> = (0..t)
                    .map(|j| eq_addr[self.indices[i][j] as usize])
                    .collect();
                MultilinearPolynomial::from(col)
            })
            .collect();

        let table_vals = self.table_values_at(&r_addr);
        let raf = self.raf_values_at(&r_addr);
        let eq_red = EqPolynomial::<F>::evals(&self.r_reduction, None);
        let g_col: Vec<F> = (0..t)
            .map(|j| eq_red[j] * self.cycle_value(&table_vals, raf, j))
            .collect();
        self.g = Some(MultilinearPolynomial::from(g_col));
    }
}

impl<F: Field, const XLEN: usize, const D: usize, const NE: usize> SumcheckInstance<F>
    for InstructionReadRaf<F, XLEN, D, NE>
{
    fn num_rounds(&self) -> usize {
        Self::LOG_K + self.log_t
    }

    fn degree(&self) -> usize {
        D + 1
    }

    /// Mixed binding orders: the address phase binds HighToLow (already MSB-first) while the cycle
    /// phase binds LowToHigh. So the canonical BIG_ENDIAN point `[r_cycle ‖ r_addr]` reverses ONLY the
    /// cycle challenges and keeps the address challenges as-is — NOT the default uniform reverse.
    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (addr, cycle) = challenges.split_at(Self::LOG_K);
        let mut point = Vec::with_capacity(challenges.len());
        point.extend(cycle.iter().rev().copied());
        point.extend_from_slice(addr);
        OpeningPoint::new(point)
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // The honest claim Σ_j eq(r_reduction,j)·(Val_{table[j]}(idx[j]) + raf(idx[j])), with the
        // table/operand values at the integer (binary) index point. (In the e2e this is seeded by the
        // upstream InstructionClaimReduction; computing it directly here yields the same value.)
        let tables = LookupTableKind::<XLEN>::all();
        let eq_red = EqPolynomial::<F>::evals(&self.r_reduction, None);
        let g2 = self.gamma * self.gamma;
        (0..(1usize << self.log_t))
            .map(|j| {
                let idx = self.lookup_indices[j];
                let read = self.lookup_table_indices[j].map_or_else(
                    || F::from_u64(0),
                    |t| F::from_u64(tables[t].materialize_entry(idx)),
                );
                let bp = binary_point::<F>(idx, Self::LOG_K);
                let raf = if self.is_interleaved[j] {
                    self.gamma * operand_polynomial_eval(&bp, true)
                        + g2 * operand_polynomial_eval(&bp, false)
                } else {
                    g2 * identity_polynomial_eval(&bp)
                };
                eq_red[j] * (read + raf)
            })
            .fold(F::from_u64(0), |acc, v| acc + v)
    }

    #[expect(
        clippy::expect_used,
        reason = "g is materialized at the round LOG_K-1 hand-off, before any cycle round runs"
    )]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UnivariatePoly<F> {
        if round < Self::LOG_K {
            return self.address.round_poly(previous_claim);
        }
        let g = self
            .g
            .as_ref()
            .expect("cycle column materialized at hand-off");
        let half = self.ra[0].len() / 2;
        let mut acc = [F::from_u64(0); NE];
        for i in 0..half {
            let mut ra_prod = [F::from_u64(1); NE];
            for chunk in &self.ra {
                let e = chunk.sumcheck_evals_array::<NE>(i, BindingOrder::LowToHigh);
                for (a, &ep) in ra_prod.iter_mut().zip(e.iter()) {
                    *a *= ep;
                }
            }
            let ge = g.sumcheck_evals_array::<NE>(i, BindingOrder::LowToHigh);
            for (a, (&rp, &gp)) in acc.iter_mut().zip(ra_prod.iter().zip(ge.iter())) {
                *a += rp * gp;
            }
        }
        UnivariatePoly::from_evals(&acc)
    }

    fn bind(&mut self, r: F, round: usize) {
        if round < Self::LOG_K {
            self.address.bind(r);
            if round == Self::LOG_K - 1 {
                self.materialize_cycle();
            }
        } else {
            for chunk in &mut self.ra {
                chunk.bind_parallel(r, BindingOrder::LowToHigh);
            }
            if let Some(g) = self.g.as_mut() {
                g.bind_parallel(r, BindingOrder::LowToHigh);
            }
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        // Address bound first ⇒ BIG_ENDIAN point is [r_cycle ‖ r_addr]; split at log_t.
        let point = self.normalize_opening_point(challenges);
        let (r_cycle, r_addr) = point.split_at(self.log_t);
        let chunk_bits = Self::RA_CHUNK_BITS;

        // The D per-chunk ra_i openings (M7 pushforward inputs).
        for i in 0..D {
            let r_k_i = &r_addr.r[i * chunk_bits..(i + 1) * chunk_bits];
            let chunk_point = OpeningPoint::new([r_k_i, r_cycle.r.as_slice()].concat());
            accumulator.append_dense(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
                chunk_point,
                self.ra[i].final_sumcheck_claim(),
            );
        }

        // Per-table and raf flags: double-eq weighted Σ_j eq(r_cycle,j)·eq(r_reduction,j)·[…]. The
        // verifier reconstructs g̃(r_cycle) = Σ_t flag_t·Val_t(r_addr) + raf_mle·flags.
        let t = 1usize << self.log_t;
        let eq_cycle = EqPolynomial::<F>::evals(&r_cycle.r, None);
        let eq_red = EqPolynomial::<F>::evals(&self.r_reduction, None);
        let dbleq: Vec<F> = (0..t).map(|j| eq_cycle[j] * eq_red[j]).collect();

        let tables = LookupTableKind::<XLEN>::all();
        let mut table_flags = vec![F::from_u64(0); tables.len()];
        let mut identity_flag = F::from_u64(0);
        for ((&de, table_opt), &interleaved) in dbleq
            .iter()
            .zip(self.lookup_table_indices.iter())
            .zip(self.is_interleaved.iter())
        {
            if let Some(table) = table_opt {
                table_flags[*table] += de;
            }
            if !interleaved {
                identity_flag += de;
            }
        }
        let cycle_point = OpeningPoint::new(r_cycle.r.clone());
        for (table, &flag) in table_flags.iter().enumerate() {
            accumulator.append_virtual(
                VirtualPolynomial::LookupTableFlag(table),
                SumcheckId::InstructionReadRaf,
                cycle_point.clone(),
                flag,
            );
        }
        accumulator.append_virtual(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            cycle_point,
            identity_flag,
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let (r_cycle, r_addr) = point.split_at(self.log_t);

        let mut ra_prod = F::from_u64(1);
        for i in 0..D {
            let (_, ra_i) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
            );
            ra_prod *= ra_i;
        }

        let tables = LookupTableKind::<XLEN>::all();
        let mut read = F::from_u64(0);
        for (table, kind) in tables.iter().enumerate() {
            let (_, flag) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupTableFlag(table),
                SumcheckId::InstructionReadRaf,
            );
            read += flag * kind.evaluate_mle::<F, F>(&r_addr.r);
        }

        let (_, identity_flag) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );
        // total double-eq flag = Σ_j eq(r_cycle,j)·eq(r_reduction,j) = eq(r_reduction, r_cycle).
        let total_flag = EqPolynomial::<F>::mle(&self.r_reduction, &r_cycle.r);
        let interleaved_flag = total_flag - identity_flag;
        let g2 = self.gamma * self.gamma;
        let raf = interleaved_flag
            * (self.gamma * operand_polynomial_eval(&r_addr.r, true)
                + g2 * operand_polynomial_eval(&r_addr.r, false))
            + identity_flag * g2 * identity_polynomial_eval(&r_addr.r);

        ra_prod * (read + raf)
    }
}

/// MSB-first binary point of `idx` over `n` variables.
fn binary_point<F: Field>(idx: u128, n: usize) -> Vec<F> {
    (0..n)
        .map(|i| F::from_u64(((idx >> (n - 1 - i)) & 1) as u64))
        .collect()
}

/// Instruction read-raf stage failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructionReadRafError {
    Sumcheck,
    OutputClaim,
}

/// The instruction read-raf stage proof. Interim Fork-2 form: carries the input claim and the
/// openings the verifier (which lacks the trace) cannot recompute — the `D` cached `ra_i(r_k_i,
/// r_cycle)` (also the M7 pushforward inputs) and the per-table / raf flag virtual openings. The
/// sumcheck round polynomials live in the shared NARG. (Binding the input claim to the upstream
/// `InstructionClaimReduction` is the deferred Fork-2 closure — see the e2e seeding.)
#[derive(Clone, Debug)]
pub struct InstructionReadRafProof<F: Field> {
    pub input_claim: F,
    pub ra_openings: Vec<F>,
    pub table_flags: Vec<F>,
    pub raf_flag: F,
}

/// Prove the instruction-lookup read-raf as a composable stage on the shared transcript +
/// accumulator: draw `γ`, build the [`InstructionReadRaf`] instance from the trace + the shared
/// reduction point, run the sumcheck, and extract the `ra_i` + flag openings into the proof.
pub fn prove_instruction_read_raf<F, T, const XLEN: usize, const D: usize, const NE: usize>(
    trace: InstructionTrace<'_, D>,
    r_reduction: Vec<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> InstructionReadRafProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let gamma = transcript.challenge();
    let mut instance = InstructionReadRaf::<F, XLEN, D, NE>::new_prover(trace, r_reduction, gamma);
    let input_claim = instance.input_claim(&*accumulator);
    let _ = prove(&mut instance, accumulator, transcript);

    let table_count = LookupTableKind::<XLEN>::all().len();
    let ra_openings = (0..D)
        .map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionReadRaf,
                )
                .1
        })
        .collect();
    let table_flags = (0..table_count)
        .map(|t| {
            accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupTableFlag(t),
                    SumcheckId::InstructionReadRaf,
                )
                .1
        })
        .collect();
    let raf_flag = accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        )
        .1;
    InstructionReadRafProof {
        input_claim,
        ra_openings,
        table_flags,
        raf_flag,
    }
}

/// Verify the instruction read-raf stage (mirror of [`prove_instruction_read_raf`]): draw `γ` in
/// lockstep, replay the sumcheck against `proof.input_claim`, re-seed the proof-carried `ra_i` + flag
/// openings at the recomputed points, and check the reduced claim closes against
/// [`SumcheckInstance::expected_output_claim`]. A tampered opening fails that check.
pub fn verify_instruction_read_raf<F, T, const XLEN: usize, const D: usize, const NE: usize>(
    proof: &InstructionReadRafProof<F>,
    r_reduction: Vec<F>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), InstructionReadRafError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let log_t = r_reduction.len();
    let log_k = 2 * XLEN;
    let chunk_bits = log_k / D;
    let gamma = transcript.challenge();
    let instance = InstructionReadRaf::<F, XLEN, D, NE>::new_verifier(r_reduction, gamma);

    let claim = SumcheckClaim {
        num_vars: log_k + log_t,
        degree: D + 1,
        claimed_sum: proof.input_claim,
    };
    let eval = verify(&claim, transcript).map_err(|_| InstructionReadRafError::Sumcheck)?;

    // Address bound first ⇒ BIG_ENDIAN point is [r_cycle ‖ r_addr]; split at log_t.
    let point = instance.normalize_opening_point(&eval.point);
    let (r_cycle, r_addr) = point.split_at(log_t);
    for (i, &c) in proof.ra_openings.iter().enumerate() {
        let r_k_i = &r_addr.r[i * chunk_bits..(i + 1) * chunk_bits];
        let chunk_point = OpeningPoint::new([r_k_i, r_cycle.r.as_slice()].concat());
        accumulator.append_dense(
            CommittedPolynomial::InstructionRa(i),
            SumcheckId::InstructionReadRaf,
            chunk_point,
            c,
        );
    }
    let cycle_point = OpeningPoint::new(r_cycle.r.clone());
    for (t, &flag) in proof.table_flags.iter().enumerate() {
        accumulator.append_virtual(
            VirtualPolynomial::LookupTableFlag(t),
            SumcheckId::InstructionReadRaf,
            cycle_point.clone(),
            flag,
        );
    }
    accumulator.append_virtual(
        VirtualPolynomial::InstructionRafFlag,
        SumcheckId::InstructionReadRaf,
        cycle_point,
        proof.raf_flag,
    );

    if eval.value != instance.expected_output_claim(accumulator, &eval.point) {
        return Err(InstructionReadRafError::OutputClaim);
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests may unwrap/expect freely"
)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_lookup_tables::interleave_bits;

    const XLEN: usize = 8;
    const D: usize = 4; // LOG_K=16, RA_CHUNK_BITS=4
    const NE: usize = D + 2;

    type SampleTrace = (Vec<u128>, Vec<Option<usize>>, Vec<bool>, [Vec<u32>; D]);

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    /// Split a `2·XLEN`-bit index into `D` chunk values of `RA_CHUNK_BITS` bits (chunk 0 = MSB).
    fn chunk_index(idx: u128, chunk: usize) -> u32 {
        let chunk_bits = (2 * XLEN) / D;
        let shift = 2 * XLEN - (chunk + 1) * chunk_bits;
        ((idx >> shift) & ((1u128 << chunk_bits) - 1)) as u32
    }

    #[test]
    fn instruction_read_raf_full_round_trip() {
        let and = 2usize; // interleaved arithmetic table
        let idx_a = interleave_bits(0b1011_0110u64, 0b0110_1001u64);
        let idx_b = interleave_bits(0b0011_1101u64, 0b1100_0011u64);
        let idx_c = 0b1010_0101_1001_0110u128; // non-interleaved, no table
        let idx_d = interleave_bits(0b0101_0101u64, 0b1010_1010u64);
        let lookup_indices = vec![idx_a, idx_b, idx_c, idx_d];
        let lookup_table_indices = vec![Some(and), Some(and), None, Some(and)];
        let is_interleaved = vec![true, true, false, true];

        let indices: [Vec<u32>; D] = std::array::from_fn(|i| {
            lookup_indices
                .iter()
                .map(|&idx| chunk_index(idx, i))
                .collect()
        });
        let trace = InstructionTrace {
            lookup_indices: &lookup_indices,
            lookup_table_indices: &lookup_table_indices,
            is_interleaved: &is_interleaved,
            indices: &indices,
        };

        let r_reduction = vec![f(5), f(9)]; // log_t = 2 (trace_len = 4)
        let gamma = f(17);
        let mut instance =
            InstructionReadRaf::<F, XLEN, D, NE>::new_prover(trace, r_reduction, gamma);

        let acc_in = Openings::<F>::new(2);
        let mut claim = instance.input_claim(&acc_in);
        let num_rounds = instance.num_rounds();
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut ra_finals = Vec::new();
        let mut g_final = f(0);
        for round in 0..num_rounds {
            let poly = instance.compute_message(round, claim);
            let c = f((round as u64) * 7 + 3);
            challenges.push(c);
            claim = poly.evaluate(c);
            instance.bind(c, round);
            if round == num_rounds - 1 {
                ra_finals = instance
                    .ra
                    .iter()
                    .map(|p| p.final_sumcheck_claim())
                    .collect();
                let g = instance.g.as_ref().unwrap();
                g_final = g.final_sumcheck_claim();
            }
        }

        // (1) Prover reduction (endianness-safe): the product sumcheck reduces to ∏ ra_i · g.
        let prover_reduced = ra_finals.iter().fold(f(1), |acc, &x| acc * x) * g_final;
        assert_eq!(
            claim, prover_reduced,
            "cycle reduction == ∏ ra_i.final · g.final"
        );

        // (2) Verifier reconstruction via cached flag + ra openings.
        let mut acc = Openings::<F>::new(2);
        instance.cache_openings(&mut acc, &challenges);
        let expected = instance.expected_output_claim(&acc, &challenges);
        assert_eq!(
            claim, expected,
            "reduced claim == verifier expected_output_claim"
        );
    }

    fn sample_trace() -> SampleTrace {
        let and = 2usize;
        let idx_a = interleave_bits(0b1011_0110u64, 0b0110_1001u64);
        let idx_b = interleave_bits(0b0011_1101u64, 0b1100_0011u64);
        let idx_c = 0b1010_0101_1001_0110u128; // non-interleaved, no table
        let idx_d = interleave_bits(0b0101_0101u64, 0b1010_1010u64);
        let lookup_indices = vec![idx_a, idx_b, idx_c, idx_d];
        let lookup_table_indices = vec![Some(and), Some(and), None, Some(and)];
        let is_interleaved = vec![true, true, false, true];
        let indices: [Vec<u32>; D] = std::array::from_fn(|i| {
            lookup_indices
                .iter()
                .map(|&idx| chunk_index(idx, i))
                .collect()
        });
        (
            lookup_indices,
            lookup_table_indices,
            is_interleaved,
            indices,
        )
    }

    /// Prove → NARG → verify through the real framework drivers + WHIR spongefish transcript.
    #[test]
    fn instruction_read_raf_stage_round_trip() {
        let (li, lti, ii, indices) = sample_trace();
        let r_reduction = vec![f(5), f(9)];
        let trace = InstructionTrace {
            lookup_indices: &li,
            lookup_table_indices: &lti,
            is_interleaved: &ii,
            indices: &indices,
        };

        let mut prover_acc = Openings::<F>::new(2);
        let mut pt = ProverTranscript::new("instr-read-raf");
        let proof = prove_instruction_read_raf::<F, _, XLEN, D, NE>(
            trace,
            r_reduction.clone(),
            &mut prover_acc,
            &mut pt,
        );
        let narg = pt.into_proof();

        let mut verifier_acc = Openings::<F>::new(2);
        let mut vt = VerifierTranscript::new("instr-read-raf", &narg);
        verify_instruction_read_raf::<F, _, XLEN, D, NE>(
            &proof,
            r_reduction,
            &mut verifier_acc,
            &mut vt,
        )
        .expect("instruction read-raf must verify");
    }

    /// A tampered `ra_i` opening breaks the output-claim check (the NARG replay still succeeds).
    #[test]
    fn instruction_read_raf_tampered_rejected() {
        let (li, lti, ii, indices) = sample_trace();
        let r_reduction = vec![f(5), f(9)];
        let trace = InstructionTrace {
            lookup_indices: &li,
            lookup_table_indices: &lti,
            is_interleaved: &ii,
            indices: &indices,
        };

        let mut prover_acc = Openings::<F>::new(2);
        let mut pt = ProverTranscript::new("instr-read-raf");
        let mut proof = prove_instruction_read_raf::<F, _, XLEN, D, NE>(
            trace,
            r_reduction.clone(),
            &mut prover_acc,
            &mut pt,
        );
        let narg = pt.into_proof();
        proof.ra_openings[0] += f(1);

        let mut verifier_acc = Openings::<F>::new(2);
        let mut vt = VerifierTranscript::new("instr-read-raf", &narg);
        let res = verify_instruction_read_raf::<F, _, XLEN, D, NE>(
            &proof,
            r_reduction,
            &mut verifier_acc,
            &mut vt,
        );
        assert_eq!(res, Err(InstructionReadRafError::OutputClaim));
    }
}
