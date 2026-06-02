//! Shared Shout read + RAF batched sumcheck — the common structure behind jolt-core's
//! `zkvm/bytecode/read_raf_checking.rs` and `zkvm/instruction_lookups/read_raf_checking.rs`,
//! ported onto [`crate::framework`] over the lean `Field` (`C = F = Fp3`). jolt-core is the
//! parity oracle.
//!
//! Both ports prove the same batched read identity over the `(address, cycle)` hypercube:
//!
//! ```text
//! Σ_{j,k} ra(k,j) · Σ_s γ^s · eq_s(j) · Val_s(k) = Σ_s γ^s · rv_s,
//! ```
//!
//! with the one-hot read indicator `ra(k,j) = ∏_{i=0}^{D-1} ra_i(k_i, j)` (the `D`-chunk product),
//! one-hot in `k` per cycle: cycle `j` reads the single address `combined(idx[j]) = Σ_i idx_i[j]·suffix_i`.
//! - **Bytecode** (`D = 3..4`) uses per-stage cycle points `r_cycle_s` (distinct `eq_s`) and `Val_s`
//!   encoding circuit flags / RAF identity.
//! - **Instruction-lookups** shares `r_cycle = r_reduction` across stages; at production `LOG_K = 128`
//!   the dense `K_total` tables below are infeasible and the address phase must use prefix/suffix —
//!   that variant is deferred (see the `goldilocks-migration-plan` memory, P-Sparse-e).
//!
//! ## Address-first two-phase (sparse, `O(K_total + D·T)`)
//!
//! Ported from jolt-core's `BytecodeReadRafSumcheckProver`: never materializes the `K_total·T`
//! hypercube. `num_rounds = log_K + log_T`, all bound `LowToHigh`.
//! - **Address phase** (`rounds < log_K`): bind the per-stage address marginal
//!   `F_s[k] = Σ_{j: read(j)=k} eq_s(j)` (built by an `O(T)` scatter) against `Val_s[k]` (length
//!   `K_total`). The round message is `Σ_s γ^s · F_s · Val_s` (degree 2). Its `s(0)+s(1)` telescopes
//!   to `Σ_s γ^s · Σ_k F_s[k]·Val_s[k] = Σ_s γ^s · rv_s` — the input claim.
//! - **Hand-off** (after the last address bit): `r_addr` splits into the `D` chunk points `r_k_i`;
//!   materialize the sparse `ra_i(r_k_i, j) = eq(r_k_i, idx_i[j])` columns (length `T`) and capture
//!   `Val_s(r_addr)` scalars. The address-phase final claim equals the cycle-phase initial claim
//!   (`Σ_s γ^s·Val_s(r_addr)·Σ_j ra(r_addr,j)·eq_s(j)`), so the framework's running-claim tracking
//!   carries the hand-off with no special seam.
//! - **Cycle phase** (`rounds ≥ log_K`): bind the `D` `ra_i` columns + the per-stage `eq_s` (length
//!   `T`); the message is `(∏_i ra_i)·Σ_s γ^s·eq_s·Val_s(r_addr)` (degree `D + 1`).
//!
//! `NE = D + 2` evaluation points interpolate the degree-`(D+1)` cycle message; the degree-2 address
//! message uses the same `NE` points (its high coefficients are zero, written padded). The per-chunk
//! `ra_i(r_k_i, r_cycle)` openings cached by [`SumcheckInstance::cache_openings`] are the §4.5.2
//! inputs the M7 pushforward (P7) consumes — unchanged by this sparse rewrite.
//!
//! **Decoupled from the trace** (the M5 convention): takes the dense `u32` chunk-index columns
//! (`idx_i[j] < 2^{log_K_i}`, the `CommittedWitness.ra_dense` form) + the public per-stage
//! `(r_cycle_s, Val_s)` columns. Deferred: jolt-core's prefix/suffix `Val_s` materialization (so the
//! dense `Val_s`/`F_s` length-`K_total` tables only suit small `K_total`, i.e. bytecode), the Gruen
//! split-eq cycle-phase optimization (we use plain length-`T` `eq_s`), the entry-point constraint,
//! and the flag/lookup-table-specific `Val_s` construction.

use crate::framework::transcript::{Challenge, ProverFs, VerifierFs};
use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckClaim;

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};

/// One batched stage: a per-stage cycle-eq point `r_cycle`, the public address-only value column
/// `val_addr` (length `K_total = ∏_i 2^{log_K_i}`, indexed by the chunk-combined address), and the
/// accumulator key of the upstream read claim `rv_s`.
#[derive(Clone, Debug)]
pub struct ReadRafStage<F: Field> {
    pub r_cycle: Vec<F>,
    pub val_addr: Vec<F>,
    pub rv_key: (VirtualPolynomial, SumcheckId),
}

/// Batching/opening parameters, parameterized by the chunk count `D`, the committed RA family, and
/// the sumcheck id so the bytecode and instruction-lookups ports share one implementation.
#[derive(Clone, Debug)]
pub struct OneHotReadRafParams<F: Field, const D: usize> {
    /// `[γ^0, …, γ^{S-1}]` for `S` stages.
    pub gamma_powers: Vec<F>,
    /// Address-chunk bit widths `[log_K_0, …, log_K_{D-1}]` (chunk 0 is the most significant).
    pub log_k_chunks: [usize; D],
    pub log_t: usize,
    pub stages: Vec<ReadRafStage<F>>,
    /// Maps chunk index → its committed RA polynomial (e.g. `CommittedPolynomial::BytecodeRa`).
    pub ra_family: fn(usize) -> CommittedPolynomial,
    /// The sumcheck id the `ra_i` leaf openings are cached under.
    pub sumcheck_id: SumcheckId,
}

impl<F: Field, const D: usize> OneHotReadRafParams<F, D> {
    /// Draws `γ` and forms `num_stages` powers.
    pub fn new(
        ra_family: fn(usize) -> CommittedPolynomial,
        sumcheck_id: SumcheckId,
        log_k_chunks: [usize; D],
        log_t: usize,
        stages: Vec<ReadRafStage<F>>,
        transcript: &mut impl Challenge<F>,
    ) -> Self {
        let gamma = transcript.challenge();
        let mut gamma_powers = Vec::with_capacity(stages.len());
        let mut p = F::one();
        for _ in 0..stages.len() {
            gamma_powers.push(p);
            p *= gamma;
        }
        Self {
            gamma_powers,
            log_k_chunks,
            log_t,
            stages,
            ra_family,
            sumcheck_id,
        }
    }

    #[inline]
    fn log_k(&self) -> usize {
        self.log_k_chunks.iter().sum()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.stages
            .iter()
            .zip(self.gamma_powers.iter())
            .fold(F::zero(), |acc, (stage, &g)| {
                let (_, rv) =
                    accumulator.get_virtual_polynomial_opening(stage.rv_key.0, stage.rv_key.1);
                acc + g * rv
            })
    }
}

/// Mixed-radix suffix products `[∏_{l>0} K_l, …, ∏_{l>D-2} K_l, 1]` for the flat address
/// `addr = Σ_i k_i · suffix[i]` (chunk 0 most significant): `k_i = (addr / suffix[i]) % K_i`.
#[inline]
fn suffix_products<const D: usize>(k_dims: &[usize; D]) -> [usize; D] {
    let mut suffix = [1usize; D];
    for i in (1..D).rev() {
        suffix[i - 1] = suffix[i] * k_dims[i];
    }
    suffix
}

/// Prover/verifier instance for the address-first two-phase read-raf sumcheck.
///
/// `NE = D + 2` is the round-polynomial evaluation-point count (interpolating the degree-`(D+1)`
/// cycle message); threaded as a separate const because stable Rust cannot compute it in a turbofish.
pub struct OneHotReadRaf<F: Field, const D: usize, const NE: usize> {
    pub params: OneHotReadRafParams<F, D>,
    /// Raw chunk-index columns `idx_i[j] < 2^{log_K_i}` (length `T`), kept for the hand-off `ra_i` lift.
    indices: [Vec<u32>; D],
    /// Address phase: per-stage marginal `F_s[k]` (length `K_total`), bound `LowToHigh`.
    f: Vec<MultilinearPolynomial<F>>,
    /// Address phase: per-stage `Val_s[k]` (length `K_total`), bound `LowToHigh`.
    val: Vec<MultilinearPolynomial<F>>,
    /// Cycle phase: per-chunk `ra_i(r_k_i, ·)` (length `T`), materialized at the hand-off.
    ra: Vec<MultilinearPolynomial<F>>,
    /// Cycle phase: per-stage `eq_s(·)` (length `T`), materialized at the hand-off.
    eq: Vec<MultilinearPolynomial<F>>,
    /// Cycle phase: per-stage `Val_s(r_addr)` scalars, captured at the hand-off.
    bound_val: Vec<F>,
    /// Address challenges accumulated `LowToHigh` (reversed to MSB-first `r_addr` at the hand-off).
    addr_challenges: Vec<F>,
}

impl<F: Field, const D: usize, const NE: usize> OneHotReadRaf<F, D, NE> {
    /// `indices[i]` is chunk `i`'s dense index column (`idx_i[j] < 2^{log_K_i}`, length `T`). Builds
    /// the per-stage address marginals `F_s` (`O(T)` scatter) + `Val_s` tables; the sparse `ra_i` are
    /// materialized at the address→cycle hand-off.
    pub fn new_prover(params: OneHotReadRafParams<F, D>, indices: [Vec<u32>; D]) -> Self {
        debug_assert_eq!(NE, D + 2, "NE must equal D + 2");
        let log_k = params.log_k();
        assert!(
            log_k < usize::BITS as usize,
            "read-raf address width {log_k} must fit in usize for the F_s/Val_s tables"
        );
        let t = 1usize << params.log_t;
        let k_dims: [usize; D] = std::array::from_fn(|i| 1usize << params.log_k_chunks[i]);
        let k_total: usize = k_dims.iter().product();
        let suffix = suffix_products(&k_dims);

        // The single address read at each cycle: combined(idx[j]) = Σ_i idx_i[j] · suffix[i].
        let combined: Vec<usize> = (0..t)
            .map(|j| {
                indices
                    .iter()
                    .zip(suffix.iter())
                    .fold(0usize, |acc, (idx, &s)| acc + (idx[j] as usize) * s)
            })
            .collect();

        let mut f = Vec::with_capacity(params.stages.len());
        let mut val = Vec::with_capacity(params.stages.len());
        for stage in &params.stages {
            let eq_cycle = EqPolynomial::<F>::evals(&stage.r_cycle, None);
            let mut fs = vec![F::zero(); k_total];
            for (j, &k) in combined.iter().enumerate() {
                fs[k] += eq_cycle[j];
            }
            f.push(MultilinearPolynomial::from(fs));
            val.push(MultilinearPolynomial::from(stage.val_addr.clone()));
        }

        Self {
            params,
            indices,
            f,
            val,
            ra: Vec::new(),
            eq: Vec::new(),
            bound_val: Vec::new(),
            addr_challenges: Vec::new(),
        }
    }

    pub fn new_verifier(params: OneHotReadRafParams<F, D>) -> Self {
        debug_assert_eq!(NE, D + 2, "NE must equal D + 2");
        Self {
            params,
            indices: std::array::from_fn(|_| Vec::new()),
            f: Vec::new(),
            val: Vec::new(),
            ra: Vec::new(),
            eq: Vec::new(),
            bound_val: Vec::new(),
            addr_challenges: Vec::new(),
        }
    }

    /// Address→cycle hand-off: split `r_addr` into the `D` chunk points, lift the sparse
    /// `ra_i(r_k_i, ·)` cycle columns, build the per-stage `eq_s(·)` columns, and capture the bound
    /// `Val_s(r_addr)` scalars. Called once after the final address bit is bound.
    fn materialize_cycle_phase(&mut self) {
        let t = 1usize << self.params.log_t;
        // r_addr MSB-first (the address challenges were accumulated LowToHigh = LSB-first).
        let mut r_addr = self.addr_challenges.clone();
        r_addr.reverse();

        let mut offset = 0;
        self.ra = Vec::with_capacity(D);
        for i in 0..D {
            let w = self.params.log_k_chunks[i];
            let eq_addr = EqPolynomial::<F>::evals(&r_addr[offset..offset + w], None);
            offset += w;
            let col: Vec<F> = (0..t)
                .map(|j| eq_addr[self.indices[i][j] as usize])
                .collect();
            self.ra.push(MultilinearPolynomial::from(col));
        }

        self.eq = self
            .params
            .stages
            .iter()
            .map(|stage| {
                MultilinearPolynomial::from(EqPolynomial::<F>::evals(&stage.r_cycle, None))
            })
            .collect();
        self.bound_val = self.val.iter().map(|v| v.final_sumcheck_claim()).collect();
    }
}

impl<F: Field, const D: usize, const NE: usize> SumcheckInstance<F> for OneHotReadRaf<F, D, NE> {
    fn num_rounds(&self) -> usize {
        self.params.log_k() + self.params.log_t
    }

    fn degree(&self) -> usize {
        D + 1
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn compute_message(&mut self, round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let mut acc = [F::zero(); NE];
        if round < self.params.log_k() {
            // Address phase: Σ_s γ^s · F_s · Val_s (degree 2, padded to NE points).
            let half = self.f[0].len() / 2;
            for i in 0..half {
                for (s, &g) in self.params.gamma_powers.iter().enumerate() {
                    let fe = self.f[s].sumcheck_evals_array::<NE>(i, BindingOrder::LowToHigh);
                    let ve = self.val[s].sumcheck_evals_array::<NE>(i, BindingOrder::LowToHigh);
                    for (a, (&fp, &vp)) in acc.iter_mut().zip(fe.iter().zip(ve.iter())) {
                        *a += g * fp * vp;
                    }
                }
            }
        } else {
            // Cycle phase: (∏_i ra_i) · Σ_s γ^s · eq_s · Val_s(r_addr) (degree D+1).
            let half = self.ra[0].len() / 2;
            for i in 0..half {
                let mut ra_prod = [F::one(); NE];
                for chunk in &self.ra {
                    let e = chunk.sumcheck_evals_array::<NE>(i, BindingOrder::LowToHigh);
                    for (a, &ep) in ra_prod.iter_mut().zip(e.iter()) {
                        *a *= ep;
                    }
                }
                let mut stage_sum = [F::zero(); NE];
                for (s, &g) in self.params.gamma_powers.iter().enumerate() {
                    let e = self.eq[s].sumcheck_evals_array::<NE>(i, BindingOrder::LowToHigh);
                    let gv = g * self.bound_val[s];
                    for (a, &ep) in stage_sum.iter_mut().zip(e.iter()) {
                        *a += gv * ep;
                    }
                }
                for (a, (&rp, &ss)) in acc.iter_mut().zip(ra_prod.iter().zip(stage_sum.iter())) {
                    *a += rp * ss;
                }
            }
        }
        UnivariatePoly::from_evals(&acc)
    }

    fn bind(&mut self, r: F, round: usize) {
        if round < self.params.log_k() {
            for poly in self.f.iter_mut().chain(self.val.iter_mut()) {
                poly.bind_parallel(r, BindingOrder::LowToHigh);
            }
            self.addr_challenges.push(r);
            if round == self.params.log_k() - 1 {
                self.materialize_cycle_phase();
            }
        } else {
            for poly in self.ra.iter_mut().chain(self.eq.iter_mut()) {
                poly.bind_parallel(r, BindingOrder::LowToHigh);
            }
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        // Address bound first ⇒ BIG_ENDIAN point is [r_cycle ‖ r_addr]; split at log_t.
        let point = self.normalize_opening_point(challenges);
        let (r_cycle, r_addr) = point.split_at(self.params.log_t);
        let mut offset = 0;
        for i in 0..D {
            let w = self.params.log_k_chunks[i];
            let r_k_i = &r_addr.r[offset..offset + w];
            offset += w;
            let chunk_point = OpeningPoint::new([r_k_i, r_cycle.r.as_slice()].concat());
            accumulator.append_dense(
                (self.params.ra_family)(i),
                self.params.sumcheck_id,
                chunk_point,
                self.ra[i].final_sumcheck_claim(),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let (r_cycle, r_addr) = point.split_at(self.params.log_t);

        let mut ra_prod = F::one();
        for i in 0..D {
            let (_, ra_i) = accumulator.get_committed_polynomial_opening(
                (self.params.ra_family)(i),
                self.params.sumcheck_id,
            );
            ra_prod *= ra_i;
        }

        let eq_addr = EqPolynomial::<F>::evals(&r_addr.r, None);
        let mut stage_sum = F::zero();
        for (stage, &g) in self
            .params
            .stages
            .iter()
            .zip(self.params.gamma_powers.iter())
        {
            let eq_s = EqPolynomial::<F>::mle(&stage.r_cycle, &r_cycle.r);
            let val_s = stage
                .val_addr
                .iter()
                .zip(eq_addr.iter())
                .fold(F::zero(), |acc, (v, e)| acc + *v * *e);
            stage_sum += g * eq_s * val_s;
        }
        ra_prod * stage_sum
    }
}

/// Read-raf stage verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadRafStageError {
    Sumcheck,
    OutputClaim,
}

/// The read-raf stage's proof: the `D` cached `ra_i(r_k_i, r_cycle)` opening claims (chunk order) —
/// the M7 §4.5.2 inputs the per-chunk pushforward GKR (P7) consumes. The opening *points* are
/// recomputed from the sumcheck challenges; the committed `ra_dense` columns are PCS-opened at
/// stage 8. The sumcheck round polynomials live in the shared NARG.
#[derive(Clone, Debug)]
pub struct ReadRafStageProof<F: Field> {
    pub ra_openings: Vec<F>,
}

/// One family's read-raf stage inputs (the un-`γ`'d [`OneHotReadRafParams`]): which committed RA
/// family + sumcheck id, the `D` chunk widths, `log_t`, and the per-stage `(r_cycle, val_addr,
/// rv_key)` columns. Both [`prove_read_raf`] and [`verify_read_raf`] build the same
/// [`OneHotReadRafParams`] from this (drawing `γ` in lockstep).
#[derive(Clone, Debug)]
pub struct ReadRafInputs<F: Field, const D: usize> {
    pub ra_family: fn(usize) -> CommittedPolynomial,
    pub sumcheck_id: SumcheckId,
    pub log_k_chunks: [usize; D],
    pub log_t: usize,
    pub stages: Vec<ReadRafStage<F>>,
}

/// Prove one family's read-raf as a composable stage on the shared transcript + accumulator
/// (analogous to [`prove_registers`](crate::zkvm::registers::stage::prove_registers)): build the
/// [`OneHotReadRaf`] instance from the dense `u32` chunk-index columns, run it, and extract the `D`
/// cached `ra_i` openings into the proof.
///
/// The per-stage `(r_cycle, val_addr, rv_key)` and the upstream `rv_s` claims are supplied via
/// `inputs.stages` + the seeded `accumulator` (in the e2e they come from Spartan / the claim
/// reductions).
pub fn prove_read_raf<F, T, const D: usize, const NE: usize>(
    indices: [Vec<u32>; D],
    inputs: ReadRafInputs<F, D>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> ReadRafStageProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let params = OneHotReadRafParams::<F, D>::new(
        inputs.ra_family,
        inputs.sumcheck_id,
        inputs.log_k_chunks,
        inputs.log_t,
        inputs.stages,
        transcript,
    );
    let mut instance = OneHotReadRaf::<F, D, NE>::new_prover(params, indices);
    let _ = prove(&mut instance, accumulator, transcript);
    let ra_family = instance.params.ra_family;
    let sumcheck_id = instance.params.sumcheck_id;
    let ra_openings = (0..D)
        .map(|i| {
            accumulator
                .get_committed_polynomial_opening(ra_family(i), sumcheck_id)
                .1
        })
        .collect();
    ReadRafStageProof { ra_openings }
}

/// Verify one family's read-raf stage (mirror of [`prove_read_raf`]): replay the sumcheck, re-seed
/// the `D` proof-carried `ra_i` openings at their recomputed chunk points, then check the reduced
/// claim closes against `(∏_i ra_i)·Σ_s γ^s·eq_s·Val_s`. A tampered `ra_opening` fails that check.
pub fn verify_read_raf<F, T, const D: usize, const NE: usize>(
    proof: &ReadRafStageProof<F>,
    inputs: ReadRafInputs<F, D>,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), ReadRafStageError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let log_k_chunks = inputs.log_k_chunks;
    let log_t = inputs.log_t;
    let params = OneHotReadRafParams::<F, D>::new(
        inputs.ra_family,
        inputs.sumcheck_id,
        log_k_chunks,
        log_t,
        inputs.stages,
        transcript,
    );
    let ra_family = params.ra_family;
    let sumcheck_id = params.sumcheck_id;
    let log_k: usize = log_k_chunks.iter().sum();
    let instance = OneHotReadRaf::<F, D, NE>::new_verifier(params);
    let input_claim = instance.input_claim(accumulator);
    let claim = SumcheckClaim {
        num_vars: log_k + log_t,
        degree: D + 1,
        claimed_sum: input_claim,
    };
    let eval = verify(&claim, transcript).map_err(|_| ReadRafStageError::Sumcheck)?;

    // Address bound first ⇒ BIG_ENDIAN point is [r_cycle ‖ r_addr]; split at log_t.
    let point = instance.normalize_opening_point(&eval.point);
    let (r_cycle, r_addr) = point.split_at(log_t);
    let mut offset = 0;
    for (i, &c) in proof.ra_openings.iter().enumerate() {
        let w = log_k_chunks[i];
        let r_k_i = &r_addr.r[offset..offset + w];
        offset += w;
        let chunk_point = OpeningPoint::new([r_k_i, r_cycle.r.as_slice()].concat());
        accumulator.append_dense(ra_family(i), sumcheck_id, chunk_point, c);
    }
    if eval.value != instance.expected_output_claim(accumulator, &eval.point) {
        return Err(ReadRafStageError::OutputClaim);
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::zkvm::witness::one_hot_ra_column;
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_sumcheck::EvaluationClaim;

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

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<F> {
        (0..n).map(|_| F::from_u64(rng.next())).collect()
    }

    /// `(family, sumcheck_id, stage rv-keys)` for a concrete instantiation.
    struct Config {
        family: fn(usize) -> CommittedPolynomial,
        sumcheck_id: SumcheckId,
        rv_keys: Vec<(VirtualPolynomial, SumcheckId)>,
        /// Bytecode uses distinct per-stage cycle points; instruction-lookups shares one.
        shared_cycle: bool,
    }

    fn bytecode_config() -> Config {
        Config {
            family: CommittedPolynomial::BytecodeRa,
            sumcheck_id: SumcheckId::BytecodeReadRaf,
            rv_keys: vec![
                (VirtualPolynomial::UnexpandedPC, SumcheckId::SpartanOuter),
                (VirtualPolynomial::Imm, SumcheckId::SpartanShift),
                (VirtualPolynomial::PC, SumcheckId::SpartanOuter),
            ],
            shared_cycle: false,
        }
    }

    fn instruction_config() -> Config {
        Config {
            family: CommittedPolynomial::InstructionRa,
            sumcheck_id: SumcheckId::InstructionReadRaf,
            rv_keys: vec![
                (
                    VirtualPolynomial::LookupOutput,
                    SumcheckId::InstructionClaimReduction,
                ),
                (
                    VirtualPolynomial::LeftLookupOperand,
                    SumcheckId::InstructionClaimReduction,
                ),
                (
                    VirtualPolynomial::RightLookupOperand,
                    SumcheckId::SpartanProductVirtualization,
                ),
            ],
            shared_cycle: true,
        }
    }

    /// `addr = Σ_i k_i · suffix[i]` mixed-radix combine (chunk 0 most significant).
    fn combine<const D: usize>(idx: &[Vec<u32>; D], j: usize, suffix: &[usize; D]) -> usize {
        idx.iter()
            .zip(suffix.iter())
            .fold(0usize, |acc, (col, &s)| acc + (col[j] as usize) * s)
    }

    /// Generic round-trip over `D` one-hot chunks (`NE = D + 2`): random per-chunk index columns →
    /// the genuine read sum `rv_s = Σ_j eq_s(j)·Val(combined(j))` → prove the sparse address-first
    /// read-raf → check the reduced claim closes against `(∏_i ra_i)·Σ_s γ^s·eq_s·Val_s`, and the
    /// cached `ra_i` openings equal the genuine one-hot MLE evals.
    fn round_trip<const D: usize, const NE: usize>(
        cfg: &Config,
        seed: u64,
        log_k_chunks: [usize; D],
        log_t: usize,
        num_stages: usize,
    ) {
        let mut rng = Rng(seed);
        let k_dims: [usize; D] = std::array::from_fn(|i| 1usize << log_k_chunks[i]);
        let suffix = suffix_products(&k_dims);
        let k_total: usize = k_dims.iter().product();
        let t = 1usize << log_t;

        let indices: [Vec<u32>; D] = std::array::from_fn(|i| {
            (0..t)
                .map(|_| (rng.next() as u32) % (k_dims[i] as u32))
                .collect()
        });
        let combined: Vec<usize> = (0..t).map(|j| combine(&indices, j, &suffix)).collect();

        let shared_r = rand_vec(&mut rng, log_t);
        let stages: Vec<ReadRafStage<F>> = (0..num_stages)
            .map(|s| ReadRafStage {
                r_cycle: if cfg.shared_cycle {
                    shared_r.clone()
                } else {
                    rand_vec(&mut rng, log_t)
                },
                val_addr: rand_vec(&mut rng, k_total),
                rv_key: cfg.rv_keys[s],
            })
            .collect();

        let seed_acc = |acc: &mut Openings<F>| {
            for stage in &stages {
                let eq = EqPolynomial::<F>::evals(&stage.r_cycle, None);
                let rv = (0..t).fold(F::from_u64(0), |a, j| {
                    a + eq[j] * stage.val_addr[combined[j]]
                });
                acc.append_virtual(
                    stage.rv_key.0,
                    stage.rv_key.1,
                    OpeningPoint::new(stage.r_cycle.clone()),
                    rv,
                );
            }
        };

        let log_k: usize = log_k_chunks.iter().sum();

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut prover_t = ProverTranscript::new("shout-read-raf");
        let params = OneHotReadRafParams::<F, D>::new(
            cfg.family,
            cfg.sumcheck_id,
            log_k_chunks,
            log_t,
            stages.clone(),
            &mut prover_t,
        );
        let input_claim = params.input_claim(&prover_acc);
        let mut prover = OneHotReadRaf::<F, D, NE>::new_prover(params, indices.clone());
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut verifier_t = VerifierTranscript::new("shout-read-raf", &narg);
        let vparams = OneHotReadRafParams::<F, D>::new(
            cfg.family,
            cfg.sumcheck_id,
            log_k_chunks,
            log_t,
            stages,
            &mut verifier_t,
        );
        let verifier = OneHotReadRaf::<F, D, NE>::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_k + log_t,
            degree: D + 1,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("shout read-raf must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for i in 0..D {
            let (pt, c) =
                prover_acc.get_committed_polynomial_opening((cfg.family)(i), cfg.sumcheck_id);
            verifier_acc.append_dense((cfg.family)(i), cfg.sumcheck_id, pt, c);
        }
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match (∏_i ra_i)·Σ γ^s·eq_s·Val_s"
        );

        // The cached ra_i openings equal the genuine one-hot MLE ra_i(r_k_i, r_cycle).
        for i in 0..D {
            let (pt, claim) =
                prover_acc.get_committed_polynomial_opening((cfg.family)(i), cfg.sumcheck_id);
            // pt = [r_k_i ‖ r_cycle]; evaluate the genuine one-hot column at it.
            let one_hot = one_hot_ra_column::<F>(&indices[i], log_k_chunks[i]);
            let eq_pt = EqPolynomial::<F>::evals(&pt.r, None);
            let genuine = one_hot
                .iter()
                .zip(eq_pt.iter())
                .fold(F::from_u64(0), |a, (v, e)| a + *v * *e);
            assert_eq!(claim, genuine, "cached ra_{i} == genuine one-hot MLE eval");
        }
    }

    #[test]
    fn bytecode_read_raf_round_trip() {
        let cfg = bytecode_config();
        round_trip::<2, 4>(&cfg, 0xB100, [1, 1], 2, 3);
        round_trip::<2, 4>(&cfg, 0xB101, [2, 2], 3, 3);
        round_trip::<2, 4>(&cfg, 0xB102, [1, 2], 4, 2);
        round_trip::<2, 4>(&cfg, 0xB103, [2, 1], 3, 1);
    }

    #[test]
    fn bytecode_read_raf_round_trip_d3_d4() {
        let cfg = bytecode_config();
        round_trip::<3, 5>(&cfg, 0xB300, [1, 2, 1], 3, 2);
        round_trip::<3, 5>(&cfg, 0xB301, [2, 1, 2], 2, 3);
        round_trip::<4, 6>(&cfg, 0xB400, [1, 1, 2, 1], 3, 2);
    }

    #[test]
    fn instruction_read_raf_round_trip() {
        let cfg = instruction_config();
        round_trip::<2, 4>(&cfg, 0x1100, [1, 1], 2, 3);
        round_trip::<2, 4>(&cfg, 0x1101, [2, 2], 3, 3);
        round_trip::<5, 7>(&cfg, 0x1500, [1, 1, 1, 1, 1], 3, 3);
        round_trip::<5, 7>(&cfg, 0x1501, [2, 1, 1, 2, 1], 2, 2);
    }

    #[test]
    fn tampered_proof_rejected() {
        let cfg = instruction_config();
        const D: usize = 2;
        const NE: usize = 4;
        let log_k_chunks = [2usize, 2usize];
        let log_t = 3usize;
        let mut rng = Rng(0x11FE);
        let k_dims: [usize; D] = std::array::from_fn(|i| 1usize << log_k_chunks[i]);
        let suffix = suffix_products(&k_dims);
        let k_total: usize = k_dims.iter().product();
        let t = 1usize << log_t;
        let indices: [Vec<u32>; D] = std::array::from_fn(|i| {
            (0..t)
                .map(|_| (rng.next() as u32) % (k_dims[i] as u32))
                .collect()
        });
        let combined: Vec<usize> = (0..t).map(|j| combine(&indices, j, &suffix)).collect();
        let r = rand_vec(&mut rng, log_t);
        let stages: Vec<ReadRafStage<F>> = (0..2)
            .map(|s| ReadRafStage {
                r_cycle: r.clone(),
                val_addr: rand_vec(&mut rng, k_total),
                rv_key: cfg.rv_keys[s],
            })
            .collect();
        let mut acc = Openings::<F>::new(log_t);
        for stage in &stages {
            let eq = EqPolynomial::<F>::evals(&stage.r_cycle, None);
            let rv = (0..t).fold(F::from_u64(0), |a, j| {
                a + eq[j] * stage.val_addr[combined[j]]
            });
            acc.append_virtual(
                stage.rv_key.0,
                stage.rv_key.1,
                OpeningPoint::new(stage.r_cycle.clone()),
                rv,
            );
        }
        let mut prover_t = ProverTranscript::new("t");
        let params = OneHotReadRafParams::<F, D>::new(
            cfg.family,
            cfg.sumcheck_id,
            log_k_chunks,
            log_t,
            stages.clone(),
            &mut prover_t,
        );
        let input_claim = params.input_claim(&acc);
        let mut prover = OneHotReadRaf::<F, D, NE>::new_prover(params, indices);
        let _ = prove(&mut prover, &mut acc, &mut prover_t);
        let mut narg = prover_t.into_proof();

        narg.narg_string[0] ^= 0x01;
        let claim = SumcheckClaim {
            num_vars: log_k_chunks[0] + log_k_chunks[1] + log_t,
            degree: D + 1,
            claimed_sum: input_claim,
        };
        let mut verifier_t = VerifierTranscript::new("t", &narg);
        // Replay the prover's pre-round γ squeeze to keep the verifier transcript aligned.
        let _ = OneHotReadRafParams::<F, D>::new(
            cfg.family,
            cfg.sumcheck_id,
            log_k_chunks,
            log_t,
            stages,
            &mut verifier_t,
        );
        assert!(
            verify(&claim, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }

    /// Drive the read-raf STAGE end-to-end: genuine per-chunk index columns → seed the genuine read
    /// sum → [`prove_read_raf`]/[`verify_read_raf`]. With `tamper`, corrupting a proof-carried `ra_i`
    /// opening must trip the output-claim check.
    fn stage_round_trip<const D: usize, const NE: usize>(
        cfg: &Config,
        seed: u64,
        log_k_chunks: [usize; D],
        log_t: usize,
        num_stages: usize,
        tamper: bool,
    ) {
        let mut rng = Rng(seed);
        let k_dims: [usize; D] = std::array::from_fn(|i| 1usize << log_k_chunks[i]);
        let suffix = suffix_products(&k_dims);
        let k_total: usize = k_dims.iter().product();
        let t = 1usize << log_t;

        let indices: [Vec<u32>; D] = std::array::from_fn(|i| {
            (0..t)
                .map(|_| (rng.next() as u32) % (k_dims[i] as u32))
                .collect()
        });
        let combined: Vec<usize> = (0..t).map(|j| combine(&indices, j, &suffix)).collect();

        let shared_r = rand_vec(&mut rng, log_t);
        let stages: Vec<ReadRafStage<F>> = (0..num_stages)
            .map(|s| ReadRafStage {
                r_cycle: if cfg.shared_cycle {
                    shared_r.clone()
                } else {
                    rand_vec(&mut rng, log_t)
                },
                val_addr: rand_vec(&mut rng, k_total),
                rv_key: cfg.rv_keys[s],
            })
            .collect();

        let seed_acc = |acc: &mut Openings<F>| {
            for stage in &stages {
                let eq = EqPolynomial::<F>::evals(&stage.r_cycle, None);
                let rv = (0..t).fold(F::from_u64(0), |a, j| {
                    a + eq[j] * stage.val_addr[combined[j]]
                });
                acc.append_virtual(
                    stage.rv_key.0,
                    stage.rv_key.1,
                    OpeningPoint::new(stage.r_cycle.clone()),
                    rv,
                );
            }
        };

        let inputs = || ReadRafInputs::<F, D> {
            ra_family: cfg.family,
            sumcheck_id: cfg.sumcheck_id,
            log_k_chunks,
            log_t,
            stages: stages.clone(),
        };

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut prover_t = ProverTranscript::new("read-raf-stage");
        let mut proof = prove_read_raf::<F, _, D, NE>(
            indices.clone(),
            inputs(),
            &mut prover_acc,
            &mut prover_t,
        );
        assert_eq!(proof.ra_openings.len(), D, "one cached opening per chunk");
        let narg = prover_t.into_proof();
        if tamper {
            proof.ra_openings[0] += F::from_u64(1);
        }

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut verifier_t = VerifierTranscript::new("read-raf-stage", &narg);
        let result =
            verify_read_raf::<F, _, D, NE>(&proof, inputs(), &mut verifier_acc, &mut verifier_t);
        if tamper {
            assert!(result.is_err(), "tampered ra_i opening must be rejected");
        } else {
            result.expect("read-raf stage must verify");
            for i in 0..D {
                let (pp, pc) =
                    prover_acc.get_committed_polynomial_opening((cfg.family)(i), cfg.sumcheck_id);
                let (vp, vc) =
                    verifier_acc.get_committed_polynomial_opening((cfg.family)(i), cfg.sumcheck_id);
                assert_eq!(pp, vp, "chunk {i} opening point agrees");
                assert_eq!(pc, vc, "chunk {i} opening claim agrees");
            }
        }
    }

    #[test]
    fn read_raf_stage_round_trip() {
        stage_round_trip::<2, 4>(&bytecode_config(), 0xB5A1, [2, 2], 4, 3, false);
        stage_round_trip::<3, 5>(&bytecode_config(), 0xB5A2, [1, 2, 1], 3, 2, false);
        stage_round_trip::<5, 7>(&instruction_config(), 0x15A1, [1, 1, 1, 1, 1], 4, 3, false);
    }

    #[test]
    fn read_raf_stage_tampered_opening_rejected() {
        stage_round_trip::<2, 4>(&bytecode_config(), 0xB5A9, [2, 2], 4, 2, true);
        stage_round_trip::<5, 7>(&instruction_config(), 0x15A9, [1, 1, 1, 1, 1], 3, 3, true);
    }
}
