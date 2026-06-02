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
//! with the one-hot read indicator `ra(k,j) = ∏_{i=0}^{D-1} ra_i(k_i, j)` (the `D`-chunk product).
//! - **Bytecode** (`D = 2`) uses per-stage cycle points `r_cycle_s` (distinct `eq_s`) and `Val_s`
//!   encoding circuit flags / RAF identity.
//! - **Instruction-lookups** (`D = 5`) is the special case where every stage shares
//!   `r_cycle = r_reduction` (a single `eq`), with `Val_s` ∈ {lookup-output table value, left/right
//!   operand}; the LHS is `rv + γ·left_op + γ²·right_op`.
//!
//! ## Const-generic `D`
//!
//! Generalized over the number of address chunks via **two** const params
//! [`OneHotReadRaf<F, D, NE>`] with `NE = D + 2` (stable Rust cannot evaluate `D + 2` inside a
//! `sumcheck_evals_array::<{D+2}>` turbofish — generic-const-expr is nightly — so `NE` is threaded
//! explicitly, mirroring [`crate::framework::univariate_skip`]'s multi-const pattern).
//! [`OneHotReadRafParams<F, D>`] carries `log_k_chunks: [usize; D]`.
//!
//! **Degree = `D + 1`.** In a cycle round the product `∏_i ra_i · eq` has degree `D + 1` (all `D`
//! `ra_i` plus `eq` are non-constant in the bound cycle bit; `val` is address-only). In an address
//! round only one `ra_i` and `val` are non-constant (degree 2). Declaring the uniform bound `D + 1`
//! over-states the address rounds (their round poly's high coefficients are zero, written padded by
//! [`crate::framework::sumcheck::write_round_poly`]). `NE = D + 2` evaluation points interpolate the
//! degree-`(D+1)` round poly exactly.
//!
//! The per-chunk `ra_i` leaf openings `ra_i(r_addr_chunk_i, r_cycle)` are cached under
//! `(ra_family(i), sumcheck_id)` — these are exactly the §4.5.2 inputs the M7 LogUp\*-GKR pushforward
//! consumes. The read-raf sumcheck itself is unchanged by M7; only how the `ra_i` leaves are
//! committed/opened changes (one-hot → `ra_dense` + pushforward-GKR). See the
//! `m7-logupstar-readraf-relationship` design note.
//!
//! **Decoupled from the trace** (the M5 convention): takes the materialized one-hot chunk columns
//! (broadcast to the full hypercube for uniform single-phase binding) + the public per-stage
//! `(eq cycle point, Val address column)`. Deferred: jolt-core's prefix/suffix + two-phase
//! address-then-cycle materialization, the Gruen split-eq, the entry-point constraint, the
//! flag/lookup-table-specific `Val_s` construction (incl. multi-table selection and the wide-limb
//! range-check stages that fold in here per design §4.2), and the `D`-chunk one-hot *commitment*.

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

/// One batched stage: a per-stage cycle-eq point `r_cycle`, the public address-only value column
/// `val_addr` (length `K = ∏_i 2^{log_K_i}`), and the accumulator key of the upstream read claim
/// `rv_s`.
#[derive(Clone, Debug)]
pub struct ReadRafStage<F: Field> {
    pub r_cycle: Vec<F>,
    pub val_addr: Vec<F>,
    pub rv_key: (VirtualPolynomial, SumcheckId),
}

/// Batching/opening parameters, parameterized by the chunk count `D`, the committed RA family, and
/// the sumcheck id so the bytecode (`D = 2`) and instruction-lookups (`D = 5`) ports share one
/// implementation.
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

/// Mixed-radix suffix products `[∏_{l>0} K_l, …, ∏_{l>D-2} K_l, 1]` for decomposing a flat address
/// `addr = Σ_i k_i · suffix[i]` (chunk 0 most significant): `k_i = (addr / suffix[i]) % K_i`.
#[inline]
fn suffix_products<const D: usize>(k_dims: &[usize; D]) -> [usize; D] {
    let mut suffix = [1usize; D];
    for i in (1..D).rev() {
        suffix[i - 1] = suffix[i] * k_dims[i];
    }
    suffix
}

/// Prover/verifier instance. The prover holds the broadcast chunk + stage columns; the verifier
/// keeps the per-stage `r_cycle`/`val_addr` (public, in `params`) to recompute `eq_s(ρ)`/`Val_s(ρ)`.
///
/// `NE = D + 2` is the number of round-polynomial evaluation points (interpolating the degree-`(D+1)`
/// round message); threaded as a separate const because stable Rust cannot compute it in a turbofish.
pub struct OneHotReadRaf<F: Field, const D: usize, const NE: usize> {
    pub params: OneHotReadRafParams<F, D>,
    ra: [MultilinearPolynomial<F>; D],
    eq_full: Vec<MultilinearPolynomial<F>>,
    val_full: Vec<MultilinearPolynomial<F>>,
}

impl<F: Field, const D: usize, const NE: usize> OneHotReadRaf<F, D, NE> {
    /// `ra_chunks[i]` is the one-hot column over `(chunk_i ∈ [0,2^{log_K_i}), cycle)` (length
    /// `2^{log_K_i}·T`). Broadcast to the full `K·T` hypercube via the mixed-radix index
    /// `(…((k_0·K_1 + k_1)·K_2 + k_2)…)·T + j` (chunk 0 most significant).
    pub fn new_prover(params: OneHotReadRafParams<F, D>, ra_chunks: [Vec<F>; D]) -> Self {
        debug_assert_eq!(NE, D + 2, "NE must equal D + 2");
        let t = 1usize << params.log_t;
        let k_dims: [usize; D] = std::array::from_fn(|i| 1usize << params.log_k_chunks[i]);
        let k_total: usize = k_dims.iter().product();
        let n = k_total * t;
        let suffix = suffix_products(&k_dims);

        let ra: [MultilinearPolynomial<F>; D] = std::array::from_fn(|i| {
            let col: Vec<F> = (0..n)
                .map(|idx| {
                    let j = idx % t;
                    let addr = idx / t;
                    let k_i = (addr / suffix[i]) % k_dims[i];
                    ra_chunks[i][k_i * t + j]
                })
                .collect();
            MultilinearPolynomial::from(col)
        });

        let mut eq_full = Vec::with_capacity(params.stages.len());
        let mut val_full = Vec::with_capacity(params.stages.len());
        for stage in &params.stages {
            let eq_cycle = EqPolynomial::<F>::evals(&stage.r_cycle, None);
            let eqf: Vec<F> = (0..n).map(|idx| eq_cycle[idx % t]).collect();
            let valf: Vec<F> = (0..n).map(|idx| stage.val_addr[idx / t]).collect();
            eq_full.push(MultilinearPolynomial::from(eqf));
            val_full.push(MultilinearPolynomial::from(valf));
        }

        Self {
            params,
            ra,
            eq_full,
            val_full,
        }
    }

    pub fn new_verifier(params: OneHotReadRafParams<F, D>) -> Self {
        debug_assert_eq!(NE, D + 2, "NE must equal D + 2");
        let ra: [MultilinearPolynomial<F>; D] =
            std::array::from_fn(|_| MultilinearPolynomial::from(vec![F::zero()]));
        Self {
            params,
            ra,
            eq_full: vec![],
            val_full: vec![],
        }
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

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-(D+1): (∏_i ra_i)·(Σ_s γ^s·eq_s·Val_s) ⇒ NE = D+2 evaluation points (0..=D+1).
        let half = self.ra[0].len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); NE];
        for idx in 0..half {
            let mut ra_prod = [F::one(); NE];
            for chunk in &self.ra {
                let evals = chunk.sumcheck_evals_array::<NE>(idx, BindingOrder::LowToHigh);
                for (acc_p, &e) in ra_prod.iter_mut().zip(evals.iter()) {
                    *acc_p *= e;
                }
            }
            let mut stage_sum = [F::zero(); NE];
            for (s, &g) in self.params.gamma_powers.iter().enumerate() {
                let eq = self.eq_full[s].sumcheck_evals_array::<NE>(idx, BindingOrder::LowToHigh);
                let val = self.val_full[s].sumcheck_evals_array::<NE>(idx, BindingOrder::LowToHigh);
                for (slot, (&e, &v)) in stage_sum.iter_mut().zip(eq.iter().zip(val.iter())) {
                    *slot += g * e * v;
                }
            }
            for (acc_p, (&rp, &ss)) in acc.iter_mut().zip(ra_prod.iter().zip(stage_sum.iter())) {
                acc_p.fmadd(rp, ss);
            }
        }
        let evals: [F; NE] = std::array::from_fn(|p| acc[p].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        for chunk in &mut self.ra {
            chunk.bind_parallel(r, BindingOrder::LowToHigh);
        }
        for poly in self.eq_full.iter_mut().chain(self.val_full.iter_mut()) {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        let (r_addr, r_cycle) = point.split_at(self.params.log_k());
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
        let (r_addr, r_cycle) = point.split_at(self.params.log_k());

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

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use crate::framework::sumcheck::{prove, verify};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};

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

    /// `addr = Σ_i k_i · suffix[i]` reconstruction of the mixed-radix index used by `new_prover`.
    fn chunk_index<const D: usize>(
        addr: usize,
        i: usize,
        suffix: &[usize; D],
        k_dims: &[usize; D],
    ) -> usize {
        (addr / suffix[i]) % k_dims[i]
    }

    /// Generic round-trip over `D` chunks (`NE = D + 2`). Builds random one-hot chunk columns +
    /// per-stage `(r_cycle, val_addr)`, seeds the `rv_s` accumulator from the explicit hypercube sum,
    /// proves the read-raf sumcheck, and checks the reduced claim closes against
    /// `(∏_i ra_i)·Σ_s γ^s·eq_s·Val_s`.
    fn round_trip<const D: usize, const NE: usize>(
        cfg: &Config,
        seed: u64,
        log_k_chunks: [usize; D],
        log_t: usize,
        num_stages: usize,
    ) {
        let mut rng = Rng(seed);
        let k_dims: [usize; D] = std::array::from_fn(|i| 1usize << log_k_chunks[i]);
        let suffix = super::suffix_products(&k_dims);
        let k_total: usize = k_dims.iter().product();
        let t = 1usize << log_t;

        let ra_chunks: [Vec<F>; D] = std::array::from_fn(|i| rand_vec(&mut rng, k_dims[i] * t));

        // Shared cycle point (instruction) or distinct per stage (bytecode).
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
                let eq_cycle = EqPolynomial::<F>::evals(&stage.r_cycle, None);
                let mut rv = F::from_u64(0);
                for addr in 0..k_total {
                    for j in 0..t {
                        let mut ra = F::from_u64(1);
                        for (i, chunk) in ra_chunks.iter().enumerate() {
                            let k_i = chunk_index(addr, i, &suffix, &k_dims);
                            ra *= chunk[k_i * t + j];
                        }
                        rv += ra * eq_cycle[j] * stage.val_addr[addr];
                    }
                }
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
        let degree = D + 1;
        let input_claim = params.input_claim(&prover_acc);
        let mut prover = OneHotReadRaf::<F, D, NE>::new_prover(params, ra_chunks.clone());
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
            degree,
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
    }

    #[test]
    fn bytecode_read_raf_round_trip() {
        // Bytecode: D = 2 (NE = 4).
        let cfg = bytecode_config();
        round_trip::<2, 4>(&cfg, 0xB100, [1, 1], 2, 3);
        round_trip::<2, 4>(&cfg, 0xB101, [2, 2], 3, 3);
        round_trip::<2, 4>(&cfg, 0xB102, [1, 2], 4, 2);
        round_trip::<2, 4>(&cfg, 0xB103, [2, 1], 3, 1);
    }

    #[test]
    fn instruction_read_raf_round_trip_d2() {
        // Instruction-lookups exercised at D = 2 (NE = 4).
        let cfg = instruction_config();
        round_trip::<2, 4>(&cfg, 0x1100, [1, 1], 2, 3);
        round_trip::<2, 4>(&cfg, 0x1101, [2, 2], 3, 3);
        round_trip::<2, 4>(&cfg, 0x1102, [2, 2], 4, 2);
        round_trip::<2, 4>(&cfg, 0x1103, [1, 2], 3, 1);
    }

    #[test]
    fn instruction_read_raf_round_trip_d5() {
        // Instruction-lookups at the production D = 5 (NE = 7): 5 one-bit chunks (K = 32) over a
        // shared cycle eq, 3 batched stages.
        let cfg = instruction_config();
        round_trip::<5, 7>(&cfg, 0x1500, [1, 1, 1, 1, 1], 3, 3);
        round_trip::<5, 7>(&cfg, 0x1501, [2, 1, 1, 2, 1], 2, 2);
        round_trip::<5, 7>(&cfg, 0x1502, [1, 2, 1, 1, 1], 4, 1);
    }

    /// A third chunk count (`D = 3`, `NE = 5`) to guard the const-generic broadcast/cache split at a
    /// value neither caller uses.
    #[test]
    fn bytecode_read_raf_round_trip_d3() {
        let cfg = bytecode_config();
        round_trip::<3, 5>(&cfg, 0xB300, [1, 2, 1], 3, 2);
        round_trip::<3, 5>(&cfg, 0xB301, [2, 1, 2], 2, 3);
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
        let suffix = super::suffix_products(&k_dims);
        let k_total: usize = k_dims.iter().product();
        let t = 1usize << log_t;
        let ra_chunks: [Vec<F>; D] = std::array::from_fn(|i| rand_vec(&mut rng, k_dims[i] * t));
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
            let eq_cycle = EqPolynomial::<F>::evals(&stage.r_cycle, None);
            let mut rv = F::from_u64(0);
            for addr in 0..k_total {
                for j in 0..t {
                    let mut ra = F::from_u64(1);
                    for (i, chunk) in ra_chunks.iter().enumerate() {
                        let k_i = chunk_index(addr, i, &suffix, &k_dims);
                        ra *= chunk[k_i * t + j];
                    }
                    rv += ra * eq_cycle[j] * stage.val_addr[addr];
                }
            }
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
        let mut prover = OneHotReadRaf::<F, D, NE>::new_prover(params, ra_chunks);
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
}
