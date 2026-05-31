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
//! with the one-hot read indicator `ra(k,j) = ∏_{i=0}^{d-1} ra_i(k_i, j)` (the d-chunk product).
//! - **Bytecode** uses per-stage cycle points `r_cycle_s` (distinct `eq_s`) and `Val_s` encoding
//!   circuit flags / RAF identity.
//! - **Instruction-lookups** is the special case where every stage shares `r_cycle = r_reduction`
//!   (a single `eq`), with `Val_s` ∈ {lookup-output table value, left/right operand}; the LHS is
//!   `rv + γ·left_op + γ²·right_op`.
//!
//! This port fixes **d = 2** (degree-3, the handoff's stated degree). The per-chunk `ra_i` leaf
//! openings `ra_i(r_addr_chunk_i, r_cycle)` are cached under `(ra_family(i), sumcheck_id)` — these
//! are exactly the §4.5.2 inputs the M7 LogUp\*-GKR pushforward consumes. The read-raf sumcheck
//! itself is unchanged by M7; only how the `ra_i` leaves are committed/opened changes (one-hot →
//! `ra_dense` + pushforward-GKR). See the `m7-logupstar-readraf-relationship` design note.
//!
//! **Decoupled from the trace** (the M5 convention): takes the materialized one-hot chunk columns
//! (broadcast to the full hypercube for uniform single-phase binding) + the public per-stage
//! `(eq cycle point, Val address column)`. Deferred: jolt-core's prefix/suffix + two-phase
//! address-then-cycle materialization, the Gruen split-eq, the entry-point constraint, the
//! flag/lookup-table-specific `Val_s` construction (incl. multi-table selection and the wide-limb
//! range-check stages that fold in here per design §4.2), and the d-chunk one-hot *commitment*.

use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

/// d = 2 one-hot address chunks ⇒ cycle rounds are degree `d + 1 = 3`.
pub const NUM_CHUNKS: usize = 2;
const DEGREE: usize = NUM_CHUNKS + 1;

/// One batched stage: a per-stage cycle-eq point `r_cycle`, the public address-only value column
/// `val_addr` (length `K`), and the accumulator key of the upstream read claim `rv_s`.
#[derive(Clone, Debug)]
pub struct ReadRafStage<F: Field> {
    pub r_cycle: Vec<F>,
    pub val_addr: Vec<F>,
    pub rv_key: (VirtualPolynomial, SumcheckId),
}

/// Batching/opening parameters, parameterized by the committed RA family and the sumcheck id so
/// both the bytecode and instruction-lookups ports share one implementation.
#[derive(Clone, Debug)]
pub struct OneHotReadRafParams<F: Field> {
    /// `[γ^0, …, γ^{S-1}]` for `S` stages.
    pub gamma_powers: Vec<F>,
    /// Address-chunk bit widths `[log_K_0, log_K_1]`.
    pub log_k_chunks: [usize; NUM_CHUNKS],
    pub log_t: usize,
    pub stages: Vec<ReadRafStage<F>>,
    /// Maps chunk index → its committed RA polynomial (e.g. `CommittedPolynomial::BytecodeRa`).
    pub ra_family: fn(usize) -> CommittedPolynomial,
    /// The sumcheck id the `ra_i` leaf openings are cached under.
    pub sumcheck_id: SumcheckId,
}

impl<F: Field> OneHotReadRafParams<F> {
    /// Draws `γ` and forms `num_stages` powers.
    pub fn new(
        ra_family: fn(usize) -> CommittedPolynomial,
        sumcheck_id: SumcheckId,
        log_k_chunks: [usize; NUM_CHUNKS],
        log_t: usize,
        stages: Vec<ReadRafStage<F>>,
        transcript: &mut impl Transcript<Challenge = F>,
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
        self.log_k_chunks[0] + self.log_k_chunks[1]
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

/// Prover/verifier instance. The prover holds the broadcast chunk + stage columns; the verifier
/// keeps the per-stage `r_cycle`/`val_addr` (public, in `params`) to recompute `eq_s(ρ)`/`Val_s(ρ)`.
pub struct OneHotReadRaf<F: Field> {
    pub params: OneHotReadRafParams<F>,
    ra: [MultilinearPolynomial<F>; NUM_CHUNKS],
    eq_full: Vec<MultilinearPolynomial<F>>,
    val_full: Vec<MultilinearPolynomial<F>>,
}

impl<F: Field> OneHotReadRaf<F> {
    /// `ra_chunks[i]` is the one-hot column over `(chunk_i ∈ [0,2^{log_K_i}), cycle)` (length
    /// `2^{log_K_i}·T`). Broadcast to the full `K·T` hypercube (index `(k_0·K_1 + k_1)·T + j`).
    pub fn new_prover(params: OneHotReadRafParams<F>, ra_chunks: [Vec<F>; NUM_CHUNKS]) -> Self {
        let t = 1usize << params.log_t;
        let k0 = 1usize << params.log_k_chunks[0];
        let k1 = 1usize << params.log_k_chunks[1];
        let n = k0 * k1 * t;

        let ra0_full: Vec<F> = (0..n)
            .map(|idx| {
                let j = idx % t;
                let k0i = (idx / t) / k1;
                ra_chunks[0][k0i * t + j]
            })
            .collect();
        let ra1_full: Vec<F> = (0..n)
            .map(|idx| {
                let j = idx % t;
                let k1i = (idx / t) % k1;
                ra_chunks[1][k1i * t + j]
            })
            .collect();

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
            ra: [
                MultilinearPolynomial::from(ra0_full),
                MultilinearPolynomial::from(ra1_full),
            ],
            eq_full,
            val_full,
        }
    }

    pub fn new_verifier(params: OneHotReadRafParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::zero()]);
        Self {
            params,
            ra: [dummy(), dummy()],
            eq_full: vec![],
            val_full: vec![],
        }
    }
}

impl<F: Field> SumcheckInstance<F> for OneHotReadRaf<F> {
    fn num_rounds(&self) -> usize {
        self.params.log_k() + self.params.log_t
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-3: ra0·ra1·(Σ_s γ^s·eq_s·Val_s) ⇒ 4 evaluation points (0,1,2,3).
        let half = self.ra[0].len() / 2;
        let mut evals = [F::zero(); DEGREE + 1];
        for idx in 0..half {
            let ra0 = self.ra[0].sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let ra1 = self.ra[1].sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
            let mut stage_sum = [F::zero(); DEGREE + 1];
            for (s, &g) in self.params.gamma_powers.iter().enumerate() {
                let eq = self.eq_full[s].sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
                let val = self.val_full[s].sumcheck_evals_array::<4>(idx, BindingOrder::LowToHigh);
                for p in 0..=DEGREE {
                    stage_sum[p] += g * eq[p] * val[p];
                }
            }
            for p in 0..=DEGREE {
                evals[p] += ra0[p] * ra1[p] * stage_sum[p];
            }
        }
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.ra[0].bind_parallel(r, BindingOrder::LowToHigh);
        self.ra[1].bind_parallel(r, BindingOrder::LowToHigh);
        for poly in self.eq_full.iter_mut().chain(self.val_full.iter_mut()) {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        let (r_addr, r_cycle) = point.split_at(self.params.log_k());
        let (r_k0, r_k1) = r_addr.split_at(self.params.log_k_chunks[0]);
        let chunk_point = |r_chunk: &OpeningPoint<BIG_ENDIAN, F>| {
            OpeningPoint::new([r_chunk.r.as_slice(), r_cycle.r.as_slice()].concat())
        };
        accumulator.append_dense(
            (self.params.ra_family)(0),
            self.params.sumcheck_id,
            chunk_point(&r_k0),
            self.ra[0].final_sumcheck_claim(),
        );
        accumulator.append_dense(
            (self.params.ra_family)(1),
            self.params.sumcheck_id,
            chunk_point(&r_k1),
            self.ra[1].final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let (r_addr, r_cycle) = point.split_at(self.params.log_k());

        let (_, ra0) = accumulator
            .get_committed_polynomial_opening((self.params.ra_family)(0), self.params.sumcheck_id);
        let (_, ra1) = accumulator
            .get_committed_polynomial_opening((self.params.ra_family)(1), self.params.sumcheck_id);

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
        ra0 * ra1 * stage_sum
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::sumcheck::{prove, verify};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_sumcheck::{EvaluationClaim, SumcheckClaim};
    use jolt_transcript::{Blake2bTranscript, Transcript};

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

    fn round_trip(
        cfg: &Config,
        seed: u64,
        log_k0: usize,
        log_k1: usize,
        log_t: usize,
        num_stages: usize,
    ) {
        let mut rng = Rng(seed);
        let k0 = 1usize << log_k0;
        let k1 = 1usize << log_k1;
        let k = k0 * k1;
        let t = 1usize << log_t;

        let ra0 = rand_vec(&mut rng, k0 * t);
        let ra1 = rand_vec(&mut rng, k1 * t);

        // Shared cycle point (instruction) or distinct per stage (bytecode).
        let shared_r = rand_vec(&mut rng, log_t);
        let stages: Vec<ReadRafStage<F>> = (0..num_stages)
            .map(|s| ReadRafStage {
                r_cycle: if cfg.shared_cycle {
                    shared_r.clone()
                } else {
                    rand_vec(&mut rng, log_t)
                },
                val_addr: rand_vec(&mut rng, k),
                rv_key: cfg.rv_keys[s],
            })
            .collect();

        let seed_acc = |acc: &mut Openings<F>| {
            for stage in &stages {
                let eq_cycle = EqPolynomial::<F>::evals(&stage.r_cycle, None);
                let mut rv = F::from_u64(0);
                for k0i in 0..k0 {
                    for k1i in 0..k1 {
                        let kk = k0i * k1 + k1i;
                        for j in 0..t {
                            let ra = ra0[k0i * t + j] * ra1[k1i * t + j];
                            rv += ra * eq_cycle[j] * stage.val_addr[kk];
                        }
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

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut prover_t = Blake2bTranscript::<F>::new(b"shout-read-raf");
        let params = OneHotReadRafParams::new(
            cfg.family,
            cfg.sumcheck_id,
            [log_k0, log_k1],
            log_t,
            stages.clone(),
            &mut prover_t,
        );
        let input_claim = params.input_claim(&prover_acc);
        let mut prover = OneHotReadRaf::new_prover(params, [ra0.clone(), ra1.clone()]);
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut verifier_t = Blake2bTranscript::<F>::new(b"shout-read-raf");
        let vparams = OneHotReadRafParams::new(
            cfg.family,
            cfg.sumcheck_id,
            [log_k0, log_k1],
            log_t,
            stages,
            &mut verifier_t,
        );
        let verifier = OneHotReadRaf::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_k0 + log_k1 + log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("shout read-raf must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for i in 0..NUM_CHUNKS {
            let (pt, c) =
                prover_acc.get_committed_polynomial_opening((cfg.family)(i), cfg.sumcheck_id);
            verifier_acc.append_dense((cfg.family)(i), cfg.sumcheck_id, pt, c);
        }
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match ra0·ra1·Σ γ^s·eq_s·Val_s"
        );
    }

    #[test]
    fn bytecode_read_raf_round_trip() {
        let cfg = bytecode_config();
        round_trip(&cfg, 0xB100, 1, 1, 2, 3);
        round_trip(&cfg, 0xB101, 2, 2, 3, 3);
        round_trip(&cfg, 0xB102, 1, 2, 4, 2);
        round_trip(&cfg, 0xB103, 2, 1, 3, 1);
    }

    #[test]
    fn instruction_read_raf_round_trip() {
        let cfg = instruction_config();
        round_trip(&cfg, 0x1100, 1, 1, 2, 3);
        round_trip(&cfg, 0x1101, 2, 2, 3, 3);
        round_trip(&cfg, 0x1102, 2, 2, 4, 2);
        round_trip(&cfg, 0x1103, 1, 2, 3, 1);
    }

    #[test]
    fn tampered_proof_rejected() {
        let cfg = instruction_config();
        let (log_k0, log_k1, log_t) = (2, 2, 3);
        let mut rng = Rng(0x11FE);
        let k0 = 1usize << log_k0;
        let k1 = 1usize << log_k1;
        let t = 1usize << log_t;
        let ra0 = rand_vec(&mut rng, k0 * t);
        let ra1 = rand_vec(&mut rng, k1 * t);
        let r = rand_vec(&mut rng, log_t);
        let stages: Vec<ReadRafStage<F>> = (0..2)
            .map(|s| ReadRafStage {
                r_cycle: r.clone(),
                val_addr: rand_vec(&mut rng, k0 * k1),
                rv_key: cfg.rv_keys[s],
            })
            .collect();
        let mut acc = Openings::<F>::new(log_t);
        for stage in &stages {
            let eq_cycle = EqPolynomial::<F>::evals(&stage.r_cycle, None);
            let mut rv = F::from_u64(0);
            for k0i in 0..k0 {
                for k1i in 0..k1 {
                    let kk = k0i * k1 + k1i;
                    for j in 0..t {
                        rv +=
                            ra0[k0i * t + j] * ra1[k1i * t + j] * eq_cycle[j] * stage.val_addr[kk];
                    }
                }
            }
            acc.append_virtual(
                stage.rv_key.0,
                stage.rv_key.1,
                OpeningPoint::new(stage.r_cycle.clone()),
                rv,
            );
        }
        let mut prover_t = Blake2bTranscript::<F>::new(b"t");
        let params = OneHotReadRafParams::new(
            cfg.family,
            cfg.sumcheck_id,
            [log_k0, log_k1],
            log_t,
            stages,
            &mut prover_t,
        );
        let input_claim = params.input_claim(&acc);
        let mut prover = OneHotReadRaf::new_prover(params, [ra0, ra1]);
        let (mut proof, _) = prove(&mut prover, &mut acc, &mut prover_t);

        proof.round_polynomials[0] = UnivariatePoly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let claim = SumcheckClaim {
            num_vars: log_k0 + log_k1 + log_t,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"t");
        assert!(
            verify(&claim, &proof, &mut verifier_t).is_err(),
            "tampered proof must be rejected"
        );
    }
}
