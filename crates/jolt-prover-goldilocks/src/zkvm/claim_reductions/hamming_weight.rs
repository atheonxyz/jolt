//! Fused HammingWeight + RA-address-reduction sumcheck — ported from jolt-core's
//! `zkvm/claim_reductions/hamming_weight.rs` onto the framework ([`crate::framework`]) over the lean
//! `Field` (`C = F = Fp3`). jolt-core is the parity oracle.
//!
//! Operates on the per-chunk pushforward `G_i(k) = Σ_j eq(r_cycle, j)·ra_i(k, j)` (a poly over the
//! `log_k_chunk` address-chunk variables). Each `ra_i` carries three Stage-6 claims at the **shared**
//! cycle point `r_cycle` but different address points; this reduction fuses HammingWeight + the two
//! address reductions (Booleanity / Virtualization) into one degree-2 sumcheck collapsing all three
//! to a single `ra_i(ρ ‖ r_cycle)` opening:
//!
//! ```text
//! input:  Σ_i ( γ^{3i}·H_i + γ^{3i+1}·claim_bool_i + γ^{3i+2}·claim_virt_i )
//! sumcheck (log_k_chunk address rounds, degree 2):
//!   Σ_k Σ_i G_i(k)·( γ^{3i} + γ^{3i+1}·eq(r_addr_bool, k) + γ^{3i+2}·eq(r_addr_virt_i, k) )
//! output: ra_i(ρ ‖ r_cycle) for each i (SumcheckId::HammingWeightClaimReduction)
//! ```
//!
//! `H_i` is the expected hamming weight: `1` for instruction/bytecode (always one access per cycle),
//! the `RamHammingWeight` opening for RAM. The `G_i` columns are taken pre-materialized (`Fp3`),
//! decoupling from the trace → `compute_all_G` pushforward (M8). Single instance (no prefix/suffix).

use crate::framework::transcript::Challenge;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
    BIG_ENDIAN,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// The number of one-hot RA chunks per family (instruction, bytecode, ram).
#[derive(Clone, Copy, Debug)]
pub struct FamilyCounts {
    pub instruction_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,
}

impl FamilyCounts {
    fn polynomial_types(&self) -> Vec<CommittedPolynomial> {
        let mut types = Vec::with_capacity(self.instruction_d + self.bytecode_d + self.ram_d);
        types.extend((0..self.instruction_d).map(CommittedPolynomial::InstructionRa));
        types.extend((0..self.bytecode_d).map(CommittedPolynomial::BytecodeRa));
        types.extend((0..self.ram_d).map(CommittedPolynomial::RamRa));
        types
    }
}

/// The Stage-6 virtualization/read-raf sumcheck a poly's virt claim is opened by.
fn virt_sumcheck_id(poly: CommittedPolynomial) -> SumcheckId {
    match poly {
        CommittedPolynomial::InstructionRa(_) => SumcheckId::InstructionRaVirtualization,
        CommittedPolynomial::BytecodeRa(_) => SumcheckId::BytecodeReadRaf,
        CommittedPolynomial::RamRa(_) => SumcheckId::RamRaVirtualization,
        _ => unreachable!("hamming-weight reduction only handles RA families"),
    }
}

/// Fiat-Shamir + opening-point parameters, fetched from the accumulator (matches jolt-core
/// `HammingWeightClaimReductionParams`).
#[derive(Clone, Debug)]
pub struct HammingWeightClaimReductionParams<F: Field> {
    /// `[γ⁰, …, γ^{3N-1}]` (3 claims per RA chunk).
    pub gamma_powers: Vec<F>,
    pub r_cycle: Vec<F>,
    pub r_addr_bool: Vec<F>,
    pub r_addr_virt: Vec<Vec<F>>,
    pub claims_hw: Vec<F>,
    pub claims_bool: Vec<F>,
    pub claims_virt: Vec<F>,
    pub log_k_chunk: usize,
    pub polynomial_types: Vec<CommittedPolynomial>,
}

impl<F: Field> HammingWeightClaimReductionParams<F> {
    pub fn new(
        counts: FamilyCounts,
        log_k_chunk: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Challenge<F>,
    ) -> Self {
        let polynomial_types = counts.polynomial_types();
        let n = polynomial_types.len();

        let gamma = transcript.challenge();
        let mut gamma_powers = Vec::with_capacity(3 * n);
        let mut power = F::from_u64(1);
        for _ in 0..(3 * n) {
            gamma_powers.push(power);
            power *= gamma;
        }

        // r_addr_bool ‖ r_cycle from the shared Booleanity opening point.
        let (bool_point, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::Booleanity,
        );
        let r_addr_bool = bool_point.r[..log_k_chunk].to_vec();
        let r_cycle = bool_point.r[log_k_chunk..].to_vec();

        let ram_hw_factor = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .1;

        let mut r_addr_virt = Vec::with_capacity(n);
        let mut claims_hw = Vec::with_capacity(n);
        let mut claims_bool = Vec::with_capacity(n);
        let mut claims_virt = Vec::with_capacity(n);
        for &poly in &polynomial_types {
            let hw = if matches!(poly, CommittedPolynomial::RamRa(_)) {
                ram_hw_factor
            } else {
                F::from_u64(1)
            };
            claims_hw.push(hw);

            let (_, bool_claim) =
                accumulator.get_committed_polynomial_opening(poly, SumcheckId::Booleanity);
            claims_bool.push(bool_claim);

            let (virt_point, virt_claim) =
                accumulator.get_committed_polynomial_opening(poly, virt_sumcheck_id(poly));
            r_addr_virt.push(virt_point.r[..log_k_chunk].to_vec());
            claims_virt.push(virt_claim);
        }

        Self {
            gamma_powers,
            r_cycle,
            r_addr_bool,
            r_addr_virt,
            claims_hw,
            claims_bool,
            claims_virt,
            log_k_chunk,
            polynomial_types,
        }
    }

    fn input_claim(&self) -> F {
        (0..self.polynomial_types.len()).fold(F::zero(), |acc, i| {
            acc + self.gamma_powers[3 * i] * self.claims_hw[i]
                + self.gamma_powers[3 * i + 1] * self.claims_bool[i]
                + self.gamma_powers[3 * i + 2] * self.claims_virt[i]
        })
    }

    /// `[reverse(challenges) ‖ r_cycle]` — the full `(address, cycle)` opening point.
    fn opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_addr: Vec<F> = challenges.iter().rev().copied().collect();
        OpeningPoint::new([r_addr, self.r_cycle.clone()].concat())
    }
}

/// Prover/verifier instance. The prover holds the `N` pushforward `G_i` columns + the shared
/// `eq(r_addr_bool,·)` column + the `N` per-chunk `eq(r_addr_virt_i,·)` columns.
pub struct HammingWeightClaimReduction<F: Field> {
    pub params: HammingWeightClaimReductionParams<F>,
    g: Vec<MultilinearPolynomial<F>>,
    eq_bool: MultilinearPolynomial<F>,
    eq_virt: Vec<MultilinearPolynomial<F>>,
}

impl<F: Field> HammingWeightClaimReduction<F> {
    /// Build the prover instance from the `N` pushforward columns `G_i` (each length `2^log_k_chunk`),
    /// in the [`HammingWeightClaimReductionParams::polynomial_types`] order.
    pub fn new_prover(params: HammingWeightClaimReductionParams<F>, g: Vec<Vec<F>>) -> Self {
        let eq_bool = EqPolynomial::<F>::evals(&params.r_addr_bool, None);
        let eq_virt: Vec<MultilinearPolynomial<F>> = params
            .r_addr_virt
            .iter()
            .map(|r| MultilinearPolynomial::from(EqPolynomial::<F>::evals(r, None)))
            .collect();
        Self {
            params,
            g: g.into_iter().map(MultilinearPolynomial::from).collect(),
            eq_bool: MultilinearPolynomial::from(eq_bool),
            eq_virt,
        }
    }

    /// Build a verifier instance (no polynomials; `expected_output_claim` reads cached reduced
    /// openings + recomputes the eq factors).
    pub fn new_verifier(params: HammingWeightClaimReductionParams<F>) -> Self {
        Self {
            params,
            g: vec![MultilinearPolynomial::from(vec![F::zero()])],
            eq_bool: MultilinearPolynomial::from(vec![F::zero()]),
            eq_virt: vec![MultilinearPolynomial::from(vec![F::zero()])],
        }
    }
}

impl<F: Field> SumcheckInstance<F> for HammingWeightClaimReduction<F> {
    fn num_rounds(&self) -> usize {
        self.params.log_k_chunk
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.input_claim()
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let n = self.params.polynomial_types.len();
        let half = self.g[0].len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for j in 0..half {
            let eq_b = self
                .eq_bool
                .sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
            for i in 0..n {
                let g = self.g[i].sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
                let eq_v = self.eq_virt[i].sumcheck_evals_array::<3>(j, BindingOrder::LowToHigh);
                let gamma_hw = self.params.gamma_powers[3 * i];
                let gamma_bool = self.params.gamma_powers[3 * i + 1];
                let gamma_virt = self.params.gamma_powers[3 * i + 2];
                for k in 0..3 {
                    let weight = gamma_hw + gamma_bool * eq_b[k] + gamma_virt * eq_v[k];
                    acc[k].fmadd(g[k], weight);
                }
            }
        }
        let evals: [F; 3] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        for g in &mut self.g {
            g.bind_parallel(r, BindingOrder::LowToHigh);
        }
        self.eq_bool.bind_parallel(r, BindingOrder::LowToHigh);
        for eq in &mut self.eq_virt {
            eq.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.params.opening_point(challenges);
        for (i, &poly) in self.params.polynomial_types.iter().enumerate() {
            accumulator.append_dense(
                poly,
                SumcheckId::HammingWeightClaimReduction,
                point.clone(),
                self.g[i].final_sumcheck_claim(),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let rho_rev: Vec<F> = challenges.iter().rev().copied().collect();
        let eq_bool_eval = EqPolynomial::<F>::mle(&rho_rev, &self.params.r_addr_bool);
        self.params
            .polynomial_types
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (i, &poly)| {
                let eq_virt_eval = EqPolynomial::<F>::mle(&rho_rev, &self.params.r_addr_virt[i]);
                let (_, g_claim) = accumulator.get_committed_polynomial_opening(
                    poly,
                    SumcheckId::HammingWeightClaimReduction,
                );
                let gamma_hw = self.params.gamma_powers[3 * i];
                let gamma_bool = self.params.gamma_powers[3 * i + 1];
                let gamma_virt = self.params.gamma_powers[3 * i + 2];
                acc + g_claim * (gamma_hw + gamma_bool * eq_bool_eval + gamma_virt * eq_virt_eval)
            })
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

    fn dot(poly: &[F], eq: &[F]) -> F {
        poly.iter()
            .zip(eq.iter())
            .fold(F::from_u64(0), |a, (p, e)| a + *p * *e)
    }

    /// Build a valid pushforward `G_i(k) = Σ_{j: addr(j)=k} eq(r_cycle, j)` from per-cycle one-hot
    /// addresses, so `Σ_k G_i(k) = Σ_{accessed j} eq(r_cycle, j)` (the hamming weight). `addr[j] =
    /// None` means no access at cycle `j` (RAM).
    fn pushforward(addr: &[Option<usize>], r_cycle: &[F], k_dom: usize) -> (Vec<F>, F) {
        let eq_cycle = EqPolynomial::<F>::evals(r_cycle, None);
        let mut g = vec![F::from_u64(0); k_dom];
        let mut hw = F::from_u64(0);
        for (j, a) in addr.iter().enumerate() {
            if let Some(k) = a {
                g[*k] += eq_cycle[j];
                hw += eq_cycle[j];
            }
        }
        (g, hw)
    }

    /// Run the fused reduction for `(instruction_d, bytecode_d, ram_d)` chunks over `log_k_chunk`
    /// address rounds and a `log_t`-bit cycle point.
    fn round_trip(seed: u64, counts: FamilyCounts, log_k_chunk: usize, log_t: usize) {
        let mut rng = Rng(seed);
        let k_dom = 1usize << log_k_chunk;
        let t = 1usize << log_t;
        let n = counts.instruction_d + counts.bytecode_d + counts.ram_d;
        let polynomial_types = counts.polynomial_types();

        let r_cycle = rand_vec(&mut rng, log_t);
        let r_addr_bool = rand_vec(&mut rng, log_k_chunk);
        let eq_bool = EqPolynomial::<F>::evals(&r_addr_bool, None);

        // RAM chunks share ONE access mask (so every RAM chunk has the same hamming weight =
        // ram_hw_factor, matching how jolt-core assigns the shared RamHammingWeight to all RAM
        // chunks). Instruction/bytecode chunks access every cycle, so their HW is exactly 1.
        let ram_mask: Vec<bool> = (0..t).map(|_| !rng.next().is_multiple_of(3)).collect();
        let mut g_cols = Vec::with_capacity(n);
        let mut hw_vals = Vec::with_capacity(n);
        let mut r_addr_virt = Vec::with_capacity(n);
        for &poly in &polynomial_types {
            let is_ram = matches!(poly, CommittedPolynomial::RamRa(_));
            let addr: Vec<Option<usize>> = (0..t)
                .map(|j| {
                    let k = (rng.next() as usize) % k_dom;
                    if is_ram && !ram_mask[j] {
                        None
                    } else {
                        Some(k)
                    }
                })
                .collect();
            let (g, hw) = pushforward(&addr, &r_cycle, k_dom);
            g_cols.push(g);
            hw_vals.push(hw);
            r_addr_virt.push(rand_vec(&mut rng, log_k_chunk));
        }

        // ram_hw_factor is shared across RAM chunks (all share `ram_mask`, so equal HW).
        let ram_hw_factor = polynomial_types
            .iter()
            .position(|p| matches!(p, CommittedPolynomial::RamRa(_)))
            .map(|i| hw_vals[i]);

        let seed_acc = |acc: &mut Openings<F>| {
            // Shared Booleanity point [r_addr_bool ‖ r_cycle] on InstructionRa(0) (params reads it);
            // and per-poly bool/virt committed openings.
            for (i, &poly) in polynomial_types.iter().enumerate() {
                let bool_point = OpeningPoint::new([r_addr_bool.clone(), r_cycle.clone()].concat());
                acc.append_dense(
                    poly,
                    SumcheckId::Booleanity,
                    bool_point,
                    dot(&g_cols[i], &eq_bool),
                );
                let eq_v = EqPolynomial::<F>::evals(&r_addr_virt[i], None);
                let virt_point =
                    OpeningPoint::new([r_addr_virt[i].clone(), r_cycle.clone()].concat());
                acc.append_dense(
                    poly,
                    virt_sumcheck_id(poly),
                    virt_point,
                    dot(&g_cols[i], &eq_v),
                );
            }
            if let Some(hw) = ram_hw_factor {
                acc.append_virtual(
                    VirtualPolynomial::RamHammingWeight,
                    SumcheckId::RamHammingBooleanity,
                    OpeningPoint::new(r_cycle.clone()),
                    hw,
                );
            }
        };

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut prover_t = ProverTranscript::new("hamming-weight-claim-reduce");
        let params =
            HammingWeightClaimReductionParams::new(counts, log_k_chunk, &prover_acc, &mut prover_t);
        let input_claim = params.input_claim();
        let mut prover = HammingWeightClaimReduction::new_prover(params.clone(), g_cols.clone());
        let challenges = prove(&mut prover, &mut prover_acc, &mut prover_t);
        let narg = prover_t.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut verifier_t = VerifierTranscript::new("hamming-weight-claim-reduce", &narg);
        let vparams = HammingWeightClaimReductionParams::new(
            counts,
            log_k_chunk,
            &verifier_acc,
            &mut verifier_t,
        );
        let verifier = HammingWeightClaimReduction::new_verifier(vparams);
        let claim = SumcheckClaim {
            num_vars: log_k_chunk,
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let EvaluationClaim { point, value } =
            verify(&claim, &mut verifier_t).expect("hamming-weight reduction must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        for &poly in &polynomial_types {
            let (_, g_claim) = prover_acc
                .get_committed_polynomial_opening(poly, SumcheckId::HammingWeightClaimReduction);
            verifier_acc.append_dense(
                poly,
                SumcheckId::HammingWeightClaimReduction,
                OpeningPoint::new(point.clone()),
                g_claim,
            );
        }
        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(
            value, expected,
            "reduced claim must match the fused output formula"
        );
    }

    #[test]
    fn hamming_weight_claim_reduction_round_trip() {
        round_trip(
            0xD001,
            FamilyCounts {
                instruction_d: 2,
                bytecode_d: 1,
                ram_d: 1,
            },
            2,
            3,
        );
        round_trip(
            0xD002,
            FamilyCounts {
                instruction_d: 1,
                bytecode_d: 2,
                ram_d: 2,
            },
            3,
            4,
        );
    }
}
