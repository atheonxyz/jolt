//! RAM RA virtualization sumcheck (ST3a) — ported from jolt-core's `zkvm/ram/ra_virtual.rs`. jolt-core
//! is the parity oracle.
//!
//! Decomposes the single consolidated dense-`RamRa` opening produced by
//! [`super::super::claim_reductions::RamRaClaimReduction`] (`RamRa(r_address ‖ ρ)@RamRaClaimReduction`)
//! into the `D` committed per-chunk `RamRa(i)` openings — the inputs the M7 per-chunk pushforward
//! (`prove_read_raf_pushforward`) discharges to `RaDense`/`Pushforward` at stage 8, exactly as the
//! bytecode/instruction read-raf families do. The RAM read/write *checking* stays dense — only the RA
//! *opening* is virtualized, matching jolt-core.
//!
//! **WIRING PREREQUISITE (the remaining dense-`RamRa` gap, ST3b).** This reduction is sound for any
//! `ra` where the dense one-hot `ra(r_address, c)` factorizes as `Π_i ra_i(r_address_i, c)` over the
//! committed chunks (the unit tests below exercise that case). But the goldilocks dense RAM witness
//! ([`super::witness::ram_witness`]) encodes a **non-access cycle as all-zero** `ra`, while the
//! committed per-chunk one-hot ([`super::super::witness`]) encodes it as **chunk index 0** (one-hot at
//! address 0) — so `Π_i ra_i[c] = eq(r_address, 0) ≠ 0` there, and the factorization fails on
//! non-access cycles (most RAM cycles). jolt-core avoids this because its RAM `ra` is one-hot at a
//! sentinel for *every* cycle (dense == committed). Wiring this into the real-trace e2e therefore
//! first needs the dense RAM model reconciled with the committed one (a reserved no-access sentinel,
//! or an access-flag factor) — the "RAM-via-read-raf" reconciliation, deferred.
//!
//! ## Identity (log_T cycle rounds, degree `D + 1`)
//! ```text
//! Σ_c eq(ρ, c) · Π_{i=0}^{D-1} ra_i(r_address_i, c) = ra_claim
//! ```
//! where `ra_i(r_address_i, c) = eq(r_address_i, idx_i(c))` is the i-th committed chunk one-hot, and
//! `r_address` (length `log_k_ram`) splits into `D` chunks of `log_m = log_k_chunk` bits via the
//! jolt-core zero-pad-prepend convention ([`r_address_chunks`]) so a non-multiple `log_k_ram` aligns
//! with the fixed-width committed chunks (the always-zero high bits match the prepended zeros).
//!
//! Caches `D` committed `RamRa(i)(r_address_i ‖ r_cycle_final)@RamRaVirtualization` openings.

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckClaim;

use crate::framework::accumulator::{
    CommittedPolynomial, OpeningAccumulator, OpeningPoint, Openings, SumcheckId, VirtualPolynomial,
};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::{prove, verify, SumcheckInstance};
use crate::framework::transcript::{ProverFs, VerifierFs};

/// Zero-pad-prepend `r_address` to `D · chunk_bits` coordinates (MSB side), then split into `D`
/// chunks of `chunk_bits`. Mirrors jolt-core `OneHotParams::compute_r_address_chunks`: a non-multiple
/// `log_k_ram` gets high-side zeros so it aligns with the fixed-width committed chunks (whose
/// top-chunk high bits are always zero).
pub fn r_address_chunks<F: Field, const D: usize>(
    r_address: &[F],
    chunk_bits: usize,
) -> [Vec<F>; D] {
    let total = D * chunk_bits;
    debug_assert!(
        r_address.len() <= total,
        "r_address wider than D·chunk_bits"
    );
    let mut padded = vec![F::from_u64(0); total - r_address.len()];
    padded.extend_from_slice(r_address);
    std::array::from_fn(|i| padded[i * chunk_bits..(i + 1) * chunk_bits].to_vec())
}

/// Parameters: the reduced cycle point `ρ`, the `D` address-chunk points, and the consolidated
/// `RamRa@RamRaClaimReduction` claim (the sumcheck's input).
#[derive(Clone, Debug)]
pub struct RamRaVirtualizationParams<F: Field> {
    pub log_t: usize,
    pub r_cycle: Vec<F>,
    pub r_address_chunks: Vec<Vec<F>>,
    pub ra_claim: F,
}

impl<F: Field> RamRaVirtualizationParams<F> {
    /// Read the consolidated `RamRa(r_address ‖ ρ)@RamRaClaimReduction` opening and split it into the
    /// `D` chunk address points (`log_m = chunk_bits` each) + the cycle point `ρ`.
    pub fn new<const D: usize>(
        accumulator: &dyn OpeningAccumulator<F>,
        log_k: usize,
        log_t: usize,
        chunk_bits: usize,
    ) -> Self {
        let (point, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
        );
        let (r_address, r_cycle) = point.split_at(log_k);
        let chunks = r_address_chunks::<F, D>(&r_address.r, chunk_bits);
        Self {
            log_t,
            r_cycle: r_cycle.r,
            r_address_chunks: chunks.to_vec(),
            ra_claim,
        }
    }
}

/// Prover/verifier instance. Degree `D + 1` (`D` one-hot factors + the `eq(ρ,·)` factor).
pub struct RamRaVirtualization<F: Field, const D: usize, const NE: usize> {
    params: RamRaVirtualizationParams<F>,
    ra: Vec<MultilinearPolynomial<F>>,
    eq: MultilinearPolynomial<F>,
}

impl<F: Field, const D: usize, const NE: usize> RamRaVirtualization<F, D, NE> {
    /// Build the prover instance. `indices[i]` is the i-th committed RAM chunk-index column
    /// (`ra_dense[ram_range.start + i].indices`, values `< 2^chunk_bits`, length `T`). Each `ra_i`
    /// column is `eq(r_address_chunk_i, idx_i[c])`.
    pub fn new_prover(params: RamRaVirtualizationParams<F>, indices: &[Vec<u32>; D]) -> Self {
        debug_assert_eq!(NE, D + 2, "NE must equal D + 2");
        let t = 1usize << params.log_t;
        let ra = (0..D)
            .map(|i| {
                let eq_chunk = EqPolynomial::<F>::evals(&params.r_address_chunks[i], None);
                let col: Vec<F> = (0..t).map(|c| eq_chunk[indices[i][c] as usize]).collect();
                MultilinearPolynomial::from(col)
            })
            .collect();
        let eq = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.r_cycle, None));
        Self { params, ra, eq }
    }

    pub fn new_verifier(params: RamRaVirtualizationParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::from_u64(0)]);
        Self {
            ra: (0..D).map(|_| dummy()).collect(),
            eq: dummy(),
            params,
        }
    }
}

impl<F: Field, const D: usize, const NE: usize> SumcheckInstance<F>
    for RamRaVirtualization<F, D, NE>
{
    fn num_rounds(&self) -> usize {
        self.params.log_t
    }

    fn degree(&self) -> usize {
        D + 1
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.params.ra_claim
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let half = self.eq.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); NE];
        for c in 0..half {
            let mut prod = [F::from_u64(1); NE];
            for ra in &self.ra {
                let e = ra.sumcheck_evals_array::<NE>(c, BindingOrder::LowToHigh);
                for (p, &ep) in prod.iter_mut().zip(e.iter()) {
                    *p *= ep;
                }
            }
            let eqe = self
                .eq
                .sumcheck_evals_array::<NE>(c, BindingOrder::LowToHigh);
            for (a, (&p, &ep)) in acc.iter_mut().zip(prod.iter().zip(eqe.iter())) {
                a.fmadd(p, ep);
            }
        }
        let evals: [F; NE] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        for ra in &mut self.ra {
            ra.bind_parallel(r, BindingOrder::LowToHigh);
        }
        self.eq.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let r_cycle_final: Vec<F> = challenges.iter().rev().copied().collect();
        for i in 0..D {
            let point = OpeningPoint::new(
                [
                    self.params.r_address_chunks[i].as_slice(),
                    r_cycle_final.as_slice(),
                ]
                .concat(),
            );
            accumulator.append_dense(
                CommittedPolynomial::RamRa(i),
                SumcheckId::RamRaVirtualization,
                point,
                self.ra[i].final_sumcheck_claim(),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let r_cycle_final: Vec<F> = challenges.iter().rev().copied().collect();
        let eq_eval = EqPolynomial::<F>::mle(&self.params.r_cycle, &r_cycle_final);
        let mut prod = F::from_u64(1);
        for i in 0..D {
            let (_, ra_i) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::RamRaVirtualization,
            );
            prod *= ra_i;
        }
        prod * eq_eval
    }
}

/// RAM RA virtualization failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RamRaVirtualizationError {
    Sumcheck,
    OutputClaim,
}

/// The stage proof: the `D` committed `RamRa(i)` openings the verifier (no witness) re-seeds (also the
/// M7 pushforward inputs). The sumcheck round polynomials live in the shared NARG.
#[derive(Clone, Debug)]
pub struct RamRaVirtualizationProof<F: Field> {
    pub ra_openings: Vec<F>,
}

/// Prove the RAM RA virtualization on the shared transcript + accumulator: read the consolidated
/// `RamRa@RamRaClaimReduction` claim, run the degree-`D+1` product sumcheck over the cycle, and
/// extract the `D` `RamRa(i)` openings.
pub fn prove_ram_ra_virtualization<F, T, const D: usize, const NE: usize>(
    indices: &[Vec<u32>; D],
    log_k: usize,
    log_t: usize,
    chunk_bits: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> RamRaVirtualizationProof<F>
where
    F: Field,
    T: ProverFs<F>,
{
    let params = RamRaVirtualizationParams::new::<D>(&*accumulator, log_k, log_t, chunk_bits);
    let mut instance = RamRaVirtualization::<F, D, NE>::new_prover(params, indices);
    let _ = prove(&mut instance, accumulator, transcript);
    let ra_openings = (0..D)
        .map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::RamRaVirtualization,
                )
                .1
        })
        .collect();
    RamRaVirtualizationProof { ra_openings }
}

/// Verify the RAM RA virtualization (mirror of [`prove_ram_ra_virtualization`]): replay the sumcheck
/// against the consolidated claim, re-seed the proof-carried `RamRa(i)` openings at the recomputed
/// per-chunk points, and check the reduced claim closes against [`SumcheckInstance::expected_output_claim`].
pub fn verify_ram_ra_virtualization<F, T, const D: usize, const NE: usize>(
    proof: &RamRaVirtualizationProof<F>,
    log_k: usize,
    log_t: usize,
    chunk_bits: usize,
    accumulator: &mut Openings<F>,
    transcript: &mut T,
) -> Result<(), RamRaVirtualizationError>
where
    F: Field,
    T: VerifierFs<F>,
{
    let params = RamRaVirtualizationParams::new::<D>(&*accumulator, log_k, log_t, chunk_bits);
    let instance = RamRaVirtualization::<F, D, NE>::new_verifier(params.clone());
    let claim = SumcheckClaim {
        num_vars: log_t,
        degree: D + 1,
        claimed_sum: params.ra_claim,
    };
    let eval = verify(&claim, transcript).map_err(|_| RamRaVirtualizationError::Sumcheck)?;

    let r_cycle_final: Vec<F> = eval.point.iter().rev().copied().collect();
    for (i, &c) in proof.ra_openings.iter().enumerate() {
        let point = OpeningPoint::new(
            [
                params.r_address_chunks[i].as_slice(),
                r_cycle_final.as_slice(),
            ]
            .concat(),
        );
        accumulator.append_dense(
            CommittedPolynomial::RamRa(i),
            SumcheckId::RamRaVirtualization,
            point,
            c,
        );
    }
    if eval.value != instance.expected_output_claim(accumulator, &eval.point) {
        return Err(RamRaVirtualizationError::OutputClaim);
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::field::{ProverTranscript, VerifierTranscript};
    use jolt_field::goldilocks::GoldilocksFp3 as F;

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    /// Seed a consistent `RamRa@RamRaClaimReduction` opening `ra(r_address ‖ ρ)` from synthetic chunk
    /// indices, then prove→NARG→verify the virtualization closes back to the `D` per-chunk openings.
    fn round_trip<const D: usize, const NE: usize>(seed: u64, log_t: usize, log_k: usize) {
        let chunk_bits = 4;
        let t = 1usize << log_t;
        let kc = 1usize << chunk_bits;
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            state >> 33
        };

        let indices: [Vec<u32>; D] =
            std::array::from_fn(|_| (0..t).map(|_| (next() % kc as u64) as u32).collect());
        let r_address: Vec<F> = (0..log_k).map(|_| f(next())).collect();
        let r_cycle: Vec<F> = (0..log_t).map(|_| f(next())).collect();

        // ra(r_address ‖ ρ) = Σ_c eq(ρ,c)·Π_i eq(r_address_chunk_i, idx_i[c]).
        let chunks = r_address_chunks::<F, D>(&r_address, chunk_bits);
        let eq_cycle = EqPolynomial::<F>::evals(&r_cycle, None);
        let eq_chunks: Vec<Vec<F>> = (0..D)
            .map(|i| EqPolynomial::<F>::evals(&chunks[i], None))
            .collect();
        let ra_claim = (0..t).fold(f(0), |acc, c| {
            let prod = (0..D).fold(f(1), |p, i| p * eq_chunks[i][indices[i][c] as usize]);
            acc + eq_cycle[c] * prod
        });

        let seed_acc = |acc: &mut Openings<F>| {
            let point = OpeningPoint::new([r_address.clone(), r_cycle.clone()].concat());
            acc.append_virtual(
                VirtualPolynomial::RamRa,
                SumcheckId::RamRaClaimReduction,
                point,
                ra_claim,
            );
        };

        let mut prover_acc = Openings::<F>::new(log_t);
        seed_acc(&mut prover_acc);
        let mut pt = ProverTranscript::new("ram-ra-virt");
        let proof = prove_ram_ra_virtualization::<F, _, D, NE>(
            &indices,
            log_k,
            log_t,
            chunk_bits,
            &mut prover_acc,
            &mut pt,
        );
        let narg = pt.into_proof();

        let mut verifier_acc = Openings::<F>::new(log_t);
        seed_acc(&mut verifier_acc);
        let mut vt = VerifierTranscript::new("ram-ra-virt", &narg);
        verify_ram_ra_virtualization::<F, _, D, NE>(
            &proof,
            log_k,
            log_t,
            chunk_bits,
            &mut verifier_acc,
            &mut vt,
        )
        .expect("RAM RA virtualization must verify");
    }

    #[test]
    fn ram_ra_virtualization_round_trip_aligned() {
        // log_k = 8 = 2·4 (aligned, D=2).
        round_trip::<2, 4>(0x4A41, 4, 8);
    }

    #[test]
    fn ram_ra_virtualization_round_trip_unaligned() {
        // log_k = 13, D = ceil(13/4) = 4 (top chunk zero-padded — the muldiv geometry).
        round_trip::<4, 6>(0x5151, 5, 13);
    }

    #[test]
    fn ram_ra_virtualization_tampered_rejected() {
        let chunk_bits = 4;
        let (log_t, log_k) = (4usize, 8usize);
        let t = 1usize << log_t;
        let indices: [Vec<u32>; 2] = std::array::from_fn(|i| {
            (0..t)
                .map(|c| ((c + i) % (1 << chunk_bits)) as u32)
                .collect()
        });
        let r_address: Vec<F> = (0..log_k).map(|i| f(i as u64 + 3)).collect();
        let r_cycle: Vec<F> = (0..log_t).map(|i| f(i as u64 + 7)).collect();
        let chunks = r_address_chunks::<F, 2>(&r_address, chunk_bits);
        let eq_cycle = EqPolynomial::<F>::evals(&r_cycle, None);
        let eq_chunks: Vec<Vec<F>> = (0..2)
            .map(|i| EqPolynomial::<F>::evals(&chunks[i], None))
            .collect();
        let ra_claim = (0..t).fold(f(0), |acc, c| {
            let prod = (0..2).fold(f(1), |p, i| p * eq_chunks[i][indices[i][c] as usize]);
            acc + eq_cycle[c] * prod
        });
        let seed_pt = OpeningPoint::new([r_address.clone(), r_cycle.clone()].concat());

        let mut prover_acc = Openings::<F>::new(log_t);
        prover_acc.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            seed_pt.clone(),
            ra_claim,
        );
        let mut pt = ProverTranscript::new("ram-ra-virt");
        let mut proof = prove_ram_ra_virtualization::<F, _, 2, 4>(
            &indices,
            log_k,
            log_t,
            chunk_bits,
            &mut prover_acc,
            &mut pt,
        );
        let narg = pt.into_proof();
        proof.ra_openings[0] += f(1);

        let mut verifier_acc = Openings::<F>::new(log_t);
        verifier_acc.append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            seed_pt,
            ra_claim,
        );
        let mut vt = VerifierTranscript::new("ram-ra-virt", &narg);
        let res = verify_ram_ra_virtualization::<F, _, 2, 4>(
            &proof,
            log_k,
            log_t,
            chunk_bits,
            &mut verifier_acc,
            &mut vt,
        );
        assert_eq!(res, Err(RamRaVirtualizationError::OutputClaim));
    }
}
