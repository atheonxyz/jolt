//! Spartan **inner** reduction (binary; the R1CSEval analog) — reduces the outer sumcheck's
//! `Az(r_x)/Bz(r_x)/Cz(r_x)` claims to a single witness opening `z(r_y)` via a sumcheck over the
//! column hypercube, built on the workspace [`jolt_r1cs::R1csKey`].
//!
//! Given the outer's reduced point `r_x` and a batching `(ρ_a, ρ_b, ρ_c)`, prove
//!
//! ```text
//! ρ_a·Az(r_x) + ρ_b·Bz(r_x) + ρ_c·Cz(r_x) = Σ_y M(r_x, y)·z(y),   M = ρ_a·A + ρ_b·B + ρ_c·C,
//! ```
//!
//! where `M(r_x, ·)` is the dense **combined row** [`R1csKey::combined_row`] and `z` is the witness
//! MLE over the columns. The degree-2 product sumcheck reduces it to `M(r_x, r_y)·z(r_y)`; the
//! verifier recomputes `M(r_x, r_y) = ρ_a·Ã + ρ_b·B̃ + ρ_c·C̃` via [`R1csKey::evaluate_matrix_mles`]
//! and reads `z(r_y)`.
//!
//! **Binary** Spartan (the M8 path; jolt-core's univariate-skip inner reduction is deferred — see
//! the [`super`] module doc). Decoupled / correctness-first: takes the materialized witness `z`
//! column (length `R1csKey::total_cols()`); in the e2e, `z(r_y)` decomposes into the committed/
//! virtual input openings via the uniform structure (M8 stage-driver wiring).

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_r1cs::R1csKey;

use crate::framework::accumulator::{OpeningAccumulator, Openings, SumcheckId, VirtualPolynomial};
use crate::framework::poly::MultilinearPolynomial;
use crate::framework::sumcheck::SumcheckInstance;

const DEGREE: usize = 2;

/// Parameters: the preprocessed R1CS key, the outer reduced point `r_x` (length
/// `R1csKey::num_row_vars`, the `(cycle ‖ constraint)` order [`R1csKey`] expects), and the batching
/// `[ρ_a, ρ_b, ρ_c]`.
#[derive(Clone, Debug)]
pub struct SpartanInnerParams<F: Field> {
    pub key: R1csKey<F>,
    pub r_x: Vec<F>,
    pub rho: [F; 3],
}

/// Prover/verifier instance. The verifier carries `params` (its `key` recomputes `M(r_x, r_y)`) and
/// ignores the empty polynomials.
pub struct SpartanInner<F: Field> {
    pub params: SpartanInnerParams<F>,
    combined: MultilinearPolynomial<F>,
    z: MultilinearPolynomial<F>,
}

impl<F: Field> SpartanInner<F> {
    /// Build the prover instance: materialize the combined row `M(r_x, ·)` and the witness `z`
    /// (length `key.total_cols()`).
    pub fn new_prover(params: SpartanInnerParams<F>, z: Vec<F>) -> Self {
        let combined =
            params
                .key
                .combined_row(&params.r_x, params.rho[0], params.rho[1], params.rho[2]);
        debug_assert_eq!(combined.len(), params.key.total_cols());
        debug_assert_eq!(z.len(), params.key.total_cols());
        Self {
            params,
            combined: MultilinearPolynomial::from(combined),
            z: MultilinearPolynomial::from(z),
        }
    }

    pub fn new_verifier(params: SpartanInnerParams<F>) -> Self {
        let dummy = || MultilinearPolynomial::from(vec![F::zero()]);
        Self {
            params,
            combined: dummy(),
            z: dummy(),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for SpartanInner<F> {
    fn num_rounds(&self) -> usize {
        self.params.key.num_col_vars()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, az) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, bz) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, cz) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
        self.params.rho[0] * az + self.params.rho[1] * bz + self.params.rho[2] * cz
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        // Degree-2: combined·z ⇒ 3 evaluation points; unreduced accumulation.
        let half = self.combined.len() / 2;
        let mut acc = [<F as Field>::Accumulator::default(); 3];
        for i in 0..half {
            let c = self
                .combined
                .sumcheck_evals_array::<3>(i, BindingOrder::LowToHigh);
            let z = self.z.sumcheck_evals_array::<3>(i, BindingOrder::LowToHigh);
            for k in 0..3 {
                acc[k].fmadd(c[k], z[k]);
            }
        }
        let evals: [F; 3] = std::array::from_fn(|k| acc[k].reduce());
        UnivariatePoly::from_evals(&evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        self.combined.bind_parallel(r, BindingOrder::LowToHigh);
        self.z.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn cache_openings(&self, accumulator: &mut Openings<F>, challenges: &[F]) {
        let point = self.normalize_opening_point(challenges);
        accumulator.append_virtual(
            VirtualPolynomial::SpartanWitnessZ,
            SumcheckId::SpartanInner,
            point,
            self.z.final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        challenges: &[F],
    ) -> F {
        let point = self.normalize_opening_point(challenges);
        let (a, b, c) = self
            .params
            .key
            .evaluate_matrix_mles(&self.params.r_x, &point.r);
        let m = self.params.rho[0] * a + self.params.rho[1] * b + self.params.rho[2] * c;
        let (_, z_ry) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SpartanWitnessZ,
            SumcheckId::SpartanInner,
        );
        m * z_ry
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::framework::accumulator::OpeningPoint;
    use crate::framework::sumcheck::{prove, verify};
    use jolt_field::goldilocks::GoldilocksFp3 as F;
    use jolt_poly::EqPolynomial;
    use jolt_r1cs::ConstraintMatrices;
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

    /// `x·x = y`, `y·x = z` — 2 constraints, 4 vars `[1, x, y, z]` (mirrors `R1csKey`'s test shape).
    fn test_matrices() -> ConstraintMatrices<F> {
        let one = F::from_u64(1);
        ConstraintMatrices::new(
            2,
            4,
            vec![vec![(1, one)], vec![(2, one)]],
            vec![vec![(1, one)], vec![(1, one)]],
            vec![vec![(2, one)], vec![(3, one)]],
        )
    }

    /// The inner reduction round-trips: the degree-2 product sumcheck over `(M(r_x,·), z)` reduces to
    /// `M(r_x, r_y)·z(r_y)`, matching `evaluate_matrix_mles`; and `input_claim` (seeded `ρ·Az/Bz/Cz`)
    /// equals the dense product `Σ_y M(r_x,y)·z(y)` (the `R1csKey` guarantee).
    fn round_trip(seed: u64, num_cycles: usize) {
        let mut rng = Rng(seed);
        let key = R1csKey::new(test_matrices(), num_cycles);
        let total_cols = key.total_cols();
        let v_pad = key.num_vars_padded;
        let cv = key.num_cycle_vars();

        let z = rand_vec(&mut rng, total_cols);
        let r_x = rand_vec(&mut rng, key.num_row_vars());
        let rho = [
            F::from_u64(rng.next()),
            F::from_u64(rng.next()),
            F::from_u64(rng.next()),
        ];

        // Seed the outer's Az/Bz/Cz(r_x) via the uniform factorization (the same eq tables R1csKey
        // uses): z_at_rx_cycle[var] = Σ_cycle eq(rx_cycle)[cycle]·z[cycle·v_pad+var].
        let (rx_cycle, rx_con) = r_x.split_at(cv);
        let eq_cycle = EqPolynomial::new(rx_cycle.to_vec()).evaluations();
        let z_at_rx_cycle: Vec<F> = (0..v_pad)
            .map(|var| {
                (0..num_cycles).fold(F::from_u64(0), |a, cycle| {
                    a + eq_cycle[cycle] * z[cycle * v_pad + var]
                })
            })
            .collect();
        let (az, bz, cz) = key.evaluate_sparse_matvec(rx_con, &z_at_rx_cycle);

        let mut prover_acc = Openings::<F>::new(cv);
        let dummy_pt = OpeningPoint::new(r_x.clone());
        prover_acc.append_virtual(
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            dummy_pt.clone(),
            az,
        );
        prover_acc.append_virtual(
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            dummy_pt.clone(),
            bz,
        );
        prover_acc.append_virtual(
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            dummy_pt,
            cz,
        );

        let params = SpartanInnerParams {
            key: key.clone(),
            r_x: r_x.clone(),
            rho,
        };
        let mut prover = SpartanInner::new_prover(params.clone(), z.clone());

        // input_claim == Σ_y M(r_x,y)·z(y) (the R1csKey combined-row ↔ matvec guarantee).
        let input_claim = prover.input_claim(&prover_acc);
        let combined = key.combined_row(&r_x, rho[0], rho[1], rho[2]);
        let dense_product = combined
            .iter()
            .zip(z.iter())
            .fold(F::from_u64(0), |a, (m, zv)| a + *m * *zv);
        assert_eq!(input_claim, dense_product, "input claim == Σ M·z");

        let mut prover_t = Blake2bTranscript::<F>::new(b"spartan-inner");
        let (proof, challenges) = prove(&mut prover, &mut prover_acc, &mut prover_t);

        let mut verifier_acc = Openings::<F>::new(cv);
        // The verifier reads the outer claims + the prover's cached z(r_y).
        for key_id in [
            VirtualPolynomial::SpartanAz,
            VirtualPolynomial::SpartanBz,
            VirtualPolynomial::SpartanCz,
        ] {
            let (pt, c) =
                prover_acc.get_virtual_polynomial_opening(key_id, SumcheckId::SpartanOuter);
            verifier_acc.append_virtual(key_id, SumcheckId::SpartanOuter, pt, c);
        }
        let (zpt, zc) = prover_acc.get_virtual_polynomial_opening(
            VirtualPolynomial::SpartanWitnessZ,
            SumcheckId::SpartanInner,
        );
        verifier_acc.append_virtual(
            VirtualPolynomial::SpartanWitnessZ,
            SumcheckId::SpartanInner,
            zpt,
            zc,
        );

        let verifier = SpartanInner::new_verifier(params);
        let claim = SumcheckClaim {
            num_vars: key.num_col_vars(),
            degree: DEGREE,
            claimed_sum: input_claim,
        };
        let mut verifier_t = Blake2bTranscript::<F>::new(b"spartan-inner");
        let EvaluationClaim { point, value } =
            verify(&claim, &proof, &mut verifier_t).expect("inner reduction must verify");
        assert_eq!(
            point, challenges,
            "verifier point matches prover challenges"
        );

        let expected = verifier.expected_output_claim(&verifier_acc, &challenges);
        assert_eq!(value, expected, "reduced claim == M(r_x,r_y)·z(r_y)");
    }

    #[test]
    fn inner_reduction_round_trip() {
        round_trip(0x1A1A, 2);
        round_trip(0x2B2B, 4);
        round_trip(0x3C3C, 8);
    }

    #[test]
    fn tampered_proof_rejected() {
        let mut rng = Rng(0x9F9F);
        let key = R1csKey::new(test_matrices(), 4);
        let z = rand_vec(&mut rng, key.total_cols());
        let r_x = rand_vec(&mut rng, key.num_row_vars());
        let rho = [
            F::from_u64(rng.next()),
            F::from_u64(rng.next()),
            F::from_u64(rng.next()),
        ];
        let mut acc = Openings::<F>::new(key.num_cycle_vars());
        let pt = OpeningPoint::new(r_x.clone());
        for k in [
            VirtualPolynomial::SpartanAz,
            VirtualPolynomial::SpartanBz,
            VirtualPolynomial::SpartanCz,
        ] {
            acc.append_virtual(
                k,
                SumcheckId::SpartanOuter,
                pt.clone(),
                F::from_u64(rng.next()),
            );
        }
        let params = SpartanInnerParams { key, r_x, rho };
        let mut prover = SpartanInner::new_prover(params, z);
        let input_claim = prover.input_claim(&acc);
        let mut prover_t = Blake2bTranscript::<F>::new(b"t");
        let (mut proof, _) = prove(&mut prover, &mut acc, &mut prover_t);
        proof.round_polynomials[0] =
            UnivariatePoly::new(vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
        let claim = SumcheckClaim {
            num_vars: proof.round_polynomials.len(),
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
