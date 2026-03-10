//! Dory polynomial commitment scheme implementation

use super::dory_globals::{DoryGlobals, DoryLayout};
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
#[cfg(all(feature = "webgpu-pairing", target_arch = "wasm32"))]
use super::wrappers::ArkG2;
use super::wrappers::{
    ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup,
    ArkworksVerifierSetup, JoltToDoryTranscript, BN254,
};
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::commitment_scheme::{
        CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
    },
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, small_scalar::SmallScalar},
};
use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::Zero;
use dory::primitives::{
    arithmetic::{Field as DoryField, Group, PairingCurve},
    poly::Polynomial,
};
use rayon::prelude::*;
use std::borrow::Borrow;
use tracing::trace_span;

#[derive(Clone)]
pub struct DoryCommitmentScheme;

#[derive(Clone, Debug, PartialEq)]
pub struct DoryOpeningProofHint(Vec<ArkG1>);

impl DoryOpeningProofHint {
    fn new(row_commitments: Vec<ArkG1>) -> Self {
        Self(row_commitments)
    }

    pub fn from_rows(row_commitments: Vec<ArkG1>) -> Self {
        Self(row_commitments)
    }

    pub(crate) fn into_rows(self) -> Vec<ArkG1> {
        self.0
    }
}

pub fn bind_opening_inputs<F: JoltField, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    opening: &F,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_scalar(b"dory_opening_eval", opening);
}

#[cfg(feature = "zk")]
pub fn bind_opening_inputs_zk<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    y_com: &C::G1,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_commitment(b"dory_eval_commitment", y_com);
}

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = DoryOpeningProofHint;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        #[cfg(not(target_arch = "wasm32"))]
        let setup = ArkworksProverSetup::new_from_urs(max_num_vars);
        #[cfg(target_arch = "wasm32")]
        let setup = ArkworksProverSetup::new(max_num_vars);

        // The prepared-point cache in dory-pcs is global and can only be initialized once.
        // In unit tests, multiple setups with different sizes are created, so initializing the
        // cache with a small setup can break later tests that need more generators.
        // We therefore disable cache initialization in `cfg(test)` builds.
        #[cfg(not(test))]
        DoryGlobals::init_prepared_cache(&setup.g1_vec, &setup.g2_vec);

        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_verifier").entered();
        setup.to_verifier_setup()
    }

    fn commit(
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let (tier_2, row_commitments, _commit_blind) =
            <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<ArkFr>>::commit::<
                BN254,
                dory::Transparent,
                JoltG1Routines,
            >(poly, nu, sigma, setup)
            .expect("commitment should succeed");

        (tier_2, DoryOpeningProofHint::new(row_commitments))
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        let _span = trace_span!("DoryCommitmentScheme::batch_commit").entered();

        polys
            .par_iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>) {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let (row_commitments, commit_blind) = hint
            .map(|h| (h.into_rows(), DoryField::zero()))
            .unwrap_or_else(|| {
                let (_commitment, row_commitments) = Self::commit(poly, setup);
                (row_commitments.into_rows(), DoryField::zero())
            });

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let reordered_point = reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point);
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        #[cfg(feature = "zk")]
        type DoryMode = dory::ZK;
        #[cfg(not(feature = "zk"))]
        type DoryMode = dory::Transparent;

        let (proof, y_blinding) =
            dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _, DoryMode>(
                poly,
                &ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                setup,
                &mut dory_transcript,
            )
            .expect("proof generation should succeed");

        (proof, y_blinding.map(|b| ark_to_jolt(&b)))
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        opening: &ark_bn254::Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let _span = trace_span!("DoryCommitmentScheme::verify").entered();

        let reordered_point = reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point);

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_eval: ArkFr = jolt_to_ark(opening);

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::verify::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _>(
            *commitment,
            ark_eval,
            &ark_point,
            proof,
            setup.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    ///
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, mut hint) in coeffs.iter().zip(hints.into_iter()) {
            hint.0.resize(num_rows, ArkG1(G1Projective::zero()));

            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(
                    hint.0.as_mut_ptr() as *mut G1Projective,
                    hint.0.len(),
                )
            };

            let rlc_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len())
            };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitments,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, hint.0);
        }

        DoryOpeningProofHint::new(rlc_hint)
    }

    /// Homomorphically combines multiple commitments using a random linear combination.
    /// Computes: sum_i(coeff_i * commitment_i) for the GT elements.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let _span = trace_span!("DoryCommitmentScheme::combine_commitments").entered();

        // Combine GT elements using parallel RLC
        let commitments_vec: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        coeffs
            .par_iter()
            .zip(commitments_vec.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                ark_coeff * **commitment
            })
            .reduce(ArkGT::identity, |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type ChunkState = Vec<ArkG1>; // Tier 1 commitment chunks

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment")]
    fn process_chunk<T: SmallScalar>(setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState {
        debug_assert_eq!(chunk.len(), DoryGlobals::get_num_columns());

        let row_len = DoryGlobals::get_num_columns();
        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        let row_commitment =
            ArkG1(T::msm(&g1_bases[..chunk.len()], chunk).expect("MSM calculation failed."));
        vec![row_commitment]
    }

    #[tracing::instrument(
        skip_all,
        name = "DoryCommitmentScheme::compute_tier1_commitment_onehot"
    )]
    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let K = onehot_k;

        let row_len = DoryGlobals::get_num_columns();
        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); K];
        for (col_index, k) in chunk.iter().enumerate() {
            if let Some(k) = k {
                indices_per_k[*k].push(col_index);
            }
        }

        let results = jolt_optimizations::batch_g1_additions_multi(&g1_bases, &indices_per_k);

        let mut row_commitments = vec![ArkG1(G1Projective::zero()); K];
        for (k, result) in results.into_iter().enumerate() {
            if !indices_per_k[k].is_empty() {
                row_commitments[k] = ArkG1(G1Projective::from(result));
            }
        }
        row_commitments
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier2_commitment")]
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let num_rows = DoryGlobals::get_max_num_rows();

        if let Some(_K) = onehot_k {
            let row_len = DoryGlobals::get_num_columns();
            let T = DoryGlobals::get_T();
            let rows_per_k = T / row_len;

            let mut row_commitments = vec![ArkG1(G1Projective::zero()); num_rows];
            for (chunk_index, commitments) in chunks.iter().enumerate() {
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }

            let g2_bases = &setup.g2_vec[..num_rows];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, DoryOpeningProofHint::new(row_commitments))
        } else {
            let row_commitments: Vec<ArkG1> =
                chunks.iter().flat_map(|chunk| chunk.clone()).collect();

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, DoryOpeningProofHint::new(row_commitments))
        }
    }
}

impl<C: JoltCurve> ZkEvalCommitment<C> for DoryCommitmentScheme
where
    C::G1: From<ArkG1>,
{
    fn eval_commitment(proof: &Self::Proof) -> Option<C::G1> {
        #[cfg(feature = "zk")]
        {
            proof.y_com.as_ref().copied().map(C::G1::from)
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = proof;
            None
        }
    }

    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = setup.0.g1_vec.first().copied().map(C::G1::from)?;
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = C::G1::from(setup.0.g1_0);
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    #[cfg(feature = "zk")]
    fn zk_generators(setup: &Self::ProverSetup, count: usize) -> Option<(Vec<C::G1>, C::G1)> {
        let count = std::cmp::min(count, setup.0.g1_vec.len());
        let g1s = setup.0.g1_vec[..count]
            .iter()
            .map(|g| C::G1::from(*g))
            .collect();
        let h1 = C::G1::from(setup.0.h1);
        Some((g1s, h1))
    }
}

#[cfg(all(feature = "webgpu-pairing", target_arch = "wasm32"))]
impl DoryCommitmentScheme {
    /// Extract row commitments from tier-1 chunks without computing the pairing.
    /// This is the CPU-only part of `aggregate_chunks`.
    pub fn collect_row_commitments(onehot_k: Option<usize>, chunks: &[Vec<ArkG1>]) -> Vec<ArkG1> {
        let num_rows = DoryGlobals::get_max_num_rows();

        if let Some(_K) = onehot_k {
            let row_len = DoryGlobals::get_num_columns();
            let T = DoryGlobals::get_T();
            let rows_per_k = T / row_len;

            let mut row_commitments = vec![ArkG1(G1Projective::zero()); num_rows];
            for (chunk_index, commitments) in chunks.iter().enumerate() {
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }
            row_commitments
        } else {
            chunks.iter().flat_map(|chunk| chunk.clone()).collect()
        }
    }

    /// Get a slice of G2 bases from the prover setup, needed for GPU pairing.
    pub fn get_g2_bases(setup: &ArkworksProverSetup, len: usize) -> Vec<super::wrappers::ArkG2> {
        setup.g2_vec[..len].to_vec()
    }

    pub async fn combine_hints_gpu(
        hints: Vec<DoryOpeningProofHint>,
        coeffs: &[ark_bn254::Fr],
    ) -> DoryOpeningProofHint {
        if !super::webgpu_pairing::is_gpu_combine_hints_available() {
            return <Self as CommitmentScheme>::combine_hints(hints, coeffs);
        }

        let hint_rows: Vec<Vec<ArkG1>> = hints
            .into_iter()
            .map(DoryOpeningProofHint::into_rows)
            .collect();
        let handle = super::webgpu_pairing::dispatch_gpu_combine_hints(&hint_rows, coeffs);
        let rows = super::webgpu_pairing::resolve_gpu_combine_hints(handle).await;
        DoryOpeningProofHint::new(rows)
    }

    /// Compute v_vec (column evaluation scalars) for GPU G2 fixed-base scalar mul.
    /// The returned scalars should be multiplied by h2 on GPU, then passed to `prove_with_gpu_v2`.
    #[cfg(all(feature = "webgpu-pairing", target_arch = "wasm32"))]
    pub fn compute_v_vec_for_gpu(
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[ark_bn254::Fr],
    ) -> Vec<ArkFr> {
        use dory::MultilinearLagrange;

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let reordered_point = if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
            let log_T = DoryGlobals::get_T().log_2();
            let log_K = opening_point.len().saturating_sub(log_T);
            let (r_address, r_cycle) = opening_point.split_at(log_K);
            [r_cycle, r_address].concat()
        } else {
            opening_point.to_vec()
        };
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let (left_vec, _right_vec) = poly.compute_evaluation_vectors(&ark_point, nu, sigma);
        poly.vector_matrix_product(&left_vec, nu, sigma)
    }

    /// Prove with a GPU-precomputed v2 vector (G2 fixed-base scalar mul result).
    /// This skips the expensive `M2::fixed_base_vector_scalar_mul` inside
    /// `create_evaluation_proof`, replacing it with the pre-computed GPU result.
    ///
    /// Only supports Transparent (non-ZK) mode — WASM builds don't use ZK.
    ///
    /// When running on WASM with WebGPU pairing available, pairings in the
    /// reduce-and-fold loop are offloaded to the GPU for rounds with
    /// >= GPU_PAIRING_THRESHOLD points. Smaller rounds fall back to CPU pairings.
    #[cfg(all(feature = "webgpu-pairing", target_arch = "wasm32"))]
    pub async fn prove_with_gpu_v2<ProofTranscript: Transcript>(
        setup: &ArkworksProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[ark_bn254::Fr],
        hint: Option<DoryOpeningProofHint>,
        transcript: &mut ProofTranscript,
        pre_computed_v2: Vec<ArkG2>,
    ) -> (ArkDoryProof, Option<ark_bn254::Fr>) {
        use super::webgpu_pairing;
        use dory::primitives::arithmetic::{DoryRoutines, Field as DoryField, Group, PairingCurve};
        use dory::primitives::transcript::Transcript as DoryTranscript;
        use dory::{
            FirstReduceMessage, MultilinearLagrange, ScalarProductMessage, SecondReduceMessage,
            VMVMessage,
        };

        const GPU_PAIRING_THRESHOLD: usize = 64;

        let _span = trace_span!("DoryCommitmentScheme::prove_with_gpu_v2").entered();

        let use_gpu_pairing = webgpu_pairing::is_gpu_pairing_available();

        let row_commitments = hint.map(|h| h.into_rows()).unwrap_or_else(|| {
            let (_commitment, row_commitments) = Self::commit(poly, setup);
            row_commitments.into_rows()
        });

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let reordered_point = if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
            let log_T = DoryGlobals::get_T().log_2();
            let log_K = opening_point.len().saturating_sub(log_T);
            let (r_address, r_cycle) = opening_point.split_at(log_K);
            [r_cycle, r_address].concat()
        } else {
            opening_point.to_vec()
        };
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev()
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let (left_vec, right_vec) = poly.compute_evaluation_vectors(&ark_point, nu, sigma);
        let v_vec = poly.vector_matrix_product(&left_vec, nu, sigma);

        let mut padded_row_commitments = row_commitments.clone();
        if nu < sigma {
            padded_row_commitments.resize(1 << sigma, <BN254 as PairingCurve>::G1::identity());
        }

        let g2_fin = &setup.g2_vec[0];

        let t_vec_v = JoltG1Routines::msm(&padded_row_commitments, &v_vec);
        let c = BN254::pair(&t_vec_v, g2_fin);

        let d2 = BN254::pair(
            &JoltG1Routines::msm(&setup.g1_vec[..1 << sigma], &v_vec),
            g2_fin,
        );

        let e1 = JoltG1Routines::msm(&row_commitments, &left_vec);

        let vmv_message = VMVMessage { c, d2, e1 };

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);
        dory_transcript.append_serde(b"vmv_c", &vmv_message.c);
        dory_transcript.append_serde(b"vmv_d2", &vmv_message.d2);
        dory_transcript.append_serde(b"vmv_e1", &vmv_message.e1);

        // Inline reduce-and-fold state (bypasses DoryProverState whose fields are private).
        // Transparent mode: all blinds are zero, M::mask() is identity, M::sample() returns zero.
        let mut v1 = padded_row_commitments;
        let mut v2 = pre_computed_v2;
        let mut v2_scalars: Option<Vec<ArkFr>> = Some(v_vec);
        let mut padded_right_vec = right_vec;
        let mut padded_left_vec = left_vec;
        if nu < sigma {
            padded_right_vec.resize(1 << sigma, ArkFr::zero());
            padded_left_vec.resize(1 << sigma, ArkFr::zero());
        }
        let mut s1 = padded_right_vec;
        let mut s2 = padded_left_vec;

        let num_rounds = nu.max(sigma);
        let mut first_messages = Vec::with_capacity(num_rounds);
        let mut second_messages = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let n = 1 << (num_rounds - round);
            let n2 = n / 2;

            // --- compute_first_message ---
            let (v1_l, v1_r) = v1.split_at(n2);
            let (v2_l, v2_r) = v2.split_at(n2);
            let g1_prime = &setup.g1_vec[..n2];
            let g2_prime = &setup.g2_vec[..n2];

            let (d1_left, d1_right, d2_left, d2_right, e1_beta, e2_beta);

            if use_gpu_pairing && n2 >= GPU_PAIRING_THRESHOLD && v2_scalars.is_none() {
                // Batch all 4 pairings in ONE GPU dispatch (d1 + d2)
                let gpu_handle = webgpu_pairing::dispatch_gpu_multi_group_pairing(&[
                    (v1_l, g2_prime), // d1_left
                    (v1_r, g2_prime), // d1_right
                    (g1_prime, v2_l), // d2_left
                    (g1_prime, v2_r), // d2_right
                ]);

                // CPU MSMs overlap with GPU dispatch
                e1_beta = JoltG1Routines::msm(&setup.g1_vec[..n], &s2);
                e2_beta = JoltG2Routines::msm(&setup.g2_vec[..n], &s1);

                let gpu_results = webgpu_pairing::resolve_gpu_multi_group_pairing(gpu_handle).await;
                let mut it = gpu_results.into_iter();
                d1_left = it.next().unwrap();
                d1_right = it.next().unwrap();
                d2_left = it.next().unwrap();
                d2_right = it.next().unwrap();
            } else if use_gpu_pairing && n2 >= GPU_PAIRING_THRESHOLD {
                // v2_scalars is Some (first round): batch d1 on GPU, d2 uses MSM + pair
                let gpu_handle = webgpu_pairing::dispatch_gpu_multi_group_pairing(&[
                    (v1_l, g2_prime), // d1_left
                    (v1_r, g2_prime), // d1_right
                ]);

                // d2 via MSM + single pair (CPU) + MSMs, overlapping with GPU
                let scalars = v2_scalars.as_ref().unwrap();
                let (s_l, s_r) = scalars.split_at(n2);
                let sum_left = JoltG1Routines::msm(g1_prime, s_l);
                let sum_right = JoltG1Routines::msm(g1_prime, s_r);
                d2_left = BN254::pair(&sum_left, g2_fin);
                d2_right = BN254::pair(&sum_right, g2_fin);
                e1_beta = JoltG1Routines::msm(&setup.g1_vec[..n], &s2);
                e2_beta = JoltG2Routines::msm(&setup.g2_vec[..n], &s1);

                let gpu_results = webgpu_pairing::resolve_gpu_multi_group_pairing(gpu_handle).await;
                let mut it = gpu_results.into_iter();
                d1_left = it.next().unwrap();
                d1_right = it.next().unwrap();
            } else {
                // CPU fallback
                if let Some(scalars) = v2_scalars.as_ref() {
                    let (s_l, s_r) = scalars.split_at(n2);
                    let sum_left = JoltG1Routines::msm(g1_prime, s_l);
                    let sum_right = JoltG1Routines::msm(g1_prime, s_r);
                    d1_left = BN254::multi_pair_g2_setup(v1_l, g2_prime);
                    d1_right = BN254::multi_pair_g2_setup(v1_r, g2_prime);
                    d2_left = BN254::pair(&sum_left, g2_fin);
                    d2_right = BN254::pair(&sum_right, g2_fin);
                } else {
                    d1_left = BN254::multi_pair_g2_setup(v1_l, g2_prime);
                    d1_right = BN254::multi_pair_g2_setup(v1_r, g2_prime);
                    d2_left = BN254::multi_pair_g1_setup(g1_prime, v2_l);
                    d2_right = BN254::multi_pair_g1_setup(g1_prime, v2_r);
                }
                e1_beta = JoltG1Routines::msm(&setup.g1_vec[..n], &s2);
                e2_beta = JoltG2Routines::msm(&setup.g2_vec[..n], &s1);
            }

            let first_msg = FirstReduceMessage {
                d1_left,
                d1_right,
                d2_left,
                d2_right,
                e1_beta,
                e2_beta,
            };

            dory_transcript.append_serde(b"d1_left", &first_msg.d1_left);
            dory_transcript.append_serde(b"d1_right", &first_msg.d1_right);
            dory_transcript.append_serde(b"d2_left", &first_msg.d2_left);
            dory_transcript.append_serde(b"d2_right", &first_msg.d2_right);
            dory_transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
            dory_transcript.append_serde(b"e2_beta", &first_msg.e2_beta);

            let beta: ArkFr = dory_transcript.challenge_scalar(b"beta");

            // --- apply_first_challenge ---
            let beta_inv = beta.inv().expect("beta must be invertible");
            JoltG1Routines::fixed_scalar_mul_bases_then_add(&setup.g1_vec[..n], &mut v1, &beta);
            JoltG2Routines::fixed_scalar_mul_bases_then_add(&setup.g2_vec[..n], &mut v2, &beta_inv);
            v2_scalars = None;

            first_messages.push(first_msg);

            // --- compute_second_message ---
            let (v1_l, v1_r) = v1.split_at(n2);
            let (v2_l, v2_r) = v2.split_at(n2);
            let (s1_l, s1_r) = s1.split_at(n2);
            let (s2_l, s2_r) = s2.split_at(n2);

            let (c_plus, c_minus, e1_plus, e1_minus, e2_plus, e2_minus);

            if use_gpu_pairing && n2 >= GPU_PAIRING_THRESHOLD {
                // Batch both pairings in ONE GPU dispatch
                let gpu_handle = webgpu_pairing::dispatch_gpu_multi_group_pairing(&[
                    (v1_l, v2_r), // c_plus
                    (v1_r, v2_l), // c_minus
                ]);

                // CPU MSMs overlap with GPU dispatch
                e1_plus = JoltG1Routines::msm(v1_l, s2_r);
                e1_minus = JoltG1Routines::msm(v1_r, s2_l);
                e2_plus = JoltG2Routines::msm(v2_r, s1_l);
                e2_minus = JoltG2Routines::msm(v2_l, s1_r);

                let gpu_results = webgpu_pairing::resolve_gpu_multi_group_pairing(gpu_handle).await;
                let mut it = gpu_results.into_iter();
                c_plus = it.next().unwrap();
                c_minus = it.next().unwrap();
            } else {
                c_plus = BN254::multi_pair(v1_l, v2_r);
                c_minus = BN254::multi_pair(v1_r, v2_l);
                e1_plus = JoltG1Routines::msm(v1_l, s2_r);
                e1_minus = JoltG1Routines::msm(v1_r, s2_l);
                e2_plus = JoltG2Routines::msm(v2_r, s1_l);
                e2_minus = JoltG2Routines::msm(v2_l, s1_r);
            };

            let second_msg = SecondReduceMessage {
                c_plus,
                c_minus,
                e1_plus,
                e1_minus,
                e2_plus,
                e2_minus,
            };

            dory_transcript.append_serde(b"c_plus", &second_msg.c_plus);
            dory_transcript.append_serde(b"c_minus", &second_msg.c_minus);
            dory_transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
            dory_transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
            dory_transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
            dory_transcript.append_serde(b"e2_minus", &second_msg.e2_minus);

            let alpha: ArkFr = dory_transcript.challenge_scalar(b"alpha");

            // --- apply_second_challenge: fold all vectors by half ---
            let alpha_inv = alpha.inv().expect("alpha must be invertible");

            {
                let (v1_l, v1_r) = v1.split_at_mut(n2);
                JoltG1Routines::fixed_scalar_mul_vs_then_add(v1_l, v1_r, &alpha);
            }
            v1.truncate(n2);

            {
                let (v2_l, v2_r) = v2.split_at_mut(n2);
                JoltG2Routines::fixed_scalar_mul_vs_then_add(v2_l, v2_r, &alpha_inv);
            }
            v2.truncate(n2);

            {
                let (s1_l, s1_r) = s1.split_at_mut(n2);
                JoltG1Routines::fold_field_vectors(s1_l, s1_r, &alpha);
            }
            s1.truncate(n2);

            {
                let (s2_l, s2_r) = s2.split_at_mut(n2);
                JoltG1Routines::fold_field_vectors(s2_l, s2_r, &alpha_inv);
            }
            s2.truncate(n2);

            second_messages.push(second_msg);
        }

        // --- compute_final_message ---
        let gamma: ArkFr = dory_transcript.challenge_scalar(b"gamma");
        let gamma_inv = gamma.inv().expect("gamma must be invertible");

        // Transparent mode: r_final1, r_final2 are zero
        let gamma_s1 = gamma * s1[0];
        let e1_final = v1[0] + gamma_s1 * setup.h1;

        let gamma_inv_s2 = gamma_inv * s2[0];
        let e2_final = v2[0] + setup.h2.scale(&gamma_inv_s2);

        let final_message = ScalarProductMessage {
            e1: e1_final,
            e2: e2_final,
        };

        dory_transcript.append_serde(b"final_e1", &final_message.e1);
        dory_transcript.append_serde(b"final_e2", &final_message.e2);
        let _d = dory_transcript.challenge_scalar(b"d");

        let proof = ArkDoryProof {
            vmv_message,
            first_messages,
            second_messages,
            final_message,
            nu,
            sigma,
            #[cfg(feature = "zk")]
            e2: None,
            #[cfg(feature = "zk")]
            y_com: None,
            #[cfg(feature = "zk")]
            sigma1_proof: None,
            #[cfg(feature = "zk")]
            sigma2_proof: None,
            #[cfg(feature = "zk")]
            scalar_product_proof: None,
        };

        (proof, None)
    }
}

/// Reorders opening_point for AddressMajor layout.
///
/// For AddressMajor layout, reorders opening_point from [r_address, r_cycle] to [r_cycle, r_address].
/// This ensures that after Dory's reversal and splitting:
/// - Column (right) vector gets address variables (matching AddressMajor column indexing)
/// - Row (left) vector gets cycle variables (matching AddressMajor row indexing)
///
/// For CycleMajor layout, returns the point unchanged.
fn reorder_opening_point_for_layout<F: JoltField>(
    opening_point: &[F::Challenge],
) -> Vec<F::Challenge> {
    if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
        let log_T = DoryGlobals::get_T().log_2();
        let log_K = opening_point.len().saturating_sub(log_T);
        let (r_address, r_cycle) = opening_point.split_at(log_K);
        [r_cycle, r_address].concat()
    } else {
        opening_point.to_vec()
    }
}
