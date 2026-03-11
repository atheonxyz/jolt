//! Custom DoryRoutines implementations using jolt_optimizations

use super::wrappers::{ArkFr, ArkG1, ArkG2};
use crate::msm::VariableBaseMSM;
use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ec::CurveGroup;
use dory::primitives::arithmetic::DoryRoutines;
use rayon::prelude::*;

/// left[i] = left[i] * scalar + right[i]
fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "Lengths must match");
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(l, r)| {
            *l = *l * *scalar + *r;
        });
}

pub struct JoltG1Routines;

impl DoryRoutines<ArkG1> for JoltG1Routines {
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        // SAFETY: ArkG1 has same memory layout as G1Projective
        let projective_points: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, bases.len())
        };
        let affines = G1Projective::normalize_batch(projective_points);

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        // Only use the first scalars.len() bases to match the scalar count
        let result = VariableBaseMSM::msm_field_elements(&affines, raw_scalars)
            .expect("msm_field_elements should not fail");

        ArkG1(result)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        if scalars.is_empty() {
            return vec![];
        }

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        let results_proj = jolt_optimizations::fixed_base_vector_msm_g1(&base.0, raw_scalars);

        results_proj.into_iter().map(ArkG1).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");

        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let bases_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, bases.len())
        };

        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        jolt_optimizations::vector_add_scalar_mul_g1_online(vs_proj, bases_proj, raw_scalar);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );

        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let addends_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G1Projective, addends.len())
        };

        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        jolt_optimizations::vector_scalar_mul_add_gamma_g1_mixed(
            vs_proj,
            raw_scalar,
            addends_proj,
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

impl JoltG1Routines {
    /// Phase 1 fold using precomputed windowed2 tables for SRS generators.
    /// `v[i] = v[i] + scalar * bases[i]` with 2-bit windowed GLV.
    /// Accepts a table slice so callers can pass `&tables[..n]` for sub-ranges.
    pub fn fixed_scalar_mul_bases_then_add_windowed2(
        vs: &mut [ArkG1],
        scalar: &ArkFr,
        tables: &[jolt_optimizations::Windowed2Signed2Table],
    ) {
        use jolt_optimizations::decomp_2d::decompose_scalar_2d;
        assert_eq!(vs.len(), tables.len());
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };
        let (coeffs, signs) = decompose_scalar_2d(raw_scalar);

        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        vs_proj
            .par_iter_mut()
            .zip(tables.par_iter())
            .for_each(|(vi, table)| {
                let prod = jolt_optimizations::glv_two::glv_two_scalar_mul_windowed2_signed_single(table, &coeffs, &signs);
                *vi += prod;
            });
    }

    /// Phase 2 fold using online windowed2 GLV (v changes each round, can't precompute).
    /// `v[i] = scalar * v[i] + addends[i]`
    pub fn fixed_scalar_mul_vs_then_add_windowed2(
        vs: &mut [ArkG1],
        addends: &[ArkG1],
        scalar: &ArkFr,
    ) {
        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let addends_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G1Projective, addends.len())
        };
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };
        jolt_optimizations::dory_g1::vector_scalar_mul_add_gamma_g1_windowed2_signed(
            vs_proj,
            raw_scalar,
            addends_proj,
        );
    }

    /// Precompute windowed2 tables for G1 SRS generators.
    pub fn precompute_windowed2(
        bases: &[ArkG1],
    ) -> jolt_optimizations::Windowed2Signed2Data {
        let bases_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, bases.len())
        };
        jolt_optimizations::glv_two_precompute_windowed2_signed(bases_proj)
    }

    /// Phase 2 fold with mixed-coordinate addition.
    /// Batch-converts v to affine, then uses Projective + Affine mixed addition
    /// in the Shamir loop (~26% cheaper per addition than projective + projective).
    /// `v[i] = scalar * v[i] + addends[i]`
    pub fn fixed_scalar_mul_vs_then_add_mixed(
        vs: &mut [ArkG1],
        addends: &[ArkG1],
        scalar: &ArkFr,
    ) {
        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let addends_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G1Projective, addends.len())
        };
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };
        jolt_optimizations::vector_scalar_mul_add_gamma_g1_mixed(
            vs_proj,
            raw_scalar,
            addends_proj,
        );
    }
}

pub struct JoltG2Routines;

impl DoryRoutines<ArkG2> for JoltG2Routines {
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        // SAFETY: ArkG2 has same memory layout as G2Projective
        let projective_points: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G2Projective, bases.len())
        };
        let affines = G2Projective::normalize_batch(projective_points);

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        // Only use the first scalars.len() bases to match the scalar count
        let result = VariableBaseMSM::msm_field_elements(&affines[..scalars.len()], raw_scalars)
            .expect("msm_field_elements should not fail");

        ArkG2(result)
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        if scalars.is_empty() {
            return vec![];
        }

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        // Use GLV-based optimization for G2
        let base_proj = base.0;

        //TODO (markosg04) this can be optimized heavily?
        let results_proj: Vec<G2Projective> = raw_scalars
            .par_iter()
            .map(|&scalar| jolt_optimizations::glv_four_scalar_mul_online(scalar, &[base_proj])[0])
            .collect();

        results_proj.into_iter().map(ArkG2).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");

        // SAFETY: ArkG2 is repr(transparent) so has same memory layout as G2Projective
        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        let bases_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G2Projective, bases.len())
        };

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        // v[i] = v[i] + scalar * bases[i]
        jolt_optimizations::vector_add_scalar_mul_g2_online(vs_proj, bases_proj, raw_scalar);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );

        // SAFETY: ArkG2 is repr(transparent) so has same memory layout as G2Projective
        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        let addends_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G2Projective, addends.len())
        };

        // SAFETY: ArkFr has same memory layout as Fr
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };

        // v[i] = scalar * v[i] + addends[i]
        jolt_optimizations::vector_scalar_mul_add_gamma_g2_online(
            vs_proj,
            raw_scalar,
            addends_proj,
        );
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors(left, right, scalar);
    }
}

impl JoltG2Routines {
    /// Phase 1 fold using precomputed windowed2 tables for SRS G2 generators.
    /// `v[i] = v[i] + scalar * bases[i]` with 2-bit windowed 4D GLV.
    /// Accepts a table slice so callers can pass `&tables[..n]` for sub-ranges.
    pub fn fixed_scalar_mul_bases_then_add_windowed2(
        vs: &mut [ArkG2],
        scalar: &ArkFr,
        tables: &[jolt_optimizations::Windowed2Signed4Table],
    ) {
        use jolt_optimizations::decomp_4d::decompose_scalar_4d;
        assert_eq!(vs.len(), tables.len());
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };
        let (coeffs, signs) = decompose_scalar_4d(raw_scalar);

        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        vs_proj
            .par_iter_mut()
            .zip(tables.par_iter())
            .for_each(|(vi, table)| {
                let prod = jolt_optimizations::glv_four_scalar_mul_windowed2_signed_single(table, &coeffs, &signs);
                *vi += prod;
            });
    }

    /// Phase 2 fold using online windowed2 4D GLV (v changes each round, can't precompute).
    /// `v[i] = scalar * v[i] + addends[i]`
    pub fn fixed_scalar_mul_vs_then_add_windowed2(
        vs: &mut [ArkG2],
        addends: &[ArkG2],
        scalar: &ArkFr,
    ) {
        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        let addends_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G2Projective, addends.len())
        };
        let raw_scalar = unsafe { std::mem::transmute_copy::<ArkFr, Fr>(scalar) };
        jolt_optimizations::dory_g2::vector_scalar_mul_add_gamma_g2_windowed2_signed(
            vs_proj,
            raw_scalar,
            addends_proj,
        );
    }

    /// Precompute windowed2 tables for G2 SRS generators.
    pub fn precompute_windowed2(
        bases: &[ArkG2],
    ) -> jolt_optimizations::Windowed2Signed4Data {
        let bases_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G2Projective, bases.len())
        };
        jolt_optimizations::glv_four_precompute_windowed2_signed(bases_proj)
    }
}
