#![cfg(feature = "gpu")]

use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_ec::CurveGroup;

use super::wrappers::{ArkG1, ArkG2, ArkGT, BN254};
use dory::primitives::arithmetic::PairingCurve;

use std::sync::OnceLock;

static SHADER_REGISTRY: OnceLock<Result<jolt_gpu::ShaderRegistry, jolt_gpu::GpuError>> =
    OnceLock::new();

fn get_registry(
    ctx: &jolt_gpu::WgpuContext,
) -> Result<&'static jolt_gpu::ShaderRegistry, &'static jolt_gpu::GpuError> {
    SHADER_REGISTRY
        .get_or_init(|| jolt_gpu::ShaderRegistry::new(&ctx.device))
        .as_ref()
}

pub fn native_gpu_multi_group_pairing(groups: &[(&[ArkG1], &[ArkG2])]) -> Vec<ArkGT> {
    let ctx = match jolt_gpu::get_or_init_context() {
        Some(ctx) => ctx,
        None => return cpu_fallback(groups),
    };

    let registry = match get_registry(&ctx) {
        Ok(r) => r,
        Err(_) => return cpu_fallback(groups),
    };

    let affine_groups: Vec<(Vec<G1Affine>, Vec<G2Affine>)> = groups
        .iter()
        .map(|(g1s, g2s)| {
            let g1_proj: Vec<_> = g1s.iter().map(|g| g.0).collect();
            let g2_proj: Vec<_> = g2s.iter().map(|g| g.0).collect();
            (
                <ark_bn254::G1Projective as CurveGroup>::normalize_batch(&g1_proj),
                <ark_bn254::G2Projective as CurveGroup>::normalize_batch(&g2_proj),
            )
        })
        .collect();

    let group_refs: Vec<(&[G1Affine], &[G2Affine])> = affine_groups
        .iter()
        .map(|(g1, g2)| (g1.as_slice(), g2.as_slice()))
        .collect();

    match jolt_gpu::hybrid_batch_multi_pairing(&ctx, registry, &group_refs) {
        Ok(miller_results) => miller_results
            .into_iter()
            .map(|fp12| {
                let result = Bn254::final_exponentiation(MillerLoopOutput::<Bn254>(fp12))
                    .expect("final exponentiation must not fail");
                ArkGT(result.0)
            })
            .collect(),
        Err(_) => cpu_fallback(groups),
    }
}

fn cpu_fallback(groups: &[(&[ArkG1], &[ArkG2])]) -> Vec<ArkGT> {
    groups
        .iter()
        .map(|(g1s, g2s)| BN254::multi_pair(g1s, g2s))
        .collect()
}
