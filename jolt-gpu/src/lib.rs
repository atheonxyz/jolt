mod context;
mod dispatch;
mod error;
mod pairing;
mod pipeline;
#[allow(dead_code)]
mod pool;
mod serialize;
pub mod shaders;

pub use context::WgpuContext;
pub use error::GpuError;
pub use pairing::{gpu_batch_multi_pairing, hybrid_batch_multi_pairing};
pub use pipeline::ShaderRegistry;
pub use serialize::{
    deserialize_fq12, fq_to_limbs, limbs8_to_fq, serialize_g1_affine, serialize_g2_affine,
    FP12_WORDS, FQ_LIMBS, G1_WORDS, G2_WORDS,
};

use ark_bn254::{Fq12, G1Affine, G2Affine};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

pub struct GpuPairingEngine {
    ctx: WgpuContext,
    registry: ShaderRegistry,
}

impl GpuPairingEngine {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, GpuError> {
        let ctx = WgpuContext::new()?;
        let registry = ShaderRegistry::new(&ctx)?;
        Ok(Self { ctx, registry })
    }

    pub async fn new_async() -> Result<Self, GpuError> {
        let ctx = WgpuContext::new_async().await?;
        let registry = ShaderRegistry::new(&ctx)?;
        Ok(Self { ctx, registry })
    }

    pub fn batch_multi_pairing(
        &self,
        groups: &[(&[G1Affine], &[G2Affine])],
    ) -> Result<Vec<Fq12>, GpuError> {
        gpu_batch_multi_pairing(&self.ctx, &self.registry, groups)
    }

    pub fn hybrid_multi_pairing(
        &self,
        groups: &[(&[G1Affine], &[G2Affine])],
        gpu_ratio: f64,
    ) -> Result<Vec<Fq12>, GpuError> {
        hybrid_batch_multi_pairing(&self.ctx, &self.registry, groups, gpu_ratio)
    }
}

#[cfg(not(target_arch = "wasm32"))]
static GPU_ENGINE: OnceLock<Option<GpuPairingEngine>> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
pub fn get_or_init_engine() -> Option<&'static GpuPairingEngine> {
    GPU_ENGINE
        .get_or_init(|| match GpuPairingEngine::new() {
            Ok(engine) => {
                tracing::info!("GPU pairing engine initialized");
                Some(engine)
            }
            Err(e) => {
                tracing::warn!("GPU pairing engine unavailable: {e}");
                None
            }
        })
        .as_ref()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_gpu_available() -> bool {
    get_or_init_engine().is_some()
}

#[cfg(test)]
mod availability_tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let first_call = is_gpu_available();
        let second_call = is_gpu_available();
        assert_eq!(first_call, second_call);
    }
}
