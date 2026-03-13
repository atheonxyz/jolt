mod backend;
mod context;
mod dispatch;
mod error;
mod pairing;
mod pipeline;
mod pool;
mod serialize;
pub mod shaders;

pub use backend::{GpuBackend, WgpuBackend};
pub use context::WgpuContext;
pub use error::GpuError;

use std::sync::{Arc, OnceLock};

static GPU_CONTEXT: OnceLock<Option<Arc<WgpuContext>>> = OnceLock::new();

pub fn is_gpu_available() -> bool {
    get_or_init_context().is_some()
}

pub fn get_or_init_context() -> Option<Arc<WgpuContext>> {
    GPU_CONTEXT
        .get_or_init(|| match WgpuContext::new() {
            Ok(ctx) => {
                tracing::info!("GPU available");
                Some(Arc::new(ctx))
            }
            Err(e) => {
                tracing::warn!("GPU unavailable: {e}");
                None
            }
        })
        .clone()
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
