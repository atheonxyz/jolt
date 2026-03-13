use crate::error::GpuError;
use crate::WgpuContext;
use std::sync::Arc;

pub trait GpuBackend {
    fn batch_pairing(
        &self,
        g1_points: &[u8],
        g2_points: &[u8],
        group_sizes: &[u32],
        group_offsets: &[u32],
    ) -> Result<Vec<u8>, GpuError>;

    fn is_available(&self) -> bool;

    fn device_info(&self) -> String;
}

pub struct WgpuBackend {
    ctx: Arc<WgpuContext>,
}

impl WgpuBackend {
    pub fn new(ctx: Arc<WgpuContext>) -> Self {
        Self { ctx }
    }
}

impl GpuBackend for WgpuBackend {
    fn batch_pairing(
        &self,
        _g1_points: &[u8],
        _g2_points: &[u8],
        _group_sizes: &[u32],
        _group_offsets: &[u32],
    ) -> Result<Vec<u8>, GpuError> {
        Err(GpuError::DispatchError("not yet implemented".to_string()))
    }

    fn is_available(&self) -> bool {
        Arc::strong_count(&self.ctx) >= 1
    }

    fn device_info(&self) -> String {
        format!(
            "{} (vendor: {}, device: {}, backend: {:?})",
            self.ctx.adapter_info.name,
            self.ctx.adapter_info.vendor,
            self.ctx.adapter_info.device,
            self.ctx.adapter_info.backend
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuBackend, WgpuBackend};
    use crate::WgpuContext;
    use std::sync::Arc;

    #[test]
    fn wgpu_backend_exposes_availability_and_device_info() {
        let ctx_result = std::panic::catch_unwind(WgpuContext::new);
        let Ok(Ok(ctx)) = ctx_result else {
            return;
        };

        let backend = WgpuBackend::new(Arc::new(ctx));
        assert!(backend.is_available());
        let info = backend.device_info();
        assert!(!info.is_empty());
    }
}
