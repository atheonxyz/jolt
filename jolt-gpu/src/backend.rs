use crate::context::WgpuContext;
use crate::error::GpuError;
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

    fn device_info(&self) -> &str;
}

pub struct WgpuBackend {
    pub(crate) ctx: Arc<WgpuContext>,
    pub(crate) info_string: String,
}

impl WgpuBackend {
    pub fn new(ctx: Arc<WgpuContext>) -> Self {
        let info_string = format!(
            "{} ({:?}, vendor: {}, device: {})",
            ctx.adapter_info.name,
            ctx.adapter_info.backend,
            ctx.adapter_info.vendor,
            ctx.adapter_info.device
        );

        Self { ctx, info_string }
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
        unimplemented!("Task 10-12: Batch pairing implementation")
    }

    fn is_available(&self) -> bool {
        let _ = &self.ctx;
        true
    }

    fn device_info(&self) -> &str {
        &self.info_string
    }
}
