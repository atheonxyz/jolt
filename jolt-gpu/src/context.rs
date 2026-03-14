use crate::error::GpuError;

pub struct WgpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    #[allow(dead_code)]
    pub(crate) adapter_info: wgpu::AdapterInfo,
    #[allow(dead_code)]
    pub(crate) limits: wgpu::Limits,
}

impl WgpuContext {
    /// Blocking init for native platforms. Uses pollster to block on the async init.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::init_internal())
    }

    /// Async init for WASM and any context where blocking is undesirable.
    pub async fn new_async() -> Result<Self, GpuError> {
        Self::init_internal().await
    }

    async fn init_internal() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        tracing::info!(
            adapter_name = %adapter_info.name,
            backend = ?adapter_info.backend,
            device_type = ?adapter_info.device_type,
            "initialized wgpu adapter"
        );

        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    label: Some("jolt-gpu-device"),
                },
                None,
            )
            .await
            .map_err(|err| GpuError::NoDevice(err.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter_info,
            limits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuContext;

    #[test]
    fn test_context_init() {
        assert!(WgpuContext::new().is_ok());
    }
}
