use crate::error::GpuError;

#[allow(dead_code)] // fields used incrementally as Wave 2-3 dispatch code lands
pub struct WgpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    pub(crate) limits: wgpu::Limits,
}

impl WgpuContext {
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::init_internal())
    }

    async fn init_internal() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });

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
