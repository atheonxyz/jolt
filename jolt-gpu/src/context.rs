use crate::error::GpuError;

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}

impl WgpuContext {
    pub fn new() -> Result<Self, GpuError> {
        let instance =
            std::panic::catch_unwind(wgpu::Instance::default).map_err(|_| GpuError::NoAdapter)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let mut requested_limits = wgpu::Limits::default();
        let adapter_limits = adapter.limits();
        requested_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        requested_limits.max_buffer_size = adapter_limits.max_buffer_size;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jolt-gpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: requested_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .map_err(|err| GpuError::NoDevice(err.to_string()))?;

        tracing::info!(
            gpu_name = %adapter_info.name,
            backend = ?adapter_info.backend,
            device_type = ?adapter_info.device_type,
            "Initialized GPU context"
        );

        let limits = device.limits();

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
    use crate::error::GpuError;

    #[test]
    fn test_context_init() {
        match WgpuContext::new() {
            Ok(context) => {
                let _ = &context.device;
                let _ = &context.queue;
            }
            Err(GpuError::NoAdapter) => {}
            Err(GpuError::NoDevice(_)) => {}
            Err(err) => panic!("unexpected GPU init error: {err}"),
        }
    }

    #[test]
    fn test_context_device_info() {
        let context = match WgpuContext::new() {
            Ok(context) => context,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        assert!(!context.adapter_info.name.trim().is_empty());
        assert!(matches!(
            context.adapter_info.backend,
            wgpu::Backend::Metal | wgpu::Backend::Vulkan | wgpu::Backend::Dx12
        ));
        assert!(context.limits.max_storage_buffer_binding_size > 0);
        assert!(context.limits.max_buffer_size > 0);
    }
}
