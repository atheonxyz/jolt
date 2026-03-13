pub fn create_storage_buffer(
    device: &wgpu::Device,
    size: u64,
    label: Option<&str>,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub fn create_uniform_buffer(
    device: &wgpu::Device,
    size: u64,
    label: Option<&str>,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn create_staging_buffer(
    device: &wgpu::Device,
    size: u64,
    label: Option<&str>,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

pub fn write_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[u8]) {
    queue.write_buffer(buffer, 0, data);
}

#[allow(dead_code)]
pub fn read_buffer_sync(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let staging = create_staging_buffer(device, size, Some("readback-staging-buffer"));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    pollster::block_on(async {
        receiver
            .recv()
            .expect("staging map callback dropped unexpectedly")
            .expect("failed to map staging buffer for read")
    });

    let bytes = slice.get_mapped_range().to_vec();
    staging.unmap();
    bytes
}

pub fn dispatch_compute(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroup_count: u32,
) {
    let span = tracing::debug_span!("dispatch_compute", workgroup_count);
    let _guard = span.enter();
    tracing::debug!("starting compute dispatch");

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute-dispatch-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}

pub fn div_ceil(x: u32, y: u32) -> u32 {
    x.div_ceil(y)
}

#[cfg(test)]
mod tests {
    use super::{create_storage_buffer, div_ceil, read_buffer_sync, write_buffer};
    use crate::context::WgpuContext;
    use crate::error::GpuError;

    #[test]
    fn test_dispatch_roundtrip() {
        let context = match WgpuContext::new() {
            Ok(context) => context,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let expected: Vec<u8> = vec![1, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let storage = create_storage_buffer(
            &context.device,
            expected.len() as u64,
            Some("test-roundtrip-storage"),
        );

        write_buffer(&context.queue, &storage, &expected);
        let actual = read_buffer_sync(
            &context.device,
            &context.queue,
            &storage,
            expected.len() as u64,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(9, 3), 3);
        assert_eq!(div_ceil(0, 3), 0);
    }
}
