use std::sync::mpsc;

use crate::error::GpuError;

pub fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jolt-gpu-bind-group"),
        layout,
        entries,
    })
}

#[allow(dead_code)]
pub fn read_buffer_sync(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, GpuError> {
    let _span = tracing::debug_span!("read_buffer_sync", size).entered();

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jolt-gpu-readback-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jolt-gpu-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = mpsc::channel();
    staging_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

    let _ = device.poll(wgpu::Maintain::Wait);

    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Err(GpuError::BufferError(err.to_string())),
        Err(err) => return Err(GpuError::BufferError(err.to_string())),
    }

    let data = staging_buffer.slice(..).get_mapped_range().to_vec();
    staging_buffer.unmap();
    tracing::debug!(bytes = data.len(), "finished synchronous GPU readback");
    Ok(data)
}

#[allow(dead_code)]
pub async fn read_buffer_async(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, GpuError> {
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jolt-gpu-readback-staging-async"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jolt-gpu-readback-encoder-async"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = mpsc::channel();
    staging_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

    device.poll(wgpu::Maintain::Wait);

    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            return Err(GpuError::BufferError(err.to_string()));
        }
        Err(err) => {
            return Err(GpuError::BufferError(err.to_string()));
        }
    }

    let data = staging_buffer.slice(..).get_mapped_range().to_vec();
    staging_buffer.unmap();
    Ok(data)
}

pub fn write_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[u8]) {
    queue.write_buffer(buffer, 0, data);
}

pub fn div_ceil(x: u32, y: u32) -> u32 {
    x.div_ceil(y)
}

#[cfg(test)]
mod tests {
    use crate::{dispatch, WgpuContext};

    #[test]
    fn test_dispatch_roundtrip() {
        let ctx_result = std::panic::catch_unwind(WgpuContext::new);
        let Ok(Ok(ctx)) = ctx_result else {
            return;
        };

        let expected: Vec<u8> = (0_u8..=127_u8).collect();
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dispatch-roundtrip-buffer"),
            size: expected.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        dispatch::write_buffer(&ctx.queue, &buffer, &expected);
        let actual =
            dispatch::read_buffer_sync(&ctx.device, &ctx.queue, &buffer, expected.len() as u64)
                .expect("readback should succeed");
        assert_eq!(actual, expected);
    }
}
