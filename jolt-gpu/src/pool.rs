use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

use crate::WgpuContext;

#[allow(dead_code)]
type PoolKey = (u64, wgpu::BufferUsages);

#[allow(dead_code)]
pub(crate) fn size_class(size: u64) -> u64 {
    match size {
        0 => 1,
        _ => size.checked_next_power_of_two().unwrap_or(u64::MAX),
    }
}

#[allow(dead_code)]
pub(crate) struct GpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) actual_size: u64,
    pub(crate) pool_size: u64,
    usage: wgpu::BufferUsages,
}

#[allow(dead_code)]
pub(crate) struct BufferPool {
    buffers: Mutex<HashMap<PoolKey, Vec<wgpu::Buffer>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

#[allow(dead_code)]
impl BufferPool {
    pub(crate) fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub(crate) fn get_or_create(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> GpuBuffer {
        let pool_size = size_class(size);
        let key = (pool_size, usage);

        if let Some(buffer) = self
            .buffers
            .lock()
            .expect("buffer pool mutex poisoned")
            .get_mut(&key)
            .and_then(Vec::pop)
        {
            let hits = self.hits.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::debug!(requested_size = size, pool_size, usage = ?usage, hits, "buffer pool hit");

            return GpuBuffer {
                buffer,
                actual_size: size,
                pool_size,
                usage,
            };
        }

        let misses = self.misses.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::debug!(requested_size = size, pool_size, usage = ?usage, misses, "buffer pool miss");

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jolt-gpu-buffer-pool"),
            size: pool_size,
            usage,
            mapped_at_creation: false,
        });

        GpuBuffer {
            buffer,
            actual_size: size,
            pool_size,
            usage,
        }
    }

    pub(crate) fn release(&self, buffer: GpuBuffer) {
        self.buffers
            .lock()
            .expect("buffer pool mutex poisoned")
            .entry((buffer.pool_size, buffer.usage))
            .or_default()
            .push(buffer.buffer);
    }
}

#[allow(dead_code)]
impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl WgpuContext {
    pub(crate) fn create_buffer_pool(&self) -> BufferPool {
        BufferPool::new()
    }

    pub(crate) fn get_or_create_buffer(
        &self,
        pool: &BufferPool,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> GpuBuffer {
        pool.get_or_create(&self.device, size, usage)
    }

    pub(crate) fn release_buffer(&self, pool: &BufferPool, buffer: GpuBuffer) {
        pool.release(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::size_class;
    use crate::WgpuContext;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_buffer_pool() {
        let Ok(ctx) = WgpuContext::new() else {
            return;
        };

        let pool = ctx.create_buffer_pool();
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE;
        let first = ctx.get_or_create_buffer(&pool, 1_500, usage);

        assert_eq!(first.actual_size, 1_500);
        assert_eq!(first.pool_size, 2_048);
        assert_eq!(pool.hits.load(Ordering::Relaxed), 0);
        assert_eq!(pool.misses.load(Ordering::Relaxed), 1);

        ctx.release_buffer(&pool, first);

        let second = ctx.get_or_create_buffer(&pool, 1_500, usage);
        assert_eq!(second.actual_size, 1_500);
        assert_eq!(second.pool_size, 2_048);
        assert_eq!(pool.hits.load(Ordering::Relaxed), 1);
        assert_eq!(pool.misses.load(Ordering::Relaxed), 1);

        ctx.release_buffer(&pool, second);
    }

    #[test]
    fn test_size_class() {
        assert_eq!(size_class(0), 1);
        assert_eq!(size_class(1), 1);
        assert_eq!(size_class(2), 2);
        assert_eq!(size_class(3), 4);
        assert_eq!(size_class(1_500), 2_048);
    }
}
