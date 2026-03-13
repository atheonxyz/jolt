use std::collections::HashMap;

pub struct BufferPool {
    buffers: HashMap<(u64, wgpu::BufferUsages), Vec<wgpu::Buffer>>,
    hits: u64,
    misses: u64,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> wgpu::Buffer {
        let rounded_size = size_class(size);
        let key = (rounded_size, usage);

        if let Some(buffer) = self.buffers.get_mut(&key).and_then(Vec::pop) {
            self.hits += 1;
            tracing::debug!(
                requested_size = size,
                rounded_size,
                usage = usage.bits(),
                hits = self.hits,
                misses = self.misses,
                "Buffer pool hit"
            );
            return buffer;
        }

        self.misses += 1;
        tracing::debug!(
            requested_size = size,
            rounded_size,
            usage = usage.bits(),
            hits = self.hits,
            misses = self.misses,
            "Buffer pool miss"
        );

        device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: rounded_size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn release(&mut self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) {
        let rounded_size = size_class(buffer.size());
        self.buffers
            .entry((rounded_size, usage))
            .or_default()
            .push(buffer);
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

fn size_class(size: u64) -> u64 {
    size.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::{size_class, BufferPool};
    use crate::{GpuError, WgpuContext};

    #[test]
    fn test_buffer_pool_reuse() {
        let context = match WgpuContext::new() {
            Ok(context) => context,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let mut pool = BufferPool::new();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        let buffer = pool.get_or_create(&context.device, 1000, usage, Some("buffer-pool-test"));
        assert_eq!(buffer.size(), 1024);
        assert_eq!(pool.stats(), (0, 1));

        pool.release(buffer, usage);
        assert_eq!(pool.buffers.get(&(1024, usage)).map(Vec::len), Some(1));

        let reused = pool.get_or_create(&context.device, 1000, usage, Some("second-label"));
        assert_eq!(reused.size(), 1024);
        assert_eq!(pool.stats(), (1, 1));
        assert_eq!(pool.buffers.get(&(1024, usage)).map(Vec::len), Some(0));

        drop(reused);
    }

    #[test]
    fn test_buffer_pool_size_class() {
        assert_eq!(size_class(1000), 1024);
        assert_eq!(size_class(1024), 1024);
        assert_eq!(size_class(1025), 2048);
    }
}
