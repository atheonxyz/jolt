mod availability;
mod backend;
mod context;
mod dispatch;
mod error;
mod pairing;
mod pipeline;
mod pool;
mod serialize;
#[allow(unused)]
mod shaders;

pub use availability::{get_or_init_context, is_gpu_available};
pub use backend::{GpuBackend, WgpuBackend};
pub use context::WgpuContext;
pub use error::GpuError;
pub use pairing::{gpu_batch_multi_pairing, hybrid_batch_multi_pairing};
pub use pipeline::ShaderRegistry;
pub use pool::BufferPool;
