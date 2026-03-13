use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("failed to create GPU device: {0}")]
    NoDevice(String),
    #[error("GPU device lost")]
    DeviceLost,
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),
    #[error("buffer operation failed: {0}")]
    BufferError(String),
    #[error("compute dispatch failed: {0}")]
    DispatchError(String),
}
