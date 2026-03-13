use crate::context::WgpuContext;
use std::sync::{Arc, OnceLock};

static GPU_CONTEXT: OnceLock<Option<Arc<WgpuContext>>> = OnceLock::new();

/// Detects GPU availability and caches the result process-wide.
///
/// First call attempts to initialize a WgpuContext. Subsequent calls return
/// the cached result instantly. If GPU is unavailable, logs a warning and
/// returns false.
pub fn is_gpu_available() -> bool {
    GPU_CONTEXT
        .get_or_init(|| match WgpuContext::new() {
            Ok(context) => {
                tracing::debug!("GPU context initialized successfully");
                Some(Arc::new(context))
            }
            Err(err) => {
                tracing::warn!("GPU unavailable, falling back to CPU: {}", err);
                None
            }
        })
        .is_some()
}

/// Returns the cached GPU context if available, None otherwise.
///
/// Must call `is_gpu_available()` first to initialize the cache.
/// Thread-safe by design (OnceLock guarantees single initialization).
pub fn get_or_init_context() -> Option<Arc<WgpuContext>> {
    GPU_CONTEXT
        .get_or_init(|| match WgpuContext::new() {
            Ok(context) => {
                tracing::debug!("GPU context initialized successfully");
                Some(Arc::new(context))
            }
            Err(err) => {
                tracing::warn!("GPU unavailable, falling back to CPU: {}", err);
                None
            }
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let result1 = is_gpu_available();
        let result2 = is_gpu_available();

        // Results should be consistent across calls
        assert_eq!(result1, result2, "GPU availability should be consistent");
    }

    #[test]
    fn test_gpu_detection_cached() {
        // First call initializes
        let _result1 = is_gpu_available();

        // Second call should hit the cache (instant)
        let _result2 = is_gpu_available();

        // Both should return the same value
        assert_eq!(is_gpu_available(), is_gpu_available());
    }

    #[test]
    fn test_get_or_init_context() {
        let ctx1 = get_or_init_context();
        let ctx2 = get_or_init_context();

        // Both should return the same cached value
        match (ctx1, ctx2) {
            (Some(c1), Some(c2)) => {
                // Both should be the same Arc
                assert!(Arc::ptr_eq(&c1, &c2), "Context should be cached");
            }
            (None, None) => {
                // Both unavailable is fine
            }
            _ => panic!("Context availability should be consistent"),
        }
    }

    #[test]
    fn test_context_consistency() {
        let is_available = is_gpu_available();
        let context = get_or_init_context();

        // Consistency check: if is_gpu_available() returns true, context should be Some
        if is_available {
            assert!(
                context.is_some(),
                "Context should be available if is_gpu_available() is true"
            );
        } else {
            assert!(
                context.is_none(),
                "Context should be None if is_gpu_available() is false"
            );
        }
    }
}
