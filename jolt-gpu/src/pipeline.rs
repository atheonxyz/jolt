use std::collections::HashMap;

use crate::{
    error::GpuError,
    shaders::{
        SHADER_BN254_COMMON, SHADER_BN254_MILLER, SHADER_BN254_REDUCE, SHADER_COMBINE_HINTS,
        SHADER_JAC2AFFINE, SHADER_MSM_G1_CURVE,
    },
};

pub struct ShaderRegistry {
    modules: HashMap<String, wgpu::ShaderModule>,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl ShaderRegistry {
    pub fn new(device: &wgpu::Device) -> Result<Self, GpuError> {
        let module_sources = [
            (
                "miller",
                [SHADER_BN254_COMMON, SHADER_BN254_MILLER].join("\n"),
            ),
            (
                "reduce",
                [SHADER_BN254_COMMON, SHADER_BN254_REDUCE].join("\n"),
            ),
            (
                "jac2affine",
                [SHADER_BN254_COMMON, SHADER_MSM_G1_CURVE, SHADER_JAC2AFFINE].join("\n"),
            ),
            (
                "combine_hints",
                [
                    SHADER_BN254_COMMON,
                    SHADER_MSM_G1_CURVE,
                    SHADER_COMBINE_HINTS,
                ]
                .join("\n"),
            ),
        ];

        let mut modules = HashMap::with_capacity(module_sources.len());
        for (name, code) in module_sources {
            let module = Self::compile_module(device, name, code)?;
            modules.insert(name.to_string(), module);
        }

        let pipeline_defs = [
            ("miller_loop_kernel", "miller"),
            ("fp12_reduce", "reduce"),
            ("jac2affine_kernel", "jac2affine"),
            ("combine_hints_kernel", "combine_hints"),
        ];

        let mut pipelines = HashMap::with_capacity(pipeline_defs.len());
        for (entry_point, module_name) in pipeline_defs {
            let module = modules.get(module_name).ok_or_else(|| {
                GpuError::ShaderCompilation(format!(
                    "missing compiled shader module `{module_name}`"
                ))
            })?;
            let pipeline = Self::create_compute_pipeline(device, module, entry_point)?;
            pipelines.insert(entry_point.to_string(), pipeline);
        }

        Ok(Self { modules, pipelines })
    }

    fn compile_module(
        device: &wgpu::Device,
        name: &str,
        code: String,
    ) -> Result<wgpu::ShaderModule, GpuError> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });

        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(GpuError::ShaderCompilation(format!(
                "failed to compile shader module `{name}`: {err}"
            )));
        }

        Ok(module)
    }

    fn create_compute_pipeline(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, GpuError> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: None,
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(GpuError::ShaderCompilation(format!(
                "failed to create compute pipeline `{entry_point}`: {err}"
            )));
        }

        Ok(pipeline)
    }

    pub fn get_pipeline(&self, name: &str) -> Option<&wgpu::ComputePipeline> {
        self.pipelines.get(name)
    }

    pub fn get_module(&self, name: &str) -> Option<&wgpu::ShaderModule> {
        self.modules.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderRegistry;
    use crate::{context::WgpuContext, error::GpuError};

    #[test]
    fn test_shader_compilation() {
        let context = match WgpuContext::new() {
            Ok(context) => context,
            Err(GpuError::NoAdapter) | Err(GpuError::NoDevice(_)) => return,
            Err(err) => panic!("unexpected GPU init error: {err}"),
        };

        let registry = ShaderRegistry::new(&context.device)
            .unwrap_or_else(|err| panic!("failed to build shader registry: {err}"));

        assert!(registry.get_module("miller").is_some());
        assert!(registry.get_module("reduce").is_some());
        assert!(registry.get_pipeline("miller_loop_kernel").is_some());
        assert!(registry.get_pipeline("fp12_reduce").is_some());
    }
}
