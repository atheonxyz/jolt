#![allow(dead_code)]

use std::collections::HashMap;

use crate::{error::GpuError, shaders, WgpuContext};

const PAIRING_PIPELINES: &[(&str, &str)] = &[
    ("bn254_miller", "miller_loop_kernel"),
    ("bn254_reduce", "fp12_reduce"),
];

pub struct ShaderRegistry {
    shaders: HashMap<String, wgpu::ShaderModule>,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl ShaderRegistry {
    pub fn new(context: &WgpuContext) -> Result<Self, GpuError> {
        let mut shaders_map = HashMap::new();

        for &(shader_name, _) in PAIRING_PIPELINES {
            let shader_source = full_shader_source(shader_name)?;
            let shader = compile_shader_module(&context.device, shader_name, shader_source)?;
            shaders_map.insert(shader_name.to_string(), shader);
        }

        let mut pipelines = HashMap::new();
        for &(shader_name, entry_point) in PAIRING_PIPELINES {
            let shader = shaders_map.get(shader_name).ok_or_else(|| {
                GpuError::ShaderCompilation(format!(
                    "missing compiled shader module: {shader_name}"
                ))
            })?;
            let pipeline =
                create_compute_pipeline(&context.device, shader_name, shader, entry_point)?;
            pipelines.insert(shader_name.to_string(), pipeline);
        }

        Ok(Self {
            shaders: shaders_map,
            pipelines,
        })
    }

    pub fn get_pipeline(&self, name: &str) -> Option<&wgpu::ComputePipeline> {
        self.pipelines.get(name)
    }

    pub fn get_shader(&self, name: &str) -> Option<&wgpu::ShaderModule> {
        self.shaders.get(name)
    }
}

fn full_shader_source(name: &str) -> Result<String, GpuError> {
    let source = shaders::get_shader_source(name)
        .ok_or_else(|| GpuError::ShaderCompilation(format!("unknown shader source: {name}")))?;

    let full_source = match name {
        "bn254_miller"
        | "bn254_reduce"
        | "g2_fixed_base_mul"
        | "g2_srs_fold_opt"
        | "g2_var_base_scalar_mul"
        | "jac2affine"
        | "msm_g1_curve" => format!("{}\n{}", shaders::COMMON, source),
        "combine_hints" | "msm_glv" | "msm_horner" | "msm_pbpr" | "msm_smvp" | "onehot_gather" => {
            format!("{}\n{}\n{}", shaders::COMMON, shaders::MSM_G1_CURVE, source)
        }
        "msm_csc_setup" => source.to_string(),
        _ => {
            return Err(GpuError::ShaderCompilation(format!(
                "unsupported shader registry source: {name}"
            )));
        }
    };

    Ok(full_source)
}

fn compile_shader_module(
    device: &wgpu::Device,
    name: &str,
    source: String,
) -> Result<wgpu::ShaderModule, GpuError> {
    let label = format!("jolt-gpu-shader-{name}");
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label.as_str()),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let mut errors = Vec::new();
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        errors.push(error.to_string());
    }

    let compilation_info = pollster::block_on(shader.get_compilation_info());
    errors.extend(
        compilation_info
            .messages
            .into_iter()
            .filter(|message| message.message_type == wgpu::CompilationMessageType::Error)
            .map(|message| {
                if let Some(location) = message.location {
                    format!(
                        "{}:{}:{}: {}",
                        location.line_number,
                        location.line_position,
                        location.offset,
                        message.message
                    )
                } else {
                    message.message
                }
            }),
    );

    if errors.is_empty() {
        Ok(shader)
    } else {
        Err(GpuError::ShaderCompilation(format!(
            "{name}: {}",
            errors.join(" | ")
        )))
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    name: &str,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> Result<wgpu::ComputePipeline, GpuError> {
    let label = format!("jolt-gpu-pipeline-{name}");
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label.as_str()),
        layout: None,
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        Err(GpuError::ShaderCompilation(format!(
            "{name} ({entry_point}): {error}"
        )))
    } else {
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderRegistry;
    use crate::WgpuContext;

    #[test]
    fn test_shader_compilation() {
        let context = WgpuContext::new().expect("failed to create WGPU context");
        let registry = ShaderRegistry::new(&context).expect("failed to compile shader registry");

        assert!(registry.get_shader("bn254_miller").is_some());
        assert!(registry.get_shader("bn254_reduce").is_some());
        assert!(registry.get_pipeline("bn254_miller").is_some());
        assert!(registry.get_pipeline("bn254_reduce").is_some());
    }
}
