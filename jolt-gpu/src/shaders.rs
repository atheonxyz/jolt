pub const SHADER_BN254_COMMON: &str = include_str!("shaders/bn254_common.wgsl");

// MSM
pub const SHADER_MSM_G1_CURVE: &str = include_str!("shaders/msm_g1_curve.wgsl");
pub const SHADER_MSM_CSC_SETUP: &str = include_str!("shaders/msm_csc_setup.wgsl");
pub const SHADER_MSM_SMVP: &str = include_str!("shaders/msm_smvp.wgsl");
pub const SHADER_MSM_PBPR: &str = include_str!("shaders/msm_pbpr.wgsl");
pub const SHADER_MSM_HORNER: &str = include_str!("shaders/msm_horner.wgsl");
pub const SHADER_MSM_GLV: &str = include_str!("shaders/msm_glv.wgsl");

// PAIRING
pub const SHADER_BN254_MILLER: &str = include_str!("shaders/bn254_miller.wgsl");
pub const SHADER_BN254_REDUCE: &str = include_str!("shaders/bn254_reduce.wgsl");

// ONEHOT
pub const SHADER_ONEHOT_GATHER: &str = include_str!("shaders/onehot_gather.wgsl");

// G2
pub const SHADER_G2_FIXED_BASE_MUL: &str = include_str!("shaders/g2_fixed_base_mul.wgsl");
pub const SHADER_G2_VAR_BASE_SCALAR_MUL: &str = include_str!("shaders/g2_var_base_scalar_mul.wgsl");
pub const SHADER_G2_SRS_FOLD_OPT: &str = include_str!("shaders/g2_srs_fold_opt.wgsl");

// UTIL
pub const SHADER_JAC2AFFINE: &str = include_str!("shaders/jac2affine.wgsl");
pub const SHADER_COMBINE_HINTS: &str = include_str!("shaders/combine_hints.wgsl");

pub fn get_shader_source(name: &str) -> Option<&'static str> {
    match name {
        "bn254_common" => Some(SHADER_BN254_COMMON),
        "msm_g1_curve" => Some(SHADER_MSM_G1_CURVE),
        "msm_csc_setup" => Some(SHADER_MSM_CSC_SETUP),
        "msm_smvp" => Some(SHADER_MSM_SMVP),
        "msm_pbpr" => Some(SHADER_MSM_PBPR),
        "msm_horner" => Some(SHADER_MSM_HORNER),
        "msm_glv" => Some(SHADER_MSM_GLV),
        "bn254_miller" => Some(SHADER_BN254_MILLER),
        "bn254_reduce" => Some(SHADER_BN254_REDUCE),
        "onehot_gather" => Some(SHADER_ONEHOT_GATHER),
        "g2_fixed_base_mul" => Some(SHADER_G2_FIXED_BASE_MUL),
        "g2_var_base_scalar_mul" => Some(SHADER_G2_VAR_BASE_SCALAR_MUL),
        "g2_srs_fold_opt" => Some(SHADER_G2_SRS_FOLD_OPT),
        "jac2affine" => Some(SHADER_JAC2AFFINE),
        "combine_hints" => Some(SHADER_COMBINE_HINTS),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_sources() {
        // Verify all 15 shader constants are non-empty
        assert!(
            !SHADER_BN254_COMMON.is_empty(),
            "SHADER_BN254_COMMON is empty"
        );
        assert!(
            !SHADER_MSM_G1_CURVE.is_empty(),
            "SHADER_MSM_G1_CURVE is empty"
        );
        assert!(
            !SHADER_MSM_CSC_SETUP.is_empty(),
            "SHADER_MSM_CSC_SETUP is empty"
        );
        assert!(!SHADER_MSM_SMVP.is_empty(), "SHADER_MSM_SMVP is empty");
        assert!(!SHADER_MSM_PBPR.is_empty(), "SHADER_MSM_PBPR is empty");
        assert!(!SHADER_MSM_HORNER.is_empty(), "SHADER_MSM_HORNER is empty");
        assert!(!SHADER_MSM_GLV.is_empty(), "SHADER_MSM_GLV is empty");
        assert!(
            !SHADER_BN254_MILLER.is_empty(),
            "SHADER_BN254_MILLER is empty"
        );
        assert!(
            !SHADER_BN254_REDUCE.is_empty(),
            "SHADER_BN254_REDUCE is empty"
        );
        assert!(
            !SHADER_ONEHOT_GATHER.is_empty(),
            "SHADER_ONEHOT_GATHER is empty"
        );
        assert!(
            !SHADER_G2_FIXED_BASE_MUL.is_empty(),
            "SHADER_G2_FIXED_BASE_MUL is empty"
        );
        assert!(
            !SHADER_G2_VAR_BASE_SCALAR_MUL.is_empty(),
            "SHADER_G2_VAR_BASE_SCALAR_MUL is empty"
        );
        assert!(
            !SHADER_G2_SRS_FOLD_OPT.is_empty(),
            "SHADER_G2_SRS_FOLD_OPT is empty"
        );
        assert!(!SHADER_JAC2AFFINE.is_empty(), "SHADER_JAC2AFFINE is empty");
        assert!(
            !SHADER_COMBINE_HINTS.is_empty(),
            "SHADER_COMBINE_HINTS is empty"
        );

        // Test get_shader_source function
        assert!(
            get_shader_source("bn254_common").is_some(),
            "get_shader_source should return Some for bn254_common"
        );
        assert!(
            !get_shader_source("bn254_common").unwrap().is_empty(),
            "bn254_common source should be non-empty"
        );

        assert_eq!(
            get_shader_source("nonexistent"),
            None,
            "get_shader_source should return None for nonexistent shader"
        );
    }

    #[test]
    fn test_naga_validation() {
        use naga::front::wgsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        struct ShaderConfig {
            name: &'static str,
            dependencies: Vec<&'static str>,
        }

        fn build_shader_source(config: &ShaderConfig) -> String {
            let mut source = String::new();
            for dep in &config.dependencies {
                if let Some(src) = get_shader_source(dep) {
                    source.push_str(src);
                    source.push('\n');
                }
            }
            if let Some(src) = get_shader_source(config.name) {
                source.push_str(src);
            }
            source
        }

        let shaders = vec![
            ShaderConfig {
                name: "bn254_common",
                dependencies: vec![],
            },
            ShaderConfig {
                name: "msm_csc_setup",
                dependencies: vec![],
            },
            ShaderConfig {
                name: "bn254_miller",
                dependencies: vec!["bn254_common"],
            },
            ShaderConfig {
                name: "bn254_reduce",
                dependencies: vec!["bn254_common"],
            },
            ShaderConfig {
                name: "g2_fixed_base_mul",
                dependencies: vec!["bn254_common"],
            },
            ShaderConfig {
                name: "g2_var_base_scalar_mul",
                dependencies: vec!["bn254_common"],
            },
            ShaderConfig {
                name: "g2_srs_fold_opt",
                dependencies: vec!["bn254_common"],
            },
            ShaderConfig {
                name: "msm_g1_curve",
                dependencies: vec!["bn254_common"],
            },
            ShaderConfig {
                name: "jac2affine",
                dependencies: vec!["bn254_common", "msm_g1_curve"],
            },
            ShaderConfig {
                name: "combine_hints",
                dependencies: vec!["bn254_common", "msm_g1_curve"],
            },
            ShaderConfig {
                name: "msm_smvp",
                dependencies: vec!["bn254_common", "msm_g1_curve"],
            },
            ShaderConfig {
                name: "msm_pbpr",
                dependencies: vec!["bn254_common", "msm_g1_curve"],
            },
            ShaderConfig {
                name: "msm_horner",
                dependencies: vec!["bn254_common", "msm_g1_curve"],
            },
            ShaderConfig {
                name: "msm_glv",
                dependencies: vec!["bn254_common", "msm_g1_curve"],
            },
            ShaderConfig {
                name: "onehot_gather",
                dependencies: vec!["bn254_common", "msm_g1_curve"],
            },
        ];

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let mut results = Vec::new();
        let mut all_passed = true;

        println!("\n=== WGSL Shader Validation Report ===\n");

        for config in shaders {
            let source = build_shader_source(&config);
            let result = wgsl::parse_str(&source);

            match result {
                Ok(module) => match validator.validate(&module) {
                    Ok(_) => {
                        println!("✓ {} - PASS", config.name);
                        results.push((config.name, true, None));
                    }
                    Err(e) => {
                        println!("✗ {} - FAIL (validation error)", config.name);
                        println!("  Error: {}", e);
                        results.push((config.name, false, Some(format!("{}", e))));
                        all_passed = false;
                    }
                },
                Err(e) => {
                    println!("✗ {} - FAIL (parse error)", config.name);
                    println!("  Error: {}", e);
                    results.push((config.name, false, Some(format!("{}", e))));
                    all_passed = false;
                }
            }
        }

        println!("\n=== Summary ===");
        let passed = results.iter().filter(|(_, ok, _)| *ok).count();
        let total = results.len();
        println!("Passed: {}/{}", passed, total);

        if !all_passed {
            println!("\nFailed shaders:");
            for (name, ok, err) in &results {
                if !ok {
                    println!(
                        "  - {}: {}",
                        name,
                        err.as_ref().unwrap_or(&"Unknown error".to_string())
                    );
                }
            }
            panic!("Shader validation failed for {} shader(s)", total - passed);
        }
    }
}
