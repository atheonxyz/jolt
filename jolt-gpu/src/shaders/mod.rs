pub const COMMON: &str = include_str!("bn254_common.wgsl");
pub const MILLER: &str = include_str!("bn254_miller.wgsl");
pub const REDUCE: &str = include_str!("bn254_reduce.wgsl");
pub const COMBINE_HINTS: &str = include_str!("combine_hints.wgsl");

pub const G2_FIXED_BASE_MUL: &str = include_str!("g2_fixed_base_mul.wgsl");
pub const G2_SRS_FOLD_OPT: &str = include_str!("g2_srs_fold_opt.wgsl");
pub const G2_VAR_BASE_SCALAR_MUL: &str = include_str!("g2_var_base_scalar_mul.wgsl");
pub const JAC2AFFINE: &str = include_str!("jac2affine.wgsl");

pub const MSM_CSC_SETUP: &str = include_str!("msm_csc_setup.wgsl");
pub const MSM_G1_CURVE: &str = include_str!("msm_g1_curve.wgsl");
pub const MSM_GLV: &str = include_str!("msm_glv.wgsl");
pub const MSM_HORNER: &str = include_str!("msm_horner.wgsl");
pub const MSM_PBPR: &str = include_str!("msm_pbpr.wgsl");
pub const MSM_SMVP: &str = include_str!("msm_smvp.wgsl");

pub const ONEHOT_GATHER: &str = include_str!("onehot_gather.wgsl");

pub fn get_shader_source(name: &str) -> Option<&'static str> {
    match name {
        "bn254_common" => Some(COMMON),
        "bn254_miller" => Some(MILLER),
        "bn254_reduce" => Some(REDUCE),
        "combine_hints" => Some(COMBINE_HINTS),
        "g2_fixed_base_mul" => Some(G2_FIXED_BASE_MUL),
        "g2_srs_fold_opt" => Some(G2_SRS_FOLD_OPT),
        "g2_var_base_scalar_mul" => Some(G2_VAR_BASE_SCALAR_MUL),
        "jac2affine" => Some(JAC2AFFINE),
        "msm_csc_setup" => Some(MSM_CSC_SETUP),
        "msm_g1_curve" => Some(MSM_G1_CURVE),
        "msm_glv" => Some(MSM_GLV),
        "msm_horner" => Some(MSM_HORNER),
        "msm_pbpr" => Some(MSM_PBPR),
        "msm_smvp" => Some(MSM_SMVP),
        "onehot_gather" => Some(ONEHOT_GATHER),
        _ => None,
    }
}

pub const ALL_SHADER_NAMES: &[&str] = &[
    "bn254_common",
    "bn254_miller",
    "bn254_reduce",
    "combine_hints",
    "g2_fixed_base_mul",
    "g2_srs_fold_opt",
    "g2_var_base_scalar_mul",
    "jac2affine",
    "msm_csc_setup",
    "msm_g1_curve",
    "msm_glv",
    "msm_horner",
    "msm_pbpr",
    "msm_smvp",
    "onehot_gather",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_shaders_parse() {
        let shaders_requiring_common = [
            "bn254_miller",
            "bn254_reduce",
            "g2_fixed_base_mul",
            "g2_srs_fold_opt",
            "g2_var_base_scalar_mul",
            "jac2affine",
            "msm_g1_curve",
        ];

        let shaders_requiring_common_and_g1_curve = [
            "combine_hints",
            "msm_glv",
            "msm_horner",
            "msm_pbpr",
            "msm_smvp",
            "onehot_gather",
        ];

        let mut passed = Vec::new();
        let mut failed = Vec::new();

        for shader_name in ALL_SHADER_NAMES {
            let source = match get_shader_source(shader_name) {
                Some(src) => src,
                None => {
                    failed.push((shader_name.to_string(), "Shader not found".to_string()));
                    continue;
                }
            };

            let full_source = if shaders_requiring_common_and_g1_curve.contains(shader_name) {
                format!("{}\n{}\n{}", COMMON, MSM_G1_CURVE, source)
            } else if shaders_requiring_common.contains(shader_name) {
                format!("{}\n{}", COMMON, source)
            } else {
                source.to_string()
            };

            match naga::front::wgsl::parse_str(&full_source) {
                Ok(_) => {
                    passed.push(shader_name.to_string());
                }
                Err(e) => {
                    failed.push((shader_name.to_string(), format!("{:?}", e)));
                }
            }
        }

        println!("\n=== Shader Parse Results ===");
        println!("Passed: {}/{}", passed.len(), ALL_SHADER_NAMES.len());
        for name in &passed {
            println!("  ✓ {}", name);
        }

        if !failed.is_empty() {
            println!("\nFailed: {}", failed.len());
            for (name, err) in &failed {
                println!("  ✗ {}: {}", name, err);
            }
        }

        assert!(
            failed.is_empty(),
            "Shader parsing failed for: {}",
            failed
                .iter()
                .map(|(n, _)| n)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}
