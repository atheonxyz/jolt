// Apply precomputed G2 EllCoeffs to G1 points — no G2 Jacobian arithmetic.
// One thread per (G1, G2) pair. G2 index derived via tid % num_g2_bases.
// Reads precomputed coefficients in iteration-major layout from the precompute pass.
//
// This file is concatenated after bn254_common.wgsl by the host.

@group(0) @binding(0) var<storage, read> g1_points: array<u32>;
@group(0) @binding(1) var<storage, read> precomputed_coeffs: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

struct ApplyParams {
    num_pairs: u32,
    num_g2_bases: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: ApplyParams;

const COEFFS_PER_G2: u32 = 91u;
const COEFF_WORDS: u32 = 48u;

fn load_ell_coeffs(g2_idx: u32, coeff_idx: u32) -> EllCoeffs {
    // Iteration-major layout: coeffs[coeff_idx][g2_idx]
    let base = (coeff_idx * params.num_g2_bases + g2_idx) * COEFF_WORDS;
    var coeffs: EllCoeffs;
    for (var i = 0u; i < 8u; i = i + 1u) {
        coeffs.ell_0.c0.limbs[i] = precomputed_coeffs[base + 0u * 8u + i];
        coeffs.ell_0.c1.limbs[i] = precomputed_coeffs[base + 1u * 8u + i];
        coeffs.ell_vw.c0.limbs[i] = precomputed_coeffs[base + 2u * 8u + i];
        coeffs.ell_vw.c1.limbs[i] = precomputed_coeffs[base + 3u * 8u + i];
        coeffs.ell_vv.c0.limbs[i] = precomputed_coeffs[base + 4u * 8u + i];
        coeffs.ell_vv.c1.limbs[i] = precomputed_coeffs[base + 5u * 8u + i];
    }
    return coeffs;
}

@compute @workgroup_size(128)
fn apply_line_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_pairs) { return; }

    // Load G1 affine point
    let g1_offset = tid * 2u * 8u;
    var p_x: BigInt;
    var p_y: BigInt;
    for (var i = 0u; i < 8u; i = i + 1u) {
        p_x.limbs[i] = g1_points[g1_offset + i];
        p_y.limbs[i] = g1_points[g1_offset + 8u + i];
    }

    // G2 index: shared across groups (tid % num_g2_bases)
    let g2_idx = tid % params.num_g2_bases;

    var f = fp12_one();
    var coeff_idx = 0u;

    // Ate loop — same traversal order as precompute kernel
    for (var i: i32 = 1; i < ATE_LOOP_NAF_LEN; i = i + 1) {
        if (i > 1) {
            f = fp12_sqr(f);
        }

        // Doubling coefficients (always present)
        let dbl_coeffs = load_ell_coeffs(g2_idx, coeff_idx);
        coeff_idx = coeff_idx + 1u;

        let naf_val = ATE_LOOP_NAF[i];
        if (naf_val == 1 || naf_val == -1) {
            // Addition coefficients
            let add_coeffs = load_ell_coeffs(g2_idx, coeff_idx);
            coeff_idx = coeff_idx + 1u;

            // Sparse × sparse, then apply to f
            let c0_d = fp2_mul_by_fp(dbl_coeffs.ell_0, p_y);
            let c3_d = fp2_mul_by_fp(dbl_coeffs.ell_vw, p_x);
            let c4_d = dbl_coeffs.ell_vv;
            let c0_a = fp2_mul_by_fp(add_coeffs.ell_0, p_y);
            let c3_a = fp2_mul_by_fp(add_coeffs.ell_vw, p_x);
            let c4_a = add_coeffs.ell_vv;
            let sparse = fp12_mul_034_by_034(c0_d, c3_d, c4_d, c0_a, c3_a, c4_a);
            f = fp12_mul_by_01234(f, sparse);
        } else {
            // Apply single line evaluation
            f = apply_line_to_f(f, dbl_coeffs, p_x, p_y);
        }
    }

    // Frobenius Q1 + (-Q2) coefficients (last 2 entries)
    let frob_q1 = load_ell_coeffs(g2_idx, coeff_idx);
    coeff_idx = coeff_idx + 1u;
    let frob_q2 = load_ell_coeffs(g2_idx, coeff_idx);

    let c0_q1 = fp2_mul_by_fp(frob_q1.ell_0, p_y);
    let c3_q1 = fp2_mul_by_fp(frob_q1.ell_vw, p_x);
    let c4_q1 = frob_q1.ell_vv;
    let c0_q2 = fp2_mul_by_fp(frob_q2.ell_0, p_y);
    let c3_q2 = fp2_mul_by_fp(frob_q2.ell_vw, p_x);
    let c4_q2 = frob_q2.ell_vv;
    let sparse_frob = fp12_mul_034_by_034(c0_q1, c3_q1, c4_q1, c0_q2, c3_q2, c4_q2);
    f = fp12_mul_by_01234(f, sparse_frob);

    // Write Fp12 result
    let out_offset = tid * 12u * 8u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        results[out_offset + 0u * 8u + i] = f.c0.c0.c0.limbs[i];
        results[out_offset + 1u * 8u + i] = f.c0.c0.c1.limbs[i];
        results[out_offset + 2u * 8u + i] = f.c0.c1.c0.limbs[i];
        results[out_offset + 3u * 8u + i] = f.c0.c1.c1.limbs[i];
        results[out_offset + 4u * 8u + i] = f.c0.c2.c0.limbs[i];
        results[out_offset + 5u * 8u + i] = f.c0.c2.c1.limbs[i];
        results[out_offset + 6u * 8u + i] = f.c1.c0.c0.limbs[i];
        results[out_offset + 7u * 8u + i] = f.c1.c0.c1.limbs[i];
        results[out_offset + 8u * 8u + i] = f.c1.c1.c0.limbs[i];
        results[out_offset + 9u * 8u + i] = f.c1.c1.c1.limbs[i];
        results[out_offset + 10u * 8u + i] = f.c1.c2.c0.limbs[i];
        results[out_offset + 11u * 8u + i] = f.c1.c2.c1.limbs[i];
    }
}
