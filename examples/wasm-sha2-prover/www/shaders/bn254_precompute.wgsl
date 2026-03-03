// G2 line evaluation precomputation kernel
// One thread per G2 point — precomputes all EllCoeffs through the ate loop.
// Output is stored in iteration-major order for coalesced reads by the apply kernel.
//
// This file is concatenated after bn254_common.wgsl by the host.

@group(0) @binding(0) var<storage, read> g2_points: array<u32>;
@group(0) @binding(1) var<storage, read_write> coeffs_out: array<u32>;

struct PrecomputeParams {
    num_g2: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: PrecomputeParams;

// 64 doublings + 25 NAF additions + 2 Frobenius additions = 91
const COEFFS_PER_G2: u32 = 91u;
// Each EllCoeffs: 3 Fp2 = 6 BigInt = 48 u32s
const COEFF_WORDS: u32 = 48u;

fn store_ell_coeffs(g2_idx: u32, coeff_idx: u32, coeffs: EllCoeffs) {
    // Iteration-major layout: coeffs[coeff_idx][g2_idx]
    let base = (coeff_idx * params.num_g2 + g2_idx) * COEFF_WORDS;
    for (var i = 0u; i < 8u; i = i + 1u) {
        coeffs_out[base + 0u * 8u + i] = coeffs.ell_0.c0.limbs[i];
        coeffs_out[base + 1u * 8u + i] = coeffs.ell_0.c1.limbs[i];
        coeffs_out[base + 2u * 8u + i] = coeffs.ell_vw.c0.limbs[i];
        coeffs_out[base + 3u * 8u + i] = coeffs.ell_vw.c1.limbs[i];
        coeffs_out[base + 4u * 8u + i] = coeffs.ell_vv.c0.limbs[i];
        coeffs_out[base + 5u * 8u + i] = coeffs.ell_vv.c1.limbs[i];
    }
}

@compute @workgroup_size(128)
fn precompute_g2_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_g2) { return; }

    // Load G2 affine point
    let g2_offset = tid * 4u * 8u;
    var Q: G2Affine;
    for (var i = 0u; i < 8u; i = i + 1u) {
        Q.x.c0.limbs[i] = g2_points[g2_offset + i];
        Q.x.c1.limbs[i] = g2_points[g2_offset + 8u + i];
        Q.y.c0.limbs[i] = g2_points[g2_offset + 16u + i];
        Q.y.c1.limbs[i] = g2_points[g2_offset + 24u + i];
    }

    var T: G2Jacobian;
    T.x = Q.x;
    T.y = Q.y;
    T.z = fp2_one();

    var coeff_idx = 0u;

    // Ate loop (same traversal as miller_loop_single)
    for (var i: i32 = 1; i < ATE_LOOP_NAF_LEN; i = i + 1) {
        let dbl_res = g2_double_eval(T);
        T = dbl_res.point;
        store_ell_coeffs(tid, coeff_idx, dbl_res.coeffs);
        coeff_idx = coeff_idx + 1u;

        let naf_val = ATE_LOOP_NAF[i];
        if (naf_val == 1 || naf_val == -1) {
            var q_to_add: G2Affine;
            if (naf_val == 1) {
                q_to_add = Q;
            } else {
                q_to_add.x = Q.x;
                q_to_add.y = fp2_neg(Q.y);
            }
            let add_res = g2_add_eval(T, q_to_add);
            T = add_res.point;
            store_ell_coeffs(tid, coeff_idx, add_res.coeffs);
            coeff_idx = coeff_idx + 1u;
        }
    }

    // Frobenius endomorphism: Q1
    var Q1: G2Affine;
    let q_x_conj = fp2_conjugate(Q.x);
    let q_y_conj = fp2_conjugate(Q.y);

    var frob_x: Fp2;
    var frob_y: Fp2;
    for (var i = 0u; i < 8u; i = i + 1u) {
        frob_x.c0.limbs[i] = FROBENIUS_COEFF_X_C0[i];
        frob_x.c1.limbs[i] = FROBENIUS_COEFF_X_C1[i];
        frob_y.c0.limbs[i] = FROBENIUS_COEFF_Y_C0[i];
        frob_y.c1.limbs[i] = FROBENIUS_COEFF_Y_C1[i];
    }

    Q1.x = fp2_mul(q_x_conj, frob_x);
    Q1.y = fp2_mul(q_y_conj, frob_y);

    let add_q1 = g2_add_eval(T, Q1);
    T = add_q1.point;
    store_ell_coeffs(tid, coeff_idx, add_q1.coeffs);
    coeff_idx = coeff_idx + 1u;

    // Frobenius endomorphism: -Q2
    var neg_Q2: G2Affine;
    var frob_x2: Fp2;
    var frob_y2: Fp2;
    for (var i = 0u; i < 8u; i = i + 1u) {
        frob_x2.c0.limbs[i] = FROBENIUS_COEFF_X2_C0[i];
        frob_x2.c1.limbs[i] = FROBENIUS_COEFF_X2_C1[i];
        frob_y2.c0.limbs[i] = FROBENIUS_COEFF_Y2_C0[i];
        frob_y2.c1.limbs[i] = FROBENIUS_COEFF_Y2_C1[i];
    }
    neg_Q2.x = fp2_mul(Q.x, frob_x2);
    neg_Q2.y = fp2_neg(fp2_mul(Q.y, frob_y2));

    let add_q2 = g2_add_eval(T, neg_Q2);
    store_ell_coeffs(tid, coeff_idx, add_q2.coeffs);
}
