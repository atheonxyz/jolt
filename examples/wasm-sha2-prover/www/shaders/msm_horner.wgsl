override WG_SIZE: u32 = 64;

@group(0) @binding(0) var<storage, read> g_points: array<BigInt>;
@group(0) @binding(1) var<storage, read_write> results: array<BigInt>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

fn load_g_point(idx: u32) -> G1XYZZ {
    let base = 4u * idx;
    var pt: G1XYZZ;
    pt.x = g_points[base];
    pt.y = g_points[base + 1u];
    pt.zz = g_points[base + 2u];
    pt.zzz = g_points[base + 3u];
    return pt;
}

fn store_result(idx: u32, pt: G1Jacobian) {
    let base = 3u * idx;
    results[base] = pt.x;
    results[base + 1u] = pt.y;
    results[base + 2u] = pt.z;
}

fn reduce_subtask(msm_idx: u32, subtask_idx: u32, subtasks_per_msm: u32, b_wg_size: u32) -> G1XYZZ {
    let global_subtask = msm_idx * subtasks_per_msm + subtask_idx;
    let point_offset = global_subtask * b_wg_size;
    var sum = xyzz_zero();
    var sum_initialized = false;
    for (var i = 0u; i < b_wg_size; i = i + 1u) {
        let pt = load_g_point(point_offset + i);
        if (is_xyzz_zero(pt)) { continue; }
        if (!sum_initialized) {
            sum = pt;
            sum_initialized = true;
        } else {
            sum = xyzz_add(sum, pt);
        }
    }
    return sum;
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn horner_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let msm_idx = gid.x;
    let batch_size = params[0];
    if (msm_idx >= batch_size) { return; }

    let subtasks_per_msm = params[1];
    let b_wg_size = params[2];
    let window_size = params[3];

    if (subtasks_per_msm == 0u) {
        store_result(msm_idx, g1_zero_mont());
        return;
    }

    let window_scalar = 1u << window_size;
    var acc = reduce_subtask(msm_idx, subtasks_per_msm - 1u, subtasks_per_msm, b_wg_size);

    for (var i = i32(subtasks_per_msm) - 2; i >= 0; i = i - 1) {
        if (!is_xyzz_zero(acc)) {
            acc = xyzz_scalar_mul(acc, window_scalar);
        }
        let subtask_point = reduce_subtask(msm_idx, u32(i), subtasks_per_msm, b_wg_size);
        if (!is_xyzz_zero(subtask_point)) {
            if (is_xyzz_zero(acc)) {
                acc = subtask_point;
            } else {
                acc = xyzz_add(acc, subtask_point);
            }
        }
    }

    // Convert XYZZ to Jacobian for output
    store_result(msm_idx, xyzz_to_jacobian(acc));
}
