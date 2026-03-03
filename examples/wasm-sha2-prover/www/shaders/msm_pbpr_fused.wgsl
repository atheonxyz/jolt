override WG_SIZE: u32 = 256;

@group(0) @binding(0) var<storage, read_write> bucket_sums: array<BigInt>;
@group(0) @binding(1) var<storage, read_write> g_points: array<BigInt>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

fn load_bucket_sum(idx: u32) -> G1Jacobian {
    let base = 3u * idx;
    var pt: G1Jacobian;
    pt.x = bucket_sums[base];
    pt.y = bucket_sums[base + 1u];
    pt.z = bucket_sums[base + 2u];
    return pt;
}

fn store_g_point(idx: u32, pt: G1Jacobian) {
    let base = 3u * idx;
    g_points[base] = pt.x;
    g_points[base + 1u] = pt.y;
    g_points[base + 2u] = pt.z;
}

// Dispatch as (1, num_subtasks, 1): each workgroup handles one subtask.
// params[0] = num_columns
// params[1] = subtask_offset (for global g_points indexing)
@compute @workgroup_size(WG_SIZE, 1, 1)
fn bpr_fused(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let thread_id = local_id.x;
    let num_threads_per_subtask = WG_SIZE;

    let subtask_idx = wg_id.y;
    let num_columns = params[0];
    let subtask_offset = params[1];

    let num_buckets_per_subtask = num_columns / 2u;
    let buckets_per_thread = num_buckets_per_subtask / num_threads_per_subtask;
    let offset = num_buckets_per_subtask * subtask_idx;

    var idx = offset;
    if (thread_id != 0u) {
        idx = (num_threads_per_subtask - thread_id) * buckets_per_thread + offset;
    }

    var m = load_bucket_sum(idx);
    var g = m;
    for (var i = 0u; i < buckets_per_thread - 1u; i = i + 1u) {
        let k = (num_threads_per_subtask - thread_id) * buckets_per_thread - 1u - i;
        let bi = offset + k;
        let b = load_bucket_sum(bi);
        m = g1_add(m, b);
        g = g1_add(g, m);
    }

    let s = buckets_per_thread * (num_threads_per_subtask - thread_id - 1u);
    g = g1_add(g, g1_scalar_mul(m, s));

    let t = (subtask_idx + subtask_offset) * num_threads_per_subtask + thread_id;
    store_g_point(t, g);
}
