override WG_SIZE: u32 = 64;

@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> val_idx: array<u32>;
@group(0) @binding(2) var<storage, read> points_raw: array<u32>;
@group(0) @binding(3) var<storage, read_write> buckets: array<BigInt>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
// binding 5 (sort_perm) removed — GPU CSC setup handles ordering directly

@compute @workgroup_size(WG_SIZE, 1, 1)
fn smvp(
    @builtin(workgroup_id) tgid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>,
) {
    let group_id = (tgid.x * num_wgs.y + tgid.y) * num_wgs.z + tgid.z;
    let id = group_id * WG_SIZE + tid.x;

    let input_size = params[0];
    let num_columns = params[1];
    let num_subtasks = params[2];
    let subtask_offset = params[3];
    let csc_base_offset = params[4];
    let PT_STRIDE = 16u;
    let half_columns = num_columns / 2u;
    let subtask_idx = id / half_columns;
    let global_subtask = subtask_idx + subtask_offset;
    if (subtask_idx >= num_subtasks) { return; }

    let local_subtask = global_subtask - csc_base_offset;
    let rp_offset = local_subtask * (num_columns + 1u);
    let val_idx_base = local_subtask * input_size;
    let mapped_col = id % half_columns;

    let inf = g1_zero_mont();
    for (var j = 0u; j < 2u; j = j + 1u) {
        var row_idx = mapped_col + half_columns;
        if (j == 1u) {
            row_idx = half_columns - mapped_col;
        }
        if (j == 0u && mapped_col == 0u) {
            row_idx = 0u;
        }

        let row_begin = row_ptr[rp_offset + row_idx];
        let row_end = row_ptr[rp_offset + row_idx + 1u];
        var sum = inf;
        var sum_initialized = false;

        for (var k = row_begin; k < row_end; k = k + 1u) {
            let idx = val_idx[val_idx_base + k];
            let pt_base = idx * PT_STRIDE;
            if (!sum_initialized) {
                for (var i = 0u; i < 8u; i = i + 1u) {
                    sum.x.limbs[i] = points_raw[pt_base + i];
                    sum.y.limbs[i] = points_raw[pt_base + 8u + i];
                }
                sum.z = g1_one_mont_z();
                sum_initialized = true;
            } else {
                var b_aff: G1Affine;
                for (var i = 0u; i < 8u; i = i + 1u) {
                    b_aff.x.limbs[i] = points_raw[pt_base + i];
                    b_aff.y.limbs[i] = points_raw[pt_base + 8u + i];
                }
                sum = g1_madd_fast(sum, b_aff);
            }
        }

        var bucket_idx = 0u;
        if (half_columns > row_idx) {
            bucket_idx = half_columns - row_idx;
            sum = g1_neg(sum);
        } else {
            bucket_idx = row_idx - half_columns;
        }

        let bi = mapped_col + subtask_idx * half_columns;
        let bucket_base_idx = 3u * bi;
        let bucket_size = half_columns * num_subtasks * 3u;
        if (bucket_idx > 0u && bucket_base_idx + 2u < bucket_size) {
            if (j == 1u) {
                var bucket_val: G1Jacobian;
                bucket_val.x = buckets[bucket_base_idx];
                bucket_val.y = buckets[bucket_base_idx + 1u];
                bucket_val.z = buckets[bucket_base_idx + 2u];
                sum = g1_add(bucket_val, sum);
            }

            buckets[bucket_base_idx] = sum.x;
            buckets[bucket_base_idx + 1u] = sum.y;
            buckets[bucket_base_idx + 2u] = sum.z;
        }
    }
}
