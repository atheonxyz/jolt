override WG_SIZE: u32 = 64;

@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> val_idx: array<u32>;
@group(0) @binding(2) var<storage, read> points_raw: array<u32>;
@group(0) @binding(3) var<storage, read_write> buckets: array<BigInt>;
@group(0) @binding(4) var<storage, read> params: array<u32>;

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

    let inf = xyzz_zero();
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

        if (row_end > row_begin) {
            // Load first point as affine
            let idx0 = val_idx[val_idx_base + row_begin];
            let pt_base0 = idx0 * PT_STRIDE;
            var p0: G1Affine;
            for (var i = 0u; i < 8u; i = i + 1u) {
                p0.x.limbs[i] = points_raw[pt_base0 + i];
                p0.y.limbs[i] = points_raw[pt_base0 + 8u + i];
            }

            if (row_end > row_begin + 1u) {
                // Load second point as affine
                let idx1 = val_idx[val_idx_base + row_begin + 1u];
                let pt_base1 = idx1 * PT_STRIDE;
                var p1: G1Affine;
                for (var i = 0u; i < 8u; i = i + 1u) {
                    p1.x.limbs[i] = points_raw[pt_base1 + i];
                    p1.y.limbs[i] = points_raw[pt_base1 + 8u + i];
                }
                // First addition: affine+affine → XYZZ (6M instead of 10M)
                sum = xyzz_add_affine_affine(p0, p1);

                // Remaining points via standard mixed addition (10M each)
                for (var k = row_begin + 2u; k < row_end; k = k + 1u) {
                    let idx = val_idx[val_idx_base + k];
                    let pt_base = idx * PT_STRIDE;
                    var b_aff: G1Affine;
                    for (var i = 0u; i < 8u; i = i + 1u) {
                        b_aff.x.limbs[i] = points_raw[pt_base + i];
                        b_aff.y.limbs[i] = points_raw[pt_base + 8u + i];
                    }
                    sum = xyzz_madd(sum, b_aff);
                }
            } else {
                // Only 1 point in bucket → copy as XYZZ with ZZ=ZZZ=1
                sum.x = p0.x;
                sum.y = p0.y;
                sum.zz = g1_one_mont_z();
                sum.zzz = g1_one_mont_z();
            }
        }

        var bucket_idx = 0u;
        if (half_columns > row_idx) {
            bucket_idx = half_columns - row_idx;
            sum = xyzz_neg(sum);
        } else {
            bucket_idx = row_idx - half_columns;
        }

        let bi = mapped_col + subtask_idx * half_columns;
        let bucket_base_idx = 4u * bi;
        let bucket_size = half_columns * num_subtasks * 4u;
        if (bucket_idx > 0u && bucket_base_idx + 3u < bucket_size) {
            if (j == 1u) {
                var bucket_val: G1XYZZ;
                bucket_val.x = buckets[bucket_base_idx];
                bucket_val.y = buckets[bucket_base_idx + 1u];
                bucket_val.zz = buckets[bucket_base_idx + 2u];
                bucket_val.zzz = buckets[bucket_base_idx + 3u];
                sum = xyzz_add(bucket_val, sum);
            }

            buckets[bucket_base_idx] = sum.x;
            buckets[bucket_base_idx + 1u] = sum.y;
            buckets[bucket_base_idx + 2u] = sum.zz;
            buckets[bucket_base_idx + 3u] = sum.zzz;
        }
    }
}
