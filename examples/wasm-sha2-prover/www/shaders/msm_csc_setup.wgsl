// msm_csc_setup.wgsl — GPU-side scalar decomposition + CSC construction
// Replaces the JS-side decomposeAndTranspose with 3 GPU kernels:
//   1. decompose_scalars: per-POINT carry chain for ALL subtasks + build histogram
//   2. prefix_sum: sequential prefix sum on histogram → colPtr
//   3. scatter_csc: per-POINT scatter across ALL subtasks → valIdxs
//
// Key optimization: each thread processes ONE point across ALL subtasks,
// computing the carry chain exactly once. This eliminates redundant carry
// recomputation (13.5x less carry work vs per-(subtask,point) approach).

override WG_SIZE: u32 = 256;

// ── Buffers ──────────────────────────────────────────────────────────────────

// Input: raw scalar limbs (8 u32s per scalar, little-endian)
@group(0) @binding(0) var<storage, read> scalars: array<u32>;
// Output: column index per (subtask, point)
@group(0) @binding(1) var<storage, read_write> col_indices: array<u32>;
// Output: CSC column pointers (histogram, then prefix-summed)
// Layout: [subtask0: numColumns+1 entries, subtask1: numColumns+1 entries, ...]
@group(0) @binding(2) var<storage, read_write> col_ptr: array<atomic<u32>>;
// Output: CSC value indices (point indices in CSC order)
@group(0) @binding(3) var<storage, read_write> val_idxs: array<u32>;
// Scatter counters (one per subtask*numColumns, for atomic scatter)
@group(0) @binding(4) var<storage, read_write> scatter_cnt: array<atomic<u32>>;
// Parameters
@group(0) @binding(5) var<storage, read> params: array<u32>;
// params[0] = numPoints
// params[1] = numColumns
// params[2] = windowSize
// params[3] = subtasksPerMSM
// params[4] = batchSize
// params[5] = totalSubtasks
// params[6] = NUM_LIMBS (8)

const NUM_LIMBS: u32 = 8u;

// ── Kernel 1: decompose_scalars ──────────────────────────────────────────────
// Each thread handles ONE point and processes ALL subtasks for that point.
// The carry chain is computed exactly once per scalar (not redundantly per subtask).
//
// Dispatch: ceil(batchSize * numPoints / WG_SIZE) workgroups

@compute @workgroup_size(WG_SIZE, 1, 1)
fn decompose_scalars(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_points = params[0];
    let num_columns = params[1];
    let window_size = params[2];
    let subtasks_per_msm = params[3];
    let batch_size = params[4];
    let half_columns = num_columns / 2u;

    let idx = gid.x;
    let total_points = batch_size * num_points;
    if (idx >= total_points) { return; }

    let msm_idx = idx / num_points;
    let point_idx = idx % num_points;

    // Read scalar once — 8 u32 limbs
    let scalar_base = idx * NUM_LIMBS;

    // Process carry chain for ALL subtasks in one pass
    var carry = 0u;
    for (var sub = 0u; sub < subtasks_per_msm; sub++) {
        let bit_offset = sub * window_size;
        let limb_idx = bit_offset / 32u;
        let bit_in_limb = bit_offset % 32u;

        var chunk_val = 0u;
        if (limb_idx < NUM_LIMBS) {
            chunk_val = scalars[scalar_base + limb_idx] >> bit_in_limb;
            if (bit_in_limb + window_size > 32u && limb_idx + 1u < NUM_LIMBS) {
                chunk_val |= scalars[scalar_base + limb_idx + 1u] << (32u - bit_in_limb);
            }
            // Mask to window_size bits
            if (window_size < 32u) {
                chunk_val &= (1u << window_size) - 1u;
            }
        }

        let slice_val = chunk_val + carry;
        var col_idx: u32;
        if (slice_val >= half_columns) {
            col_idx = half_columns - (num_columns - slice_val);
            carry = 1u;
        } else {
            col_idx = slice_val + half_columns;
            carry = 0u;
        }

        // Store column index for this (subtask, point)
        let subtask_idx = msm_idx * subtasks_per_msm + sub;
        let ci_offset = subtask_idx * num_points + point_idx;
        col_indices[ci_offset] = col_idx;

        // Atomically increment histogram
        let cp_offset = subtask_idx * (num_columns + 1u) + col_idx + 1u;
        atomicAdd(&col_ptr[cp_offset], 1u);
    }
}

// ── Kernel 2: prefix_sum ─────────────────────────────────────────────────────
// Each workgroup handles one subtask's column pointer array.
// Sequential scan over numColumns+1 entries (numColumns ≤ 1024 for windowSize ≤ 10).
//
// Dispatch: (totalSubtasks, 1, 1) workgroups

@compute @workgroup_size(1, 1, 1)
fn prefix_sum(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let num_columns = params[1];
    let total_subtasks = params[5];

    let subtask_idx = wg_id.x;
    if (subtask_idx >= total_subtasks) { return; }

    let cp_base = subtask_idx * (num_columns + 1u);

    // Sequential prefix sum
    var running = atomicLoad(&col_ptr[cp_base]);
    for (var c = 1u; c <= num_columns; c++) {
        let prev = running;
        let cur = atomicLoad(&col_ptr[cp_base + c]);
        running = prev + cur;
        atomicStore(&col_ptr[cp_base + c], running);
    }
}

// ── Kernel 3: scatter_csc ────────────────────────────────────────────────────
// Each thread handles ONE point and scatters ALL subtask entries.
// Reads col_indices for all subtasks, atomically claims positions, writes val_idxs.
//
// Dispatch: ceil(batchSize * numPoints / WG_SIZE) workgroups

@compute @workgroup_size(WG_SIZE, 1, 1)
fn scatter_csc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_points = params[0];
    let num_columns = params[1];
    let subtasks_per_msm = params[3];
    let batch_size = params[4];

    let idx = gid.x;
    let total_points = batch_size * num_points;
    if (idx >= total_points) { return; }

    let msm_idx = idx / num_points;
    let point_idx = idx % num_points;

    for (var sub = 0u; sub < subtasks_per_msm; sub++) {
        let subtask_idx = msm_idx * subtasks_per_msm + sub;
        let col_idx = col_indices[subtask_idx * num_points + point_idx];

        // Atomically claim a position within this column's range
        let cnt_offset = subtask_idx * num_columns + col_idx;
        let pos = atomicAdd(&scatter_cnt[cnt_offset], 1u);

        // col_ptr[subtask*(numColumns+1) + col_idx] is the start of this column's range
        let cp_base = subtask_idx * (num_columns + 1u);
        let range_start = atomicLoad(&col_ptr[cp_base + col_idx]);

        // Write point index into val_idxs
        let vi_base = subtask_idx * num_points;
        val_idxs[vi_base + range_start + pos] = point_idx;
    }
}
