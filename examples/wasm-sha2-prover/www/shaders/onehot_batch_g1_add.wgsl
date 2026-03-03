// OneHot Batch G1 Addition Kernel — Direct Index Scan.
//
// Each thread handles one (chunk, ki) pair. It scans the packed index array
// for its chunk, finds columns whose index == ki, and accumulates those base
// points via mixed addition. No CPU preprocessing required.
//
// Index packing: 4 × Option<u8> per u32 word. Value 0xFF = None (skip).
//
// Buffer layout:
//   [0] bases[row_len]          — G1Affine points (x:8 + y:8 u32s, Montgomery)
//   [1] packed_indices[]        — packed u8 indices (4 per u32), row-major
//   [2] results[num_chunks * k] — output G1Jacobian points (x:8, y:8, z:8)
//   [3] params                  — { num_chunks, k, row_len, _pad }
//
// Dispatch: ceil(num_chunks * k / WORKGROUP_SIZE) workgroups
//
// Depends on: bn254_common.wgsl + msm_g1_curve.wgsl (prepended at pipeline creation)

// ============================================================
// Constants
// ============================================================

const G1_AFFINE_WORDS: u32 = 16u;   // 2 * NUM_LIMBS (x:8, y:8)
const G1_JACOBIAN_WORDS: u32 = 24u; // 3 * NUM_LIMBS (x:8, y:8, z:8)
const ONEHOT_WORKGROUP_SIZE: u32 = 128u;
const NONE_SENTINEL: u32 = 0xFFu;   // Packed None marker

// ============================================================
// Bindings
// ============================================================

@group(0) @binding(0) var<storage, read> bases: array<u32>;
@group(0) @binding(1) var<storage, read> packed_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

struct OnehotParams {
    num_chunks: u32,
    k: u32,
    row_len: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: OnehotParams;

// ============================================================
// Helpers: load/store G1 points from flat u32 arrays
// ============================================================

fn load_g1_affine(index: u32) -> G1Affine {
    let base_offset = index * G1_AFFINE_WORDS;
    var pt: G1Affine;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        pt.x.limbs[i] = bases[base_offset + i];
        pt.y.limbs[i] = bases[base_offset + NUM_LIMBS + i];
    }
    return pt;
}

fn store_g1_jacobian(index: u32, pt: G1Jacobian) {
    let base_offset = index * G1_JACOBIAN_WORDS;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        results[base_offset + i] = pt.x.limbs[i];
        results[base_offset + NUM_LIMBS + i] = pt.y.limbs[i];
        results[base_offset + 2u * NUM_LIMBS + i] = pt.z.limbs[i];
    }
}

fn store_g1_zero(index: u32) {
    let base_offset = index * G1_JACOBIAN_WORDS;
    for (var i = 0u; i < G1_JACOBIAN_WORDS; i = i + 1u) {
        results[base_offset + i] = 0u;
    }
}

// ============================================================
// Main kernel — direct index scan
// ============================================================

@compute @workgroup_size(128)
fn onehot_direct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let job_idx = gid.x;
    let total_jobs = params.num_chunks * params.k;
    if (job_idx >= total_jobs) {
        return;
    }

    let chunk_idx = job_idx / params.k;
    let ki = job_idx % params.k;

    // Base offset into packed_indices for this chunk
    // Each chunk has row_len indices, packed 4 per u32
    let chunk_base = chunk_idx * params.row_len;
    // Number of u32 words per chunk: ceil(row_len / 4)
    let words_per_chunk = (params.row_len + 3u) >> 2u;
    let word_base = chunk_idx * words_per_chunk;

    var sum: G1Jacobian;
    var has_any = false;

    // Scan all columns in this chunk looking for index == ki
    for (var col = 0u; col < params.row_len; col = col + 1u) {
        // Extract the packed u8 for this column
        let word_idx = word_base + (col >> 2u);
        let byte_pos = (col & 3u) * 8u;
        let v = (packed_indices[word_idx] >> byte_pos) & 0xFFu;

        // Skip None (0xFF) or mismatched indices
        if (v != ki) {
            continue;
        }

        let pt = load_g1_affine(col);

        if (!has_any) {
            // First point: initialize Jacobian from Affine (Z = R Montgomery)
            sum.x = pt.x;
            sum.y = pt.y;
            sum.z = g1_one_mont_z();
            has_any = true;
        } else {
            // Subsequent points: fast mixed addition
            sum = g1_madd_fast(sum, pt);
        }
    }

    // Write result: zero if no points matched, accumulated sum otherwise
    if (!has_any) {
        store_g1_zero(job_idx);
    } else {
        store_g1_jacobian(job_idx, sum);
    }
}
