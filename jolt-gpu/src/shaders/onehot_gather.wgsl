// OneHot Batch G1 Addition Kernel — Gather approach.
//
// CPU preprocesses indices into per-(chunk,ki) gather lists so each GPU thread
// only touches base points it needs. No scanning, no wasted reads.
// ~7x faster than the direct-scan kernel at production scale (K=16, row_len=2048).
//
// Buffer layout:
//   [0] bases[row_len]            — G1Affine points (x:8 + y:8 u32s, Montgomery)
//   [1] gather_cols[]             — flat array of column indices per job
//   [2] jobs[num_jobs * 3]        — (start_offset, count, output_idx) per job
//   [3] results[total_outputs]    — output G1Jacobian points (x:8, y:8, z:8)
//   [4] params                    — { num_jobs, _pad, _pad, _pad }
//
// Dispatch: ceil(num_jobs / 128) workgroups
//
// Depends on: bn254_common.wgsl + msm_g1_curve.wgsl (prepended at pipeline creation)

const G1A_W: u32 = 16u;   // 2 * NUM_LIMBS (x:8, y:8)
const G1J_W: u32 = 24u;   // 3 * NUM_LIMBS (x:8, y:8, z:8)

@group(0) @binding(0) var<storage, read> bases: array<u32>;
@group(0) @binding(1) var<storage, read> gather_cols: array<u32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;

struct GatherParams { num_jobs: u32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(4) var<uniform> gparams: GatherParams;

fn ld_aff(index: u32) -> G1Affine {
    let o = index * G1A_W;
    var pt: G1Affine;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        pt.x.limbs[i] = bases[o + i];
        pt.y.limbs[i] = bases[o + NUM_LIMBS + i];
    }
    return pt;
}

fn st_jac(index: u32, pt: G1Jacobian) {
    let o = index * G1J_W;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        results[o + i] = pt.x.limbs[i];
        results[o + NUM_LIMBS + i] = pt.y.limbs[i];
        results[o + 2u * NUM_LIMBS + i] = pt.z.limbs[i];
    }
}

fn st_zero(index: u32) {
    let o = index * G1J_W;
    for (var i = 0u; i < G1J_W; i = i + 1u) {
        results[o + i] = 0u;
    }
}

@compute @workgroup_size(128)
fn onehot_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let jid = gid.x;
    if (jid >= gparams.num_jobs) { return; }

    let jo = jid * 3u;
    let start = jobs[jo];
    let count = jobs[jo + 1u];
    let out_idx = jobs[jo + 2u];

    if (count == 0u) { st_zero(out_idx); return; }

    let col0 = gather_cols[start];
    let pt0 = ld_aff(col0);
    var sum: G1Jacobian;
    sum.x = pt0.x;
    sum.y = pt0.y;
    sum.z = g1_one_mont_z();

    for (var i = 1u; i < count; i = i + 1u) {
        let col = gather_cols[start + i];
        sum = g1_madd_fast(sum, ld_aff(col));
    }

    st_jac(out_idx, sum);
}
