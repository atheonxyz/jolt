// G2 Fixed-Base Scalar Multiplication Kernel for BN254.
//
// Algorithm: Windowed fixed-base method (w=5).
//   - Scalar (254 bits) is decomposed into ceil(254/5) = 51 windows of 5 bits.
//   - Precomputed table: for each window i, stores [1*G_i, 2*G_i, ..., 31*G_i]
//     where G_i = 2^(5*i) * G (G is the fixed base point).
//   - Each thread processes one scalar: 51 mixed additions, zero doublings.
//
// Table layout (flat array of G2Affine in Montgomery form):
//   table[i * 31 + (j-1)] = j * 2^(5*i) * G   for i=0..50, j=1..31
//   Each G2Affine = 4 Fp components (x.c0, x.c1, y.c0, y.c1) = 4 * NUM_LIMBS u32s
//
// Buffers:
//   [0] table:   NUM_WINDOWS * 31 * 4 * NUM_LIMBS u32  (precomputed G2Affine points)
//   [1] scalars: N * NUM_LIMBS u32                      (Fr scalars, raw bits)
//   [2] results: N * 6 * NUM_LIMBS u32                  (G2Jacobian outputs)
//   [3] params:  1 u32                                   (number of scalars)
//
// Depends on: bn254_common.wgsl (prepended at pipeline creation time)

// ============================================================
// Constants
// ============================================================

const G2_WINDOW_SIZE: u32 = 5u;
const G2_NUM_WINDOWS: u32 = 51u;       // ceil(254 / 5)
const G2_TABLE_ENTRIES: u32 = 31u;     // 2^5 - 1
const G2_AFFINE_WORDS: u32 = 32u;     // 4 * NUM_LIMBS (x.c0, x.c1, y.c0, y.c1)
const G2_JACOBIAN_WORDS: u32 = 48u;   // 6 * NUM_LIMBS (x.c0, x.c1, y.c0, y.c1, z.c0, z.c1)

// ============================================================
// Bindings
// ============================================================

@group(0) @binding(0) var<storage, read> g2_table: array<u32>;
@group(0) @binding(1) var<storage, read> g2_scalars: array<u32>;
@group(0) @binding(2) var<storage, read_write> g2_results: array<u32>;

struct G2Params {
    count: u32,
}
@group(0) @binding(3) var<uniform> g2_params: G2Params;

// ============================================================
// G2 Jacobian Pure Arithmetic (port of g2_curve.metal)
// Uses Fp2 primitives from bn254_common.wgsl
// ============================================================

fn g2j_is_zero(a: G2Jacobian) -> bool {
    return fp2_is_zero(a.z);
}

fn g2j_zero() -> G2Jacobian {
    var r: G2Jacobian;
    r.x = fp2_one();
    r.y = fp2_one();
    r.z = fp2_zero();
    return r;
}

fn g2j_from_affine(a: G2Affine) -> G2Jacobian {
    var r: G2Jacobian;
    r.x = a.x;
    r.y = a.y;
    r.z = fp2_one();
    return r;
}

// G2 Jacobian doubling (a=0 curve)
// Formula: http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
// Cost: 2M + 5S + 6add + 3*2 + 1*3 + 1*8 (in Fp2)
fn g2j_dbl(pt: G2Jacobian) -> G2Jacobian {
    let x = pt.x;
    let y = pt.y;
    let z = pt.z;

    let a = fp2_sqr(x);             // a = X^2
    let b = fp2_sqr(y);             // b = Y^2
    let c = fp2_sqr(b);             // c = b^2 = Y^4

    let x1b = fp2_add(x, b);        // X + b
    let x1b2 = fp2_sqr(x1b);        // (X + b)^2
    let ac = fp2_add(a, c);          // a + c
    let x1b2ac = fp2_sub(x1b2, ac);  // (X+b)^2 - a - c
    let d = fp2_double(x1b2ac);      // d = 2*((X+b)^2 - a - c)

    let a2 = fp2_double(a);          // 2*a
    let e = fp2_add(a2, a);          // e = 3*a

    let f = fp2_sqr(e);              // f = e^2

    let d2 = fp2_double(d);          // 2*d
    let x3 = fp2_sub(f, d2);         // X3 = f - 2*d

    let c2 = fp2_double(c);          // 2*c
    let c4 = fp2_double(c2);         // 4*c
    let c8 = fp2_double(c4);         // 8*c

    let dx3 = fp2_sub(d, x3);        // d - X3
    let edx3 = fp2_mul(e, dx3);      // e*(d - X3)
    let y3 = fp2_sub(edx3, c8);      // Y3 = e*(d - X3) - 8*c

    let y1z1 = fp2_mul(y, z);        // Y*Z
    let z3 = fp2_double(y1z1);       // Z3 = 2*Y*Z

    var result: G2Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// G2 Jacobian mixed addition: Jacobian + Affine (Z2 = 1 implicit)
// Formula: http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
// Cost: 7M + 4S (in Fp2)
fn g2j_madd(a_pt: G2Jacobian, b_pt: G2Affine) -> G2Jacobian {
    // Handle identity: 0 + b = b (promote affine to Jacobian)
    if (g2j_is_zero(a_pt)) {
        return g2j_from_affine(b_pt);
    }

    let x1 = a_pt.x;
    let y1 = a_pt.y;
    let z1 = a_pt.z;
    let x2 = b_pt.x;
    let y2 = b_pt.y;

    // Z1Z1 = Z1^2
    let z1z1 = fp2_sqr(z1);

    // U2 = X2*Z1Z1
    let u2 = fp2_mul(x2, z1z1);

    // S2 = Y2*Z1*Z1Z1
    let temp_s2 = fp2_mul(y2, z1);
    let s2 = fp2_mul(temp_s2, z1z1);

    // H = U2 - X1
    let h = fp2_sub(u2, x1);

    // Degenerate check: H == 0 means x-coordinates match
    if (fp2_is_zero(h)) {
        let s2_minus_y1 = fp2_sub(s2, y1);
        if (fp2_is_zero(s2_minus_y1)) {
            // Points are equal -> double
            return g2j_dbl(a_pt);
        } else {
            // Points are inverses -> identity
            return g2j_zero();
        }
    }

    // HH = H^2
    let hh = fp2_sqr(h);

    // I = 4*HH
    var i_val = fp2_double(hh);
    i_val = fp2_double(i_val);

    // J = H*I
    let j = fp2_mul(h, i_val);

    // r = 2*(S2 - Y1)
    let s2_minus_y1 = fp2_sub(s2, y1);
    let r = fp2_double(s2_minus_y1);

    // V = X1*I
    let v = fp2_mul(x1, i_val);

    // X3 = r^2 - J - 2*V
    let r2 = fp2_sqr(r);
    let v2 = fp2_double(v);
    let jv2 = fp2_add(j, v2);
    let x3 = fp2_sub(r2, jv2);

    // Y3 = r*(V - X3) - 2*Y1*J
    let v_minus_x3 = fp2_sub(v, x3);
    let r_vmx3 = fp2_mul(r, v_minus_x3);
    let y1j = fp2_mul(y1, j);
    let y1j2 = fp2_double(y1j);
    let y3 = fp2_sub(r_vmx3, y1j2);

    // Z3 = (Z1+H)^2 - Z1Z1 - HH
    let z1_plus_h = fp2_add(z1, h);
    let z1_plus_h_sq = fp2_sqr(z1_plus_h);
    let temp_z = fp2_sub(z1_plus_h_sq, z1z1);
    let z3 = fp2_sub(temp_z, hh);

    var result: G2Jacobian;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// ============================================================
// Buffer read/write helpers
// ============================================================

// Read a G2Affine point from the table buffer
fn g2_read_affine(offset: u32) -> G2Affine {
    var pt: G2Affine;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        pt.x.c0.limbs[i] = g2_table[offset + 0u * NUM_LIMBS + i];
        pt.x.c1.limbs[i] = g2_table[offset + 1u * NUM_LIMBS + i];
        pt.y.c0.limbs[i] = g2_table[offset + 2u * NUM_LIMBS + i];
        pt.y.c1.limbs[i] = g2_table[offset + 3u * NUM_LIMBS + i];
    }
    return pt;
}

// Write a G2Jacobian point to the result buffer
fn g2_write_jacobian(offset: u32, pt: G2Jacobian) {
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        g2_results[offset + 0u * NUM_LIMBS + i] = pt.x.c0.limbs[i];
        g2_results[offset + 1u * NUM_LIMBS + i] = pt.x.c1.limbs[i];
        g2_results[offset + 2u * NUM_LIMBS + i] = pt.y.c0.limbs[i];
        g2_results[offset + 3u * NUM_LIMBS + i] = pt.y.c1.limbs[i];
        g2_results[offset + 4u * NUM_LIMBS + i] = pt.z.c0.limbs[i];
        g2_results[offset + 5u * NUM_LIMBS + i] = pt.z.c1.limbs[i];
    }
}

// Read a scalar (BigInt) from the scalars buffer
fn g2_read_scalar(offset: u32) -> BigInt {
    var s: BigInt;
    for (var i = 0u; i < NUM_LIMBS; i = i + 1u) {
        s.limbs[i] = g2_scalars[offset + i];
    }
    return s;
}

// Extract a w-bit window from a scalar.
// window_idx: which window (0 = LSB, 50 = MSB)
// Returns value 0..31 (for w=5)
fn g2_extract_window(scalar: BigInt, window_idx: u32) -> u32 {
    let bit_start = window_idx * G2_WINDOW_SIZE;
    let limb_idx = bit_start / 32u;
    let bit_offset = bit_start % 32u;

    var val = 0u;
    if (limb_idx < NUM_LIMBS) {
        val = scalar.limbs[limb_idx] >> bit_offset;
    }

    // Handle window crossing a limb boundary
    let bits_from_first = 32u - bit_offset;
    if (bits_from_first < G2_WINDOW_SIZE && (limb_idx + 1u) < NUM_LIMBS) {
        val = val | (scalar.limbs[limb_idx + 1u] << bits_from_first);
    }

    return val & ((1u << G2_WINDOW_SIZE) - 1u);
}

// ============================================================
// Kernel entry point
// ============================================================

@compute @workgroup_size(128)
fn g2_fixed_base_scalar_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= g2_params.count) {
        return;
    }

    // Read this thread's scalar
    let scalar = g2_read_scalar(tid * NUM_LIMBS);

    // Accumulator starts at identity
    var acc = g2j_zero();

    // Process each window: look up table entry and add
    for (var w = 0u; w < G2_NUM_WINDOWS; w = w + 1u) {
        let digit = g2_extract_window(scalar, w);

        if (digit != 0u) {
            // Table index: window * 31 + (digit - 1)
            let table_offset = (w * G2_TABLE_ENTRIES + (digit - 1u)) * G2_AFFINE_WORDS;
            let table_point = g2_read_affine(table_offset);
            acc = g2j_madd(acc, table_point);
        }
    }

    // Write result
    g2_write_jacobian(tid * G2_JACOBIAN_WORDS, acc);
}
