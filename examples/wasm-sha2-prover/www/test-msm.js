// test-msm.js — GPU MSM correctness tests via algebraic invariants
// Verifies the XYZZ + affine_affine optimizations produce correct results.
// Uses projective equivalence checks (BigInt modular arithmetic) to compare
// MSM results that should represent the same elliptic curve point.

import { initGPUMSM, executeGPUBatchMSM } from './gpu-msm.js';

const NUM_LIMBS = 8;
const PT_STRIDE = 16;
const RESULT_STRIDE = 24; // Jacobian: x(8) + y(8) + z(8)

// BN254 base field modulus
const P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47n;

// BN254 G1 generator in Montgomery form
const GEN_X = new Uint32Array([
    0xd35d438d, 0x0a85c8b8, 0x11328e64, 0x04d1bea2,
    0xfc5aa270, 0x403b0127, 0xf9c12efd, 0x1c1d1625,
]);
const GEN_Y = new Uint32Array([
    0xa74e5ea3, 0x7e94e24e, 0x339f0be6, 0x3ac87ed3,
    0x35caea54, 0x0e9d3640, 0x30816e2d, 0x0f4f9c97,
]);

const SCALAR_BIT_WIDTH = 128;

// ── Helpers ──────────────────────────────────────────────────────────────────

function log(msg) {
    const el = document.getElementById('log');
    el.textContent += msg + '\n';
    el.scrollTop = el.scrollHeight;
    console.log(msg);
}

function limbsToBigInt(arr, offset) {
    let r = 0n;
    for (let i = 7; i >= 0; i--) r = (r << 32n) | BigInt(arr[offset + i] >>> 0);
    return r;
}

function isIdentity(arr, offset) {
    for (let i = 0; i < 8; i++) {
        if (arr[offset + 16 + i] !== 0) return false;
    }
    return true;
}

// Projective equivalence: (X1,Y1,Z1) ≡ (X2,Y2,Z2) iff
// X1*Z2² ≡ X2*Z1² (mod P) and Y1*Z2³ ≡ Y2*Z1³ (mod P)
// Works in Montgomery form because the R³ factors cancel.
function jacobianEq(a, aOff, b, bOff) {
    const aId = isIdentity(a, aOff);
    const bId = isIdentity(b, bOff);
    if (aId && bId) return true;
    if (aId || bId) return false;

    const x1 = limbsToBigInt(a, aOff), y1 = limbsToBigInt(a, aOff + 8), z1 = limbsToBigInt(a, aOff + 16);
    const x2 = limbsToBigInt(b, bOff), y2 = limbsToBigInt(b, bOff + 8), z2 = limbsToBigInt(b, bOff + 16);

    const z1sq = (z1 * z1) % P, z2sq = (z2 * z2) % P;
    if ((x1 * z2sq) % P !== (x2 * z1sq) % P) return false;

    const z1cu = (z1sq * z1) % P, z2cu = (z2sq * z2) % P;
    return (y1 * z2cu) % P === (y2 * z1cu) % P;
}

function bitwiseEq(a, aOff, b, bOff, len) {
    for (let i = 0; i < len; i++) {
        if (a[aOff + i] !== b[bOff + i]) return false;
    }
    return true;
}

// All points = BN254 generator (known on-curve)
function makePoints(n) {
    const flat = new Uint32Array(n * PT_STRIDE);
    for (let i = 0; i < n; i++) {
        const base = i * PT_STRIDE;
        for (let j = 0; j < 8; j++) {
            flat[base + j] = GEN_X[j];
            flat[base + 8 + j] = GEN_Y[j];
        }
    }
    return flat;
}

// Build scalars from u32 values per row. rows[r][p] = scalar for row r, point p.
function makeScalarsU32(numPoints, rows) {
    const flat = new Uint32Array(rows.length * numPoints * NUM_LIMBS);
    for (let r = 0; r < rows.length; r++) {
        for (let p = 0; p < numPoints; p++) {
            flat[(r * numPoints + p) * NUM_LIMBS] = rows[r][p] >>> 0;
        }
    }
    return flat;
}

// Build scalars from BigInt values per row.
function makeScalarsBigInt(numPoints, rows) {
    const flat = new Uint32Array(rows.length * numPoints * NUM_LIMBS);
    for (let r = 0; r < rows.length; r++) {
        for (let p = 0; p < numPoints; p++) {
            const off = (r * numPoints + p) * NUM_LIMBS;
            let val = rows[r][p];
            for (let l = 0; l < 8; l++) {
                flat[off + l] = Number(val & 0xFFFFFFFFn);
                val >>= 32n;
            }
        }
    }
    return flat;
}

function formatPoint(arr, off) {
    const x = limbsToBigInt(arr, off).toString(16).padStart(64, '0');
    const y = limbsToBigInt(arr, off + 8).toString(16).padStart(64, '0');
    const z = limbsToBigInt(arr, off + 16).toString(16).padStart(64, '0');
    return `X=0x${x.slice(0,16)}…  Y=0x${y.slice(0,16)}…  Z=0x${z.slice(0,16)}…`;
}

// ── Test runner ──────────────────────────────────────────────────────────────

let passed = 0, failed = 0, total = 0;

function assert(cond, name, detail) {
    total++;
    if (cond) {
        passed++;
        log(`  ✓ ${name}`);
    } else {
        failed++;
        log(`  ✗ ${name}${detail ? ': ' + detail : ''}`);
    }
}

async function msm(numPoints, rows, bitWidth) {
    const pts = makePoints(numPoints);
    const isBI = rows.some(r => r.some(v => typeof v === 'bigint'));
    const scalars = isBI
        ? makeScalarsBigInt(numPoints, rows)
        : makeScalarsU32(numPoints, rows);
    return executeGPUBatchMSM(pts, scalars, numPoints, bitWidth || SCALAR_BIT_WIDTH, rows.length);
}

async function runTests() {
    log('═══ GPU MSM Correctness Tests ═══');
    log(`Testing XYZZ coordinates + affine_affine first-addition optimization\n`);

    // ── Test 0: Determinism ──
    log('Test 0: Determinism (same input → bitwise identical output)');
    {
        const pts = makePoints(8);
        const s = makeScalarsU32(8, [[3, 7, 11, 13, 17, 19, 23, 29]]);
        const r1 = await executeGPUBatchMSM(pts, s, 8, SCALAR_BIT_WIDTH, 1);
        const r2 = await executeGPUBatchMSM(pts, s, 8, SCALAR_BIT_WIDTH, 1);
        assert(bitwiseEq(r1, 0, r2, 0, RESULT_STRIDE), 'Bitwise identical across 2 runs');
        const r3 = await executeGPUBatchMSM(pts, s, 8, SCALAR_BIT_WIDTH, 1);
        assert(bitwiseEq(r1, 0, r3, 0, RESULT_STRIDE), 'Bitwise identical across 3 runs');
    }

    // ── Test 1: Zero scalars → identity ──
    // NOTE: All tests use numPoints >= 8 because numPoints < 8 triggers windowSize=12
    // which has a pre-existing edge case in the CSC pipeline (not related to XYZZ changes).
    log('\nTest 1: Zero scalars → identity');
    {
        const r = await msm(8, [[0, 0, 0, 0, 0, 0, 0, 0]]);
        assert(isIdentity(r, 0), 'All-zero scalars → identity (Z=0)');
    }

    // ── Test 2: Non-zero scalar → non-identity ──
    log('\nTest 2: Non-zero scalar → non-identity');
    {
        const r = await msm(8, [[1, 0, 0, 0, 0, 0, 0, 0]]);
        assert(!isIdentity(r, 0), 'scalar=1 → non-identity');
    }

    // ── Test 3: Summation invariant (small scalars) ──
    log('\nTest 3: Summation invariant (1+2+3+4+5+6+7+8 = 36)');
    {
        const r1 = await msm(8, [[1, 2, 3, 4, 5, 6, 7, 8]]);
        const r2 = await msm(8, [[36, 0, 0, 0, 0, 0, 0, 0]]);
        assert(jacobianEq(r1, 0, r2, 0), 'MSM [1..8] ≡ MSM [36,0,…]');
        log(`    distributed: ${formatPoint(r1, 0)}`);
        log(`    concentrated: ${formatPoint(r2, 0)}`);
    }

    // ── Test 4: xyzz_add_affine_affine path (identical scalars → same bucket) ──
    // With scalars [1,1,1,1,...], all points land in bucket 1 of window 0.
    // First two use affine+affine (6M), rest use xyzz_madd (10M each)
    log('\nTest 4: affine+affine path (8 identical scalars → 1 bucket)');
    {
        const r1 = await msm(8, [[1, 1, 1, 1, 1, 1, 1, 1]]);
        const r2 = await msm(8, [[8, 0, 0, 0, 0, 0, 0, 0]]);
        assert(jacobianEq(r1, 0, r2, 0), 'MSM [1]*8 ≡ MSM [8,0,…]');
    }

    // ── Test 5: affine+affine with exactly 2 points per bucket ──
    log('\nTest 5: affine+affine only (2 identical scalars + 6 zeros)');
    {
        const r1 = await msm(8, [[5, 5, 0, 0, 0, 0, 0, 0]]);
        const r2 = await msm(8, [[10, 0, 0, 0, 0, 0, 0, 0]]);
        assert(jacobianEq(r1, 0, r2, 0), 'MSM [5,5,0,…] ≡ MSM [10,0,…]');
    }

    // ── Test 6: Large bucket (32 identical scalars) ──
    log('\nTest 6: Large bucket (32 identical scalars → 1 bucket with 32 entries)');
    {
        const ones = new Array(32).fill(1);
        const sumArr = new Array(32).fill(0); sumArr[0] = 32;
        const r1 = await msm(32, [ones]);
        const r2 = await msm(32, [sumArr]);
        assert(jacobianEq(r1, 0, r2, 0), 'MSM [1]*32 ≡ MSM [32,0,…]');
    }

    // ── Test 7: Commutativity (reversed scalars) ──
    log('\nTest 7: Commutativity (reversed scalar order)');
    {
        const fwd = [5, 11, 7, 3, 17, 2, 13, 19];
        const rev = [...fwd].reverse();
        const r1 = await msm(8, [fwd]);
        const r2 = await msm(8, [rev]);
        assert(jacobianEq(r1, 0, r2, 0), 'MSM [fwd] ≡ MSM [rev]');
    }

    // ── Test 8: Batch independence ──
    log('\nTest 8: Batch independence (2-row batch vs individual runs)');
    {
        const rBatch = await msm(8, [
            [3, 5, 7, 11, 0, 0, 0, 0],
            [2, 4, 6, 8, 0, 0, 0, 0],
        ]);
        const r0 = await msm(8, [[3, 5, 7, 11, 0, 0, 0, 0]]);
        const r1 = await msm(8, [[2, 4, 6, 8, 0, 0, 0, 0]]);
        assert(jacobianEq(rBatch, 0, r0, 0), 'Batch row 0 ≡ single MSM row 0');
        assert(jacobianEq(rBatch, RESULT_STRIDE, r1, 0), 'Batch row 1 ≡ single MSM row 1');
    }

    // ── Test 9: Multi-window scalars (values > 2^windowSize) ──
    log('\nTest 9: Multi-window scalars (values spanning 2+ windows)');
    {
        const vals = [1027, 2053, 4099, 8209, 512, 256, 128, 64]; // > 2^10 = 1024 for some
        const sum = vals.reduce((a, b) => a + b, 0);
        const sumArr = [sum, 0, 0, 0, 0, 0, 0, 0];
        const r1 = await msm(8, [vals]);
        const r2 = await msm(8, [sumArr]);
        assert(jacobianEq(r1, 0, r2, 0), `Sum invariant with multi-window scalars (sum=${sum})`);
    }

    // ── Test 10: 64-bit scalars ──
    log('\nTest 10: 64-bit scalars (sum invariant)');
    {
        const scalars = [
            0x123456789ABCDEFn,
            0xFEDCBA987654321n,
            0xA5A5A5A5A5A5A5An,
            0x5A5A5A5A5A5A5A5n,
            0x1111111111111111n,
            0x2222222222222222n,
            0x3333333333333333n,
            0x4444444444444444n,
        ];
        const sum = scalars.reduce((a, b) => a + b, 0n);
        const sumRow = [sum, ...new Array(7).fill(0n)];
        const r1 = await msm(8, [scalars]);
        const r2 = await msm(8, [sumRow]);
        assert(jacobianEq(r1, 0, r2, 0), `64-bit sum invariant (sum=0x${sum.toString(16)})`);
    }

    // ── Test 11: Mixed bucket sizes in single MSM ──
    // Different scalars → points land in different buckets with varied sizes
    log('\nTest 11: Mixed bucket sizes (varied scalars)');
    {
        // With window=10, digit range is [-512, 512].
        // Scalars 1,1,1 → bucket 1 gets 3 entries (affine+affine then xyzz_madd)
        // Scalar 2 → bucket 2 gets 1 entry (copy path)
        // Scalar 3 → bucket 3 gets 1 entry
        // Total = 1+1+1+2+3 = 8
        const r1 = await msm(8, [[1, 1, 1, 2, 3, 0, 0, 0]]);
        const r2 = await msm(8, [[8, 0, 0, 0, 0, 0, 0, 0]]);
        assert(jacobianEq(r1, 0, r2, 0), 'Mixed bucket sizes: [1,1,1,2,3,0,0,0] ≡ [8,0,…]');
    }

    // ── Test 12: Larger batch (4 rows) ──
    log('\nTest 12: 4-row batch correctness');
    {
        const rBatch = await msm(8, [
            [1, 0, 0, 0, 0, 0, 0, 0],   // sum = 1
            [0, 1, 0, 0, 0, 0, 0, 0],   // sum = 1
            [1, 1, 0, 0, 0, 0, 0, 0],   // sum = 2
            [1, 1, 1, 1, 0, 0, 0, 0],   // sum = 4
        ]);
        // Rows 0 and 1 should be the same point (1*G)
        assert(jacobianEq(rBatch, 0 * RESULT_STRIDE, rBatch, 1 * RESULT_STRIDE),
            'Batch: row0 [1,0,…] ≡ row1 [0,1,…]');
        // Row 2 should NOT equal row 0 (2*G ≠ 1*G)
        assert(!jacobianEq(rBatch, 2 * RESULT_STRIDE, rBatch, 0 * RESULT_STRIDE),
            'Batch: row2 [1,1,…] ≢ row0 [1,0,…]');
        // Verify row 2 = 2*G and row 3 = 4*G via separate runs
        const r2 = await msm(8, [[2, 0, 0, 0, 0, 0, 0, 0]]);
        const r4 = await msm(8, [[4, 0, 0, 0, 0, 0, 0, 0]]);
        assert(jacobianEq(rBatch, 2 * RESULT_STRIDE, r2, 0), 'Batch row2 ≡ 2*G');
        assert(jacobianEq(rBatch, 3 * RESULT_STRIDE, r4, 0), 'Batch row3 ≡ 4*G');
    }

    // ── Test 13: Jolt-realistic workload size ──
    log('\nTest 13: Jolt-realistic workload (256 pts × 4 batch)');
    {
        // Deterministic pseudo-random small scalars (sum stays under 128 bits)
        let rng = 0x12345678;
        function xorshift() {
            rng ^= rng << 13; rng ^= rng >>> 17; rng ^= rng << 5;
            return (rng >>> 0) & 0xFFFF; // 16-bit values, sum of 256 ≈ 2^24
        }

        const rows = [];
        const sumRow = [];
        for (let r = 0; r < 4; r++) {
            const row = [];
            let sum = 0;
            for (let p = 0; p < 256; p++) {
                const v = xorshift();
                row.push(v);
                sum += v;
            }
            rows.push(row);
            const sr = new Array(256).fill(0); sr[0] = sum;
            sumRow.push(sr);
        }

        const rDist = await msm(256, rows);
        const rConc = await msm(256, sumRow);
        for (let r = 0; r < 4; r++) {
            assert(jacobianEq(rDist, r * RESULT_STRIDE, rConc, r * RESULT_STRIDE),
                `Jolt-size row ${r}: distributed ≡ concentrated`);
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    log(`\n═══ Results: ${passed}/${total} passed, ${failed} failed ═══`);
    const statusEl = document.getElementById('status');
    if (failed === 0) {
        log('ALL TESTS PASSED');
        statusEl.textContent = `PASS (${passed}/${total})`;
        statusEl.className = 'pass';
    } else {
        log(`${failed} TEST(S) FAILED`);
        statusEl.textContent = `FAIL (${passed}/${total})`;
        statusEl.className = 'fail';
    }
}

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
    log('Initializing WebGPU...');
    if (!navigator.gpu) {
        log('ERROR: WebGPU not available');
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        log('ERROR: No GPU adapter');
        return;
    }

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        },
    });

    log(`GPU: ${adapter.info?.description || adapter.info?.device || 'unknown'}`);
    log('Initializing MSM pipeline...');
    await initGPUMSM(device);
    log('Pipeline ready.\n');

    await runTests();
}

init().catch(e => {
    log(`FATAL: ${e.message}\n${e.stack}`);
    document.getElementById('status').textContent = 'ERROR';
    document.getElementById('status').className = 'fail';
});
