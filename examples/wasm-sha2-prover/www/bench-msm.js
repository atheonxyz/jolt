// bench-msm.js — Isolated GPU MSM benchmark for BN254 G1
// Tests the CSC + SMVP + PBPR + Horner pipeline at various workload sizes.
// No WASM dependency — pure WebGPU throughput measurement.

import { initGPUMSM, executeGPUBatchMSM, setSmvpWorkgroupSize } from './gpu-msm.js';

const NUM_LIMBS = 8;
const PT_STRIDE = 16; // 8 limbs x + 8 limbs y per affine point

// BN254 G1 generator in Montgomery form (from msm_g1_curve.wgsl constants)
const G1_GEN_X = [
    0xd35d438d, 0x0a85c8b8, 0x11328e64, 0x04d1bea2,
    0xfc5aa270, 0x403b0127, 0xf9c12efd, 0x1c1d1625,
];
const G1_GEN_Y = [
    0xa74e5ea3, 0x7e94e24e, 0x339f0be6, 0x3ac87ed3,
    0x35caea54, 0x0e9d3640, 0x30816e2d, 0x0f4f9c97,
];

// BN254 scalar field modulus (Fr) as u32 limbs — for generating valid-range scalars
const FR_MODULUS = [
    0x43e1f593, 0x79b97091, 0x2833e848, 0xb85045b6,
    0xe131a029, 0x64774b84, 0x0000000e, 0x30644e72,
];

// Generate pseudo-random u32 (simple xorshift for deterministic benchmarks)
let _rngState = 0x12345678;
function xorshift32() {
    let x = _rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    _rngState = x >>> 0;
    return x >>> 0;
}

function resetRng(seed) {
    _rngState = seed || 0x12345678;
}

// Generate random affine points (deterministic, not on curve — fine for throughput benchmarks)
// Uses generator-like limb patterns so field arithmetic behaves realistically
function generatePoints(numPoints) {
    resetRng(42);
    const flat = new Uint32Array(numPoints * PT_STRIDE);
    for (let i = 0; i < numPoints; i++) {
        const base = i * PT_STRIDE;
        // Mix generator coords with randomness to get varied but realistic limbs
        for (let j = 0; j < 8; j++) {
            flat[base + j] = G1_GEN_X[j] ^ xorshift32();
            flat[base + 8 + j] = G1_GEN_Y[j] ^ xorshift32();
        }
    }
    return flat;
}

// Generate random scalars of a given bit width
function generateScalars(numPoints, batchSize, scalarBitWidth) {
    resetRng(137);
    const totalScalars = batchSize * numPoints;
    const flat = new Uint32Array(totalScalars * NUM_LIMBS);

    const activeLimbs = Math.ceil(scalarBitWidth / 32);
    const topLimbBits = scalarBitWidth % 32 || 32;
    const topLimbMask = topLimbBits === 32 ? 0xFFFFFFFF : (1 << topLimbBits) - 1;

    for (let i = 0; i < totalScalars; i++) {
        const base = i * NUM_LIMBS;
        for (let j = 0; j < activeLimbs; j++) {
            let val = xorshift32();
            if (j === activeLimbs - 1) {
                val &= topLimbMask;
            }
            flat[base + j] = val;
        }
        // Higher limbs stay 0
    }
    return flat;
}

// Benchmark configurations — based on actual Jolt Dory commitment workloads
// Dense polynomials (RdInc, RamInc): i128 scalars, MSM size = num_cols / k_chunk
// OneHot polynomials: use batch_g1_additions (not MSM), so not benchmarked here
// Each entry: { label, numPoints, batchSize, scalarBitWidth, isJoltWorkload }
const BENCH_CONFIGS = [
    // Jolt realistic workloads (from Dory commit_tier_1 analysis)
    // log_T=20 (small trace, ~1M cycles): 256 pts × 4096 rows × 128-bit
    { label: 'Jolt small trace (logT=20)', numPoints: 256, batchSize: 512, scalarBitWidth: 128, isJoltWorkload: true },
    // log_T=22 (medium trace, ~4M cycles): 512 pts × 16384 rows × 128-bit
    { label: 'Jolt medium trace (logT=22)', numPoints: 512, batchSize: 512, scalarBitWidth: 128, isJoltWorkload: true },
    // log_T=25 threshold: 1024 pts × 65536 rows × 128-bit
    { label: 'Jolt threshold trace (logT=25)', numPoints: 1024, batchSize: 512, scalarBitWidth: 128, isJoltWorkload: true },
    // log_T=28 (large trace, ~256M cycles): 1024 pts × 262144 rows × 128-bit
    // (batchSize capped at 1024 to avoid OOM — real prover streams rows)
    { label: 'Jolt large trace (logT=28)', numPoints: 1024, batchSize: 1024, scalarBitWidth: 128, isJoltWorkload: true },

    // Scalar bit width sweep (at Jolt-like point count)
    { label: '1024 pts × 256-bit', numPoints: 1024, batchSize: 256, scalarBitWidth: 256, isJoltWorkload: false },
    { label: '1024 pts × 64-bit', numPoints: 1024, batchSize: 256, scalarBitWidth: 64, isJoltWorkload: false },
    { label: '1024 pts × 16-bit', numPoints: 1024, batchSize: 256, scalarBitWidth: 16, isJoltWorkload: false },
    { label: '1024 pts × 8-bit', numPoints: 1024, batchSize: 256, scalarBitWidth: 8, isJoltWorkload: false },

    // Point count sweep (at 256-bit scalars — classic MSM benchmark)
    { label: '2^12 pts (4096)', numPoints: 4096, batchSize: 64, scalarBitWidth: 256, isJoltWorkload: false },
    { label: '2^14 pts (16384)', numPoints: 16384, batchSize: 16, scalarBitWidth: 256, isJoltWorkload: false },
    { label: '2^16 pts (65536)', numPoints: 65536, batchSize: 4, scalarBitWidth: 256, isJoltWorkload: false },

    // Batch size sweep (at 1024 points, 128-bit — matches Jolt scalars)
    { label: '1024 pts × batch=1', numPoints: 1024, batchSize: 1, scalarBitWidth: 128, isJoltWorkload: false },
    { label: '1024 pts × batch=16', numPoints: 1024, batchSize: 16, scalarBitWidth: 128, isJoltWorkload: false },
    { label: '1024 pts × batch=128', numPoints: 1024, batchSize: 128, scalarBitWidth: 128, isJoltWorkload: false },
    { label: '1024 pts × batch=512', numPoints: 1024, batchSize: 512, scalarBitWidth: 128, isJoltWorkload: false },
];

const JOLT_ONLY_CONFIGS = BENCH_CONFIGS.filter(c => c.isJoltWorkload);

// DOM elements
const logEl = document.getElementById('log');
const resultsBody = document.getElementById('results-body');
const statusEl = document.getElementById('status');
const btnRun = document.getElementById('btn-run');
const btnJolt = document.getElementById('btn-jolt');
const btnSmvp = document.getElementById('btn-smvp');
function log(msg) {
    const ts = new Date().toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 1 });
    logEl.textContent += `[${ts}] ${msg}\n`;
    logEl.scrollTop = logEl.scrollHeight;
    console.log(msg);
}

function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = 'status-badge ' + cls;
}

function addResultRow(config, id) {
    const tr = document.createElement('tr');
    tr.id = `row-${id}`;
    if (config.isJoltWorkload) tr.className = 'jolt-row';
    tr.innerHTML = `
        <td>${config.label}</td>
        <td class="num">${config.numPoints.toLocaleString()}</td>
        <td class="num">${config.batchSize}</td>
        <td class="num">${config.scalarBitWidth}</td>
        <td class="num" id="win-${id}">—</td>
        <td class="num" id="gpu-${id}">—</td>
        <td class="num" id="tp-${id}">—</td>
        <td id="st-${id}"><span class="status-badge wait">pending</span></td>
    `;
    resultsBody.appendChild(tr);
}

function updateRow(id, { windowSize, gpuMs, throughput, status }) {
    const row = document.getElementById(`row-${id}`);
    if (windowSize != null) document.getElementById(`win-${id}`).textContent = windowSize;
    if (gpuMs != null) document.getElementById(`gpu-${id}`).textContent = gpuMs.toFixed(2);
    if (throughput != null) document.getElementById(`tp-${id}`).textContent = throughput;
    if (status) {
        const stEl = document.getElementById(`st-${id}`);
        const cls = status === 'done' ? 'ok' : status === 'error' ? 'err' : 'run';
        stEl.innerHTML = `<span class="status-badge ${cls}">${status}</span>`;
        if (status === 'done') row.classList.add('done');
        if (status === 'running') row.classList.add('running');
        else row.classList.remove('running');
    }
}

// Compute window size using same formula as gpu-msm.js
function optimalWindowSize(inputSize, scalarBits) {
    if (inputSize > 16_777_216) return 16;
    if (inputSize >= 262_144) return 15;
    const DISPATCH_OVERHEAD = 1024;
    const wMin = inputSize < 8 ? 12 : 10;
    let bestW = wMin, bestCost = Number.MAX_SAFE_INTEGER;
    for (let w = wMin; w <= 14; w++) {
        const subtasks = Math.ceil(scalarBits / w);
        const halfCols = 1 << (w - 1);
        const cost = subtasks * (inputSize + halfCols + DISPATCH_OVERHEAD);
        if (cost < bestCost) { bestCost = cost; bestW = w; }
    }
    return bestW;
}

async function runBenchmark(configs) {
    const warmupRuns = parseInt(document.getElementById('warmup').value, 10);
    const benchRuns = parseInt(document.getElementById('runs').value, 10);

    btnRun.disabled = true;
    btnJolt.disabled = true;
    resultsBody.innerHTML = '';

    // Pre-create all rows
    configs.forEach((config, i) => addResultRow(config, i));

    log(`Starting benchmark: ${configs.length} configs × ${benchRuns} runs (${warmupRuns} warmup)`);

    for (let ci = 0; ci < configs.length; ci++) {
        const config = configs[ci];
        const { numPoints, batchSize, scalarBitWidth, label } = config;
        const windowSize = optimalWindowSize(numPoints, scalarBitWidth);

        updateRow(ci, { windowSize, status: 'running' });
        setStatus(`Running: ${label}`, 'run');
        log(`\n─── ${label} ───`);
        log(`  points=${numPoints} batch=${batchSize} bits=${scalarBitWidth} window=${windowSize}`);

        // Generate test data
        const points = generatePoints(numPoints);
        const scalars = generateScalars(numPoints, batchSize, scalarBitWidth);
        log(`  data: ${(points.byteLength / 1024).toFixed(0)}KB points + ${(scalars.byteLength / 1024).toFixed(0)}KB scalars`);

        try {
            // Warmup
            for (let w = 0; w < warmupRuns; w++) {
                await executeGPUBatchMSM(points, scalars, numPoints, scalarBitWidth, batchSize);
            }
            if (warmupRuns > 0) log(`  warmup: ${warmupRuns} runs done`);

            // Benchmark
            const times = [];
            for (let r = 0; r < benchRuns; r++) {
                const t0 = performance.now();
                await executeGPUBatchMSM(points, scalars, numPoints, scalarBitWidth, batchSize);
                const t1 = performance.now();
                times.push(t1 - t0);
            }

            // Stats
            times.sort((a, b) => a - b);
            const median = times[Math.floor(times.length / 2)];
            const mean = times.reduce((a, b) => a + b, 0) / times.length;
            const min = times[0];
            const max = times[times.length - 1];

            // Throughput: total scalar-point multiplications per second
            const totalMuls = numPoints * batchSize;
            const mulsPerSec = (totalMuls / (median / 1000));
            let throughput;
            if (mulsPerSec >= 1e6) throughput = (mulsPerSec / 1e6).toFixed(1) + 'M/s';
            else if (mulsPerSec >= 1e3) throughput = (mulsPerSec / 1e3).toFixed(1) + 'K/s';
            else throughput = mulsPerSec.toFixed(0) + '/s';

            updateRow(ci, { gpuMs: median, throughput, status: 'done' });
            log(`  median=${median.toFixed(2)}ms  mean=${mean.toFixed(2)}ms  min=${min.toFixed(2)}ms  max=${max.toFixed(2)}ms`);
            log(`  throughput: ${throughput} (${totalMuls.toLocaleString()} scalar-point muls)`);
        } catch (e) {
            updateRow(ci, { status: 'error' });
            log(`  ERROR: ${e.message}`);
        }

        // Let the browser breathe between configs
        await new Promise(r => setTimeout(r, 100));
    }

    log('\n════ Benchmark complete ════');
    setStatus('Done', 'ok');
    btnRun.disabled = false;
    btnJolt.disabled = false;
}

// Initialization
async function init() {
    log('Checking WebGPU support...');

    if (!navigator.gpu) {
        setStatus('WebGPU not available', 'err');
        log('ERROR: navigator.gpu is undefined. Use Chrome 113+ with WebGPU enabled.');
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        setStatus('No GPU adapter', 'err');
        log('ERROR: No WebGPU adapter found.');
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
    log(`Max storage buffer: ${(device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`);
    log(`Max buffer size: ${(device.limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`);

    log('Loading MSM shaders...');
    try {
        await initGPUMSM(device);
        log('MSM pipeline initialized');
    } catch (e) {
        setStatus('Shader error', 'err');
        log(`ERROR loading shaders: ${e.message}`);
        return;
    }

    setStatus('Ready', 'ok');
    btnRun.disabled = false;
    btnJolt.disabled = false;
    btnSmvp.disabled = false;
    log('Ready — click a benchmark button to start');
}

async function runSmvpSweep() {
    const WG_SIZES = [32, 64, 128, 256];
    const testConfig = { label: 'SMVP sweep', numPoints: 1024, batchSize: 512, scalarBitWidth: 128, isJoltWorkload: true };
    const warmupRuns = parseInt(document.getElementById('warmup').value, 10);
    const benchRuns = parseInt(document.getElementById('runs').value, 10);

    btnRun.disabled = true;
    btnJolt.disabled = true;
    btnSmvp.disabled = true;
    resultsBody.innerHTML = '';

    log(`\nSMVP workgroup size sweep: ${WG_SIZES.join(', ')}`);
    log(`Config: pts=${testConfig.numPoints} batch=${testConfig.batchSize} bits=${testConfig.scalarBitWidth}`);
    log(`Warmup=${warmupRuns} Runs=${benchRuns}\n`);

    const points = generatePoints(testConfig.numPoints);
    const scalars = generateScalars(testConfig.numPoints, testConfig.batchSize, testConfig.scalarBitWidth);

    for (let wi = 0; wi < WG_SIZES.length; wi++) {
        const wgSize = WG_SIZES[wi];
        const label = `SMVP WG_SIZE=${wgSize}`;
        const config = { ...testConfig, label };
        addResultRow(config, wi);

        setSmvpWorkgroupSize(wgSize);
        updateRow(wi, { status: 'running' });
        setStatus(`Running: ${label}`, 'run');
        log(`─── ${label} ───`);

        try {
            for (let w = 0; w < warmupRuns; w++) {
                await executeGPUBatchMSM(points, scalars, testConfig.numPoints, testConfig.scalarBitWidth, testConfig.batchSize);
            }

            const times = [];
            for (let r = 0; r < benchRuns; r++) {
                const t0 = performance.now();
                await executeGPUBatchMSM(points, scalars, testConfig.numPoints, testConfig.scalarBitWidth, testConfig.batchSize);
                const t1 = performance.now();
                times.push(t1 - t0);
            }

            times.sort((a, b) => a - b);
            const median = times[Math.floor(times.length / 2)];
            const totalMuls = testConfig.numPoints * testConfig.batchSize;
            const mulsPerSec = totalMuls / (median / 1000);
            let throughput;
            if (mulsPerSec >= 1e6) throughput = (mulsPerSec / 1e6).toFixed(1) + 'M/s';
            else if (mulsPerSec >= 1e3) throughput = (mulsPerSec / 1e3).toFixed(1) + 'K/s';
            else throughput = mulsPerSec.toFixed(0) + '/s';

            const windowSize = optimalWindowSize(testConfig.numPoints, testConfig.scalarBitWidth);
            updateRow(wi, { windowSize, gpuMs: median, throughput, status: 'done' });
            log(`  median=${median.toFixed(2)}ms throughput=${throughput}`);
        } catch (e) {
            updateRow(wi, { status: 'error' });
            log(`  ERROR: ${e.message}`);
        }

        await new Promise(r => setTimeout(r, 100));
    }

    // Restore default
    setSmvpWorkgroupSize(64);
    log('\n════ SMVP sweep complete ════');
    setStatus('Done', 'ok');
    btnRun.disabled = false;
    btnJolt.disabled = false;
    btnSmvp.disabled = false;
}

btnRun.addEventListener('click', () => runBenchmark(BENCH_CONFIGS));
btnJolt.addEventListener('click', () => runBenchmark(JOLT_ONLY_CONFIGS));
btnSmvp.addEventListener('click', () => runSmvpSweep());

init();
