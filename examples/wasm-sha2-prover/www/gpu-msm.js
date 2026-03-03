// gpu-msm.js — WebGPU batch MSM (CUZK Pippenger) for BN254 G1
// GPU-native pipeline: scalar decomposition + CSC construction + SMVP + PBPR + Horner
// All preprocessing happens on GPU — only raw scalars are uploaded from CPU.

const NUM_LIMBS = 8;
const PT_STRIDE = 16; // 8 limbs x + 8 limbs y per affine point

// ── Cost-model window size selection ──────────────────────────────────────────
function optimalWindowSize(inputSize, scalarBits) {
    if (inputSize > 16_777_216) return 16;
    if (inputSize >= 262_144)  return 15;

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

// ── Shader loading ───────────────────────────────────────────────────────────
let _shaderCache = null;
async function loadShaders() {
    if (_shaderCache) return _shaderCache;
    const base = self.location ? '' : '';
    const [common, curve, cscSetup, smvp, pbpr, pbprFused, horner] = await Promise.all([
        fetch(`${base}shaders/bn254_common.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_g1_curve.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_csc_setup.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_smvp.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_pbpr.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_pbpr_fused.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_horner.wgsl`).then(r => r.text()),
    ]);
    _shaderCache = { common, curve, cscSetup, smvp, pbpr, pbprFused, horner };
    return _shaderCache;
}

// ── GPU MSM Pipeline ─────────────────────────────────────────────────────────
let _msmPipeline = null;

async function initMSMPipeline(device) {
    if (_msmPipeline) return _msmPipeline;

    const shaders = await loadShaders();
    const commonCode = shaders.common + '\n' + shaders.curve + '\n';

    // Create shader modules and check for compilation errors
    const modules = [
        { name: 'cscSetup', module: device.createShaderModule({ code: shaders.cscSetup }) },
        { name: 'smvp', module: device.createShaderModule({ code: commonCode + shaders.smvp }) },
        { name: 'pbpr', module: device.createShaderModule({ code: commonCode + shaders.pbpr }) },
        { name: 'pbprFused', module: device.createShaderModule({ code: commonCode + shaders.pbprFused }) },
        { name: 'horner', module: device.createShaderModule({ code: commonCode + shaders.horner }) },
    ];
    for (const { name, module } of modules) {
        const info = await module.getCompilationInfo();
        const errors = info.messages.filter(m => m.type === 'error');
        if (errors.length > 0) {
            const msg = errors.map(e => `${e.message} (line ${e.lineNum})`).join('; ');
            throw new Error(`[gpu-msm] ${name} shader compilation failed: ${msg}`);
        }
    }
    const cscModule = modules[0].module;
    const smvpModule = modules[1].module;
    const pbprModule = modules[2].module;
    const pbprFusedModule = modules[3].module;
    const hornerModule = modules[4].module;
    console.log('[gpu-msm] All MSM shaders compiled successfully');

    // ── CSC Setup bind group layout (decompose, prefix_sum, scatter share one layout)
    const cscBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // scalars
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // col_indices
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // col_ptr (atomic)
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // val_idxs
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // scatter_cnt
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // params
        ],
    });
    const cscPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [cscBGL] });

    const decomposePipeline = device.createComputePipeline({
        layout: cscPipelineLayout,
        compute: { module: cscModule, entryPoint: 'decompose_scalars', constants: { WG_SIZE: 256 } },
    });
    const prefixSumPipeline = device.createComputePipeline({
        layout: cscPipelineLayout,
        compute: { module: cscModule, entryPoint: 'prefix_sum' },
    });
    const scatterPipeline = device.createComputePipeline({
        layout: cscPipelineLayout,
        compute: { module: cscModule, entryPoint: 'scatter_csc', constants: { WG_SIZE: 256 } },
    });

    // ── SMVP bind group layout (5 entries — no sort_perm)
    const smvpBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // col_ptr
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // val_idx
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // points
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // buckets
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // params
        ],
    });

    // ── PBPR and Horner bind group layouts (unchanged)
    const pbprBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });
    const hornerBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    // ── Pipelines
    const smvpPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [smvpBGL] }),
        compute: { module: smvpModule, entryPoint: 'smvp', constants: { WG_SIZE: 64 } },
    });
    const hornerPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [hornerBGL] }),
        compute: { module: hornerModule, entryPoint: 'horner_reduce' },
    });
    const pbprPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [pbprBGL] });

    _msmPipeline = {
        device,
        cscBGL, decomposePipeline, prefixSumPipeline, scatterPipeline,
        smvpBGL, pbprBGL, hornerBGL,
        smvpPipeline, hornerPipeline,
        pbprModule, pbprFusedModule, pbprPipelineLayout,
        _pbprPipelineCache: new Map(),
        _bprFusedPipelineCache: new Map(),
        _pointsCache: null,
    };
    return _msmPipeline;
}

// Get or create PBPR pipelines for a specific workgroup size
function getPBPRPipelines(p, bWgSize) {
    if (p._pbprPipelineCache.has(bWgSize)) {
        return p._pbprPipelineCache.get(bWgSize);
    }
    const bpr1Pipeline = p.device.createComputePipeline({
        layout: p.pbprPipelineLayout,
        compute: { module: p.pbprModule, entryPoint: 'bpr_stage_1', constants: { WG_SIZE: bWgSize } },
    });
    const bpr2Pipeline = p.device.createComputePipeline({
        layout: p.pbprPipelineLayout,
        compute: { module: p.pbprModule, entryPoint: 'bpr_stage_2', constants: { WG_SIZE: bWgSize } },
    });
    const entry = { bpr1Pipeline, bpr2Pipeline };
    p._pbprPipelineCache.set(bWgSize, entry);
    return entry;
}

function getBPRFusedPipeline(p, bWgSize) {
    if (p._bprFusedPipelineCache.has(bWgSize)) {
        return p._bprFusedPipelineCache.get(bWgSize);
    }
    const bprFusedPipeline = p.device.createComputePipeline({
        layout: p.pbprPipelineLayout,
        compute: { module: p.pbprFusedModule, entryPoint: 'bpr_fused', constants: { WG_SIZE: bWgSize } },
    });
    p._bprFusedPipelineCache.set(bWgSize, bprFusedPipeline);
    return bprFusedPipeline;
}

// ── Main entry point: batch MSM ──────────────────────────────────────────────
//
// pointsFlat: Uint32Array — PT_STRIDE (16) u32s per point (x:8, y:8 in Montgomery form)
// scalarsFlat: Uint32Array — NUM_LIMBS (8) u32s per scalar (Montgomery form)
//   Layout: [msm0_scalar0, msm0_scalar1, ..., msm0_scalarN, msm1_scalar0, ...]
// numPoints: number of points per MSM (all MSMs share same bases)
// scalarBitWidth: bit width of scalars (e.g., 8 for u8, 256 for Fr)
// batchSize: number of independent MSMs
//
// Returns: Uint32Array — 24 u32s per result (Jacobian x:8, y:8, z:8)
async function gpuBatchMSM(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) {
    const device = _msmPipeline.device;
    const p = _msmPipeline;

    const t0 = performance.now();
    const windowSize = optimalWindowSize(numPoints, scalarBitWidth);
    const numColumns = 1 << windowSize;
    const halfColumns = numColumns >>> 1;
    const subtasksPerMSM = Math.ceil(scalarBitWidth / windowSize);
    const totalSubtasks = batchSize * subtasksPerMSM;

    // PBPR workgroup size
    const bWgSize = Math.max(Math.min(Math.floor(halfColumns / 128), 64), 32);
    const { bpr1Pipeline, bpr2Pipeline } = getPBPRPipelines(p, bWgSize);

    // ── Buffer creation ─────────────────────────────────────────────────────

    // Point buffer: cache across calls (same bases reused for all polynomials)
    let pointsBuf;
    if (p._pointsCache && p._pointsCache.numPoints === numPoints && p._pointsCache.byteLength === pointsFlat.byteLength) {
        pointsBuf = p._pointsCache.buf;
    } else {
        if (p._pointsCache) p._pointsCache.buf.destroy();
        pointsBuf = device.createBuffer({
            size: pointsFlat.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pointsBuf, 0, pointsFlat);
        p._pointsCache = { buf: pointsBuf, numPoints, byteLength: pointsFlat.byteLength };
    }

    // Raw scalars buffer (ONLY data uploaded from CPU besides points)
    const scalarsBuf = device.createBuffer({
        size: scalarsFlat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(scalarsBuf, 0, scalarsFlat);

    // GPU-local buffers (no upload — populated by GPU kernels)
    const totalWork = totalSubtasks * numPoints;
    const colIndicesBuf = device.createBuffer({
        size: Math.max(totalWork * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    const colPtrLen = totalSubtasks * (numColumns + 1);
    const colPtrBuf = device.createBuffer({
        size: Math.max(colPtrLen * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    const valIdxsBuf = device.createBuffer({
        size: Math.max(totalWork * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    const scatterCntBuf = device.createBuffer({
        size: Math.max(totalSubtasks * numColumns * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    // CSC setup params: [numPoints, numColumns, windowSize, subtasksPerMSM, batchSize, totalSubtasks, NUM_LIMBS]
    const cscParams = new Uint32Array([numPoints, numColumns, windowSize, subtasksPerMSM, batchSize, totalSubtasks, NUM_LIMBS]);
    const cscParamsBuf = device.createBuffer({
        size: cscParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(cscParamsBuf, 0, cscParams);

    // Bucket buffer for SMVP output
    const bucketLen = halfColumns * totalSubtasks * 3 * NUM_LIMBS;
    const bucketBuf = device.createBuffer({
        size: Math.max(bucketLen * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // g_points buffer for PBPR output
    const gPointsLen = totalSubtasks * bWgSize * 3 * NUM_LIMBS;
    const gPointsBuf = device.createBuffer({
        size: Math.max(gPointsLen * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Result + staging buffers
    const resultLen = batchSize * 3 * NUM_LIMBS;
    const resultBuf = device.createBuffer({
        size: Math.max(resultLen * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const stagingBuf = device.createBuffer({
        size: resultLen * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Bind groups ─────────────────────────────────────────────────────────

    // CSC setup bind group (shared by decompose, prefix_sum, scatter)
    const cscBG = device.createBindGroup({
        layout: p.cscBGL,
        entries: [
            { binding: 0, resource: { buffer: scalarsBuf } },
            { binding: 1, resource: { buffer: colIndicesBuf } },
            { binding: 2, resource: { buffer: colPtrBuf } },
            { binding: 3, resource: { buffer: valIdxsBuf } },
            { binding: 4, resource: { buffer: scatterCntBuf } },
            { binding: 5, resource: { buffer: cscParamsBuf } },
        ],
    });

    // SMVP bind group (reads CSC from GPU buffers)
    const smvpWgSize = 64;
    const smvpTotalThreads = halfColumns * totalSubtasks;
    const smvpParams = new Uint32Array([numPoints, numColumns, totalSubtasks, 0, 0]);
    const smvpParamsBuf = device.createBuffer({
        size: smvpParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(smvpParamsBuf, 0, smvpParams);

    const smvpBG = device.createBindGroup({
        layout: p.smvpBGL,
        entries: [
            { binding: 0, resource: { buffer: colPtrBuf } },
            { binding: 1, resource: { buffer: valIdxsBuf } },
            { binding: 2, resource: { buffer: pointsBuf } },
            { binding: 3, resource: { buffer: bucketBuf } },
            { binding: 4, resource: { buffer: smvpParamsBuf } },
        ],
    });

    // PBPR bind group
    const bprParams = new Uint32Array([numColumns, 0]);
    const bprParamsBuf = device.createBuffer({
        size: bprParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bprParamsBuf, 0, bprParams);

    const bprBG = device.createBindGroup({
        layout: p.pbprBGL,
        entries: [
            { binding: 0, resource: { buffer: bucketBuf } },
            { binding: 1, resource: { buffer: gPointsBuf } },
            { binding: 2, resource: { buffer: bprParamsBuf } },
        ],
    });

    // Horner bind group
    const hornerParams = new Uint32Array([batchSize, subtasksPerMSM, bWgSize, windowSize]);
    const hornerParamsBuf = device.createBuffer({
        size: hornerParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(hornerParamsBuf, 0, hornerParams);

    const hornerBG = device.createBindGroup({
        layout: p.hornerBGL,
        entries: [
            { binding: 0, resource: { buffer: gPointsBuf } },
            { binding: 1, resource: { buffer: resultBuf } },
            { binding: 2, resource: { buffer: hornerParamsBuf } },
        ],
    });

    const t1 = performance.now();

    // ── SINGLE command encoder: CSC setup → SMVP → PBPR → Horner → readback
    const encoder = device.createCommandEncoder();

    // Pass 1: Decompose scalars → column indices + histogram
    // Per-point dispatch: each thread handles ONE point across ALL subtasks
    const totalPoints = batchSize * numPoints;
    const decompWG = Math.ceil(totalPoints / 256);
    const decompPass = encoder.beginComputePass();
    decompPass.setPipeline(p.decomposePipeline);
    decompPass.setBindGroup(0, cscBG);
    decompPass.dispatchWorkgroups(decompWG, 1, 1);
    decompPass.end();

    // Pass 2: Prefix sum on histogram → colPtr
    const pfxPass = encoder.beginComputePass();
    pfxPass.setPipeline(p.prefixSumPipeline);
    pfxPass.setBindGroup(0, cscBG);
    pfxPass.dispatchWorkgroups(totalSubtasks, 1, 1);
    pfxPass.end();

    // Pass 3: Scatter CSC → valIdxs (same per-point dispatch)
    const scatterPass = encoder.beginComputePass();
    scatterPass.setPipeline(p.scatterPipeline);
    scatterPass.setBindGroup(0, cscBG);
    scatterPass.dispatchWorkgroups(decompWG, 1, 1);
    scatterPass.end();

    // Pass 4: SMVP bucket accumulation
    const smvpNumWG = Math.ceil(smvpTotalThreads / smvpWgSize);
    const smvpPass = encoder.beginComputePass();
    smvpPass.setPipeline(p.smvpPipeline);
    smvpPass.setBindGroup(0, smvpBG);
    smvpPass.dispatchWorkgroups(smvpNumWG, 1, 1);
    smvpPass.end();

    // Pass 5: PBPR stage 1 — running-sum bucket reduction
    const bpr1Pass = encoder.beginComputePass();
    bpr1Pass.setPipeline(bpr1Pipeline);
    bpr1Pass.setBindGroup(0, bprBG);
    bpr1Pass.dispatchWorkgroups(1, totalSubtasks, 1);
    bpr1Pass.end();

    // Pass 6: PBPR stage 2 — scalar-mul correction
    const bpr2Pass = encoder.beginComputePass();
    bpr2Pass.setPipeline(bpr2Pipeline);
    bpr2Pass.setBindGroup(0, bprBG);
    bpr2Pass.dispatchWorkgroups(1, totalSubtasks, 1);
    bpr2Pass.end();

    // Pass 7: Horner reduction
    const hornerWgSize = 64;
    const hornerNumWG = Math.ceil(batchSize / hornerWgSize);
    const hornerPass = encoder.beginComputePass();
    hornerPass.setPipeline(p.hornerPipeline);
    hornerPass.setBindGroup(0, hornerBG);
    hornerPass.dispatchWorkgroups(hornerNumWG, 1, 1);
    hornerPass.end();

    // Copy results to staging buffer
    encoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, resultLen * 4);

    // ONE submit for the entire pipeline
    device.queue.submit([encoder.finish()]);
    const t2 = performance.now();

    // Read back results
    await stagingBuf.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuf.getMappedRange().slice(0));
    stagingBuf.unmap();
    const t3 = performance.now();

    // Cleanup (point buffer is cached, don't destroy it)
    scalarsBuf.destroy();
    colIndicesBuf.destroy();
    colPtrBuf.destroy();
    valIdxsBuf.destroy();
    scatterCntBuf.destroy();
    cscParamsBuf.destroy();
    bucketBuf.destroy();
    gPointsBuf.destroy();
    smvpParamsBuf.destroy();
    bprParamsBuf.destroy();
    hornerParamsBuf.destroy();
    resultBuf.destroy();
    stagingBuf.destroy();

    const uploadMB = ((scalarsFlat.byteLength + (p._pointsCache ? 0 : pointsFlat.byteLength)) / 1048576).toFixed(1);
    console.log(`[gpu-msm] batch=${batchSize} pts=${numPoints} bits=${scalarBitWidth} w=${windowSize} ` +
        `upload=${uploadMB}MB bufs=${(t1-t0).toFixed(1)}ms submit=${(t2-t1).toFixed(1)}ms read=${(t3-t2).toFixed(1)}ms total=${(t3-t0).toFixed(1)}ms`);

    return resultData; // 24 u32s per MSM result (Jacobian x:8, y:8, z:8)
}

async function gpuBatchMSMChunked(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) {
    const device = _msmPipeline.device;
    const p = _msmPipeline;

    const t0 = performance.now();
    const windowSize = optimalWindowSize(numPoints, scalarBitWidth);
    const numColumns = 1 << windowSize;
    const halfColumns = numColumns >>> 1;
    const subtasksPerMSM = Math.ceil(scalarBitWidth / windowSize);
    const totalSubtasks = batchSize * subtasksPerMSM;
    const CHUNK_SIZE = 128;

    const bWgSize = Math.max(Math.min(Math.floor(halfColumns / 128), 64), 32);
    const bprFusedPipeline = getBPRFusedPipeline(p, bWgSize);

    let pointsBuf;
    if (p._pointsCache && p._pointsCache.numPoints === numPoints && p._pointsCache.byteLength === pointsFlat.byteLength) {
        pointsBuf = p._pointsCache.buf;
    } else {
        if (p._pointsCache) p._pointsCache.buf.destroy();
        pointsBuf = device.createBuffer({
            size: pointsFlat.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pointsBuf, 0, pointsFlat);
        p._pointsCache = { buf: pointsBuf, numPoints, byteLength: pointsFlat.byteLength };
    }

    const scalarsBuf = device.createBuffer({
        size: scalarsFlat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(scalarsBuf, 0, scalarsFlat);

    const totalWork = totalSubtasks * numPoints;
    const colIndicesBuf = device.createBuffer({
        size: Math.max(totalWork * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    const colPtrLen = totalSubtasks * (numColumns + 1);
    const colPtrBuf = device.createBuffer({
        size: Math.max(colPtrLen * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    const valIdxsBuf = device.createBuffer({
        size: Math.max(totalWork * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    const scatterCntBuf = device.createBuffer({
        size: Math.max(totalSubtasks * numColumns * 4, 4),
        usage: GPUBufferUsage.STORAGE,
    });

    const cscParams = new Uint32Array([numPoints, numColumns, windowSize, subtasksPerMSM, batchSize, totalSubtasks, NUM_LIMBS]);
    const cscParamsBuf = device.createBuffer({
        size: cscParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(cscParamsBuf, 0, cscParams);

    const bucketSubtasks = Math.min(CHUNK_SIZE, totalSubtasks);
    const bucketLen = halfColumns * bucketSubtasks * 3 * NUM_LIMBS;
    const bucketBuf = device.createBuffer({
        size: Math.max(bucketLen * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const gPointsLen = totalSubtasks * bWgSize * 3 * NUM_LIMBS;
    const gPointsBuf = device.createBuffer({
        size: Math.max(gPointsLen * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const resultLen = batchSize * 3 * NUM_LIMBS;
    const resultBuf = device.createBuffer({
        size: Math.max(resultLen * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const stagingBuf = device.createBuffer({
        size: resultLen * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const cscBG = device.createBindGroup({
        layout: p.cscBGL,
        entries: [
            { binding: 0, resource: { buffer: scalarsBuf } },
            { binding: 1, resource: { buffer: colIndicesBuf } },
            { binding: 2, resource: { buffer: colPtrBuf } },
            { binding: 3, resource: { buffer: valIdxsBuf } },
            { binding: 4, resource: { buffer: scatterCntBuf } },
            { binding: 5, resource: { buffer: cscParamsBuf } },
        ],
    });

    const smvpParams = new Uint32Array([numPoints, numColumns, 0, 0, 0]);
    const smvpParamsBuf = device.createBuffer({
        size: smvpParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const smvpBG = device.createBindGroup({
        layout: p.smvpBGL,
        entries: [
            { binding: 0, resource: { buffer: colPtrBuf } },
            { binding: 1, resource: { buffer: valIdxsBuf } },
            { binding: 2, resource: { buffer: pointsBuf } },
            { binding: 3, resource: { buffer: bucketBuf } },
            { binding: 4, resource: { buffer: smvpParamsBuf } },
        ],
    });

    const bprParams = new Uint32Array([numColumns, 0]);
    const bprParamsBuf = device.createBuffer({
        size: bprParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bprParamsBuf, 0, bprParams);

    const bprBG = device.createBindGroup({
        layout: p.pbprBGL,
        entries: [
            { binding: 0, resource: { buffer: bucketBuf } },
            { binding: 1, resource: { buffer: gPointsBuf } },
            { binding: 2, resource: { buffer: bprParamsBuf } },
        ],
    });

    const hornerParams = new Uint32Array([batchSize, subtasksPerMSM, bWgSize, windowSize]);
    const hornerParamsBuf = device.createBuffer({
        size: hornerParams.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(hornerParamsBuf, 0, hornerParams);

    const hornerBG = device.createBindGroup({
        layout: p.hornerBGL,
        entries: [
            { binding: 0, resource: { buffer: gPointsBuf } },
            { binding: 1, resource: { buffer: resultBuf } },
            { binding: 2, resource: { buffer: hornerParamsBuf } },
        ],
    });

    const t1 = performance.now();

    const totalPoints = batchSize * numPoints;
    const decompWG = Math.ceil(totalPoints / 256);

    const cscEncoder = device.createCommandEncoder();
    const decompPass = cscEncoder.beginComputePass();
    decompPass.setPipeline(p.decomposePipeline);
    decompPass.setBindGroup(0, cscBG);
    decompPass.dispatchWorkgroups(decompWG, 1, 1);
    decompPass.end();

    const pfxPass = cscEncoder.beginComputePass();
    pfxPass.setPipeline(p.prefixSumPipeline);
    pfxPass.setBindGroup(0, cscBG);
    pfxPass.dispatchWorkgroups(totalSubtasks, 1, 1);
    pfxPass.end();

    const scatterPass = cscEncoder.beginComputePass();
    scatterPass.setPipeline(p.scatterPipeline);
    scatterPass.setBindGroup(0, cscBG);
    scatterPass.dispatchWorkgroups(decompWG, 1, 1);
    scatterPass.end();

    device.queue.submit([cscEncoder.finish()]);

    const smvpWgSize = 64;
    for (let chunkStart = 0; chunkStart < totalSubtasks; chunkStart += CHUNK_SIZE) {
        const chunkSize = Math.min(CHUNK_SIZE, totalSubtasks - chunkStart);

        smvpParams[2] = chunkSize;
        smvpParams[3] = chunkStart;
        smvpParams[4] = 0;  // csc_base_offset = 0: CSC covers ALL subtasks
        device.queue.writeBuffer(smvpParamsBuf, 0, smvpParams);

        bprParams[1] = chunkStart;
        device.queue.writeBuffer(bprParamsBuf, 0, bprParams);

        const chunkEncoder = device.createCommandEncoder();
        chunkEncoder.clearBuffer(bucketBuf);

        const smvpPass = chunkEncoder.beginComputePass();
        smvpPass.setPipeline(p.smvpPipeline);
        smvpPass.setBindGroup(0, smvpBG);
        smvpPass.dispatchWorkgroups(Math.ceil((halfColumns * chunkSize) / smvpWgSize), 1, 1);
        smvpPass.end();

        const bprPass = chunkEncoder.beginComputePass();
        bprPass.setPipeline(bprFusedPipeline);
        bprPass.setBindGroup(0, bprBG);
        bprPass.dispatchWorkgroups(1, chunkSize, 1);
        bprPass.end();

        device.queue.submit([chunkEncoder.finish()]);
    }

    const finalEncoder = device.createCommandEncoder();
    const hornerWgSize = 64;
    const hornerNumWG = Math.ceil(batchSize / hornerWgSize);
    const hornerPass = finalEncoder.beginComputePass();
    hornerPass.setPipeline(p.hornerPipeline);
    hornerPass.setBindGroup(0, hornerBG);
    hornerPass.dispatchWorkgroups(hornerNumWG, 1, 1);
    hornerPass.end();

    finalEncoder.copyBufferToBuffer(resultBuf, 0, stagingBuf, 0, resultLen * 4);
    device.queue.submit([finalEncoder.finish()]);
    const t2 = performance.now();

    await stagingBuf.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuf.getMappedRange().slice(0));
    stagingBuf.unmap();
    const t3 = performance.now();

    scalarsBuf.destroy();
    colIndicesBuf.destroy();
    colPtrBuf.destroy();
    valIdxsBuf.destroy();
    scatterCntBuf.destroy();
    cscParamsBuf.destroy();
    bucketBuf.destroy();
    gPointsBuf.destroy();
    smvpParamsBuf.destroy();
    bprParamsBuf.destroy();
    hornerParamsBuf.destroy();
    resultBuf.destroy();
    stagingBuf.destroy();

    const uploadMB = ((scalarsFlat.byteLength + (p._pointsCache ? 0 : pointsFlat.byteLength)) / 1048576).toFixed(1);
    console.log(`[gpu-msm] chunked batch=${batchSize} pts=${numPoints} bits=${scalarBitWidth} w=${windowSize} chunk=${CHUNK_SIZE} ` +
        `upload=${uploadMB}MB bufs=${(t1-t0).toFixed(1)}ms submit=${(t2-t1).toFixed(1)}ms read=${(t3-t2).toFixed(1)}ms total=${(t3-t0).toFixed(1)}ms`);

    return resultData;
}

// ── Hybrid CPU+GPU batch MSM ─────────────────────────────────────────────
// Splits batch into GPU and CPU portions, running them in parallel.
// cpuMsmFn: (pointsFlat, scalarsFlat, numPoints, batchSize) => Uint32Array (Jacobian)
// gpuFn: either gpuBatchMSM or gpuBatchMSMChunked
async function gpuBatchMSMHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn, gpuFn) {
    if (!cpuMsmFn) {
        // No CPU function available — fall back to pure GPU
        return gpuFn(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize);
    }

    const t0 = performance.now();

    // Throughput ratio from benchmarks: GPU ~5.8× faster than CPU
    // GPU: 128 MSMs in 227ms = 0.564 MSMs/ms
    // CPU: 128 MSMs in 1328ms = 0.096 MSMs/ms
    // Optimal split: GPU gets ~85% of batch, CPU gets ~15%
    const GPU_SPEED_RATIO = 5.8;
    const cpuBatch = Math.max(1, Math.min(batchSize - 1, Math.round(batchSize / (1 + GPU_SPEED_RATIO))));
    const gpuBatch = batchSize - cpuBatch;

    const scalarsPerMSM = numPoints * NUM_LIMBS;
    const resultStride = 3 * NUM_LIMBS; // 24 u32s per Jacobian result

    // Split scalars: GPU gets first gpuBatch rows, CPU gets remaining cpuBatch rows
    const gpuScalars = scalarsFlat.subarray(0, gpuBatch * scalarsPerMSM);
    const cpuScalars = scalarsFlat.subarray(gpuBatch * scalarsPerMSM);

    // Run GPU and CPU in parallel
    const [gpuResult, cpuResult] = await Promise.all([
        gpuFn(pointsFlat, gpuScalars, numPoints, scalarBitWidth, gpuBatch),
        new Promise((resolve) => {
            // CPU MSM runs synchronously on worker threads (rayon),
            // but we wrap it in a microtask to not block the GPU submission
            setTimeout(() => {
                resolve(cpuMsmFn(pointsFlat, cpuScalars, numPoints, cpuBatch));
            }, 0);
        }),
    ]);

    const t1 = performance.now();

    // Merge results: [gpuResults..., cpuResults...]
    const merged = new Uint32Array(batchSize * resultStride);
    merged.set(gpuResult, 0);
    merged.set(cpuResult, gpuBatch * resultStride);

    console.log(`[gpu-msm] hybrid batch=${batchSize} (gpu=${gpuBatch} cpu=${cpuBatch}) ` +
        `pts=${numPoints} bits=${scalarBitWidth} total=${(t1-t0).toFixed(1)}ms`);

    return merged;
}

// ── Public API ───────────────────────────────────────────────────────────────
export async function initGPUMSM(device) {
    return initMSMPipeline(device);
}

export async function executeGPUBatchMSM(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) {
    return gpuBatchMSM(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize);
}

export async function executeGPUBatchMSMChunked(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) {
    return gpuBatchMSMChunked(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize);
}

export async function executeGPUBatchMSMHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn) {
    return gpuBatchMSMHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn, gpuBatchMSM);
}

export async function executeGPUBatchMSMChunkedHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn) {
    return gpuBatchMSMHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn, gpuBatchMSMChunked);
}

// Register on globalThis for wasm-bindgen FFI
if (typeof globalThis !== 'undefined') {
    globalThis.__jolt_gpu_msm_init = async (device) => {
        await initGPUMSM(device);
    };

    globalThis.__jolt_gpu_batch_msm = async (pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) => {
        if (!_msmPipeline) {
            throw new Error('MSM pipeline not initialized. Call __jolt_gpu_msm_init first.');
        }
        return executeGPUBatchMSM(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize);
    };

    globalThis.__jolt_gpu_batch_msm_chunked = async (pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) => {
        if (!_msmPipeline) throw new Error('MSM pipeline not initialized');
        return executeGPUBatchMSMChunked(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize);
    };

    globalThis.__jolt_gpu_batch_msm_hybrid = async (pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn) => {
        if (!_msmPipeline) throw new Error('MSM pipeline not initialized');
        return executeGPUBatchMSMHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn);
    };

    globalThis.__jolt_gpu_batch_msm_chunked_hybrid = async (pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn) => {
        if (!_msmPipeline) throw new Error('MSM pipeline not initialized');
        return executeGPUBatchMSMChunkedHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn);
    };
}
