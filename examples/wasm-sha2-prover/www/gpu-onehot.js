// WebGPU OneHot Batch G1 Addition Module — Direct Index Scan
//
// Each GPU thread handles one (chunk, ki) pair: scans packed indices for its
// chunk, accumulates matching base points via mixed addition. No CPU
// preprocessing required.
//
// API:
//   initGpuOnehot(device)                                — compile shader pipeline
//   gpuOnehotBatchG1Add(bases, packedIndices, ...)       — scan dispatch (primary)
//   gpuOnehotBatchG1AddFire(bases, packedIndices, ...)   — non-blocking wrapper
//   gpuOnehotGatherDirect(bases, gatherCols, jobs, n)    — stub (unused with scan)
//   isGpuOnehotAvailable()                               — check if initialized

const NUM_LIMBS = 8;
const G1_AFFINE_WORDS = 2 * NUM_LIMBS;   // 16 u32s per G1Affine
const G1_JACOBIAN_WORDS = 3 * NUM_LIMBS; // 24 u32s per G1Jacobian
const WORKGROUP_SIZE = 128; // must match shader @workgroup_size

let scanDevice = null;
let scanQueue = null;
let scanPipeline = null;
let gatherPipeline = null;
let _onehotInitialized = false;

// Bases buffer cache: upload once, reuse across all dispatches for same bases.
// All OneHot polys share the same g1_generators, so bases never change within a proof.
let _cachedBasesBuffer = null;
let _cachedBasesSize = 0;

function divCeil(x, y) { return Math.ceil(x / y); }

export function isGpuOnehotAvailable() {
    return _onehotInitialized;
}

/**
 * Initialize GPU OneHot scan pipeline.
 * Must be called after WebGPU device is obtained.
 *
 * @param {GPUDevice} device - WebGPU device (shared with other GPU modules)
 * @returns {Promise<boolean>} - true if initialization succeeded
 */
export async function initGpuOnehot(device) {
    if (_onehotInitialized) return true;
    if (!device) {
        console.warn('[gpu-onehot] No GPU device provided');
        return false;
    }

    try {
        scanDevice = device;
        scanQueue = device.queue;

        // Load shader sources: common + G1 curve + onehot scan kernel
        const commonSrc = await (await fetch('shaders/bn254_common.wgsl')).text();
        const g1CurveSrc = await (await fetch('shaders/msm_g1_curve.wgsl')).text();
        const scanSrc = await (await fetch('./shaders/onehot_batch_g1_add.wgsl')).text();
        const gatherSrc = await (await fetch('./shaders/onehot_gather.wgsl')).text();

        const scanShaderModule = device.createShaderModule({
            code: commonSrc + '\n' + g1CurveSrc + '\n' + scanSrc,
        });
        const scanCompilationInfo = await scanShaderModule.getCompilationInfo();
        const scanErrors = scanCompilationInfo.messages.filter(m => m.type === 'error');
        if (scanErrors.length > 0) {
            const detail = scanErrors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n');
            throw new Error(`WGSL scan shader compile failed:\n${detail}`);
        }
        scanPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: scanShaderModule, entryPoint: 'onehot_direct' },
        });

        const gatherShaderModule = device.createShaderModule({
            code: commonSrc + '\n' + g1CurveSrc + '\n' + gatherSrc,
        });
        const gatherCompilationInfo = await gatherShaderModule.getCompilationInfo();
        const gatherErrors = gatherCompilationInfo.messages.filter(m => m.type === 'error');
        if (gatherErrors.length > 0) {
            console.warn('[gpu-onehot] Gather shader compile failed:', gatherErrors.map(e => e.message));
        } else {
            gatherPipeline = device.createComputePipeline({
                layout: 'auto',
                compute: { module: gatherShaderModule, entryPoint: 'onehot_gather' },
            });
        }

        _onehotInitialized = true;
        console.log('[gpu-onehot] WebGPU OneHot scan pipeline compiled (onehot_batch_g1_add)');
        if (gatherPipeline) console.log('[gpu-onehot] WebGPU OneHot gather pipeline compiled (onehot_gather)');
        return true;
    } catch (e) {
        console.warn('[gpu-onehot] WebGPU OneHot initialization failed:', e);
        return false;
    }
}

/**
 * Get or create cached bases GPU buffer.
 * Bases are the same for all OneHot polys within a proof, so we upload once.
 * @param {Uint32Array} basesFlat
 * @returns {GPUBuffer}
 */
function getOrCreateBasesBuffer(basesFlat) {
    if (_cachedBasesBuffer && _cachedBasesSize === basesFlat.byteLength) {
        return _cachedBasesBuffer;
    }
    if (_cachedBasesBuffer) _cachedBasesBuffer.destroy();
    _cachedBasesBuffer = scanDevice.createBuffer({
        size: basesFlat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    scanQueue.writeBuffer(_cachedBasesBuffer, 0, basesFlat);
    _cachedBasesSize = basesFlat.byteLength;
    return _cachedBasesBuffer;
}

/**
 * Run OneHot batch G1 addition on GPU using direct index scan.
 * Each GPU thread scans one (chunk, ki) pair — no CPU preprocessing needed.
 *
 * @param {Uint32Array} basesFlat        - G1Affine bases (G1_AFFINE_WORDS * row_len u32s)
 * @param {Uint32Array} packedIndices    - Packed u8 indices (4 per u32), row-major by chunk
 * @param {number} numChunks             - Number of chunks (rows_per_k)
 * @param {number} k                     - Bucket count (e.g. 16)
 * @param {number} rowLen                - Row length from DoryGlobals
 * @returns {Promise<Uint32Array>}       - G1Jacobian results (G1_JACOBIAN_WORDS * numChunks * k u32s)
 */
export async function gpuOnehotBatchG1Add(basesFlat, packedIndices, numChunks, k, rowLen) {
    if (!_onehotInitialized) throw new Error('GPU OneHot not initialized. Call initGpuOnehot() first.');

    if (typeof numChunks !== 'number' || typeof k !== 'number' || typeof rowLen !== 'number') {
        throw new Error(`[gpu-onehot] Invalid args: numChunks=${typeof numChunks}, k=${typeof k}, rowLen=${typeof rowLen}. Stale WASM build?`);
    }

    const totalOutputs = numChunks * k;
    if (totalOutputs === 0) return new Uint32Array(0);

    // --- GPU buffers ---

    // Bases (cached across calls — same g1_generators for all OneHot polys)
    const basesBuffer = getOrCreateBasesBuffer(basesFlat);

    // Packed indices (per-poly, created fresh each call)
    const indicesBuffer = scanDevice.createBuffer({
        size: Math.max(packedIndices.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (packedIndices.byteLength > 0) scanQueue.writeBuffer(indicesBuffer, 0, packedIndices);

    // Results
    const resultsSize = totalOutputs * G1_JACOBIAN_WORDS * 4;
    const resultsBuffer = scanDevice.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Params uniform: { num_chunks, k, row_len, _pad }
    const paramsData = new Uint32Array([numChunks, k, rowLen, 0]);
    const paramsBuffer = scanDevice.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    scanQueue.writeBuffer(paramsBuffer, 0, paramsData);

    // --- Dispatch ---
    const encoder = scanDevice.createCommandEncoder();
    const bindGroup = scanDevice.createBindGroup({
        layout: scanPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: basesBuffer } },
            { binding: 1, resource: { buffer: indicesBuffer } },
            { binding: 2, resource: { buffer: resultsBuffer } },
            { binding: 3, resource: { buffer: paramsBuffer } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(scanPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(totalOutputs, WORKGROUP_SIZE));
    pass.end();

    // --- Readback ---
    const stagingBuffer = scanDevice.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(resultsBuffer, 0, stagingBuffer, 0, resultsSize);

    scanQueue.submit([encoder.finish()]);
    await scanDevice.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    // Cleanup (NOT basesBuffer — it's cached)
    indicesBuffer.destroy();
    resultsBuffer.destroy();
    paramsBuffer.destroy();
    stagingBuffer.destroy();

    return resultData;
}

/**
 * Non-blocking dispatch: starts GPU work and returns a Promise.
 * Use this to overlap GPU work with CPU work before awaiting.
 */
export function gpuOnehotBatchG1AddFire(basesFlat, packedIndices, numChunks, k, rowLen) {
    return gpuOnehotBatchG1Add(basesFlat, packedIndices, numChunks, k, rowLen);
}

/**
 * Direct gather dispatch stub — not used with scan kernel.
 * Kept for API compatibility (worker.js imports it).
 *
 * @param {Uint32Array} basesFlat
 * @param {Uint32Array} gatherCols
 * @param {Uint32Array} jobs
 * @param {number} numJobs
 * @returns {Promise<Uint32Array>}
 */
export async function gpuOnehotGatherDirect(basesFlat, gatherCols, jobs, numJobs) {
    if (!gatherPipeline) throw new Error('[gpu-onehot] Gather pipeline not compiled');

    const totalOutputs = numJobs;
    if (totalOutputs === 0) return new Uint32Array(0);

    // Bases (cached across calls)
    const basesBuffer = getOrCreateBasesBuffer(basesFlat);

    // Gather cols
    const gatherColsBuffer = scanDevice.createBuffer({
        size: Math.max(gatherCols.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (gatherCols.byteLength > 0) scanQueue.writeBuffer(gatherColsBuffer, 0, gatherCols);

    // Jobs: (start_offset, count, output_idx) × numJobs
    const jobsBuffer = scanDevice.createBuffer({
        size: Math.max(jobs.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (jobs.byteLength > 0) scanQueue.writeBuffer(jobsBuffer, 0, jobs);

    // Results
    const resultsSize = totalOutputs * G1_JACOBIAN_WORDS * 4;
    const resultsBuffer = scanDevice.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Params uniform: { num_jobs, _p1, _p2, _p3 }
    const paramsData = new Uint32Array([numJobs, 0, 0, 0]);
    const paramsBuffer = scanDevice.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    scanQueue.writeBuffer(paramsBuffer, 0, paramsData);

    // Dispatch
    const encoder = scanDevice.createCommandEncoder();
    const bindGroup = scanDevice.createBindGroup({
        layout: gatherPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: basesBuffer } },
            { binding: 1, resource: { buffer: gatherColsBuffer } },
            { binding: 2, resource: { buffer: jobsBuffer } },
            { binding: 3, resource: { buffer: resultsBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(gatherPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(numJobs, WORKGROUP_SIZE));
    pass.end();

    // Readback
    const stagingBuffer = scanDevice.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(resultsBuffer, 0, stagingBuffer, 0, resultsSize);

    scanQueue.submit([encoder.finish()]);
    await scanDevice.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    // Cleanup (NOT basesBuffer — it's cached)
    gatherColsBuffer.destroy();
    jobsBuffer.destroy();
    resultsBuffer.destroy();
    paramsBuffer.destroy();
    stagingBuffer.destroy();

    return resultData;
}

export async function gpuOnehotGatherDirectRetainBuffer(basesFlat, gatherCols, jobs, numJobs) {
    if (!gatherPipeline) throw new Error('[gpu-onehot] Gather pipeline not compiled');

    const totalOutputs = numJobs;
    if (totalOutputs === 0) {
        return {
            cpuData: new Uint32Array(0),
            gpuBuffer: null,
            gpuBufferSize: 0,
        };
    }

    const basesBuffer = getOrCreateBasesBuffer(basesFlat);

    const gatherColsBuffer = scanDevice.createBuffer({
        size: Math.max(gatherCols.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (gatherCols.byteLength > 0) scanQueue.writeBuffer(gatherColsBuffer, 0, gatherCols);

    const jobsBuffer = scanDevice.createBuffer({
        size: Math.max(jobs.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (jobs.byteLength > 0) scanQueue.writeBuffer(jobsBuffer, 0, jobs);

    const resultsSize = totalOutputs * G1_JACOBIAN_WORDS * 4;
    const resultsBuffer = scanDevice.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const paramsData = new Uint32Array([numJobs, 0, 0, 0]);
    const paramsBuffer = scanDevice.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    scanQueue.writeBuffer(paramsBuffer, 0, paramsData);

    const encoder = scanDevice.createCommandEncoder();
    const bindGroup = scanDevice.createBindGroup({
        layout: gatherPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: basesBuffer } },
            { binding: 1, resource: { buffer: gatherColsBuffer } },
            { binding: 2, resource: { buffer: jobsBuffer } },
            { binding: 3, resource: { buffer: resultsBuffer } },
            { binding: 4, resource: { buffer: paramsBuffer } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(gatherPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(numJobs, WORKGROUP_SIZE));
    pass.end();

    const stagingBuffer = scanDevice.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(resultsBuffer, 0, stagingBuffer, 0, resultsSize);

    scanQueue.submit([encoder.finish()]);
    await scanDevice.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    gatherColsBuffer.destroy();
    jobsBuffer.destroy();
    paramsBuffer.destroy();
    stagingBuffer.destroy();

    return {
        cpuData: resultData,
        gpuBuffer: resultsBuffer,
        gpuBufferSize: resultsSize,
    };
}
