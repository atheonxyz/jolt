// WebGPU OneHot Batch G1 Addition Module — Gather Dispatch
//
// Rust builds per-(chunk,ki) gather lists with rayon, sends pre-built lists
// to JS for direct GPU dispatch. Each GPU thread only loads the base points
// it needs.
//
// API:
//   initGpuOnehot(device)                                        — compile shader pipeline
//   gpuOnehotGatherDirect(bases, gatherCols, jobs, n)             — gather dispatch
//   gpuOnehotGatherDirectRetainBuffer(bases, gatherCols, jobs, n) — gather dispatch (retains GPU buffer)

const NUM_LIMBS = 8;
const G1_AFFINE_WORDS = 2 * NUM_LIMBS;   // 16 u32s per G1Affine
const G1_JACOBIAN_WORDS = 3 * NUM_LIMBS; // 24 u32s per G1Jacobian
const WORKGROUP_SIZE = 128; // must match shader @workgroup_size

let scanDevice = null;
let scanQueue = null;
let gatherPipeline = null;
let _onehotInitialized = false;

// Bases buffer cache: upload once, reuse across all dispatches for same bases.
// All OneHot polys share the same g1_generators, so bases never change within a proof.
let _cachedBasesBuffer = null;
let _cachedBasesSize = 0;

// Pending buffer: set synchronously during dispatch, before first await.
// Allows chaining GPU operations (e.g. pairing) without waiting for onehot resolve.
let _pendingGpuBuffer = null;
let _pendingGpuBufferSize = 0;
function divCeil(x, y) { return Math.ceil(x / y); }

/**
 * Initialize GPU OneHot gather pipeline.
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

        // Load shader sources: common + G1 curve + onehot gather kernel
        const commonSrc = await (await fetch('shaders/bn254_common.wgsl')).text();
        const g1CurveSrc = await (await fetch('shaders/msm_g1_curve.wgsl')).text();
        const gatherSrc = await (await fetch('./shaders/onehot_gather.wgsl')).text();
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
        console.log('[gpu-onehot] WebGPU OneHot gather pipeline compiled (onehot_gather)');
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

    // Set pending buffer BEFORE any await — this enables GPU pipeline chaining.
    // The resultsBuffer exists on GPU and commands are submitted. Another shader
    // can reference this buffer; WebGPU guarantees command ordering within a queue.
    _pendingGpuBuffer = resultsBuffer;
    _pendingGpuBufferSize = resultsSize;

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


/**
 * Get the GPU buffer from the most recent onehot dispatch.
 * MUST be called after gpuOnehotGatherDirectRetainBuffer dispatch returns
 * (the Promise is created, sync code up to first await has executed).
 * The buffer is valid for use in subsequent GPU dispatches (e.g. pairing)
 * because WebGPU guarantees command ordering within a queue.
 * @returns {{ gpuBuffer: GPUBuffer|null, gpuBufferSize: number }}
 */
export function gpuOnehotGetPendingBuffer() {
    return { gpuBuffer: _pendingGpuBuffer, gpuBufferSize: _pendingGpuBufferSize };
}