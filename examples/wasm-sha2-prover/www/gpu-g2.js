// WebGPU G2 Fixed-Base Scalar Multiplication Module
//
// API:
//   initGpuG2(device)                                       — compile shader pipeline
//   gpuG2UploadTable(tableLimbs)                            — upload table to persistent GPU buffer
//   gpuG2ScalarMulCached(scalarLimbs, numScalars)           — scalar mul using cached table
//   gpuG2FixedBaseScalarMul(tableLimbs, scalarLimbs, numScalars) — legacy: full table + scalar mul
//   isGpuG2Available()                                      — check if GPU G2 was initialized
//
// Table caching: The precomputed table (~198KB) is uploaded once via gpuG2UploadTable()
// and kept as a persistent GPUBuffer. Subsequent scalar mul calls via gpuG2ScalarMulCached()
// only upload the scalars, eliminating table transfer overhead.

const NUM_LIMBS = 8;
const G2_AFFINE_WORDS = 4 * NUM_LIMBS;
const G2_JACOBIAN_WORDS = 6 * NUM_LIMBS; // 48 u32s per G2Jacobian
const G2_WORKGROUP_SIZE = 128;

let g2Device = null;
let g2Queue = null;
let g2Pipeline = null;
let g2VarBasePipeline = null;
let g2SrsFoldOptPipeline = null;
let _g2Initialized = false;

// Persistent cached table buffer (survives across scalar mul calls)
let g2CachedTableBuffer = null;
let g2CachedTableSize = 0; // byte length of cached table
let g2SrsBuffer = null;  // Affine format for optimized SRS fold
let g2SrsCount = 0;

// Reusable buffer pool for var-base fold dispatches (avoid per-dispatch alloc)
let g2PoolPoints = null;    // STORAGE + COPY_DST (for var-base: both points inputs)
let g2PoolAddends = null;   // STORAGE + COPY_DST
let g2PoolResults = null;   // STORAGE + COPY_SRC
let g2PoolStaging = null;   // COPY_DST + MAP_READ
let g2PoolScalar = null;    // STORAGE + COPY_DST (32 bytes, fixed)
let g2PoolParams = null;    // UNIFORM + COPY_DST (16 bytes, fixed)
let g2PoolMaxPoints = 0;    // current pool capacity
function divCeil(x, y) { return Math.ceil(x / y); }

function ensurePool(numPoints) {
    if (numPoints <= g2PoolMaxPoints) return;
    if (g2PoolPoints) g2PoolPoints.destroy();
    if (g2PoolAddends) g2PoolAddends.destroy();
    if (g2PoolResults) g2PoolResults.destroy();
    if (g2PoolStaging) g2PoolStaging.destroy();
    const dataSize = numPoints * G2_JACOBIAN_WORDS * 4;
    g2PoolPoints = g2Device.createBuffer({
        size: dataSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    g2PoolAddends = g2Device.createBuffer({
        size: dataSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    g2PoolResults = g2Device.createBuffer({
        size: dataSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    g2PoolStaging = g2Device.createBuffer({
        size: dataSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    if (!g2PoolScalar) {
        g2PoolScalar = g2Device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }
    if (!g2PoolParams) {
        g2PoolParams = g2Device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }
    g2PoolMaxPoints = numPoints;
}

export function isGpuG2Available() {
    return _g2Initialized;
}

/**
 * Initialize GPU G2 fixed-base scalar multiplication pipeline.
 * Must be called after WebGPU device is obtained (e.g., after initGpuPairing).
 *
 * @param {GPUDevice} device - WebGPU device (shared with pairing/msm)
 * @returns {Promise<boolean>} - true if initialization succeeded
 */
export async function initGpuG2(device) {
    if (_g2Initialized) return true;
    if (!device) {
        console.warn('[gpu-g2] No GPU device provided');
        return false;
    }

    try {
        g2Device = device;
        g2Queue = device.queue;

        const commonSrc = await (await fetch('shaders/bn254_common.wgsl')).text();
        const g2KernelSrc = await (await fetch('shaders/g2_fixed_base_mul.wgsl')).text();
        const vbKernelSrc = await (await fetch('shaders/g2_var_base_scalar_mul.wgsl')).text();
        const srsFoldOptSrc = await (await fetch('shaders/g2_srs_fold_opt.wgsl')).text();

        const shaderModule = device.createShaderModule({
            code: commonSrc + '\n' + g2KernelSrc,
        });
        const vbShaderModule = device.createShaderModule({
            code: commonSrc + '\n' + vbKernelSrc,
        });
        const srsFoldOptModule = device.createShaderModule({
            code: commonSrc + '\n' + srsFoldOptSrc,
        });

        g2Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'g2_fixed_base_scalar_mul' },
        });
        g2VarBasePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: vbShaderModule, entryPoint: 'g2_var_base_scalar_mul' },
        });
        g2SrsFoldOptPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: srsFoldOptModule, entryPoint: 'g2_srs_fold_opt' },
        });

        _g2Initialized = true;
        console.log('[gpu-g2] WebGPU G2 scalar mul pipeline compiled');
        return true;
    } catch (e) {
        console.warn('[gpu-g2] WebGPU G2 initialization failed:', e);
        return false;
    }
}

/**
 * Upload precomputed table to a persistent GPU buffer.
 * Called once per base point (typically the G2 generator, so just once).
 * The buffer is kept alive and reused by gpuG2ScalarMulCached().
 *
 * @param {Uint32Array} tableLimbs - Precomputed table (51 * 31 * 32 u32s)
 */
export function gpuG2UploadTable(tableLimbs) {
    if (!_g2Initialized) throw new Error('GPU G2 not initialized. Call initGpuG2() first.');

    // Destroy previous cached buffer if any
    if (g2CachedTableBuffer) {
        g2CachedTableBuffer.destroy();
        g2CachedTableBuffer = null;
    }

    g2CachedTableBuffer = g2Device.createBuffer({
        size: tableLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    g2Queue.writeBuffer(g2CachedTableBuffer, 0, tableLimbs);
    g2CachedTableSize = tableLimbs.byteLength;
    console.log(`[gpu-g2] Table uploaded to GPU (${tableLimbs.byteLength} bytes, persistent)`);
}

export function gpuG2UploadSrs(affineLimbs) {
    if (!_g2Initialized) throw new Error('GPU G2 not initialized');
    if (g2SrsBuffer) {
        g2SrsBuffer.destroy();
        g2SrsBuffer = null;
    }
    g2SrsBuffer = g2Device.createBuffer({
        size: affineLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    g2Queue.writeBuffer(g2SrsBuffer, 0, affineLimbs);
    g2SrsCount = affineLimbs.length / G2_AFFINE_WORDS;
    console.log(`[gpu-g2] SRS G2 uploaded in affine (${g2SrsCount} points, ${affineLimbs.byteLength} bytes)`);
}

export async function gpuG2SrsFold(addendsJacobianLimbs, scalarLimbs, numPoints) {
    if (!_g2Initialized) throw new Error('GPU G2 not initialized');
    if (!g2SrsBuffer) throw new Error('No SRS buffer. Call gpuG2UploadSrs() first');
    if (numPoints === 0) return new Uint32Array(0);

    ensurePool(numPoints);
    const resultsSize = numPoints * G2_JACOBIAN_WORDS * 4;

    g2Queue.writeBuffer(g2PoolAddends, 0, addendsJacobianLimbs);
    g2Queue.writeBuffer(g2PoolScalar, 0, scalarLimbs);
    g2Queue.writeBuffer(g2PoolParams, 0, new Uint32Array([numPoints]));

    const encoder = g2Device.createCommandEncoder();
    const bindGroup = g2Device.createBindGroup({
        layout: g2SrsFoldOptPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: g2SrsBuffer } },
            { binding: 1, resource: { buffer: g2PoolAddends } },
            { binding: 2, resource: { buffer: g2PoolResults } },
            { binding: 3, resource: { buffer: g2PoolScalar } },
            { binding: 4, resource: { buffer: g2PoolParams } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(g2SrsFoldOptPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(numPoints, 64));
    pass.end();

    encoder.copyBufferToBuffer(g2PoolResults, 0, g2PoolStaging, 0, resultsSize);
    g2Queue.submit([encoder.finish()]);
    await g2Device.queue.onSubmittedWorkDone();
    await g2PoolStaging.mapAsync(GPUMapMode.READ, 0, resultsSize);
    const resultData = new Uint32Array(g2PoolStaging.getMappedRange(0, resultsSize).slice(0));
    g2PoolStaging.unmap();

    return resultData;
}

/**
 * Run G2 fixed-base scalar multiplication using the cached table buffer.
 * Only scalars are uploaded per call — the table stays on GPU.
 *
 * @param {Uint32Array} scalarLimbs - Scalars (NUM_LIMBS * numScalars u32s, raw bits not Montgomery)
 * @param {number} numScalars       - Number of scalars to process
 * @returns {Promise<Uint32Array>}  - G2Jacobian results (G2_JACOBIAN_WORDS * numScalars u32s)
 */
export async function gpuG2ScalarMulCached(scalarLimbs, numScalars) {
    if (!_g2Initialized) throw new Error('GPU G2 not initialized. Call initGpuG2() first.');
    if (!g2CachedTableBuffer) throw new Error('No cached table. Call gpuG2UploadTable() first.');
    if (numScalars === 0) return new Uint32Array(0);

    // Create per-call buffers (scalars, results, params)
    const scalarBuffer = g2Device.createBuffer({
        size: scalarLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    g2Queue.writeBuffer(scalarBuffer, 0, scalarLimbs);

    const resultsSize = numScalars * G2_JACOBIAN_WORDS * 4;
    const resultsBuffer = g2Device.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const params = new Uint32Array([numScalars]);
    const paramsBuf = g2Device.createBuffer({
        size: 16, // align to 16 bytes for uniform
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    g2Queue.writeBuffer(paramsBuf, 0, params);

    // Dispatch compute using cached table buffer
    const encoder = g2Device.createCommandEncoder();

    const bindGroup = g2Device.createBindGroup({
        layout: g2Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: g2CachedTableBuffer } },
            { binding: 1, resource: { buffer: scalarBuffer } },
            { binding: 2, resource: { buffer: resultsBuffer } },
            { binding: 3, resource: { buffer: paramsBuf } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(g2Pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(numScalars, G2_WORKGROUP_SIZE));
    pass.end();

    // Readback results
    const stagingBuffer = g2Device.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(resultsBuffer, 0, stagingBuffer, 0, resultsSize);

    g2Queue.submit([encoder.finish()]);
    await g2Device.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    // Cleanup per-call buffers only (NOT the cached table buffer)
    scalarBuffer.destroy();
    resultsBuffer.destroy();
    paramsBuf.destroy();
    stagingBuffer.destroy();

    return resultData;
}

export async function gpuG2VarBaseScalarMul(pointsLimbs, addendsLimbs, scalarLimbs, numPoints) {
    if (!_g2Initialized) throw new Error('GPU G2 not initialized. Call initGpuG2() first.');
    if (numPoints === 0) return new Uint32Array(0);

    if (pointsLimbs.length !== numPoints * G2_JACOBIAN_WORDS) {
        throw new Error(`pointsLimbs length mismatch: got ${pointsLimbs.length}, expected ${numPoints * G2_JACOBIAN_WORDS}`);
    }
    if (addendsLimbs.length !== numPoints * G2_JACOBIAN_WORDS) {
        throw new Error(`addendsLimbs length mismatch: got ${addendsLimbs.length}, expected ${numPoints * G2_JACOBIAN_WORDS}`);
    }

    ensurePool(numPoints);
    const resultsSize = numPoints * G2_JACOBIAN_WORDS * 4;

    g2Queue.writeBuffer(g2PoolPoints, 0, pointsLimbs);
    g2Queue.writeBuffer(g2PoolAddends, 0, addendsLimbs);
    g2Queue.writeBuffer(g2PoolScalar, 0, scalarLimbs);
    g2Queue.writeBuffer(g2PoolParams, 0, new Uint32Array([numPoints]));

    const encoder = g2Device.createCommandEncoder();
    const bindGroup = g2Device.createBindGroup({
        layout: g2VarBasePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: g2PoolPoints } },
            { binding: 1, resource: { buffer: g2PoolAddends } },
            { binding: 2, resource: { buffer: g2PoolResults } },
            { binding: 3, resource: { buffer: g2PoolScalar } },
            { binding: 4, resource: { buffer: g2PoolParams } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(g2VarBasePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(numPoints, 64));
    pass.end();

    encoder.copyBufferToBuffer(g2PoolResults, 0, g2PoolStaging, 0, resultsSize);
    g2Queue.submit([encoder.finish()]);
    await g2Device.queue.onSubmittedWorkDone();
    await g2PoolStaging.mapAsync(GPUMapMode.READ, 0, resultsSize);
    const resultData = new Uint32Array(g2PoolStaging.getMappedRange(0, resultsSize).slice(0));
    g2PoolStaging.unmap();

    return resultData;
}

/**
 * Legacy API: Run G2 fixed-base scalar multiplication on GPU.
 * Creates a temporary table buffer per call (no caching).
 * Kept for backward compatibility with tests.
 *
 * @param {Uint32Array} tableLimbs  - Precomputed table (51 * 31 * 32 u32s)
 * @param {Uint32Array} scalarLimbs - Scalars (NUM_LIMBS * numScalars u32s, raw bits not Montgomery)
 * @param {number} numScalars       - Number of scalars to process
 * @returns {Promise<Uint32Array>}  - G2Jacobian results (G2_JACOBIAN_WORDS * numScalars u32s)
 */
export async function gpuG2FixedBaseScalarMul(tableLimbs, scalarLimbs, numScalars) {
    if (!_g2Initialized) throw new Error('GPU G2 not initialized. Call initGpuG2() first.');
    if (numScalars === 0) return new Uint32Array(0);

    // Create GPU buffers
    const tableBuffer = g2Device.createBuffer({
        size: tableLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    g2Queue.writeBuffer(tableBuffer, 0, tableLimbs);

    const scalarBuffer = g2Device.createBuffer({
        size: scalarLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    g2Queue.writeBuffer(scalarBuffer, 0, scalarLimbs);

    const resultsSize = numScalars * G2_JACOBIAN_WORDS * 4;
    const resultsBuffer = g2Device.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const params = new Uint32Array([numScalars]);
    const paramsBuf = g2Device.createBuffer({
        size: 16, // align to 16 bytes for uniform
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    g2Queue.writeBuffer(paramsBuf, 0, params);

    // Dispatch compute
    const encoder = g2Device.createCommandEncoder();

    const bindGroup = g2Device.createBindGroup({
        layout: g2Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: tableBuffer } },
            { binding: 1, resource: { buffer: scalarBuffer } },
            { binding: 2, resource: { buffer: resultsBuffer } },
            { binding: 3, resource: { buffer: paramsBuf } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(g2Pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(numScalars, G2_WORKGROUP_SIZE));
    pass.end();

    // Readback results
    const stagingBuffer = g2Device.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(resultsBuffer, 0, stagingBuffer, 0, resultsSize);

    g2Queue.submit([encoder.finish()]);
    await g2Device.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    // Cleanup
    tableBuffer.destroy();
    scalarBuffer.destroy();
    resultsBuffer.destroy();
    paramsBuf.destroy();
    stagingBuffer.destroy();

    return resultData;
}
