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
const G2_JACOBIAN_WORDS = 6 * NUM_LIMBS; // 48 u32s per G2Jacobian
const G2_WORKGROUP_SIZE = 128;

let g2Device = null;
let g2Queue = null;
let g2Pipeline = null;
let _g2Initialized = false;

// Persistent cached table buffer (survives across scalar mul calls)
let g2CachedTableBuffer = null;
let g2CachedTableSize = 0; // byte length of cached table

function divCeil(x, y) { return Math.ceil(x / y); }

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

        const shaderModule = device.createShaderModule({
            code: commonSrc + '\n' + g2KernelSrc,
        });

        g2Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'g2_fixed_base_scalar_mul' },
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
