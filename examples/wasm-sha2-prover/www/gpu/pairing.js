// WebGPU BN254 Batch Multi-Pairing Module
// Single-pass Miller loop with G2 dedup: G2 bases sent once, indexed via modular arithmetic.
//
// API:
//   initGpuPairing()         — initialize WebGPU device and compile pipelines
//   gpuBatchMultiPairing(g1Flat, g2Flat, groupSizes, groupOffsets) — run batch pairing
//   isGpuAvailable()         — check if WebGPU was initialized

const NUM_LIMBS = 8;
const FP12_WORDS = 12 * NUM_LIMBS; // 96 u32s per Fp12
const G1_WORDS_PER_PAIR = 2 * NUM_LIMBS;  // 16
const G2_WORDS_PER_PAIR = 4 * NUM_LIMBS;  // 32
const MILLER_WORKGROUP_SIZE = 64;
const REDUCE_WORKGROUP_SIZE = 256;

let device = null;
let queue = null;
let jac2affinePipeline = null;
let millerPipeline = null;
let reducePipeline = null;
let combineHintsPipeline = null;
let _initialized = false;

function divCeil(x, y) { return Math.ceil(x / y); }


export function getGpuDevice() {
    return device;
}

export async function initGpuPairing() {
    if (_initialized) return true;
    if (!navigator.gpu) {
        console.warn('[gpu-pairing] WebGPU not supported in this browser');
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) {
            console.warn('[gpu-pairing] No GPU adapter found');
            return false;
        }

        const adapterInfo = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
        console.log(`[gpu-pairing] GPU: ${adapterInfo.description || adapterInfo.device || adapterInfo.architecture || 'unknown'}`);

        device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            }
        });
        queue = device.queue;

        const cacheBust = '?v=' + Date.now();
        const commonSrc = await (await fetch('shaders/bn254_common.wgsl' + cacheBust)).text();
        const g1CurveSrc = await (await fetch('shaders/msm_g1_curve.wgsl' + cacheBust)).text();
        const jac2AffineEntry = await (await fetch('shaders/jac2affine.wgsl' + cacheBust)).text();
        const combineHintsEntry = await (await fetch('shaders/combine_hints.wgsl' + cacheBust)).text();
        const millerEntry = await (await fetch('shaders/bn254_miller.wgsl' + cacheBust)).text();
        const reduceEntry = await (await fetch('shaders/bn254_reduce.wgsl' + cacheBust)).text();

        const jac2AffineShader = device.createShaderModule({ code: commonSrc + '\n' + g1CurveSrc + '\n' + jac2AffineEntry });
        const combineHintsShader = device.createShaderModule({ code: commonSrc + '\n' + g1CurveSrc + '\n' + combineHintsEntry });
        const millerShader = device.createShaderModule({ code: commonSrc + '\n' + millerEntry });
        const reduceShader = device.createShaderModule({ code: commonSrc + '\n' + reduceEntry });

        jac2affinePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: jac2AffineShader, entryPoint: 'jac2affine_kernel' },
        });
        millerPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: millerShader, entryPoint: 'miller_loop_kernel' },
        });
        combineHintsPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: combineHintsShader, entryPoint: 'combine_hints_kernel' },
        });
        reducePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: reduceShader, entryPoint: 'fp12_reduce' },
        });

        _initialized = true;
        console.log('[gpu-pairing] WebGPU pipelines compiled');
        return true;
    } catch (e) {
        console.warn('[gpu-pairing] WebGPU initialization failed:', e);
        return false;
    }
}

/**
 * Run batch multi-pairing on GPU.
 * G2 bases are sent ONCE (deduped). The shader indexes G2 via tid % numG2Bases.
 *
 * @param {Uint32Array} g1Flat - Flattened G1 points (all groups concatenated)
 * @param {Uint32Array} g2Flat - G2 bases (sent once, NOT duplicated per group)
 * @param {Uint32Array} groupSizes - Number of pairs per group
 * @param {Uint32Array} groupOffsets - Pair offset for each group
 * @returns {Promise<Uint32Array>} - Fp12 Miller loop results, FP12_WORDS per group
 */
export async function gpuBatchMultiPairing(g1Flat, g2Flat, groupSizes, groupOffsets, numG2Bases) {
    if (!_initialized) throw new Error('GPU not initialized. Call initGpuPairing() first.');

    const totalPairs = g1Flat.length / G1_WORDS_PER_PAIR;

    const g1Buffer = device.createBuffer({
        size: g1Flat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(g1Buffer, 0, g1Flat);

    const g2Buffer = device.createBuffer({
        size: g2Flat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(g2Buffer, 0, g2Flat);

    const resultsSize = totalPairs * FP12_WORDS * 4;
    const resultsBuffer = device.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const millerParams = new Uint32Array([totalPairs, numG2Bases || (g2Flat.length / G2_WORDS_PER_PAIR), 0, 0]);
    const millerParamsBuf = device.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(millerParamsBuf, 0, millerParams);

    const encoder = device.createCommandEncoder();

    // Miller loop pass — one thread per pair, G2 indexed via modular arithmetic
    const millerBG = device.createBindGroup({
        layout: millerPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: g1Buffer } },
            { binding: 1, resource: { buffer: g2Buffer } },
            { binding: 2, resource: { buffer: resultsBuffer } },
            { binding: 3, resource: { buffer: millerParamsBuf } },
        ],
    });
    const millerPass = encoder.beginComputePass();
    millerPass.setPipeline(millerPipeline);
    millerPass.setBindGroup(0, millerBG);
    millerPass.dispatchWorkgroups(divCeil(totalPairs, MILLER_WORKGROUP_SIZE));
    millerPass.end();

    for (let g = 0; g < groupSizes.length; g++) {
        let count = groupSizes[g];
        const elementOffset = groupOffsets[g];
        if (count <= 1) continue;
        while (count > 1) {
            const stride = divCeil(count, 2);
            const numThreads = count - stride;
            const rp = new Uint32Array([stride, numThreads, elementOffset, 0]);
            const rpBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            queue.writeBuffer(rpBuf, 0, rp);
            const rBG = device.createBindGroup({
                layout: reducePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: resultsBuffer } },
                    { binding: 1, resource: { buffer: rpBuf } },
                ],
            });
            const rPass = encoder.beginComputePass();
            rPass.setPipeline(reducePipeline);
            rPass.setBindGroup(0, rBG);
            rPass.dispatchWorkgroups(divCeil(numThreads, REDUCE_WORKGROUP_SIZE));
            rPass.end();
            count = stride;
        }
    }

    const numGroups = groupSizes.length;
    const readbackSize = numGroups * FP12_WORDS * 4;
    const stagingBuffer = device.createBuffer({
        size: readbackSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    for (let g = 0; g < numGroups; g++) {
        encoder.copyBufferToBuffer(
            resultsBuffer, groupOffsets[g] * FP12_WORDS * 4,
            stagingBuffer, g * FP12_WORDS * 4,
            FP12_WORDS * 4
        );
    }

    queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    // Cleanup
    g1Buffer.destroy();
    g2Buffer.destroy();
    resultsBuffer.destroy();
    millerParamsBuf.destroy();
    stagingBuffer.destroy();

    return resultData;
}

export async function gpuBatchMultiPairingFromBuffer(
    onehotBuffer,
    onehotBufferSize,
    polyLayoutFlat,
    totalAffinePoints,
    g2Flat,
    groupSizes,
    groupOffsets,
    numG2Bases,
) {
    if (!_initialized) throw new Error('GPU not initialized. Call initGpuPairing() first.');
    if (!jac2affinePipeline) throw new Error('[gpu-pairing] jac2affine pipeline not initialized');
    if (!onehotBuffer) throw new Error('[gpu-pairing] onehotBuffer is required');

    const numPolys = Math.floor(polyLayoutFlat.length / 4);
    const totalPairs = groupSizes.reduce((acc, n) => acc + n, 0);
    if (totalPairs === 0) return new Uint32Array(groupSizes.length * FP12_WORDS);

    const affineBufferSize = totalAffinePoints * G1_WORDS_PER_PAIR * 4;
    if (onehotBufferSize < totalAffinePoints * 24 * 4) {
        throw new Error('[gpu-pairing] onehotBufferSize too small for expected Jacobian points');
    }
    const affineBuffer = device.createBuffer({
        size: affineBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const g1PackedBuffer = device.createBuffer({
        size: Math.max(totalPairs * G1_WORDS_PER_PAIR * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const polyLayoutBuffer = device.createBuffer({
        size: Math.max(polyLayoutFlat.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (polyLayoutFlat.byteLength > 0) queue.writeBuffer(polyLayoutBuffer, 0, polyLayoutFlat);

    const totalPoints = polyLayoutFlat.reduce((acc, value, idx) => {
        if ((idx % 4) === 0) {
            return acc + value * polyLayoutFlat[idx + 1];
        }
        return acc;
    }, 0);
    const jac2affineParams = new Uint32Array([totalPoints, numPolys, 0, 0]);
    const jac2affineParamsBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(jac2affineParamsBuf, 0, jac2affineParams);

    const g2Buffer = device.createBuffer({
        size: Math.max(g2Flat.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (g2Flat.byteLength > 0) queue.writeBuffer(g2Buffer, 0, g2Flat);

    const resultsSize = totalPairs * FP12_WORDS * 4;
    const millerResultsBuffer = device.createBuffer({
        size: resultsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const numG2 = numG2Bases || (g2Flat.length / G2_WORDS_PER_PAIR);
    const millerParams = new Uint32Array([totalPairs, numG2, 0, 0]);
    const millerParamsBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(millerParamsBuf, 0, millerParams);

    const encoder = device.createCommandEncoder();

    const jac2AffineBG = device.createBindGroup({
        layout: jac2affinePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: onehotBuffer } },
            { binding: 1, resource: { buffer: affineBuffer } },
            { binding: 2, resource: { buffer: polyLayoutBuffer } },
            { binding: 3, resource: { buffer: jac2affineParamsBuf } },
        ],
    });
    const jacPass = encoder.beginComputePass();
    jacPass.setPipeline(jac2affinePipeline);
    jacPass.setBindGroup(0, jac2AffineBG);
    jacPass.dispatchWorkgroups(divCeil(totalPoints, 128));
    jacPass.end();

    for (let g = 0; g < groupSizes.length; g++) {
        const polyAffineOffset = polyLayoutFlat[g * 4 + 3];
        const packedOffset = groupOffsets[g];
        const count = groupSizes[g];
        if (count === 0) continue;
        encoder.copyBufferToBuffer(
            affineBuffer,
            polyAffineOffset * G1_WORDS_PER_PAIR * 4,
            g1PackedBuffer,
            packedOffset * G1_WORDS_PER_PAIR * 4,
            count * G1_WORDS_PER_PAIR * 4,
        );
    }

    const millerBG = device.createBindGroup({
        layout: millerPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: g1PackedBuffer } },
            { binding: 1, resource: { buffer: g2Buffer } },
            { binding: 2, resource: { buffer: millerResultsBuffer } },
            { binding: 3, resource: { buffer: millerParamsBuf } },
        ],
    });
    const millerPass = encoder.beginComputePass();
    millerPass.setPipeline(millerPipeline);
    millerPass.setBindGroup(0, millerBG);
    millerPass.dispatchWorkgroups(divCeil(totalPairs, MILLER_WORKGROUP_SIZE));
    millerPass.end();

    const reduceParamBuffers = [];
    for (let g = 0; g < groupSizes.length; g++) {
        let count = groupSizes[g];
        const elementOffset = groupOffsets[g];
        if (count <= 1) continue;
        while (count > 1) {
            const stride = divCeil(count, 2);
            const numThreads = count - stride;
            const rp = new Uint32Array([stride, numThreads, elementOffset, 0]);
            const rpBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            queue.writeBuffer(rpBuf, 0, rp);
            reduceParamBuffers.push(rpBuf);
            const rBG = device.createBindGroup({
                layout: reducePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: millerResultsBuffer } },
                    { binding: 1, resource: { buffer: rpBuf } },
                ],
            });
            const rPass = encoder.beginComputePass();
            rPass.setPipeline(reducePipeline);
            rPass.setBindGroup(0, rBG);
            rPass.dispatchWorkgroups(divCeil(numThreads, REDUCE_WORKGROUP_SIZE));
            rPass.end();
            count = stride;
        }
    }

    const numGroups = groupSizes.length;
    const readbackSize = numGroups * FP12_WORDS * 4;
    const stagingBuffer = device.createBuffer({
        size: readbackSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    for (let g = 0; g < numGroups; g++) {
        encoder.copyBufferToBuffer(
            millerResultsBuffer,
            groupOffsets[g] * FP12_WORDS * 4,
            stagingBuffer,
            g * FP12_WORDS * 4,
            FP12_WORDS * 4,
        );
    }

    queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    polyLayoutBuffer.destroy();
    affineBuffer.destroy();
    g1PackedBuffer.destroy();
    g2Buffer.destroy();
    millerResultsBuffer.destroy();
    jac2affineParamsBuf.destroy();
    millerParamsBuf.destroy();
    for (const rpBuf of reduceParamBuffers) rpBuf.destroy();
    stagingBuffer.destroy();

    return resultData;
}

export async function gpuCombineHints(pointsFlat, scalarsFlat, numRows, numPolys) {
    if (!_initialized) throw new Error('GPU not initialized. Call initGpuPairing() first.');
    if (!combineHintsPipeline) throw new Error('[gpu-pairing] combine_hints pipeline not initialized');

    const pointsBuffer = device.createBuffer({
        size: Math.max(pointsFlat.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (pointsFlat.byteLength > 0) queue.writeBuffer(pointsBuffer, 0, pointsFlat);

    const scalarsBuffer = device.createBuffer({
        size: Math.max(scalarsFlat.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (scalarsFlat.byteLength > 0) queue.writeBuffer(scalarsBuffer, 0, scalarsFlat);

    const resultsSize = numRows * 24 * 4;
    const resultsBuffer = device.createBuffer({
        size: Math.max(resultsSize, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numRows, numPolys, 0, 0]));

    const stagingBuffer = device.createBuffer({
        size: Math.max(resultsSize, 4),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    const bindGroup = device.createBindGroup({
        layout: combineHintsPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointsBuffer } },
            { binding: 1, resource: { buffer: scalarsBuffer } },
            { binding: 2, resource: { buffer: resultsBuffer } },
            { binding: 3, resource: { buffer: paramsBuffer } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(combineHintsPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(divCeil(numRows, 128));
    pass.end();

    if (resultsSize > 0) {
        encoder.copyBufferToBuffer(resultsBuffer, 0, stagingBuffer, 0, resultsSize);
    }

    queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(stagingBuffer.getMappedRange().slice(0, resultsSize));
    stagingBuffer.unmap();

    pointsBuffer.destroy();
    scalarsBuffer.destroy();
    resultsBuffer.destroy();
    paramsBuffer.destroy();
    stagingBuffer.destroy();

    return resultData;
}


