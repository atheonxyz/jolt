// gpu-msm.js — WebGPU batch MSM (CUZK Pippenger) for BN254 G1
// GPU-native pipeline: scalar decomposition + CSC construction + SMVP + PBPR + Horner
// All preprocessing happens on GPU — only raw scalars are uploaded from CPU.

const NUM_LIMBS = 8;
const PT_STRIDE = 16; // 8 limbs x + 8 limbs y per affine point

const GLV_U32_MASK = 0xFFFFFFFFn;
const GLV_SHIFT_256 = 1n << 256n;

const BN254_SCALAR_MODULUS = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001n;
const GLV_LAMBDA = 0x59e26bcea0d48bacd4f5211b058fa68n;
const GLV_N_A = 0x8211bbeb7d4f11286f4d8248eeb859fcn;
const GLV_N_B = 0x89d3256894d213e3n;
const GLV_N_C = 0x0be4e1541221250b6f4d8248eeb859fdn;

function roundDiv(numerator, denominator) {
    return (numerator + (denominator >> 1n)) / denominator;
}

const GLV_G1_PRECOMP = roundDiv(GLV_SHIFT_256 * GLV_N_C, BN254_SCALAR_MODULUS);
const GLV_G2_PRECOMP = roundDiv(GLV_SHIFT_256 * GLV_N_B, BN254_SCALAR_MODULUS);

function readScalarLE(words, offset, scalarBitWidth) {
    const activeLimbs = Math.ceil(scalarBitWidth / 32);
    let value = 0n;
    for (let i = activeLimbs - 1; i >= 0; i--) {
        value = (value << 32n) | BigInt(words[offset + i]);
    }
    const rem = scalarBitWidth & 31;
    if (rem !== 0) {
        value &= (1n << BigInt(scalarBitWidth)) - 1n;
    }
    return value;
}

function writeSignedScalarLE(words, offset, value, bitWidth) {
    const width = BigInt(bitWidth);
    const modulus = 1n << width;
    let encoded = value;
    if (encoded < 0n) {
        encoded += modulus;
    }
    encoded &= modulus - 1n;
    for (let i = 0; i < NUM_LIMBS; i++) {
        words[offset + i] = Number(encoded & GLV_U32_MASK);
        encoded >>= 32n;
    }
}

function glvDecomposeScalar(k) {
    const b1 = (k * GLV_G1_PRECOMP) >> 256n;
    const b2 = (k * GLV_G2_PRECOMP) >> 256n;
    const k1 = k - b1 * GLV_N_A - b2 * GLV_N_B;
    const k2 = b1 * GLV_N_B - b2 * GLV_N_C;
    return { k1, k2 };
}

function glvDecomposeScalars(scalarsFlat, numPoints, scalarBitWidth, batchSize) {
    const doubledNumPoints = numPoints * 2;
    const out = new Uint32Array(batchSize * doubledNumPoints * NUM_LIMBS);
    const glvBitWidth = Math.max(64, Math.ceil(scalarBitWidth / 2));
    const scalarBitWidthBig = BigInt(scalarBitWidth);
    const signBit = 1n << (scalarBitWidthBig - 1n);
    const signedWrap = 1n << scalarBitWidthBig;
    let checked = false;

    for (let msm = 0; msm < batchSize; msm++) {
        const msmInBase = msm * numPoints * NUM_LIMBS;
        const msmOutBase = msm * doubledNumPoints * NUM_LIMBS;
        for (let point = 0; point < numPoints; point++) {
            const inOffset = msmInBase + point * NUM_LIMBS;
            const kRaw = readScalarLE(scalarsFlat, inOffset, scalarBitWidth);
            const signedK = (kRaw & signBit) !== 0n ? kRaw - signedWrap : kRaw;
            let k = signedK % BN254_SCALAR_MODULUS;
            if (k < 0n) {
                k += BN254_SCALAR_MODULUS;
            }
            const { k1, k2 } = glvDecomposeScalar(k);

            if (!checked) {
                let recomposed = (k1 + k2 * GLV_LAMBDA) % BN254_SCALAR_MODULUS;
                if (recomposed < 0n) {
                    recomposed += BN254_SCALAR_MODULUS;
                }
                if (recomposed !== k) {
                    throw new Error('[gpu-msm] GLV decomposition self-check failed');
                }
                checked = true;
            }

            const k1Offset = msmOutBase + point * NUM_LIMBS;
            const k2Offset = msmOutBase + (numPoints + point) * NUM_LIMBS;
            writeSignedScalarLE(out, k1Offset, k1, glvBitWidth);
            writeSignedScalarLE(out, k2Offset, k2, glvBitWidth);
        }
    }

    return { scalars: out, glvBitWidth };
}

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

let _smvpWgSize = 64;
// ── Shader loading ───────────────────────────────────────────────────────────
let _shaderCache = null;
async function loadShaders() {
    if (_shaderCache) return _shaderCache;
    const base = self.location ? '' : '';
    const [common, curve, cscSetup, smvp, pbpr, horner, glv] = await Promise.all([
        fetch(`${base}shaders/bn254_common.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_g1_curve.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_csc_setup.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_smvp.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_pbpr.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_horner.wgsl`).then(r => r.text()),
        fetch(`${base}shaders/msm_glv.wgsl`).then(r => r.text()),
    ]);
    _shaderCache = { common, curve, cscSetup, smvp, pbpr, horner, glv };
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
        { name: 'horner', module: device.createShaderModule({ code: commonCode + shaders.horner }) },
        { name: 'glv', module: device.createShaderModule({ code: commonCode + shaders.glv }) },
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
    const hornerModule = modules[3].module;
    const glvModule = modules[4].module;
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

    // ── SMVP bind group layout
    const smvpBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // col_ptr
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // val_idx
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // points
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // buckets
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // params
        ],
    });

    // ── PBPR bind group layout
    const pbprBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    // ── Horner bind group layout
    const hornerBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    const glvBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });

    // ── Create all compute pipelines asynchronously for better init performance
    const smvpPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [smvpBGL] });
    const hornerPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [hornerBGL] });
    const pbprPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [pbprBGL] });
    const glvPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [glvBGL] });

    const [decomposePipeline, prefixSumPipeline, scatterPipeline, hornerPipeline, glvPipeline] =
        await Promise.all([
            device.createComputePipelineAsync({
                layout: cscPipelineLayout,
                compute: { module: cscModule, entryPoint: 'decompose_scalars', constants: { WG_SIZE: 256 } },
            }),
            device.createComputePipelineAsync({
                layout: cscPipelineLayout,
                compute: { module: cscModule, entryPoint: 'prefix_sum' },
            }),
            device.createComputePipelineAsync({
                layout: cscPipelineLayout,
                compute: { module: cscModule, entryPoint: 'scatter_csc', constants: { WG_SIZE: 256 } },
            }),
            device.createComputePipelineAsync({
                layout: hornerPipelineLayout,
                compute: { module: hornerModule, entryPoint: 'horner_reduce' },
            }),
            device.createComputePipelineAsync({
                layout: glvPipelineLayout,
                compute: { module: glvModule, entryPoint: 'apply_endomorphism', constants: { WG_SIZE: 256 } },
            }),
        ]);
    console.log('[gpu-msm] All pipelines created asynchronously');

    _msmPipeline = {
        device,
        cscBGL, decomposePipeline, prefixSumPipeline, scatterPipeline,
        smvpBGL, pbprBGL, hornerBGL,
        hornerPipeline,
        glvBGL, glvPipeline,
        smvpModule, smvpPipelineLayout,
        pbprModule, pbprPipelineLayout,
        _smvpPipelineCache: new Map(),
        _pbprPipelineCache: new Map(),
        _pointsCache: null,
        _glvPointsCache: null,
    };
    return _msmPipeline;
}

// Get or create PBPR pipelines for a specific workgroup size
async function getPBPRPipelines(p, bWgSize) {
    if (p._pbprPipelineCache.has(bWgSize)) {
        return p._pbprPipelineCache.get(bWgSize);
    }
    const [bpr1Pipeline, bpr2Pipeline] = await Promise.all([
        p.device.createComputePipelineAsync({
            layout: p.pbprPipelineLayout,
            compute: { module: p.pbprModule, entryPoint: 'bpr_stage_1', constants: { WG_SIZE: bWgSize } },
        }),
        p.device.createComputePipelineAsync({
            layout: p.pbprPipelineLayout,
            compute: { module: p.pbprModule, entryPoint: 'bpr_stage_2', constants: { WG_SIZE: bWgSize } },
        }),
    ]);
    const entry = { bpr1Pipeline, bpr2Pipeline };
    p._pbprPipelineCache.set(bWgSize, entry);
    return entry;
}

async function getSMVPPipeline(p, wgSize) {
    if (p._smvpPipelineCache.has(wgSize)) {
        return p._smvpPipelineCache.get(wgSize);
    }
    const pipeline = await p.device.createComputePipelineAsync({
        layout: p.smvpPipelineLayout,
        compute: { module: p.smvpModule, entryPoint: 'smvp', constants: { WG_SIZE: wgSize } },
    });
    p._smvpPipelineCache.set(wgSize, pipeline);
    return pipeline;
}

async function computeEndomorphismPoints(pointsFlat, numPoints, negateFlags) {
    if (numPoints === 0) {
        return new Uint32Array(0);
    }

    const p = _msmPipeline;
    const device = p.device;
    const flags = negateFlags || new Uint32Array(numPoints);

    const pointsInBuf = device.createBuffer({
        size: pointsFlat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pointsInBuf, 0, pointsFlat);

    const pointsOutBuf = device.createBuffer({
        size: pointsFlat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const flagsBuf = device.createBuffer({
        size: Math.max(flags.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (flags.byteLength > 0) {
        device.queue.writeBuffer(flagsBuf, 0, flags);
    }

    const params = new Uint32Array([numPoints, 0, 0, 0]);
    const paramsBuf = device.createBuffer({
        size: params.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, params);

    const glvBG = device.createBindGroup({
        layout: p.glvBGL,
        entries: [
            { binding: 0, resource: { buffer: pointsInBuf } },
            { binding: 1, resource: { buffer: pointsOutBuf } },
            { binding: 2, resource: { buffer: flagsBuf } },
            { binding: 3, resource: { buffer: paramsBuf } },
        ],
    });

    const stagingBuf = device.createBuffer({
        size: pointsFlat.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(p.glvPipeline);
    pass.setBindGroup(0, glvBG);
    pass.dispatchWorkgroups(Math.ceil(numPoints / 256), 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(pointsOutBuf, 0, stagingBuf, 0, pointsFlat.byteLength);
    device.queue.submit([encoder.finish()]);

    await stagingBuf.mapAsync(GPUMapMode.READ);
    const endoPoints = new Uint32Array(stagingBuf.getMappedRange().slice(0));
    stagingBuf.unmap();

    pointsInBuf.destroy();
    pointsOutBuf.destroy();
    flagsBuf.destroy();
    paramsBuf.destroy();
    stagingBuf.destroy();

    return endoPoints;
}

async function getOrCreateGLVPoints(pointsFlat, numPoints) {
    const p = _msmPipeline;
    if (
        p._glvPointsCache &&
        p._glvPointsCache.pointsFlat === pointsFlat &&
        p._glvPointsCache.numPoints === numPoints
    ) {
        return p._glvPointsCache.doubledPoints;
    }

    const endoPoints = await computeEndomorphismPoints(pointsFlat, numPoints);
    const doubledPoints = new Uint32Array(numPoints * 2 * PT_STRIDE);
    doubledPoints.set(pointsFlat, 0);
    doubledPoints.set(endoPoints, numPoints * PT_STRIDE);

    p._glvPointsCache = {
        pointsFlat,
        numPoints,
        doubledPoints,
    };

    return doubledPoints;
}

async function gpuBatchMSMWithGLV(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) {
    if (scalarBitWidth <= 64) {
        return gpuBatchMSM(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize);
    }

    const t0 = performance.now();
    const { scalars: glvScalars, glvBitWidth } = glvDecomposeScalars(
        scalarsFlat,
        numPoints,
        scalarBitWidth,
        batchSize,
    );
    const doubledPoints = await getOrCreateGLVPoints(pointsFlat, numPoints);
    const out = await gpuBatchMSM(doubledPoints, glvScalars, numPoints * 2, glvBitWidth, batchSize);
    const t1 = performance.now();

    console.log(`[gpu-msm] glv batch=${batchSize} pts=${numPoints} bits=${scalarBitWidth} ` +
        `expanded_pts=${numPoints * 2} expanded_bits=${glvBitWidth} total=${(t1 - t0).toFixed(1)}ms`);

    return out;
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
    const { bpr1Pipeline, bpr2Pipeline } = await getPBPRPipelines(p, bWgSize);

    // ── Point buffer: cache across calls (same bases reused for all polynomials)
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

    // ── Buffer cache: reuse all intermediate buffers + bind groups when dimensions match
    // Only scalars change between calls; all buffer sizes depend on (numPoints, batchSize, scalarBitWidth).
    // Atomic buffers (colPtr, scatterCnt) are cleared via encoder.clearBuffer() each call.
    const cacheKey = `${numPoints}-${batchSize}-${scalarBitWidth}`;
    let c = p._bufferCache;

    if (!c || c.key !== cacheKey || c.pointsBuf !== pointsBuf) {
        // Destroy old cache
        if (c) {
            for (const buf of c.allBuffers) buf.destroy();
        }

        const totalWork = totalSubtasks * numPoints;
        const colPtrLen = totalSubtasks * (numColumns + 1);
        const resultLen = batchSize * 3 * NUM_LIMBS;

        const scalarsBuf = device.createBuffer({
            size: scalarsFlat.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const colIndicesBuf = device.createBuffer({
            size: Math.max(totalWork * 4, 4),
            usage: GPUBufferUsage.STORAGE,
        });
        // COPY_DST needed for encoder.clearBuffer() between calls
        const colPtrBuf = device.createBuffer({
            size: Math.max(colPtrLen * 4, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const valIdxsBuf = device.createBuffer({
            size: Math.max(totalWork * 4, 4),
            usage: GPUBufferUsage.STORAGE,
        });
        const scatterCntBuf = device.createBuffer({
            size: Math.max(totalSubtasks * numColumns * 4, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const cscParams = new Uint32Array([numPoints, numColumns, windowSize, subtasksPerMSM, batchSize, totalSubtasks, NUM_LIMBS]);
        const cscParamsBuf = device.createBuffer({
            size: cscParams.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(cscParamsBuf, 0, cscParams);

        const bucketLen = halfColumns * totalSubtasks * 4 * NUM_LIMBS;
        const bucketBuf = device.createBuffer({
            size: Math.max(bucketLen * 4, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const gPointsLen = totalSubtasks * bWgSize * 4 * NUM_LIMBS;
        const gPointsBuf = device.createBuffer({
            size: Math.max(gPointsLen * 4, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const resultBuf = device.createBuffer({
            size: Math.max(resultLen * 4, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const stagingBuf = device.createBuffer({
            size: resultLen * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Param buffers (constant for this dimension set)
        const smvpParams = new Uint32Array([numPoints, numColumns, totalSubtasks, 0, 0]);
        const smvpParamsBuf = device.createBuffer({
            size: smvpParams.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(smvpParamsBuf, 0, smvpParams);

        const bprParams = new Uint32Array([numColumns, 0]);
        const bprParamsBuf = device.createBuffer({
            size: bprParams.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bprParamsBuf, 0, bprParams);

        const hornerParams = new Uint32Array([batchSize, subtasksPerMSM, bWgSize, windowSize]);
        const hornerParamsBuf = device.createBuffer({
            size: hornerParams.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(hornerParamsBuf, 0, hornerParams);

        // Bind groups (reference cached buffers — valid as long as cache lives)
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
        const bprBG = device.createBindGroup({
            layout: p.pbprBGL,
            entries: [
                { binding: 0, resource: { buffer: bucketBuf } },
                { binding: 1, resource: { buffer: gPointsBuf } },
                { binding: 2, resource: { buffer: bprParamsBuf } },
            ],
        });
        const hornerBG = device.createBindGroup({
            layout: p.hornerBGL,
            entries: [
                { binding: 0, resource: { buffer: gPointsBuf } },
                { binding: 1, resource: { buffer: resultBuf } },
                { binding: 2, resource: { buffer: hornerParamsBuf } },
            ],
        });

        c = {
            key: cacheKey, pointsBuf,
            scalarsBuf, colPtrBuf, scatterCntBuf,
            resultBuf, stagingBuf, resultLen,
            cscBG, smvpBG, bprBG, hornerBG,
            allBuffers: [
                scalarsBuf, colIndicesBuf, colPtrBuf, valIdxsBuf, scatterCntBuf,
                cscParamsBuf, bucketBuf, gPointsBuf, resultBuf, stagingBuf,
                smvpParamsBuf, bprParamsBuf, hornerParamsBuf,
            ],
        };
        p._bufferCache = c;
    }

    // Upload new scalars (the only per-call CPU→GPU transfer)
    device.queue.writeBuffer(c.scalarsBuf, 0, scalarsFlat);

    const t1 = performance.now();

    // ── Command encoder: clear atomics → CSC → SMVP → PBPR → Horner → readback
    const encoder = device.createCommandEncoder();

    // Zero-fill atomic buffers (histogram and scatter counters must start at 0)
    encoder.clearBuffer(c.colPtrBuf);
    encoder.clearBuffer(c.scatterCntBuf);

    const MAX_WG_DIM = device.limits.maxComputeWorkgroupsPerDimension || 65535;
    function splitDispatch(totalWG) {
        if (totalWG <= MAX_WG_DIM) return [totalWG, 1, 1];
        const x = MAX_WG_DIM;
        const y = Math.ceil(totalWG / MAX_WG_DIM);
        if (y <= MAX_WG_DIM) return [x, y, 1];
        return [x, MAX_WG_DIM, Math.ceil(totalWG / (MAX_WG_DIM * MAX_WG_DIM))];
    }

    const totalPoints = batchSize * numPoints;
    const decompWG = Math.ceil(totalPoints / 256);
    const smvpWgSize = _smvpWgSize;
    const smvpPipeline = await getSMVPPipeline(p, smvpWgSize);
    const smvpTotalThreads = halfColumns * totalSubtasks;
    const smvpNumWG = Math.ceil(smvpTotalThreads / smvpWgSize);
    const [smvpX, smvpY, smvpZ] = splitDispatch(smvpNumWG);

    // Pass 1: Decompose scalars → column indices + histogram
    const decompPass = encoder.beginComputePass();
    decompPass.setPipeline(p.decomposePipeline);
    decompPass.setBindGroup(0, c.cscBG);
    decompPass.dispatchWorkgroups(decompWG, 1, 1);
    decompPass.end();

    // Pass 2: Prefix sum on histogram → colPtr
    const pfxPass = encoder.beginComputePass();
    pfxPass.setPipeline(p.prefixSumPipeline);
    pfxPass.setBindGroup(0, c.cscBG);
    pfxPass.dispatchWorkgroups(totalSubtasks, 1, 1);
    pfxPass.end();

    // Pass 3: Scatter CSC → valIdxs
    const scatterPass = encoder.beginComputePass();
    scatterPass.setPipeline(p.scatterPipeline);
    scatterPass.setBindGroup(0, c.cscBG);
    scatterPass.dispatchWorkgroups(decompWG, 1, 1);
    scatterPass.end();

    // Pass 4: SMVP bucket accumulation
    const smvpPass = encoder.beginComputePass();
    smvpPass.setPipeline(smvpPipeline);
    smvpPass.setBindGroup(0, c.smvpBG);
    smvpPass.dispatchWorkgroups(smvpX, smvpY, smvpZ);
    smvpPass.end();

    // Pass 5: PBPR stage 1 — running-sum bucket reduction
    const bpr1Pass = encoder.beginComputePass();
    bpr1Pass.setPipeline(bpr1Pipeline);
    bpr1Pass.setBindGroup(0, c.bprBG);
    bpr1Pass.dispatchWorkgroups(1, totalSubtasks, 1);
    bpr1Pass.end();

    // Pass 6: PBPR stage 2 — scalar-mul correction
    const bpr2Pass = encoder.beginComputePass();
    bpr2Pass.setPipeline(bpr2Pipeline);
    bpr2Pass.setBindGroup(0, c.bprBG);
    bpr2Pass.dispatchWorkgroups(1, totalSubtasks, 1);
    bpr2Pass.end();

    // Pass 7: Horner reduction
    const hornerWgSize = 64;
    const hornerNumWG = Math.ceil(batchSize / hornerWgSize);
    const hornerPass = encoder.beginComputePass();
    hornerPass.setPipeline(p.hornerPipeline);
    hornerPass.setBindGroup(0, c.hornerBG);
    hornerPass.dispatchWorkgroups(hornerNumWG, 1, 1);
    hornerPass.end();

    // Copy results to staging buffer
    encoder.copyBufferToBuffer(c.resultBuf, 0, c.stagingBuf, 0, c.resultLen * 4);

    device.queue.submit([encoder.finish()]);
    const t2 = performance.now();

    // Read back results
    await c.stagingBuf.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(c.stagingBuf.getMappedRange().slice(0));
    c.stagingBuf.unmap();
    const t3 = performance.now();

    // Buffers are cached — NOT destroyed

    const uploadMB = ((scalarsFlat.byteLength + (p._pointsCache ? 0 : pointsFlat.byteLength)) / 1048576).toFixed(1);
    console.log(`[gpu-msm] batch=${batchSize} pts=${numPoints} bits=${scalarBitWidth} w=${windowSize} ` +
        `upload=${uploadMB}MB bufs=${(t1-t0).toFixed(1)}ms submit=${(t2-t1).toFixed(1)}ms read=${(t3-t2).toFixed(1)}ms total=${(t3-t0).toFixed(1)}ms`);

    return resultData; // 24 u32s per MSM result (Jacobian x:8, y:8, z:8)
}
// ── Hybrid CPU+GPU batch MSM ─────────────────────────────────────────────
// Splits batch into GPU and CPU portions, running them in parallel.
// cpuMsmFn: (pointsFlat, scalarsFlat, numPoints, batchSize) => Uint32Array (Jacobian)
// gpuFn: the GPU MSM function to use (gpuBatchMSM)
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

export async function executeGPUBatchMSMHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn) {
    return gpuBatchMSMHybrid(pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize, cpuMsmFn, gpuBatchMSMWithGLV);
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

}
