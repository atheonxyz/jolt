import init, {
    initThreadPool,
    init_inlines,
    init_tracing,
    get_trace_json,
    clear_trace,
    cpu_batch_msm,
    WasmProver,
    WasmVerifier,
} from '../pkg/jolt_wasm_sha2_prover.js';
// GPU imports — these JS modules still load fine, they just won't be called without the feature
import { initGpuPairing, gpuBatchMultiPairing, gpuBatchMultiPairingFromBuffer, gpuCombineHints, gpuFlatMultiPairing, isGpuAvailable, getGpuDevice } from './gpu-pairing.js';
import { initGPUMSM, executeGPUBatchMSMHybrid } from './gpu-msm.js';
import { initGpuG2, gpuG2FixedBaseScalarMul, gpuG2UploadTable, gpuG2ScalarMulCached, isGpuG2Available } from './gpu-g2.js';
import { initGpuOnehot, gpuOnehotBatchG1Add, gpuOnehotGatherDirect, gpuOnehotGatherDirectRetainBuffer, isGpuOnehotAvailable } from './gpu-onehot.js';

let wasmExports = null;
const provers = {};
const verifiers = {};

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init': {
                wasmExports = await init();
                await initThreadPool(data.numThreads);
                init_tracing();
                init_inlines();

                // Initialize WebGPU pairing and MSM, register globals for WASM FFI
                let gpuReady = false;
                let msmReady = false;
                let g2Ready = false;  // disabled — CPU fallback
                let onehotReady = false;  // disabled — CPU fallback
                try {
                    gpuReady = await initGpuPairing();
                    if (gpuReady) {
                        const gpuDevice = getGpuDevice();
                        if (gpuDevice) {
                            await initGPUMSM(gpuDevice);
                            msmReady = true;
                            console.log('[worker] GPU MSM initialized successfully');

                            // Wire hybrid CPU+GPU MSM as the default.
                            // Splits ~85% GPU / ~15% CPU, running in parallel.
                            globalThis.__jolt_gpu_batch_msm = async (pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize) => {
                                return executeGPUBatchMSMHybrid(
                                    pointsFlat, scalarsFlat, numPoints, scalarBitWidth, batchSize,
                                    cpu_batch_msm,
                                );
                            };
                            console.log('[worker] Hybrid CPU+GPU MSM wired as default');

                            // GPU G2 — accelerates Stage 8 Dory opening proof (G2 scalar mul)
                            g2Ready = await initGpuG2(gpuDevice);
                            onehotReady = await initGpuOnehot(gpuDevice);
                        } else {
                            console.warn('[worker] GPU MSM/G2 init skipped: no gpuDevice');
                        }
                    }
                } catch (e) {
                    console.warn('[worker] WebGPU init failed:', e);
                }

                globalThis.__jolt_gpu_pairing_available = () => gpuReady;
                globalThis.__jolt_gpu_msm_available = () => msmReady;
                globalThis.__jolt_gpu_g2_available = () => g2Ready;
                globalThis.__jolt_gpu_onehot_available = () => onehotReady;

                // Register the batch pairing function callable from WASM
                // WASM calls this via wasm_bindgen extern import
                globalThis.__jolt_gpu_batch_pairing = async (g1Flat, g2Flat, groupSizes, groupOffsets) => {
                    return await gpuBatchMultiPairing(
                        new Uint32Array(g1Flat),
                        new Uint32Array(g2Flat),
                        new Uint32Array(groupSizes),
                        new Uint32Array(groupOffsets),
                    );
                };

                // Register the flat multi-pairing function callable from WASM
                globalThis.__jolt_gpu_multi_pairing = async (g1Flat, g2Flat, numPairs) => {
                    return await gpuFlatMultiPairing(
                        new Uint32Array(g1Flat),
                        new Uint32Array(g2Flat),
                        numPairs,
                    );
                };

                // Register G2 fixed-base scalar mul callable from WASM
                // Legacy: full table + scalar mul in one call
                globalThis.__jolt_gpu_g2_scalar_mul = async (tableLimbs, scalarLimbs, numScalars) => {
                    return await gpuG2FixedBaseScalarMul(
                        new Uint32Array(tableLimbs),
                        new Uint32Array(scalarLimbs),
                        numScalars,
                    );
                };

                // Cached path: upload table once, then scalar mul with cached table
                globalThis.__jolt_gpu_g2_upload_table = (tableLimbs) => {
                    gpuG2UploadTable(new Uint32Array(tableLimbs));
                };

                globalThis.__jolt_gpu_g2_scalar_mul_cached = async (scalarLimbs, numScalars) => {
                    return await gpuG2ScalarMulCached(
                        new Uint32Array(scalarLimbs),
                        numScalars,
                    );
                };

                // Register OneHot batch G1 addition callable from WASM
                globalThis.__jolt_gpu_onehot_batch_g1_add = async (basesFlat, packedIndices, numChunks, k, rowLen) => {
                    return await gpuOnehotBatchG1Add(
                        new Uint32Array(basesFlat),
                        new Uint32Array(packedIndices),
                        numChunks,
                        k,
                        rowLen,
                    );
                };

                // Direct gather dispatch: Rust sends pre-built gather lists (no JS preprocessing)
                globalThis.__jolt_gpu_onehot_gather_direct = async (basesFlat, gatherCols, jobs, numJobs) => {
                    return await gpuOnehotGatherDirect(
                        new Uint32Array(basesFlat),
                        new Uint32Array(gatherCols),
                        new Uint32Array(jobs),
                        numJobs,
                    );
                };

                globalThis.gpuOnehotGatherDirectRetainBufferFire = function(basesFlat, gatherCols, jobs, numJobs) {
                    return gpuOnehotGatherDirectRetainBuffer(
                        new Uint32Array(basesFlat),
                        new Uint32Array(gatherCols),
                        new Uint32Array(jobs),
                        numJobs,
                    );
                };

                globalThis.gpuBatchMultiPairingFromBufferFire = function(onehotBuffer, onehotBufferSize, polyLayoutFlat, totalAffinePoints, g2Flat, groupSizes, groupOffsets) {
                    return gpuBatchMultiPairingFromBuffer(
                        onehotBuffer,
                        onehotBufferSize,
                        new Uint32Array(polyLayoutFlat),
                        totalAffinePoints,
                        new Uint32Array(g2Flat),
                        new Uint32Array(groupSizes),
                        new Uint32Array(groupOffsets),
                    );
                };

                globalThis.gpuCombineHintsFire = function(pointsFlat, scalarsFlat, numRows, numPolys) {
                    return gpuCombineHints(
                        new Uint32Array(pointsFlat),
                        new Uint32Array(scalarsFlat),
                        numRows,
                        numPolys,
                    );
                };

                self.postMessage({ type: 'init-done', gpuAvailable: gpuReady });
                break;
            }

            case 'load-program': {
                const name = data.program;
                provers[name] = new WasmProver(
                    new Uint8Array(data.proverPreprocessing),
                    new Uint8Array(data.elfBytes)
                );
                verifiers[name] = new WasmVerifier(
                    new Uint8Array(data.verifierPreprocessing)
                );
                self.postMessage({ type: 'program-loaded', program: name });
                break;
            }

            case 'prove': {
                const prover = provers[data.program];
                const mode = data.mode || 'gpu';
                const start = performance.now();
                let result;

                if (mode === 'gpu') {
                    switch (data.program) {
                        case 'sha2':
                            result = await prover.prove_sha2_gpu(new Uint8Array(data.input));
                            break;
                        case 'ecdsa':
                            result = await prover.prove_ecdsa_gpu(
                                BigUint64Array.from(data.z.map(BigInt)),
                                BigUint64Array.from(data.r.map(BigInt)),
                                BigUint64Array.from(data.s.map(BigInt)),
                                BigUint64Array.from(data.q.map(BigInt)),
                            );
                            break;
                        case 'keccak':
                            result = await prover.prove_keccak_chain_gpu(
                                new Uint8Array(data.input),
                                data.numIters
                            );
                            break;
                    }
                } else {
                    switch (data.program) {
                        case 'sha2':
                            result = prover.prove_sha2(new Uint8Array(data.input));
                            break;
                        case 'ecdsa':
                            result = prover.prove_ecdsa(
                                BigUint64Array.from(data.z.map(BigInt)),
                                BigUint64Array.from(data.r.map(BigInt)),
                                BigUint64Array.from(data.s.map(BigInt)),
                                BigUint64Array.from(data.q.map(BigInt)),
                            );
                            break;
                        case 'keccak':
                            result = prover.prove_keccak_chain(
                                new Uint8Array(data.input),
                                data.numIters
                            );
                            break;
                    }
                }

                const elapsed = performance.now() - start;
                const peakMemory = wasmExports.memory.buffer.byteLength;

                // Extract proof fields before freeing WASM result
                const proof = result.proof;
                const proofSize = result.proof_size;
                const compressedProofSize = result.compressed_proof_size;
                const programIo = result.program_io;
                const numCycles = result.num_cycles;
                if (result.free) result.free(); // free WASM heap early

                // Export trace NOW while memory is available (before GC pressure)
                let traceJson = null;
                try { traceJson = get_trace_json(); } catch (e) {
                    console.warn('[worker] trace export failed:', e.message);
                }

                self.postMessage({
                    type: 'prove-done',
                    program: data.program,
                    mode,
                    proof,
                    proofSize,
                    compressedProofSize,
                    programIo,
                    numCycles,
                    peakMemory,
                    elapsed,
                    trace: traceJson,
                });
                break;
            }

            case 'verify': {
                const verifier = verifiers[data.program];
                const start = performance.now();
                const valid = verifier.verify(data.proof, data.programIo);
                const elapsed = performance.now() - start;

                self.postMessage({
                    type: 'verify-done',
                    program: data.program,
                    valid,
                    elapsed,
                });
                break;
            }

            case 'get-trace': {
                const traceJson = get_trace_json();
                self.postMessage({
                    type: 'trace',
                    trace: traceJson,
                });
                break;
            }

            case 'clear-trace': {
                clear_trace();
                self.postMessage({ type: 'trace-cleared' });
                break;
            }
        }
    } catch (err) {
        self.postMessage({ type: 'error', error: err.message || String(err) });
    }
};
