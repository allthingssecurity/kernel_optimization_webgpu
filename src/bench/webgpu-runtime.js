import q4MatvecShader from "../kernels/q4_matvec.wgsl?raw";
import fp32MatvecShader from "../kernels/fp32_matvec.wgsl?raw";
import q8MatvecScalarShader from "../kernels/q8_matvec_scalar.wgsl?raw";
import q8MatvecTiledShader from "../kernels/q8_matvec_tiled.wgsl?raw";
import matmulNBitsQ4Shader from "../kernels/matmul_nbits_q4.wgsl?raw";
import rmsNormShader from "../kernels/rms_norm.wgsl?raw";
import headRmsNormShader from "../kernels/head_rms_norm.wgsl?raw";
import ropeShader from "../kernels/rope.wgsl?raw";
import siluMulShader from "../kernels/silu_mul.wgsl?raw";
import gqaDecodeShader from "../kernels/gqa_decode.wgsl?raw";
import residualAddShader from "../kernels/residual_add.wgsl?raw";
import cacheWriteShader from "../kernels/cache_write.wgsl?raw";
import lmHeadF16Shader from "../kernels/lm_head_f16.wgsl?raw";
import argmaxStage1Shader from "../kernels/argmax_stage1.wgsl?raw";
import argmaxStage2Shader from "../kernels/argmax_stage2.wgsl?raw";
// Benchmark-only: the original kernels, so ab.html can measure against them.
import baselineMatmulNBitsQ4Shader from "../kernels/baseline/matmul_nbits_q4.wgsl?raw";
import baselineLmHeadChunkShader from "../kernels/baseline/lm_head_chunk.wgsl?raw";

const GPUBufferUsageFlags =
  globalThis.GPUBufferUsage ?? {
    MAP_READ: 1,
    COPY_SRC: 4,
    COPY_DST: 8,
    INDEX: 16,
    VERTEX: 32,
    UNIFORM: 64,
    STORAGE: 128,
  };

const GPUMapModeFlags = globalThis.GPUMapMode ?? {
  READ: 1,
  WRITE: 2,
};

// 32768 rows x 1024 cols x 2 bytes = 64 MiB per chunk, under the 128 MiB
// default maxStorageBufferBindingSize. Five chunks cover the 151936 vocab.
const LM_HEAD_CHUNK_ROWS = 32768;
const ARGMAX_GROUPS = 256;
// Bound how many prefill tokens queue up before we let the GPU drain.
const PREFILL_SYNC_INTERVAL = 16;

export async function createWebGpuContext() {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("No WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [],
  });

  return { adapter, device };
}

export async function runKernelBenchmarks(device, profile) {
  const q8 = await runQ8Comparison(device, profile.hiddenSize, profile.intermediateSize);
  const block = await runQwenMlpBlockComparison(device, profile);
  const realQ4 = await runRealQwenQ4Projection(device);
  const gqa = await runGqaDecodeComparison(device, profile);
  const layer0 = await runRealQwenLayer0Decode(device, profile);
  const rms = await runRmsNorm(device, profile.hiddenSize, profile.normEpsilon);
  return {
    baselineMatvecMs: q8.scalarMs,
    optimizedMatvecMs: q8.tiledMs,
    matvecSpeedup: q8.scalarMs / q8.tiledMs,
    baselineBytes: q8.bytesRead,
    optimizedBytes: q8.bytesRead,
    memoryReduction: 1,
    maxAbsError: q8.maxAbsError,
    meanAbsError: q8.meanAbsError,
    blockDefaultMs: block.defaultMs,
    blockOptimizedMs: block.optimizedMs,
    blockSpeedup: block.defaultMs / block.optimizedMs,
    blockMaxAbsError: block.maxAbsError,
    blockMeanAbsError: block.meanAbsError,
    realQ4Ms: realQ4.ms,
    realQ4MaxAbsError: realQ4.maxAbsError,
    realQ4MeanAbsError: realQ4.meanAbsError,
    gqaMs: gqa.ms,
    gqaMaxAbsError: gqa.maxAbsError,
    gqaMeanAbsError: gqa.meanAbsError,
    gqaSeqLen: gqa.seqLen,
    layer0DecodeMs: layer0.ms,
    layer0Finite: layer0.finite,
    layer0MeanAbs: layer0.meanAbs,
    rmsMs: rms.ms,
    estimatedGbps: q8.estimatedGbps,
  };
}

export async function runFullDecoderStackBenchmark(device, profile) {
  const base = "/qwen/runtime";
  const manifest = await fetch(`${base}/manifest.json`).then((response) => response.json());
  const hiddenSize = profile.hiddenSize;
  const pipelines = {
    rms: createPipeline(device, rmsNormShader),
    headRms: createPipeline(device, headRmsNormShader),
    q4: createPipeline(device, matmulNBitsQ4Shader),
    rope: createPipeline(device, ropeShader),
    gqa: createPipeline(device, gqaDecodeShader),
    silu: createPipeline(device, siluMulShader),
    add: createPipeline(device, residualAddShader),
  };
  const epsBits = new Uint32Array(new Float32Array([profile.normEpsilon]).buffer)[0];
  const [cosCache, sinCache, finalNormWeight] = await Promise.all([
    fetchF16AsF32(`${base}/${manifest.shared.cosCache.file}`),
    fetchF16AsF32(`${base}/${manifest.shared.sinCache.file}`),
    fetchF16AsF32(`${base}/${manifest.finalNorm.file}`),
  ]);
  const shared = {
    cosBuffer: storageBuffer(device, cosCache),
    sinBuffer: storageBuffer(device, sinCache),
    epsBits,
  };

  const hiddenA = storageBuffer(device, createVector(hiddenSize, 0.013, Math.sin));
  const hiddenB = emptyStorageBuffer(device, hiddenSize * 4);
  const finalNormBuffer = storageBuffer(device, finalNormWeight);
  const finalOutBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const layers = [];

  for (let i = 0; i < manifest.layers.length; i++) {
    const inputBuffer = i % 2 === 0 ? hiddenA : hiddenB;
    const outputBuffer = i % 2 === 0 ? hiddenB : hiddenA;
    layers.push(await createQwenDecodeLayerRuntime(device, profile, base, manifest.layers[i], pipelines, shared, inputBuffer, outputBuffer, i));
  }

  const stackOutBuffer = manifest.layers.length % 2 === 0 ? hiddenA : hiddenB;
  const finalNorm = createRmsBindGroup(device, pipelines.rms, stackOutBuffer, finalNormBuffer, finalOutBuffer, hiddenSize, epsBits);

  const warmupStarted = performance.now();
  dispatchFullDecoderStack(device, { layers, pipelines, finalNorm, finalOutBuffer });
  await device.queue.onSubmittedWorkDone();
  const warmupMs = performance.now() - warmupStarted;

  const started = performance.now();
  dispatchFullDecoderStack(device, { layers, pipelines, finalNorm, finalOutBuffer });
  await device.queue.onSubmittedWorkDone();
  const ms = performance.now() - started;

  const output = await readFloat32Buffer(device, finalOutBuffer, hiddenSize);
  let finite = 0;
  let sumAbs = 0;
  for (const value of output) {
    if (Number.isFinite(value)) finite++;
    sumAbs += Math.abs(value);
  }

  return {
    ms,
    warmupMs,
    layers: manifest.layers.length,
    finite,
    meanAbs: sumAbs / output.length,
  };
}

export async function runCustomQwenGreedyPrompt({
  device,
  profile,
  inputIds,
  maxNewTokens,
  onToken,
  decodeToken,
  onProgress,
  variant = "optimized",
}) {
  if (inputIds.length + maxNewTokens > 256) {
    throw new Error("Custom WebGPU path currently supports prompt + generation length up to 256 tokens.");
  }

  const runtime = await createAutoregressiveQwenRuntime(device, profile, inputIds.length + maxNewTokens, variant);
  const generated = [];
  const started = performance.now();
  let firstTokenMs = null;

  // Prefill has no CPU-side dependency between tokens, so queue them all and
  // drain periodically instead of syncing after every token.
  for (let i = 0; i < inputIds.length; i++) {
    onProgress?.(`Custom prefill ${i + 1}/${inputIds.length}`);
    enqueueAutoregressiveToken(device, runtime, inputIds[i], i);
    if ((i + 1) % PREFILL_SYNC_INTERVAL === 0) {
      await device.queue.onSubmittedWorkDone();
    }
  }
  await device.queue.onSubmittedWorkDone();

  for (let step = 0; step < maxNewTokens; step++) {
    onProgress?.(`Custom decode ${step + 1}/${maxNewTokens}`);
    if (step > 0) {
      // greedyNextToken's readback is the sync point for this token.
      enqueueAutoregressiveToken(device, runtime, generated[generated.length - 1], inputIds.length + step - 1);
    }
    const nextToken = await greedyNextToken(device, runtime);
    if (firstTokenMs === null) firstTokenMs = performance.now() - started;
    generated.push(nextToken);
    onToken?.(decodeToken ? decodeToken([nextToken]) : String(nextToken));
  }

  return {
    generated,
    generatedTokens: generated.length,
    generateMs: performance.now() - started,
    firstTokenMs,
    outputText: decodeToken ? decodeToken(generated) : generated.join(" "),
  };
}

// Every variant runs entirely on the GPU. The embedding table is always resident;
// the variants differ only in which kernels consume it.
//
// "original"  - original q4 matvec; original LM head (f32 weights, 75 chunk
//               submit/readback round-trips, argmax scanned on the CPU)
// "mixed"     - optimized q4 matvec, original LM head (isolates the matvec's share)
// "optimized" - optimized q4 matvec; resident f16 LM head with two-stage GPU argmax
async function createAutoregressiveQwenRuntime(device, profile, maxSeqLen, variant = "optimized") {
  const useBaselineQ4 = variant === "original";
  const lmHeadMode = variant === "optimized" ? "resident" : "gpu-chunk";
  const isBaseline = lmHeadMode !== "resident";
  const base = "/qwen/runtime";
  const manifest = await fetch(`${base}/manifest.json`).then((response) => response.json());
  const hiddenSize = profile.hiddenSize;
  const kvSize = profile.kvHeads * profile.headDim;
  const epsBits = new Uint32Array(new Float32Array([profile.normEpsilon]).buffer)[0];
  const [embedF16, cosCache, sinCache, finalNormWeight] = await Promise.all([
    fetch(`${base}/${manifest.shared.embedTokens.file}`).then((response) => response.arrayBuffer()),
    fetchF16AsF32(`${base}/${manifest.shared.cosCache.file}`),
    fetchF16AsF32(`${base}/${manifest.shared.sinCache.file}`),
    fetchF16AsF32(`${base}/${manifest.finalNorm.file}`),
  ]);

  const pipelines = {
    rms: createPipeline(device, rmsNormShader),
    headRms: createPipeline(device, headRmsNormShader),
    rope: createPipeline(device, ropeShader),
    gqa: createPipeline(device, gqaDecodeShader),
    silu: createPipeline(device, siluMulShader),
    add: createPipeline(device, residualAddShader),
    cacheWrite: createPipeline(device, cacheWriteShader),
    q4: createPipeline(device, useBaselineQ4 ? baselineMatmulNBitsQ4Shader : matmulNBitsQ4Shader),
    ...(isBaseline
      ? { lmHeadChunk: createPipeline(device, baselineLmHeadChunkShader) }
      : {
          lmHead: createPipeline(device, lmHeadF16Shader),
          argmax1: createPipeline(device, argmaxStage1Shader),
          argmax2: createPipeline(device, argmaxStage2Shader),
        }),
  };

  // Every layer sees the same head counts and the same position, so one set of
  // uniform buffers serves all 28 layers. Per token that is 4 writeBuffer calls
  // instead of 5 per layer.
  const shared = {
    cosBuffer: storageBuffer(device, cosCache),
    sinBuffer: storageBuffer(device, sinCache),
    epsBits,
    qRopeDimsBuffer: uniformBuffer(device, new Uint32Array([profile.attentionHeads, profile.headDim, 0, 0])),
    kRopeDimsBuffer: uniformBuffer(device, new Uint32Array([profile.kvHeads, profile.headDim, 0, 0])),
    cacheDimsBuffer: uniformBuffer(device, new Uint32Array([kvSize, 0, 0, 0])),
    gqaDimsBuffer: uniformBuffer(device, new Uint32Array([1, profile.attentionHeads, profile.kvHeads, profile.headDim])),
  };

  const hiddenA = emptyStorageBuffer(device, hiddenSize * 4);
  const hiddenB = emptyStorageBuffer(device, hiddenSize * 4);
  const finalOutBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const finalNormBuffer = storageBuffer(device, finalNormWeight);
  const layers = [];
  for (let i = 0; i < manifest.layers.length; i++) {
    const inputBuffer = i % 2 === 0 ? hiddenA : hiddenB;
    const outputBuffer = i % 2 === 0 ? hiddenB : hiddenA;
    layers.push(await createAutoregressiveLayerRuntime(
      device,
      profile,
      base,
      manifest.layers[i],
      pipelines,
      shared,
      inputBuffer,
      outputBuffer,
      maxSeqLen,
    ));
  }
  const stackOutBuffer = manifest.layers.length % 2 === 0 ? hiddenA : hiddenB;
  const finalNorm = createRmsBindGroup(device, pipelines.rms, stackOutBuffer, finalNormBuffer, finalOutBuffer, hiddenSize, epsBits);

  const embedDims = manifest.shared.embedTokens.dims;

  return {
    base,
    variant,
    profile,
    manifest,
    pipelines,
    layers,
    shared,
    hiddenA,
    finalOutBuffer,
    finalNorm,
    embedF16: new Uint16Array(embedF16),
    embedDims,
    lmHead: isBaseline ? null : createLmHeadRuntime(device, pipelines, finalOutBuffer, embedF16, embedDims),
    lmHeadMode,
    baselineGpuChunks: lmHeadMode === "gpu-chunk"
      ? createBaselineGpuLmHead(device, pipelines, finalOutBuffer, new Uint16Array(embedF16), embedDims, 2048)
      : null,
  };
}

// Original lm_head_chunk shader, but every f32 chunk is uploaded once at setup
// instead of being rebuilt on the CPU each token. Chunk size and the per-chunk
// submit/readback are unchanged, so what remains is the kernel cost.
function createBaselineGpuLmHead(device, pipelines, hiddenBuffer, embedU16, embedDims, chunkRows) {
  const [vocabSize, hiddenSize] = embedDims;
  const logitsBuffer = emptyStorageBuffer(device, chunkRows * 4);
  const chunks = [];

  for (let rowOffset = 0; rowOffset < vocabSize; rowOffset += chunkRows) {
    const rows = Math.min(chunkRows, vocabSize - rowOffset);
    const chunk = embeddingChunkF32(embedU16, embedDims, rowOffset, rows);
    chunks.push({
      rows,
      rowOffset,
      bindGroup: device.createBindGroup({
        layout: pipelines.lmHeadChunk.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: hiddenBuffer } },
          { binding: 1, resource: { buffer: storageBuffer(device, chunk) } },
          { binding: 2, resource: { buffer: logitsBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([hiddenSize, rows, rowOffset, 0])) } },
        ],
      }),
    });
  }
  return { chunks, logitsBuffer };
}

// Same per-chunk dispatch/readback/CPU-scan as the original, minus the conversion.
async function greedyNextTokenBaselineGpu(device, runtime) {
  const { chunks, logitsBuffer } = runtime.baselineGpuChunks;
  let bestToken = 0;
  let bestLogit = -Infinity;
  for (const chunk of chunks) {
    dispatchOnce(device, runtime.pipelines.lmHeadChunk, chunk.bindGroup, chunk.rows);
    await device.queue.onSubmittedWorkDone();
    const logits = await readFloat32Buffer(device, logitsBuffer, chunk.rows);
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > bestLogit) {
        bestLogit = logits[i];
        bestToken = chunk.rowOffset + i;
      }
    }
  }
  return bestToken;
}

// The tied embedding table is both the input gather and the output projection.
// Upload it once as f16 and let the shader unpack, rather than rebuilding an
// f32 copy on the CPU for every generated token.
function createLmHeadRuntime(device, pipelines, hiddenBuffer, embedBytes, embedDims) {
  const [vocabSize, hiddenSize] = embedDims;
  const embedU8 = new Uint8Array(embedBytes);
  const bytesPerRow = hiddenSize * 2;

  const chunks = [];
  const logitsBuffer = emptyStorageBuffer(device, vocabSize * 4);

  for (let rowOffset = 0; rowOffset < vocabSize; rowOffset += LM_HEAD_CHUNK_ROWS) {
    const rows = Math.min(LM_HEAD_CHUNK_ROWS, vocabSize - rowOffset);
    const weightBuffer = device.createBuffer({
      size: align4(rows * bytesPerRow),
      usage: GPUBufferUsageFlags.STORAGE | GPUBufferUsageFlags.COPY_DST,
    });
    device.queue.writeBuffer(weightBuffer, 0, embedU8, rowOffset * bytesPerRow, rows * bytesPerRow);

    chunks.push({
      rows,
      bindGroup: device.createBindGroup({
        layout: pipelines.lmHead.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: hiddenBuffer } },
          { binding: 1, resource: { buffer: weightBuffer } },
          { binding: 2, resource: { buffer: logitsBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([hiddenSize, rows, rowOffset, 0])) } },
        ],
      }),
    });
  }

  const bestValueBuffer = emptyStorageBuffer(device, ARGMAX_GROUPS * 4);
  const bestIndexBuffer = emptyStorageBuffer(device, ARGMAX_GROUPS * 4);
  const resultBuffer = emptyStorageBuffer(device, 2 * 4);

  return {
    chunks,
    logitsBuffer,
    resultBuffer,
    stage1: device.createBindGroup({
      layout: pipelines.argmax1.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: logitsBuffer } },
        { binding: 1, resource: { buffer: bestValueBuffer } },
        { binding: 2, resource: { buffer: bestIndexBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([vocabSize, ARGMAX_GROUPS, 0, 0])) } },
      ],
    }),
    stage2: device.createBindGroup({
      layout: pipelines.argmax2.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bestValueBuffer } },
        { binding: 1, resource: { buffer: bestIndexBuffer } },
        { binding: 2, resource: { buffer: resultBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([ARGMAX_GROUPS, 0, 0, 0])) } },
      ],
    }),
  };
}

async function createAutoregressiveLayerRuntime(
  device,
  profile,
  base,
  layer,
  pipelines,
  shared,
  inputBuffer,
  outputBuffer,
  maxSeqLen,
) {
  const runtime = await createQwenDecodeLayerRuntime(device, profile, base, layer, pipelines, shared, inputBuffer, outputBuffer, 0);
  const kvSize = profile.kvHeads * profile.headDim;
  runtime.kCacheBuffer = emptyStorageBuffer(device, maxSeqLen * kvSize * 4);
  runtime.vCacheBuffer = emptyStorageBuffer(device, maxSeqLen * kvSize * 4);
  runtime.kCacheWrite = device.createBindGroup({
    layout: pipelines.cacheWrite.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: runtime.kRopeOutputBuffer } },
      { binding: 1, resource: { buffer: runtime.kCacheBuffer } },
      { binding: 2, resource: { buffer: shared.cacheDimsBuffer } },
    ],
  });
  runtime.vCacheWrite = device.createBindGroup({
    layout: pipelines.cacheWrite.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: runtime.vProjOutputBuffer } },
      { binding: 1, resource: { buffer: runtime.vCacheBuffer } },
      { binding: 2, resource: { buffer: shared.cacheDimsBuffer } },
    ],
  });
  runtime.gqa = createGqaBindGroupWithDims(
    device,
    pipelines.gqa,
    runtime.qRopeOutputBuffer,
    runtime.kCacheBuffer,
    runtime.vCacheBuffer,
    runtime.attnOutBuffer,
    shared.gqaDimsBuffer,
  );
  return runtime;
}

// Enqueues one token's forward pass. queue.writeBuffer is ordered against
// submits, so consecutive tokens can be queued without a GPU sync between them
// even though they share the hidden-state and uniform buffers.
function enqueueAutoregressiveToken(device, runtime, tokenId, position) {
  const { profile, shared } = runtime;
  const kvSize = profile.kvHeads * profile.headDim;

  const hidden = embeddingRowF32(runtime.embedF16, runtime.embedDims, tokenId);
  device.queue.writeBuffer(runtime.hiddenA, 0, hidden);
  device.queue.writeBuffer(shared.qRopeDimsBuffer, 0, new Uint32Array([profile.attentionHeads, profile.headDim, position, 0]));
  device.queue.writeBuffer(shared.kRopeDimsBuffer, 0, new Uint32Array([profile.kvHeads, profile.headDim, position, 0]));
  device.queue.writeBuffer(shared.cacheDimsBuffer, 0, new Uint32Array([kvSize, position, 0, 0]));
  device.queue.writeBuffer(shared.gqaDimsBuffer, 0, new Uint32Array([position + 1, profile.attentionHeads, profile.kvHeads, profile.headDim]));

  dispatchAutoregressiveToken(device, runtime);
}

function dispatchAutoregressiveToken(device, runtime) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const layer of runtime.layers) {
    dispatchAutoregressiveLayerPass(pass, runtime.pipelines, layer);
  }
  pass.setPipeline(runtime.pipelines.rms);
  pass.setBindGroup(0, runtime.finalNorm);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function dispatchAutoregressiveLayerPass(pass, pipelines, runtime) {
  pass.setPipeline(pipelines.rms);
  pass.setBindGroup(0, runtime.inputRms);
  pass.dispatchWorkgroups(1);

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.qProj);
  pass.dispatchWorkgroups(runtime.qSize);
  pass.setBindGroup(0, runtime.kProj);
  pass.dispatchWorkgroups(runtime.kvSize);
  pass.setBindGroup(0, runtime.vProj);
  pass.dispatchWorkgroups(runtime.kvSize);

  pass.setPipeline(pipelines.headRms);
  pass.setBindGroup(0, runtime.qHeadNorm);
  pass.dispatchWorkgroups(runtime.qHeads);
  pass.setBindGroup(0, runtime.kHeadNorm);
  pass.dispatchWorkgroups(runtime.kvHeads);

  pass.setPipeline(pipelines.rope);
  pass.setBindGroup(0, runtime.qRope);
  pass.dispatchWorkgroups(Math.ceil(runtime.qSize / 256));
  pass.setBindGroup(0, runtime.kRope);
  pass.dispatchWorkgroups(Math.ceil(runtime.kvSize / 256));

  pass.setPipeline(pipelines.cacheWrite);
  pass.setBindGroup(0, runtime.kCacheWrite);
  pass.dispatchWorkgroups(Math.ceil(runtime.kvSize / 256));
  pass.setBindGroup(0, runtime.vCacheWrite);
  pass.dispatchWorkgroups(Math.ceil(runtime.kvSize / 256));

  pass.setPipeline(pipelines.gqa);
  pass.setBindGroup(0, runtime.gqa);
  pass.dispatchWorkgroups(runtime.qHeads);

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.oProj);
  pass.dispatchWorkgroups(runtime.hiddenSize);

  pass.setPipeline(pipelines.add);
  pass.setBindGroup(0, runtime.attnResidual);
  pass.dispatchWorkgroups(Math.ceil(runtime.hiddenSize / 256));

  pass.setPipeline(pipelines.rms);
  pass.setBindGroup(0, runtime.postRms);
  pass.dispatchWorkgroups(1);

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.gate);
  pass.dispatchWorkgroups(runtime.intermediateSize);
  pass.setBindGroup(0, runtime.up);
  pass.dispatchWorkgroups(runtime.intermediateSize);

  pass.setPipeline(pipelines.silu);
  pass.setBindGroup(0, runtime.silu);
  pass.dispatchWorkgroups(Math.ceil(runtime.intermediateSize / 256));

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.down);
  pass.dispatchWorkgroups(runtime.hiddenSize);

  pass.setPipeline(pipelines.add);
  pass.setBindGroup(0, runtime.finalResidual);
  pass.dispatchWorkgroups(Math.ceil(runtime.hiddenSize / 256));
}

async function greedyNextToken(device, runtime) {
  return runtime.lmHeadMode === "gpu-chunk"
    ? greedyNextTokenBaselineGpu(device, runtime)
    : greedyNextTokenResident(device, runtime);
}

// Used once at setup to build the f32 chunks the original LM head shader expects.
function embeddingChunkF32(embedF16, dims, rowOffset, rows) {
  const hiddenSize = dims[1];
  const out = new Float32Array(rows * hiddenSize);
  const start = rowOffset * hiddenSize;
  for (let i = 0; i < out.length; i++) out[i] = halfToFloat(embedF16[start + i]);
  return out;
}

// Whole vocab projection plus argmax in one submit, reading back 8 bytes.
async function greedyNextTokenResident(device, runtime) {
  const { lmHead, pipelines } = runtime;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();

  pass.setPipeline(pipelines.lmHead);
  for (const chunk of lmHead.chunks) {
    pass.setBindGroup(0, chunk.bindGroup);
    pass.dispatchWorkgroups(chunk.rows);
  }

  pass.setPipeline(pipelines.argmax1);
  pass.setBindGroup(0, lmHead.stage1);
  pass.dispatchWorkgroups(ARGMAX_GROUPS);

  pass.setPipeline(pipelines.argmax2);
  pass.setBindGroup(0, lmHead.stage2);
  pass.dispatchWorkgroups(1);

  pass.end();
  device.queue.submit([encoder.finish()]);

  const result = await readUint32Buffer(device, lmHead.resultBuffer, 2);
  return result[0];
}

function embeddingRowF32(embedF16, dims, tokenId) {
  const hiddenSize = dims[1];
  const row = new Float32Array(hiddenSize);
  const offset = tokenId * hiddenSize;
  for (let i = 0; i < hiddenSize; i++) row[i] = halfToFloat(embedF16[offset + i]);
  return row;
}

async function runRealQwenLayer0Decode(device, profile) {
  const base = "/qwen/runtime";
  const manifest = await fetch(`${base}/manifest.json`).then((response) => response.json());
  const layer = manifest.layers[0];
  const hiddenSize = profile.hiddenSize;
  const intermediateSize = profile.intermediateSize;
  const qHeads = profile.attentionHeads;
  const kvHeads = profile.kvHeads;
  const headDim = profile.headDim;
  const qSize = qHeads * headDim;
  const kvSize = kvHeads * headDim;
  const epsBits = new Uint32Array(new Float32Array([profile.normEpsilon]).buffer)[0];

  const [
    inputNormWeight,
    postNormWeight,
    qNormWeight,
    kNormWeight,
    cosCache,
    sinCache,
  ] = await Promise.all([
    fetchF16AsF32(`${base}/${layer.norms.input.file}`),
    fetchF16AsF32(`${base}/${layer.norms.postAttention.file}`),
    fetchF16AsF32(`${base}/${layer.norms.q.file}`),
    fetchF16AsF32(`${base}/${layer.norms.k.file}`),
    fetchF16AsF32(`${base}/${manifest.shared.cosCache.file}`),
    fetchF16AsF32(`${base}/${manifest.shared.sinCache.file}`),
  ]);

  const hidden = createVector(hiddenSize, 0.013, Math.sin);
  const hiddenBuffer = storageBuffer(device, hidden);
  const inputNormBuffer = storageBuffer(device, inputNormWeight);
  const postNormBuffer = storageBuffer(device, postNormWeight);
  const qNormWeightBuffer = storageBuffer(device, qNormWeight);
  const kNormWeightBuffer = storageBuffer(device, kNormWeight);
  const cosBuffer = storageBuffer(device, cosCache);
  const sinBuffer = storageBuffer(device, sinCache);

  const normOutBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const qProjBuffer = emptyStorageBuffer(device, qSize * 4);
  const kProjBuffer = emptyStorageBuffer(device, kvSize * 4);
  const vProjBuffer = emptyStorageBuffer(device, kvSize * 4);
  const qNormBuffer = emptyStorageBuffer(device, qSize * 4);
  const kNormBuffer = emptyStorageBuffer(device, kvSize * 4);
  const qRopeBuffer = emptyStorageBuffer(device, qSize * 4);
  const kRopeBuffer = emptyStorageBuffer(device, kvSize * 4);
  const attnOutBuffer = emptyStorageBuffer(device, qSize * 4);
  const oProjBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const attnResidualBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const postNormOutBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const gateBuffer = emptyStorageBuffer(device, intermediateSize * 4);
  const upBuffer = emptyStorageBuffer(device, intermediateSize * 4);
  const actBuffer = emptyStorageBuffer(device, intermediateSize * 4);
  const downBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const outputBuffer = emptyStorageBuffer(device, hiddenSize * 4);

  const rmsPipeline = createPipeline(device, rmsNormShader);
  const headRmsPipeline = createPipeline(device, headRmsNormShader);
  const q4Pipeline = createPipeline(device, matmulNBitsQ4Shader);
  const ropePipeline = createPipeline(device, ropeShader);
  const gqaPipeline = createPipeline(device, gqaDecodeShader);
  const siluPipeline = createPipeline(device, siluMulShader);
  const addPipeline = createPipeline(device, residualAddShader);

  const inputRms = createRmsBindGroup(device, rmsPipeline, hiddenBuffer, inputNormBuffer, normOutBuffer, hiddenSize, epsBits);
  const qProj = await createQ4RuntimeProjection(device, q4Pipeline, normOutBuffer, qProjBuffer, base, layer.projections.q);
  const kProj = await createQ4RuntimeProjection(device, q4Pipeline, normOutBuffer, kProjBuffer, base, layer.projections.k);
  const vProj = await createQ4RuntimeProjection(device, q4Pipeline, normOutBuffer, vProjBuffer, base, layer.projections.v);
  const qHeadNorm = createHeadRmsBindGroup(device, headRmsPipeline, qProjBuffer, qNormWeightBuffer, qNormBuffer, headDim, qHeads, epsBits);
  const kHeadNorm = createHeadRmsBindGroup(device, headRmsPipeline, kProjBuffer, kNormWeightBuffer, kNormBuffer, headDim, kvHeads, epsBits);
  const qRope = createRopeBindGroup(device, ropePipeline, qNormBuffer, cosBuffer, sinBuffer, qRopeBuffer, qHeads, headDim, 0);
  const kRope = createRopeBindGroup(device, ropePipeline, kNormBuffer, cosBuffer, sinBuffer, kRopeBuffer, kvHeads, headDim, 0);
  const gqa = createGqaBindGroup(device, gqaPipeline, qRopeBuffer, kRopeBuffer, vProjBuffer, attnOutBuffer, 1, qHeads, kvHeads, headDim);
  const oProj = await createQ4RuntimeProjection(device, q4Pipeline, attnOutBuffer, oProjBuffer, base, layer.projections.o);
  const attnResidual = createAddBindGroup(device, addPipeline, hiddenBuffer, oProjBuffer, attnResidualBuffer, hiddenSize);
  const postRms = createRmsBindGroup(device, rmsPipeline, attnResidualBuffer, postNormBuffer, postNormOutBuffer, hiddenSize, epsBits);
  const gate = await createQ4RuntimeProjection(device, q4Pipeline, postNormOutBuffer, gateBuffer, base, layer.projections.gate);
  const up = await createQ4RuntimeProjection(device, q4Pipeline, postNormOutBuffer, upBuffer, base, layer.projections.up);
  const silu = createSiluBindGroup(device, siluPipeline, gateBuffer, upBuffer, actBuffer, intermediateSize);
  const down = await createQ4RuntimeProjection(device, q4Pipeline, actBuffer, downBuffer, base, layer.projections.down);
  const finalResidual = createAddBindGroup(device, addPipeline, attnResidualBuffer, downBuffer, outputBuffer, hiddenSize);

  const runtime = {
    hiddenSize,
    intermediateSize,
    qSize,
    kvSize,
    qHeads,
    kvHeads,
    rmsPipeline,
    headRmsPipeline,
    q4Pipeline,
    ropePipeline,
    gqaPipeline,
    siluPipeline,
    addPipeline,
    inputRms,
    qProj,
    kProj,
    vProj,
    qHeadNorm,
    kHeadNorm,
    qRope,
    kRope,
    gqa,
    oProj,
    attnResidual,
    postRms,
    gate,
    up,
    silu,
    down,
    finalResidual,
    outputBuffer,
  };

  const ms = await timeLayer0Decode(device, runtime, 8);
  dispatchLayer0Decode(device, runtime);
  await device.queue.onSubmittedWorkDone();
  const output = await readFloat32Buffer(device, outputBuffer, hiddenSize);
  let finite = 0;
  let sumAbs = 0;
  for (const value of output) {
    if (Number.isFinite(value)) finite++;
    sumAbs += Math.abs(value);
  }
  return { ms, finite, meanAbs: sumAbs / output.length };
}

async function createQwenDecodeLayerRuntime(
  device,
  profile,
  base,
  layer,
  pipelines,
  shared,
  inputBuffer,
  outputBuffer,
  position,
) {
  const hiddenSize = profile.hiddenSize;
  const intermediateSize = profile.intermediateSize;
  const qHeads = profile.attentionHeads;
  const kvHeads = profile.kvHeads;
  const headDim = profile.headDim;
  const qSize = qHeads * headDim;
  const kvSize = kvHeads * headDim;

  const [inputNormWeight, postNormWeight, qNormWeight, kNormWeight] = await Promise.all([
    fetchF16AsF32(`${base}/${layer.norms.input.file}`),
    fetchF16AsF32(`${base}/${layer.norms.postAttention.file}`),
    fetchF16AsF32(`${base}/${layer.norms.q.file}`),
    fetchF16AsF32(`${base}/${layer.norms.k.file}`),
  ]);

  const inputNormBuffer = storageBuffer(device, inputNormWeight);
  const postNormBuffer = storageBuffer(device, postNormWeight);
  const qNormWeightBuffer = storageBuffer(device, qNormWeight);
  const kNormWeightBuffer = storageBuffer(device, kNormWeight);
  const normOutBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const qProjBuffer = emptyStorageBuffer(device, qSize * 4);
  const kProjBuffer = emptyStorageBuffer(device, kvSize * 4);
  const vProjBuffer = emptyStorageBuffer(device, kvSize * 4);
  const qNormBuffer = emptyStorageBuffer(device, qSize * 4);
  const kNormBuffer = emptyStorageBuffer(device, kvSize * 4);
  const qRopeBuffer = emptyStorageBuffer(device, qSize * 4);
  const kRopeBuffer = emptyStorageBuffer(device, kvSize * 4);
  const attnOutBuffer = emptyStorageBuffer(device, qSize * 4);
  const oProjBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const attnResidualBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const postNormOutBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const gateBuffer = emptyStorageBuffer(device, intermediateSize * 4);
  const upBuffer = emptyStorageBuffer(device, intermediateSize * 4);
  const actBuffer = emptyStorageBuffer(device, intermediateSize * 4);
  const downBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  // The autoregressive runtime hands down one rope uniform per role and rewrites
  // the position once per token; standalone benchmarks get a per-layer copy.
  const qRopeDimsBuffer = shared.qRopeDimsBuffer
    ?? uniformBuffer(device, new Uint32Array([qHeads, headDim, position, 0]));
  const kRopeDimsBuffer = shared.kRopeDimsBuffer
    ?? uniformBuffer(device, new Uint32Array([kvHeads, headDim, position, 0]));

  return {
    hiddenSize,
    intermediateSize,
    qSize,
    kvSize,
    qHeads,
    kvHeads,
    inputRms: createRmsBindGroup(device, pipelines.rms, inputBuffer, inputNormBuffer, normOutBuffer, hiddenSize, shared.epsBits),
    qProj: await createQ4RuntimeProjection(device, pipelines.q4, normOutBuffer, qProjBuffer, base, layer.projections.q),
    kProj: await createQ4RuntimeProjection(device, pipelines.q4, normOutBuffer, kProjBuffer, base, layer.projections.k),
    vProj: await createQ4RuntimeProjection(device, pipelines.q4, normOutBuffer, vProjBuffer, base, layer.projections.v),
    qHeadNorm: createHeadRmsBindGroup(device, pipelines.headRms, qProjBuffer, qNormWeightBuffer, qNormBuffer, headDim, qHeads, shared.epsBits),
    kHeadNorm: createHeadRmsBindGroup(device, pipelines.headRms, kProjBuffer, kNormWeightBuffer, kNormBuffer, headDim, kvHeads, shared.epsBits),
    qRope: createRopeBindGroupWithDims(device, pipelines.rope, qNormBuffer, shared.cosBuffer, shared.sinBuffer, qRopeBuffer, qRopeDimsBuffer),
    kRope: createRopeBindGroupWithDims(device, pipelines.rope, kNormBuffer, shared.cosBuffer, shared.sinBuffer, kRopeBuffer, kRopeDimsBuffer),
    gqa: createGqaBindGroup(device, pipelines.gqa, qRopeBuffer, kRopeBuffer, vProjBuffer, attnOutBuffer, 1, qHeads, kvHeads, headDim),
    oProj: await createQ4RuntimeProjection(device, pipelines.q4, attnOutBuffer, oProjBuffer, base, layer.projections.o),
    attnResidual: createAddBindGroup(device, pipelines.add, inputBuffer, oProjBuffer, attnResidualBuffer, hiddenSize),
    postRms: createRmsBindGroup(device, pipelines.rms, attnResidualBuffer, postNormBuffer, postNormOutBuffer, hiddenSize, shared.epsBits),
    gate: await createQ4RuntimeProjection(device, pipelines.q4, postNormOutBuffer, gateBuffer, base, layer.projections.gate),
    up: await createQ4RuntimeProjection(device, pipelines.q4, postNormOutBuffer, upBuffer, base, layer.projections.up),
    silu: createSiluBindGroup(device, pipelines.silu, gateBuffer, upBuffer, actBuffer, intermediateSize),
    down: await createQ4RuntimeProjection(device, pipelines.q4, actBuffer, downBuffer, base, layer.projections.down),
    finalResidual: createAddBindGroup(device, pipelines.add, attnResidualBuffer, downBuffer, outputBuffer, hiddenSize),
    qRopeDimsBuffer,
    kRopeDimsBuffer,
    kRopeOutputBuffer: kRopeBuffer,
    vProjOutputBuffer: vProjBuffer,
    qRopeOutputBuffer: qRopeBuffer,
    attnOutBuffer,
  };
}

function dispatchFullDecoderStack(device, runtime) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const layer of runtime.layers) {
    dispatchQwenDecodeLayerPass(pass, runtime.pipelines, layer);
  }
  pass.setPipeline(runtime.pipelines.rms);
  pass.setBindGroup(0, runtime.finalNorm);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function dispatchQwenDecodeLayerPass(pass, pipelines, runtime) {
  pass.setPipeline(pipelines.rms);
  pass.setBindGroup(0, runtime.inputRms);
  pass.dispatchWorkgroups(1);

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.qProj);
  pass.dispatchWorkgroups(runtime.qSize);
  pass.setBindGroup(0, runtime.kProj);
  pass.dispatchWorkgroups(runtime.kvSize);
  pass.setBindGroup(0, runtime.vProj);
  pass.dispatchWorkgroups(runtime.kvSize);

  pass.setPipeline(pipelines.headRms);
  pass.setBindGroup(0, runtime.qHeadNorm);
  pass.dispatchWorkgroups(runtime.qHeads);
  pass.setBindGroup(0, runtime.kHeadNorm);
  pass.dispatchWorkgroups(runtime.kvHeads);

  pass.setPipeline(pipelines.rope);
  pass.setBindGroup(0, runtime.qRope);
  pass.dispatchWorkgroups(Math.ceil(runtime.qSize / 256));
  pass.setBindGroup(0, runtime.kRope);
  pass.dispatchWorkgroups(Math.ceil(runtime.kvSize / 256));

  pass.setPipeline(pipelines.gqa);
  pass.setBindGroup(0, runtime.gqa);
  pass.dispatchWorkgroups(runtime.qHeads);

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.oProj);
  pass.dispatchWorkgroups(runtime.hiddenSize);

  pass.setPipeline(pipelines.add);
  pass.setBindGroup(0, runtime.attnResidual);
  pass.dispatchWorkgroups(Math.ceil(runtime.hiddenSize / 256));

  pass.setPipeline(pipelines.rms);
  pass.setBindGroup(0, runtime.postRms);
  pass.dispatchWorkgroups(1);

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.gate);
  pass.dispatchWorkgroups(runtime.intermediateSize);
  pass.setBindGroup(0, runtime.up);
  pass.dispatchWorkgroups(runtime.intermediateSize);

  pass.setPipeline(pipelines.silu);
  pass.setBindGroup(0, runtime.silu);
  pass.dispatchWorkgroups(Math.ceil(runtime.intermediateSize / 256));

  pass.setPipeline(pipelines.q4);
  pass.setBindGroup(0, runtime.down);
  pass.dispatchWorkgroups(runtime.hiddenSize);

  pass.setPipeline(pipelines.add);
  pass.setBindGroup(0, runtime.finalResidual);
  pass.dispatchWorkgroups(Math.ceil(runtime.hiddenSize / 256));
}

async function runGqaDecodeComparison(device, profile) {
  const seqLen = 128;
  const qHeads = profile.attentionHeads;
  const kvHeads = profile.kvHeads;
  const headDim = profile.headDim;
  const q = createVector(qHeads * headDim, 0.011, Math.sin);
  const kCache = createVector(seqLen * kvHeads * headDim, 0.007, Math.cos);
  const vCache = createVector(seqLen * kvHeads * headDim, 0.013, Math.sin);

  const qBuffer = storageBuffer(device, q);
  const kBuffer = storageBuffer(device, kCache);
  const vBuffer = storageBuffer(device, vCache);
  const yBuffer = emptyStorageBuffer(device, qHeads * headDim * 4);
  const dims = uniformBuffer(device, new Uint32Array([seqLen, qHeads, kvHeads, headDim]));

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: gqaDecodeShader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: qBuffer } },
      { binding: 1, resource: { buffer: kBuffer } },
      { binding: 2, resource: { buffer: vBuffer } },
      { binding: 3, resource: { buffer: yBuffer } },
      { binding: 4, resource: { buffer: dims } },
    ],
  });

  const ms = await timeDispatch(device, pipeline, bindGroup, qHeads, 160);
  dispatchOnce(device, pipeline, bindGroup, qHeads);
  await device.queue.onSubmittedWorkDone();
  const gpu = await readFloat32Buffer(device, yBuffer, qHeads * headDim);
  const cpu = cpuGqaDecode({ q, kCache, vCache, seqLen, qHeads, kvHeads, headDim });
  return { ms, seqLen, ...errorStats(gpu, cpu) };
}

function cpuGqaDecode({ q, kCache, vCache, seqLen, qHeads, kvHeads, headDim }) {
  const y = new Float32Array(qHeads * headDim);
  const headsPerKv = qHeads / kvHeads;
  const scale = 1 / Math.sqrt(headDim);
  const scores = new Float32Array(seqLen);

  for (let qHead = 0; qHead < qHeads; qHead++) {
    const kvHead = Math.floor(qHead / headsPerKv);
    let maxScore = -Infinity;
    for (let pos = 0; pos < seqLen; pos++) {
      let dot = 0;
      const qOffset = qHead * headDim;
      const kOffset = (pos * kvHeads + kvHead) * headDim;
      for (let d = 0; d < headDim; d++) {
        dot += q[qOffset + d] * kCache[kOffset + d];
      }
      const score = dot * scale;
      scores[pos] = score;
      maxScore = Math.max(maxScore, score);
    }

    let sum = 0;
    for (let pos = 0; pos < seqLen; pos++) {
      const e = Math.exp(scores[pos] - maxScore);
      scores[pos] = e;
      sum += e;
    }

    for (let d = 0; d < headDim; d++) {
      let acc = 0;
      for (let pos = 0; pos < seqLen; pos++) {
        const vOffset = (pos * kvHeads + kvHead) * headDim;
        acc += (scores[pos] / sum) * vCache[vOffset + d];
      }
      y[qHead * headDim + d] = acc;
    }
  }

  return y;
}

async function runRealQwenQ4Projection(device) {
  const base = "/qwen/layer0-gate-proj";
  const [manifest, weightBytes, scalesBuffer] = await Promise.all([
    fetch(`${base}/manifest.json`).then((response) => response.json()),
    fetch(`${base}/weight_q4.bin`).then((response) => response.arrayBuffer()),
    fetch(`${base}/scales_f32.bin`).then((response) => response.arrayBuffer()),
  ]);
  const { inSize, outSize, blocksPerRow, blockSize } = manifest.matmulNBits;
  const x = createVector(inSize, 0.017, Math.sin);
  const scales = new Float32Array(scalesBuffer);
  const weights = new Uint32Array(weightBytes);

  const xBuffer = storageBuffer(device, x);
  const wBuffer = rawStorageBuffer(device, weightBytes);
  const sBuffer = storageBuffer(device, scales);
  const yBuffer = emptyStorageBuffer(device, outSize * 4);
  const dims = uniformBuffer(device, new Uint32Array([inSize, outSize, blocksPerRow, blockSize]));

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: matmulNBitsQ4Shader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: wBuffer } },
      { binding: 2, resource: { buffer: sBuffer } },
      { binding: 3, resource: { buffer: yBuffer } },
      { binding: 4, resource: { buffer: dims } },
    ],
  });

  const ms = await timeDispatch(device, pipeline, bindGroup, outSize, 80);
  dispatchOnce(device, pipeline, bindGroup, outSize);
  await device.queue.onSubmittedWorkDone();
  const gpu = await readFloat32Buffer(device, yBuffer, outSize);
  const cpu = cpuQ4MatMulNBits({ inSize, outSize, blocksPerRow, blockSize, x, weights, scales });
  return { ms, ...errorStats(gpu, cpu) };
}

function cpuQ4MatMulNBits({ inSize, outSize, blocksPerRow, blockSize, x, weights, scales }) {
  const wordsPerBlock = blockSize / 8;
  const y = new Float32Array(outSize);
  for (let row = 0; row < outSize; row++) {
    let acc = 0;
    const rowWordOffset = row * blocksPerRow * wordsPerBlock;
    const rowScaleOffset = row * blocksPerRow;
    for (let block = 0; block < blocksPerRow; block++) {
      const scale = scales[rowScaleOffset + block];
      const colBase = block * blockSize;
      const blockWordOffset = rowWordOffset + block * wordsPerBlock;
      for (let wordIndex = 0; wordIndex < wordsPerBlock; wordIndex++) {
        const packedWord = weights[blockWordOffset + wordIndex];
        for (let byteLane = 0; byteLane < 4; byteLane++) {
          const packedByte = (packedWord >>> (byteLane * 8)) & 0xff;
          const pairBase = colBase + wordIndex * 8 + byteLane * 2;
          if (pairBase < inSize) {
            acc += (((packedByte & 0x0f) - 8) * scale * x[pairBase]);
          }
          if (pairBase + 1 < inSize) {
            acc += ((((packedByte >>> 4) & 0x0f) - 8) * scale * x[pairBase + 1]);
          }
        }
      }
    }
    y[row] = acc;
  }
  return y;
}

async function runQwenMlpBlockComparison(device, profile) {
  const tensors = createMlpBlockInputs(profile.hiddenSize, profile.intermediateSize);
  const defaultRuntime = createMlpBlockRuntime(device, q8MatvecScalarShader, tensors);
  const optimizedRuntime = createMlpBlockRuntime(device, q8MatvecTiledShader, tensors);

  const defaultMs = await timeMlpBlock(device, defaultRuntime, "scalar", 16);
  const optimizedMs = await timeMlpBlock(device, optimizedRuntime, "tiled", 32);

  dispatchMlpBlock(device, optimizedRuntime, "tiled");
  await device.queue.onSubmittedWorkDone();
  const gpu = await readFloat32Buffer(device, optimizedRuntime.downOutBuffer, tensors.hiddenSize);
  const cpu = cpuMlpBlock(tensors, profile.normEpsilon);
  const errors = errorStats(gpu, cpu);

  return { defaultMs, optimizedMs, ...errors };
}

function createMlpBlockInputs(hiddenSize, intermediateSize) {
  return {
    hiddenSize,
    intermediateSize,
    x: createVector(hiddenSize, 0.017, Math.sin),
    normWeight: new Float32Array(hiddenSize).fill(1),
    gate: createQ8Inputs(hiddenSize, intermediateSize),
    up: createQ8Inputs(hiddenSize, intermediateSize),
    down: createQ8Inputs(intermediateSize, hiddenSize),
  };
}

function createMlpBlockRuntime(device, matvecShader, tensors) {
  const xBuffer = storageBuffer(device, tensors.x);
  const normWeightBuffer = storageBuffer(device, tensors.normWeight);
  const normOutBuffer = emptyStorageBuffer(device, tensors.hiddenSize * 4);
  const gateOutBuffer = emptyStorageBuffer(device, tensors.intermediateSize * 4);
  const upOutBuffer = emptyStorageBuffer(device, tensors.intermediateSize * 4);
  const activationBuffer = emptyStorageBuffer(device, tensors.intermediateSize * 4);
  const downOutBuffer = emptyStorageBuffer(device, tensors.hiddenSize * 4);

  const rmsPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: rmsNormShader }),
      entryPoint: "main",
    },
  });
  const epsBits = new Uint32Array(new Float32Array([1e-6]).buffer)[0];
  const rmsBindGroup = device.createBindGroup({
    layout: rmsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: normWeightBuffer } },
      { binding: 2, resource: { buffer: normOutBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([tensors.hiddenSize, epsBits, 0, 0])) } },
    ],
  });

  const matvecPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: matvecShader }),
      entryPoint: "main",
    },
  });
  const gate = createQ8BindGroup(device, matvecPipeline, normOutBuffer, gateOutBuffer, tensors.gate);
  const up = createQ8BindGroup(device, matvecPipeline, normOutBuffer, upOutBuffer, tensors.up);
  const down = createQ8BindGroup(device, matvecPipeline, activationBuffer, downOutBuffer, tensors.down);

  const siluPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: siluMulShader }),
      entryPoint: "main",
    },
  });
  const siluBindGroup = device.createBindGroup({
    layout: siluPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gateOutBuffer } },
      { binding: 1, resource: { buffer: upOutBuffer } },
      { binding: 2, resource: { buffer: activationBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([tensors.intermediateSize, 0, 0, 0])) } },
    ],
  });

  return {
    hiddenSize: tensors.hiddenSize,
    intermediateSize: tensors.intermediateSize,
    rmsPipeline,
    rmsBindGroup,
    matvecPipeline,
    gate,
    up,
    down,
    siluPipeline,
    siluBindGroup,
    downOutBuffer,
  };
}

function createPipeline(device, shader) {
  return device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: shader }),
      entryPoint: "main",
    },
  });
}

function createRmsBindGroup(device, pipeline, inputBuffer, weightBuffer, outputBuffer, hiddenSize, epsBits) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: weightBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([hiddenSize, epsBits, 0, 0])) } },
    ],
  });
}

function createHeadRmsBindGroup(device, pipeline, inputBuffer, weightBuffer, outputBuffer, headDim, heads, epsBits) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: weightBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([headDim, heads, epsBits, 0])) } },
    ],
  });
}

async function createQ4RuntimeProjection(device, pipeline, inputBuffer, outputBuffer, base, projection) {
  const [weightBytes, scalesBuffer] = await Promise.all([
    fetch(`${base}/${projection.weight.file}`).then((response) => response.arrayBuffer()),
    fetch(`${base}/${projection.scales.file}`).then((response) => response.arrayBuffer()),
  ]);
  const dims = projection.matmulNBits;
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: rawStorageBuffer(device, weightBytes) } },
      { binding: 2, resource: { buffer: storageBuffer(device, new Float32Array(scalesBuffer)) } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: uniformBuffer(device, new Uint32Array([dims.inSize, dims.outSize, dims.blocksPerRow, dims.blockSize])) } },
    ],
  });
}

function createRopeBindGroup(device, pipeline, inputBuffer, cosBuffer, sinBuffer, outputBuffer, heads, headDim, position) {
  return createRopeBindGroupWithDims(
    device,
    pipeline,
    inputBuffer,
    cosBuffer,
    sinBuffer,
    outputBuffer,
    uniformBuffer(device, new Uint32Array([heads, headDim, position, 0])),
  );
}

function createRopeBindGroupWithDims(device, pipeline, inputBuffer, cosBuffer, sinBuffer, outputBuffer, dimsBuffer) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: cosBuffer } },
      { binding: 2, resource: { buffer: sinBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: dimsBuffer } },
    ],
  });
}

function createGqaBindGroup(device, pipeline, qBuffer, kBuffer, vBuffer, outputBuffer, seqLen, qHeads, kvHeads, headDim) {
  return createGqaBindGroupWithDims(
    device,
    pipeline,
    qBuffer,
    kBuffer,
    vBuffer,
    outputBuffer,
    uniformBuffer(device, new Uint32Array([seqLen, qHeads, kvHeads, headDim])),
  );
}

function createGqaBindGroupWithDims(device, pipeline, qBuffer, kBuffer, vBuffer, outputBuffer, dimsBuffer) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: qBuffer } },
      { binding: 1, resource: { buffer: kBuffer } },
      { binding: 2, resource: { buffer: vBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: dimsBuffer } },
    ],
  });
}

function createSiluBindGroup(device, pipeline, gateBuffer, upBuffer, outputBuffer, size) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gateBuffer } },
      { binding: 1, resource: { buffer: upBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([size, 0, 0, 0])) } },
    ],
  });
}

function createAddBindGroup(device, pipeline, aBuffer, bBuffer, outputBuffer, size) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer(device, new Uint32Array([size, 0, 0, 0])) } },
    ],
  });
}

async function timeLayer0Decode(device, runtime, iterations) {
  const started = performance.now();
  for (let i = 0; i < iterations; i++) {
    dispatchLayer0Decode(device, runtime);
  }
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - started) / iterations;
}

function dispatchLayer0Decode(device, runtime) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();

  pass.setPipeline(runtime.rmsPipeline);
  pass.setBindGroup(0, runtime.inputRms);
  pass.dispatchWorkgroups(1);

  pass.setPipeline(runtime.q4Pipeline);
  pass.setBindGroup(0, runtime.qProj);
  pass.dispatchWorkgroups(runtime.qSize);
  pass.setBindGroup(0, runtime.kProj);
  pass.dispatchWorkgroups(runtime.kvSize);
  pass.setBindGroup(0, runtime.vProj);
  pass.dispatchWorkgroups(runtime.kvSize);

  pass.setPipeline(runtime.headRmsPipeline);
  pass.setBindGroup(0, runtime.qHeadNorm);
  pass.dispatchWorkgroups(runtime.qHeads);
  pass.setBindGroup(0, runtime.kHeadNorm);
  pass.dispatchWorkgroups(runtime.kvHeads);

  pass.setPipeline(runtime.ropePipeline);
  pass.setBindGroup(0, runtime.qRope);
  pass.dispatchWorkgroups(Math.ceil(runtime.qSize / 256));
  pass.setBindGroup(0, runtime.kRope);
  pass.dispatchWorkgroups(Math.ceil(runtime.kvSize / 256));

  pass.setPipeline(runtime.gqaPipeline);
  pass.setBindGroup(0, runtime.gqa);
  pass.dispatchWorkgroups(runtime.qHeads);

  pass.setPipeline(runtime.q4Pipeline);
  pass.setBindGroup(0, runtime.oProj);
  pass.dispatchWorkgroups(runtime.hiddenSize);

  pass.setPipeline(runtime.addPipeline);
  pass.setBindGroup(0, runtime.attnResidual);
  pass.dispatchWorkgroups(Math.ceil(runtime.hiddenSize / 256));

  pass.setPipeline(runtime.rmsPipeline);
  pass.setBindGroup(0, runtime.postRms);
  pass.dispatchWorkgroups(1);

  pass.setPipeline(runtime.q4Pipeline);
  pass.setBindGroup(0, runtime.gate);
  pass.dispatchWorkgroups(runtime.intermediateSize);
  pass.setBindGroup(0, runtime.up);
  pass.dispatchWorkgroups(runtime.intermediateSize);

  pass.setPipeline(runtime.siluPipeline);
  pass.setBindGroup(0, runtime.silu);
  pass.dispatchWorkgroups(Math.ceil(runtime.intermediateSize / 256));

  pass.setPipeline(runtime.q4Pipeline);
  pass.setBindGroup(0, runtime.down);
  pass.dispatchWorkgroups(runtime.hiddenSize);

  pass.setPipeline(runtime.addPipeline);
  pass.setBindGroup(0, runtime.finalResidual);
  pass.dispatchWorkgroups(Math.ceil(runtime.hiddenSize / 256));

  pass.end();
  device.queue.submit([encoder.finish()]);
}

function createQ8BindGroup(device, pipeline, inputBuffer, outputBuffer, tensors) {
  const weightsBuffer = storageBuffer(device, tensors.packed);
  const scalesBuffer = storageBuffer(device, tensors.scales);
  const dimsBuffer = uniformBuffer(
    device,
    new Uint32Array([tensors.inSize, tensors.outSize, tensors.wordsPerRow, 0]),
  );
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: weightsBuffer } },
      { binding: 2, resource: { buffer: scalesBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: dimsBuffer } },
    ],
  });
}

async function timeMlpBlock(device, runtime, mode, iterations) {
  const started = performance.now();
  for (let i = 0; i < iterations; i++) {
    dispatchMlpBlock(device, runtime, mode);
  }
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - started) / iterations;
}

function dispatchMlpBlock(device, runtime, mode) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();

  pass.setPipeline(runtime.rmsPipeline);
  pass.setBindGroup(0, runtime.rmsBindGroup);
  pass.dispatchWorkgroups(1);

  pass.setPipeline(runtime.matvecPipeline);
  pass.setBindGroup(0, runtime.gate);
  pass.dispatchWorkgroups(mode === "scalar" ? Math.ceil(runtime.intermediateSize / 64) : runtime.intermediateSize);
  pass.setBindGroup(0, runtime.up);
  pass.dispatchWorkgroups(mode === "scalar" ? Math.ceil(runtime.intermediateSize / 64) : runtime.intermediateSize);

  pass.setPipeline(runtime.siluPipeline);
  pass.setBindGroup(0, runtime.siluBindGroup);
  pass.dispatchWorkgroups(Math.ceil(runtime.intermediateSize / 256));

  pass.setPipeline(runtime.matvecPipeline);
  pass.setBindGroup(0, runtime.down);
  pass.dispatchWorkgroups(mode === "scalar" ? Math.ceil(runtime.hiddenSize / 64) : runtime.hiddenSize);

  pass.end();
  device.queue.submit([encoder.finish()]);
}

function cpuMlpBlock(tensors, epsilon) {
  const norm = cpuRmsNorm(tensors.x, tensors.normWeight, epsilon);
  const gate = cpuQ8Matvec({ ...tensors.gate, x: norm });
  const up = cpuQ8Matvec({ ...tensors.up, x: norm });
  const activation = new Float32Array(tensors.intermediateSize);
  for (let i = 0; i < activation.length; i++) {
    activation[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
  }
  return cpuQ8Matvec({ ...tensors.down, x: activation });
}

function cpuRmsNorm(x, weight, epsilon) {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  const invRms = 1 / Math.sqrt(sum / x.length + epsilon);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * invRms * weight[i];
  return out;
}

function createVector(size, step, fn) {
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) out[i] = fn(i * step);
  return out;
}

async function runQ8Comparison(device, inSize, outSize) {
  const tensors = createQ8Inputs(inSize, outSize);
  const scalar = createQ8MatvecPipeline(device, q8MatvecScalarShader, tensors);
  const tiled = createQ8MatvecPipeline(device, q8MatvecTiledShader, tensors);

  const scalarMs = await timeDispatch(device, scalar.pipeline, scalar.bindGroup, Math.ceil(outSize / 64), 40);
  const tiledMs = await timeDispatch(device, tiled.pipeline, tiled.bindGroup, outSize, 80);
  dispatchOnce(device, tiled.pipeline, tiled.bindGroup, outSize);
  await device.queue.onSubmittedWorkDone();

  const gpu = await readFloat32Buffer(device, tiled.yBuffer, outSize);
  const cpu = cpuQ8Matvec(tensors);
  const errors = errorStats(gpu, cpu);

  return {
    scalarMs,
    tiledMs,
    bytesRead: tensors.x.byteLength + tensors.packed.byteLength + tensors.scales.byteLength,
    estimatedGbps:
      (tensors.x.byteLength + tensors.packed.byteLength + tensors.scales.byteLength) /
      (tiledMs / 1000) /
      1e9,
    ...errors,
  };
}

function createQ8Inputs(inSize, outSize) {
  const wordsPerRow = Math.ceil(inSize / 4);
  const weightWords = wordsPerRow * outSize;
  const x = new Float32Array(inSize);
  const scales = new Float32Array(outSize);
  const packed = new Uint32Array(weightWords);

  for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.017);
  for (let i = 0; i < scales.length; i++) scales[i] = 0.0125 + (i % 11) * 0.0002;
  for (let row = 0; row < outSize; row++) {
    for (let word = 0; word < wordsPerRow; word++) {
      let packedWord = 0;
      for (let lane = 0; lane < 4; lane++) {
        const col = word * 4 + lane;
        const signed = col < inSize ? ((row * 13 + col * 7) % 255) - 127 : 0;
        packedWord |= (signed & 0xff) << (lane * 8);
      }
      packed[row * wordsPerRow + word] = packedWord >>> 0;
    }
  }

  return { inSize, outSize, wordsPerRow, x, scales, packed };
}

function createQ8MatvecPipeline(device, shader, tensors) {
  const xBuffer = storageBuffer(device, tensors.x);
  const wBuffer = storageBuffer(device, tensors.packed);
  const sBuffer = storageBuffer(device, tensors.scales);
  const yBuffer = emptyStorageBuffer(device, tensors.outSize * 4);
  const dims = uniformBuffer(
    device,
    new Uint32Array([tensors.inSize, tensors.outSize, tensors.wordsPerRow, 0]),
  );

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: shader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: wBuffer } },
      { binding: 2, resource: { buffer: sBuffer } },
      { binding: 3, resource: { buffer: yBuffer } },
      { binding: 4, resource: { buffer: dims } },
    ],
  });

  return { pipeline, bindGroup, yBuffer };
}

function cpuQ8Matvec({ inSize, outSize, wordsPerRow, x, scales, packed }) {
  const y = new Float32Array(outSize);
  for (let row = 0; row < outSize; row++) {
    let acc = 0;
    const rowOffset = row * wordsPerRow;
    for (let word = 0; word < wordsPerRow; word++) {
      const packedWord = packed[rowOffset + word];
      for (let lane = 0; lane < 4; lane++) {
        const col = word * 4 + lane;
        if (col < inSize) {
          const raw = (packedWord >>> (lane * 8)) & 0xff;
          const signed = raw >= 128 ? raw - 256 : raw;
          acc += signed * scales[row] * x[col];
        }
      }
    }
    y[row] = acc;
  }
  return y;
}

function errorStats(actual, expected) {
  let maxAbsError = 0;
  let sumAbsError = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    maxAbsError = Math.max(maxAbsError, error);
    sumAbsError += error;
  }
  return {
    maxAbsError,
    meanAbsError: sumAbsError / actual.length,
  };
}

async function runFp32Matvec(device, inSize, outSize) {
  const x = new Float32Array(inSize);
  const weights = new Float32Array(inSize * outSize);

  for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.017);
  for (let i = 0; i < weights.length; i++) weights[i] = Math.cos(i * 0.007) * 0.02;

  const xBuffer = storageBuffer(device, x);
  const wBuffer = storageBuffer(device, weights);
  const yBuffer = emptyStorageBuffer(device, outSize * 4);
  const dims = uniformBuffer(device, new Uint32Array([inSize, outSize, 0, 0]));

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: fp32MatvecShader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: wBuffer } },
      { binding: 2, resource: { buffer: yBuffer } },
      { binding: 3, resource: { buffer: dims } },
    ],
  });

  const ms = await timeDispatch(device, pipeline, bindGroup, Math.ceil(outSize / 64), 40);
  return { ms, bytesRead: x.byteLength + weights.byteLength };
}

async function runQ4Matvec(device, inSize, outSize) {
  const wordsPerRow = Math.ceil(inSize / 8);
  const weightWords = wordsPerRow * outSize;
  const x = new Float32Array(inSize);
  const scales = new Float32Array(outSize);
  const packed = new Uint32Array(weightWords);

  for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.017);
  for (let i = 0; i < scales.length; i++) scales[i] = 0.035 + (i % 7) * 0.001;
  for (let i = 0; i < packed.length; i++) {
    let word = 0;
    for (let lane = 0; lane < 8; lane++) {
      word |= ((i + lane * 3) & 0xf) << (lane * 4);
    }
    packed[i] = word >>> 0;
  }

  const xBuffer = storageBuffer(device, x);
  const wBuffer = storageBuffer(device, packed);
  const sBuffer = storageBuffer(device, scales);
  const yBuffer = emptyStorageBuffer(device, outSize * 4);
  const dims = uniformBuffer(
    device,
    new Uint32Array([inSize, outSize, wordsPerRow, 0]),
  );

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: q4MatvecShader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: wBuffer } },
      { binding: 2, resource: { buffer: sBuffer } },
      { binding: 3, resource: { buffer: yBuffer } },
      { binding: 4, resource: { buffer: dims } },
    ],
  });

  const ms = await timeDispatch(device, pipeline, bindGroup, Math.ceil(outSize / 64), 80);
  const bytesRead = x.byteLength + packed.byteLength + scales.byteLength;
  return { ms, bytesRead, estimatedGbps: bytesRead / (ms / 1000) / 1e9 };
}

async function runRmsNorm(device, hiddenSize, epsilon) {
  const x = new Float32Array(hiddenSize);
  const weight = new Float32Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i++) {
    x[i] = Math.cos(i * 0.013);
    weight[i] = 1.0;
  }

  const xBuffer = storageBuffer(device, x);
  const wBuffer = storageBuffer(device, weight);
  const yBuffer = emptyStorageBuffer(device, hiddenSize * 4);
  const epsBits = new Uint32Array(new Float32Array([epsilon]).buffer)[0];
  const dims = uniformBuffer(device, new Uint32Array([hiddenSize, epsBits, 0, 0]));

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: rmsNormShader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: wBuffer } },
      { binding: 2, resource: { buffer: yBuffer } },
      { binding: 3, resource: { buffer: dims } },
    ],
  });

  return { ms: await timeDispatch(device, pipeline, bindGroup, 1, 120) };
}

async function timeDispatch(device, pipeline, bindGroup, workgroups, iterations) {
  const started = performance.now();
  for (let i = 0; i < iterations; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - started) / iterations;
}

function dispatchOnce(device, pipeline, bindGroup, workgroups) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

async function readFloat32Buffer(device, source, length) {
  const readBuffer = device.createBuffer({
    size: align4(length * 4),
    usage: GPUBufferUsageFlags.COPY_DST | GPUBufferUsageFlags.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readBuffer, 0, length * 4);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapModeFlags.READ);
  const copy = readBuffer.getMappedRange().slice(0);
  readBuffer.unmap();
  readBuffer.destroy();
  return new Float32Array(copy);
}

async function readUint32Buffer(device, source, length) {
  const readBuffer = device.createBuffer({
    size: align4(length * 4),
    usage: GPUBufferUsageFlags.COPY_DST | GPUBufferUsageFlags.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readBuffer, 0, length * 4);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapModeFlags.READ);
  const copy = readBuffer.getMappedRange().slice(0);
  readBuffer.unmap();
  readBuffer.destroy();
  return new Uint32Array(copy);
}

async function fetchF16AsF32(url) {
  const bytes = new Uint16Array(await fetch(url).then((response) => response.arrayBuffer()));
  const out = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = halfToFloat(bytes[i]);
  return out;
}

function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

function storageBuffer(device, typedArray) {
  const buffer = device.createBuffer({
    size: align4(typedArray.byteLength),
    usage: GPUBufferUsageFlags.STORAGE | GPUBufferUsageFlags.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, typedArray);
  return buffer;
}

function rawStorageBuffer(device, arrayBuffer) {
  const buffer = device.createBuffer({
    size: align4(arrayBuffer.byteLength),
    usage: GPUBufferUsageFlags.STORAGE | GPUBufferUsageFlags.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, arrayBuffer);
  return buffer;
}

function emptyStorageBuffer(device, size) {
  return device.createBuffer({
    size: align4(size),
    usage: GPUBufferUsageFlags.STORAGE | GPUBufferUsageFlags.COPY_SRC | GPUBufferUsageFlags.COPY_DST,
  });
}

function uniformBuffer(device, typedArray) {
  const buffer = device.createBuffer({
    size: align4(typedArray.byteLength),
    usage: GPUBufferUsageFlags.UNIFORM | GPUBufferUsageFlags.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, typedArray);
  return buffer;
}

function align4(value) {
  return Math.ceil(value / 4) * 4;
}
