export const qwenProfile = {
  modelId: "Qwen/Qwen3-0.6B",
  parameters: "0.6B",
  nonEmbeddingParameters: "0.44B",
  layers: 28,
  hiddenSize: 1024,
  intermediateSize: 3072,
  attentionHeads: 16,
  kvHeads: 8,
  headDim: 128,
  contextLength: 32768,
  vocabSize: 151936,
  normEpsilon: 1e-6,
  quantizationTarget: "weight-only Q4 MatMulNBits for projection and MLP matvecs",
};

export const kernelPlan = [
  {
    hotPath: "Decode projection",
    kernel: "matmul_nbits_q4.wgsl",
    choice: "Matches Qwen ONNX MatMulNBits: packed uint8 q4 blocks with fp16/fp32 scales and 256-lane row reduction.",
  },
  {
    hotPath: "Layer normalization",
    kernel: "rms_norm.wgsl",
    choice: "256-lane workgroup reduction tuned for hidden_size=1024.",
  },
  {
    hotPath: "Grouped-query attention",
    kernel: "gqa_decode.wgsl",
    choice: "Single-token decode attention over KV cache layout layer, token, kv_head, head_dim for 8 KV heads.",
  },
  {
    hotPath: "MLP gate/up/down",
    kernel: "matmul_nbits_q4.wgsl",
    choice: "Reuse q4 MatMulNBits primitive across 1024->3072 and 3072->1024 projections.",
  },
];

export const fullPromptCoverage = [
  {
    component: "Tokenizer + chat template",
    status: "covered",
    note: "Handled by Transformers.js for the default prompt path.",
  },
  {
    component: "Token embedding lookup",
    status: "partial",
    note: "Real embedding weights are extracted; custom gather/upload path still needs to be wired into decode.",
  },
  {
    component: "RMSNorm",
    status: "covered",
    note: "Custom WGSL kernel is implemented and used in the wired MLP block.",
  },
  {
    component: "Q/K/V projections",
    status: "covered",
    note: "Real Qwen q4 MatMulNBits weights are extracted; custom q4 kernel is implemented.",
  },
  {
    component: "RoPE",
    status: "covered",
    note: "Custom RoPE WGSL kernel is wired into the measured layer-0 decode block with real cos/sin cache values.",
  },
  {
    component: "GQA attention + KV cache",
    status: "partial",
    note: "Single-token decode GQA kernel is CPU-validated and now wired after real q/k/v projections for layer 0; full prefill integration is pending.",
  },
  {
    component: "MLP block",
    status: "covered",
    note: "Custom q4 MatMulNBits MLP is wired inside a real layer-0 decode block using extracted Qwen weights.",
  },
  {
    component: "Residual path",
    status: "covered",
    note: "Residual adds are implemented in the layer block and in the separate 28-layer custom decoder stack benchmark.",
  },
  {
    component: "28-layer decoder stack",
    status: "partial",
    note: "Custom WebGPU path runs all 28 layers plus final norm for one token with real q4f16 weights; prompt prefill is still pending.",
  },
  {
    component: "LM head",
    status: "covered",
    note: "Tied embedding table stays resident on the GPU as f16 and is unpacked in-shader; logits for the full 151936 vocab are produced in one submit.",
  },
  {
    component: "Sampler",
    status: "covered",
    note: "Greedy argmax runs as a two-stage GPU reduction, so each token reads back 8 bytes instead of the full logit vector.",
  },
  {
    component: "Qwen ONNX weight import",
    status: "covered",
    note: "Full q4f16 ONNX artifact is downloaded and 28-layer runtime assets are extracted under /qwen/runtime.",
  },
];
