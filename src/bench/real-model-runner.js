import { AutoTokenizer, pipeline, TextStreamer } from "@huggingface/transformers";
import { createWebGpuContext, runCustomQwenGreedyPrompt } from "./webgpu-runtime.js";

const MODEL_ID = "onnx-community/Qwen3-0.6B-ONNX";

export const realModelConfigs = {
  before: {
    label: "Default: q4f16 WebGPU",
    dtype: "q4f16",
    device: "webgpu",
  },
  after: {
    label: "Optimized: custom q4f16 WebGPU",
    dtype: "q4f16",
    device: "webgpu",
    customWebGpu: true,
  },
};

export async function runRealModelPrompt({
  dtype,
  device,
  systemPrompt,
  prompt,
  maxNewTokens,
  onProgress,
  onToken,
}) {
  if (dtype === "q4f16-custom") {
    throw new Error("Use customWebGpu config instead of dtype sentinel.");
  }
  if (arguments[0]?.customWebGpu) {
    return runCustomModelPrompt(arguments[0]);
  }

  onProgress?.(`Loading ${MODEL_ID} (${dtype}, ${device})...`);
  const loadStarted = performance.now();
  const generator = await pipeline("text-generation", MODEL_ID, {
    device,
    dtype,
    progress_callback: (progress) => {
      if (progress.status === "progress") {
        const pct = Number.isFinite(progress.progress)
          ? `${progress.progress.toFixed(1)}%`
          : "downloading";
        onProgress?.(`${progress.file}: ${pct}`);
      } else if (progress.status) {
        onProgress?.(`${progress.status}${progress.file ? ` ${progress.file}` : ""}`);
      }
    },
  });
  const loadMs = performance.now() - loadStarted;

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: prompt },
  ];

  let promptTokens = null;
  let renderedPrompt = "";
  try {
    renderedPrompt = generator.tokenizer.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
    });
    const encoded = generator.tokenizer(renderedPrompt, { add_special_tokens: false });
    promptTokens = encoded.input_ids?.data?.length ?? encoded.input_ids?.size ?? null;
  } catch {
    promptTokens = null;
  }

  let outputText = "";
  let generatedTokens = 0;
  let firstTokenMs = null;
  let generationStarted = 0;

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      outputText += text;
      onToken?.(text);
    },
    token_callback_function: (tokens) => {
      generatedTokens += tokens.length;
      if (firstTokenMs === null) {
        firstTokenMs = performance.now() - generationStarted;
      }
    },
  });

  onProgress?.(`Generating with ${dtype} on ${device}...`);
  generationStarted = performance.now();
  const result = await generator(messages, {
    max_new_tokens: maxNewTokens,
    do_sample: false,
    tokenizer_encode_kwargs: {
      enable_thinking: false,
    },
    streamer,
  });
  const generateMs = performance.now() - generationStarted;

  if (!outputText.trim()) {
    outputText = extractGeneratedText(result);
  }

  await generator.dispose();

  return {
    modelId: MODEL_ID,
    dtype,
    device,
    promptTokens,
    renderedPrompt,
    generatedTokens,
    loadMs,
    generateMs,
    firstTokenMs,
    tokensPerSecond: generatedTokens > 0 ? generatedTokens / (generateMs / 1000) : null,
    outputText,
  };
}

export async function runCustomModelPrompt({
  systemPrompt,
  prompt,
  maxNewTokens,
  onProgress,
  onToken,
}) {
  onProgress?.(`Loading tokenizer and WebGPU custom runtime...`);
  const loadStarted = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const { device } = await createWebGpuContext();
  const loadMs = performance.now() - loadStarted;

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: prompt },
  ];
  const renderedPrompt = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
    enable_thinking: false,
  });
  const encoded = tokenizer(renderedPrompt, { add_special_tokens: false });
  const inputIds = normalizeInputIds(encoded.input_ids);

  const result = await runCustomQwenGreedyPrompt({
    device,
    profile: {
      hiddenSize: 1024,
      intermediateSize: 3072,
      attentionHeads: 16,
      kvHeads: 8,
      headDim: 128,
      normEpsilon: 1e-6,
    },
    inputIds,
    maxNewTokens,
    onProgress,
    onToken,
    decodeToken: (ids) => tokenizer.decode(ids, { skip_special_tokens: true }),
  });

  return {
    modelId: MODEL_ID,
    dtype: "q4f16",
    device: "custom-webgpu",
    promptTokens: inputIds.length,
    renderedPrompt,
    generatedTokens: result.generatedTokens,
    generatedTokenIds: result.generated,
    loadMs,
    generateMs: result.generateMs,
    firstTokenMs: result.firstTokenMs,
    tokensPerSecond: result.generatedTokens > 0 ? result.generatedTokens / (result.generateMs / 1000) : null,
    outputText: result.outputText,
  };
}

function normalizeInputIds(inputIds) {
  const raw = inputIds?.data ?? inputIds;
  if (!raw) return [];
  return Array.from(raw, (value) => Number(value));
}

function extractGeneratedText(result) {
  if (Array.isArray(result)) {
    return result.map((item) => extractGeneratedText(item)).join("\n");
  }
  if (Array.isArray(result?.generated_text)) {
    return result.generated_text.at(-1)?.content ?? "";
  }
  return result?.generated_text ?? "";
}
