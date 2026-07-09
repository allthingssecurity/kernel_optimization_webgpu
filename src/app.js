import {
  createWebGpuContext,
  runFullDecoderStackBenchmark,
  runKernelBenchmarks,
} from "./bench/webgpu-runtime.js";
import { realModelConfigs, runRealModelPrompt } from "./bench/real-model-runner.js";
import { fullPromptCoverage, kernelPlan, qwenProfile } from "./model/qwen3-0.6b.js";

const shape = document.querySelector("#modelShape");
const plan = document.querySelector("#kernelPlan");
const promptCoverage = document.querySelector("#promptCoverage");
const status = document.querySelector("#status");
const runButton = document.querySelector("#runBench");
const copyButton = document.querySelector("#copyProfile");
const promptInput = document.querySelector("#promptInput");
const systemInput = document.querySelector("#systemInput");
const maxTokensInput = document.querySelector("#maxTokens");
const realStatus = document.querySelector("#realStatus");
const beforeButton = document.querySelector("#runBefore");
const afterButton = document.querySelector("#runAfter");
const compareButton = document.querySelector("#runCompare");
const decoderButton = document.querySelector("#runDecoderStack");

const realResults = {
  before: null,
  after: null,
};

renderModelShape();
renderKernelPlan();
renderPromptCoverage();

runButton.addEventListener("click", async () => {
  runButton.disabled = true;
  status.textContent = "Requesting WebGPU adapter and compiling kernels...";
  try {
    const { adapter, device } = await createWebGpuContext();
    const results = await runKernelBenchmarks(device, qwenProfile);
    document.querySelector("#baselineMs").textContent = results.baselineMatvecMs.toFixed(3);
    document.querySelector("#optimizedMs").textContent = results.optimizedMatvecMs.toFixed(3);
    document.querySelector("#speedup").textContent = `${results.matvecSpeedup.toFixed(2)}x`;
    document.querySelector("#memoryReduction").textContent = `${results.maxAbsError.toExponential(2)}`;
    document.querySelector("#rmsMs").textContent = results.rmsMs.toFixed(3);
    document.querySelector("#bandwidth").textContent = results.estimatedGbps.toFixed(2);
    document.querySelector("#blockDefaultMs").textContent = results.blockDefaultMs.toFixed(3);
    document.querySelector("#blockOptimizedMs").textContent = results.blockOptimizedMs.toFixed(3);
    document.querySelector("#blockSpeedup").textContent = `${results.blockSpeedup.toFixed(2)}x`;
    document.querySelector("#blockError").textContent = results.blockMaxAbsError.toExponential(2);
    document.querySelector("#realQ4Ms").textContent = results.realQ4Ms.toFixed(3);
    document.querySelector("#realQ4Error").textContent = results.realQ4MaxAbsError.toExponential(2);
    document.querySelector("#gqaMs").textContent = results.gqaMs.toFixed(3);
    document.querySelector("#gqaError").textContent = results.gqaMaxAbsError.toExponential(2);
    document.querySelector("#gqaSeqLen").textContent = `${results.gqaSeqLen}`;
    document.querySelector("#layer0DecodeMs").textContent = results.layer0DecodeMs.toFixed(3);
    document.querySelector("#layer0Finite").textContent = `${results.layer0Finite}/1024`;
    document.querySelector("#layer0MeanAbs").textContent = results.layer0MeanAbs.toExponential(2);
    status.textContent = `Ran on ${adapter.info?.description || "available WebGPU adapter"}.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
});

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(JSON.stringify(qwenProfile, null, 2));
  copyButton.textContent = "Copied";
  setTimeout(() => {
    copyButton.textContent = "Copy model profile";
  }, 1200);
});

beforeButton.addEventListener("click", () => runRealModel("before"));
afterButton.addEventListener("click", () => runRealModel("after"));
compareButton.addEventListener("click", async () => {
  await runRealModel("before");
  await runRealModel("after");
});
decoderButton.addEventListener("click", runDecoderStack);

function renderModelShape() {
  const rows = [
    ["Model", qwenProfile.modelId],
    ["Layers", qwenProfile.layers],
    ["Hidden", qwenProfile.hiddenSize],
    ["Intermediate", qwenProfile.intermediateSize],
    ["Q / KV heads", `${qwenProfile.attentionHeads} / ${qwenProfile.kvHeads}`],
    ["Head dim", qwenProfile.headDim],
    ["Context", qwenProfile.contextLength.toLocaleString()],
    ["Target", qwenProfile.quantizationTarget],
  ];

  shape.innerHTML = rows
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("");
}

function renderKernelPlan() {
  plan.innerHTML = kernelPlan
    .map(
      (row) => `
        <tr>
          <td>${row.hotPath}</td>
          <td><code>${row.kernel}</code></td>
          <td>${row.choice}</td>
        </tr>
      `,
    )
    .join("");
}

function renderPromptCoverage() {
  promptCoverage.innerHTML = fullPromptCoverage
    .map(
      (row) => `
        <tr>
          <td>${row.component}</td>
          <td><span class="status-pill ${row.status}">${row.status}</span></td>
          <td>${row.note}</td>
        </tr>
      `,
    )
    .join("");
}

async function runRealModel(which) {
  const config = realModelConfigs[which];
  if (config.unavailableReason) {
    realStatus.textContent = config.unavailableReason;
    return;
  }

  const prompt = promptInput.value.trim();
  const systemPrompt = systemInput.value.trim();
  const maxNewTokens = Number(maxTokensInput.value);
  if (!prompt) {
    realStatus.textContent = "Enter a prompt first.";
    return;
  }

  setRealButtons(true);
  setRealOutput(which, "");
  setRealMetric(which, "dtype", config.dtype);
  setRealMetric(which, "device", config.device);
  realStatus.textContent = `${config.label}: starting...`;

  try {
    const result = await runRealModelPrompt({
      dtype: config.dtype,
      device: config.device,
      customWebGpu: config.customWebGpu,
      systemPrompt,
      prompt,
      maxNewTokens,
      onProgress: (message) => {
        realStatus.textContent = `${config.label}: ${message}`;
      },
      onToken: (text) => {
        appendRealOutput(which, text);
      },
    });
    realResults[which] = result;
    renderRealResult(which, result);
    renderRealComparison();
    realStatus.textContent = `${config.label}: complete.`;
  } catch (error) {
    realStatus.textContent = `${config.label}: ${error.message}`;
  } finally {
    setRealButtons(false);
  }
}

async function runDecoderStack() {
  decoderButton.disabled = true;
  realStatus.textContent = "Custom WebGPU decoder stack: loading real Qwen q4f16 weights...";
  try {
    const { adapter, device } = await createWebGpuContext();
    const result = await runFullDecoderStackBenchmark(device, qwenProfile);
    document.querySelector("#decoderStackMs").textContent = formatMs(result.ms);
    document.querySelector("#decoderStackWarmup").textContent = formatMs(result.warmupMs);
    document.querySelector("#decoderStackFinite").textContent = `${result.finite}/1024`;
    document.querySelector("#decoderStackMeanAbs").textContent = result.meanAbs.toExponential(2);
    realStatus.textContent = `Custom WebGPU decoder stack ran ${result.layers} Qwen layers plus final norm on ${adapter.info?.description || "available adapter"}. LM head and sampler are still pending.`;
  } catch (error) {
    realStatus.textContent = `Custom decoder stack: ${error.message}`;
  } finally {
    decoderButton.disabled = false;
  }
}

function renderRealResult(which, result) {
  setRealMetric(which, "load", formatMs(result.loadMs));
  setRealMetric(which, "device", result.device);
  setRealMetric(which, "ttft", result.firstTokenMs ? formatMs(result.firstTokenMs) : "-");
  setRealMetric(which, "gen", formatMs(result.generateMs));
  setRealMetric(
    which,
    "speed",
    result.tokensPerSecond ? `${formatTokensPerSecond(result.tokensPerSecond)} tok/s` : "-",
  );
  setRealMetric(
    which,
    "tokens",
    `${result.promptTokens ?? "?"} in / ${result.generatedTokens} out${result.generatedTokenIds ? ` / ids ${result.generatedTokenIds.join(",")}` : ""}`,
  );
  setRealMetric(which, "validity", assessOutput(result.outputText));
  if (!document.querySelector(`#${which}Output`).textContent.trim()) {
    setRealOutput(which, result.outputText);
  }
}

function assessOutput(text) {
  const normalized = text.trim().toLowerCase();
  if (!normalized && text.length > 0) return "whitespace token";
  if (!normalized) return "empty";
  if (hasRepeatedLine(text)) return "check output";
  if (normalized.includes("webgpu") || normalized.includes("quant")) return "looks valid";
  return "review";
}

function hasRepeatedLine(text) {
  const lines = text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter((line) => line.length > 12);
  return new Set(lines).size < lines.length;
}

function renderRealComparison() {
  const before = realResults.before;
  const after = realResults.after;
  if (!before || !after) return;

  const speedup = after.tokensPerSecond && before.tokensPerSecond
    ? after.tokensPerSecond / before.tokensPerSecond
    : null;
  const ttftChange = after.firstTokenMs && before.firstTokenMs
    ? before.firstTokenMs / after.firstTokenMs
    : null;

  document.querySelector("#realSpeedup").textContent = speedup
    ? `${speedup.toFixed(2)}x`
    : "-";
  document.querySelector("#realTtft").textContent = ttftChange
    ? `${ttftChange.toFixed(2)}x`
    : "-";
}

function setRealButtons(disabled) {
  beforeButton.disabled = disabled;
  afterButton.disabled = disabled;
  compareButton.disabled = disabled;
}

function setRealMetric(which, metric, value) {
  document.querySelector(`[data-result="${which}-${metric}"]`).textContent = value;
}

function setRealOutput(which, value) {
  document.querySelector(`#${which}Output`).textContent = value;
}

function appendRealOutput(which, value) {
  document.querySelector(`#${which}Output`).textContent += value;
}

function formatMs(value) {
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function formatTokensPerSecond(value) {
  if (value < 0.01) return value.toFixed(4);
  return value.toFixed(2);
}
