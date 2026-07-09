import { createWebGpuContext, runCustomQwenGreedyPrompt } from "./webgpu-runtime.js";
import { qwenProfile } from "../model/qwen3-0.6b.js";

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const promptEl = document.getElementById("prompt");
const tokensEl = document.getElementById("tokens");
const runBtn = document.getElementById("run");
const runOptBtn = document.getElementById("runOpt");

const LABELS = {
  original: "original kernels",
  mixed: "original LM head + optimized q4",
  optimized: "optimized kernels",
};
const labelOf = (v) => LABELS[v] ?? v;

const FALLBACK_IDS = [3838, 374, 279, 6722, 315, 9625, 30]; // "What is the capital of France?"
const setStatus = (t) => { statusEl.textContent = t; };

let tokenizerPromise = null;
async function getTokenizer() {
  if (!tokenizerPromise) {
    tokenizerPromise = (async () => {
      const { AutoTokenizer } = await import("@huggingface/transformers");
      return AutoTokenizer.from_pretrained("onnx-community/Qwen3-0.6B-ONNX");
    })().catch(() => null);
  }
  return tokenizerPromise;
}

async function encodePrompt(text) {
  const tok = await getTokenizer();
  if (!tok) return { ids: FALLBACK_IDS, tok: null, fellBack: true };
  const ids = Array.from(tok.encode(text));
  return { ids, tok, fellBack: false };
}

const liveEl = document.getElementById("live");

async function runVariant({ device, variant, inputIds, maxNewTokens, tok }) {
  const label = labelOf(variant);
  setStatus(`${label}: warming up…`);

  const panel = document.createElement("div");
  panel.className = "live-panel";
  panel.innerHTML = `<div class="live-head"><strong>${label}</strong>
      <span class="rate" data-rate>—</span></div><div class="out" data-text></div>`;
  liveEl.appendChild(panel);
  const textEl = panel.querySelector("[data-text]");
  const rateEl = panel.querySelector("[data-rate]");

  let seen = 0;
  let firstTokenAt = null;
  const started = performance.now();

  const result = await runCustomQwenGreedyPrompt({
    device,
    profile: qwenProfile,
    inputIds,
    maxNewTokens,
    variant,
    decodeToken: tok ? (ids) => tok.decode(ids) : undefined,
    onProgress: (m) => setStatus(`${label}: ${m}`),
    onToken: (piece) => {
      if (firstTokenAt === null) firstTokenAt = performance.now();
      seen += 1;
      textEl.textContent += piece;
      // rate over the decode loop only, excluding prefill + first-token latency
      const elapsed = performance.now() - firstTokenAt;
      const rate = seen > 1 ? (seen - 1) / (elapsed / 1000) : 0;
      rateEl.textContent = seen > 1
        ? `${rate.toFixed(2)} tok/s decode · ${(elapsed / (seen - 1)).toFixed(0)} ms/token`
        : `first token…`;
    },
  });

  // generateMs spans prefill + every decode step. Steady-state decode excludes
  // both prefill and the first token, which is what "tokens/sec" usually means.
  const decodeTokens = result.generatedTokens - 1;
  const steadyMsPerToken = decodeTokens > 0
    ? (result.generateMs - result.firstTokenMs) / decodeTokens
    : NaN;

  return {
    variant,
    ...result,
    msPerToken: result.generateMs / result.generatedTokens,
    steadyMsPerToken,
    totalMs: performance.now() - started,
  };
}

function render(rows, meta) {
  const idsMatch = rows.length < 2
    ? null
    : rows.every((r) => JSON.stringify(r.generated) === JSON.stringify(rows[0].generated));

  // With a single generated token there is no steady state to measure; say so
  // rather than quietly reporting the prefill-inclusive number under that name.
  const haveSteady = rows.every((r) => Number.isFinite(r.steadyMsPerToken));
  const usable = (r) => (haveSteady ? r.steadyMsPerToken : r.msPerToken);
  const speedup = rows.length >= 2 ? usable(rows[0]) / usable(rows[rows.length - 1]) : null;
  const speedupBasis = haveSteady ? "steady state" : "incl. prefill — run ≥2 tokens for a steady-state rate";

  const fmt = (n, d = 1) => (Number.isFinite(n)
    ? n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })
    : "—");

  resultsEl.innerHTML = `
    <table>
      <thead><tr>
        <th>Variant</th>
        <th>decode ms/token<br /><small>steady state</small></th>
        <th>decode tokens/sec</th>
        <th>ms/token<br /><small>incl. prefill</small></th>
        <th>first token (ms)</th>
        <th>total (ms)</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => `
          <tr>
            <td>${labelOf(r.variant)}</td>
            <td class="num">${fmt(r.steadyMsPerToken, 1)}</td>
            <td class="num">${fmt(1000 / r.steadyMsPerToken, 2)}</td>
            <td class="num">${fmt(r.msPerToken, 1)}</td>
            <td class="num">${fmt(r.firstTokenMs, 1)}</td>
            <td class="num">${fmt(r.totalMs, 0)}</td>
          </tr>`).join("")}
      </tbody>
    </table>
    <p style="font-size:.85rem;color:#888">
      Steady state excludes prefill and the first token. The prefill-inclusive column
      divides total generate time by tokens produced, so it looks worse at low token counts.
    </p>
    ${speedup ? `<p><strong>Speedup: ${fmt(speedup, 1)}×</strong> — ${labelOf(rows[rows.length - 1].variant)} vs ${labelOf(rows[0].variant)} (${speedupBasis})</p>` : ""}
    ${idsMatch === null ? "" : `<p>Token ids identical: <span class="${idsMatch ? "ok" : "bad"}">${idsMatch ? "yes — both paths agree" : "NO — outputs diverge"}</span></p>`}
    <p><strong>prompt</strong> (${meta.inputIds.length} tokens)${meta.fellBack ? " — tokenizer unavailable, used cached ids" : ""}:</p>
    <div class="out">${meta.promptText}</div>
    ${rows.map((r) => `
      <p><strong>${labelOf(r.variant)}</strong> generated ids <code>[${r.generated.join(", ")}]</code></p>
      <div class="out">${(r.outputText ?? "").replace(/</g, "&lt;") || "(no tokenizer)"}</div>`).join("")}
    <p style="font-size:.85rem;color:#888">
      Adapter: ${meta.adapter}. All variants run entirely on the GPU with the embedding
      table resident, so this is a kernel-for-kernel comparison.
    </p>`;
}

async function go(variants) {
  runBtn.disabled = runOptBtn.disabled = document.getElementById("runGpu").disabled = true;
  resultsEl.innerHTML = "";
  liveEl.innerHTML = "";
  try {
    setStatus("requesting WebGPU adapter…");
    const { adapter, device } = await createWebGpuContext();
    const adapterName = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(" ") || "unknown";

    // A rejected shader module yields an invalid pipeline, an invalid command
    // buffer, and silently all-zero output. Never let that pass as a result.
    const gpuErrors = [];
    device.addEventListener?.("uncapturederror", (e) => gpuErrors.push(e.error?.message ?? String(e)));

    setStatus("tokenizing…");
    const { ids: inputIds, tok, fellBack } = await encodePrompt(promptEl.value);

    // gqa_decode reduces over one 256-lane workgroup, so prompt + generation <= 256.
    const requested = Number.parseInt(tokensEl.value, 10);
    const headroom = 256 - inputIds.length;
    if (!Number.isFinite(requested) || requested < 1) {
      throw new Error(`enter a token count between 1 and ${headroom}`);
    }
    const maxNewTokens = Math.min(requested, headroom);
    if (maxNewTokens !== requested) setStatus(`clamped to ${maxNewTokens} tokens (prompt is ${inputIds.length})`);
    tokensEl.value = String(maxNewTokens);

    const rows = [];
    for (const variant of variants) {
      rows.push(await runVariant({ device, variant, inputIds, maxNewTokens, tok }));
      render(rows, { inputIds, promptText: promptEl.value, fellBack, adapter: adapterName });
    }
    if (gpuErrors.length) {
      resultsEl.insertAdjacentHTML("afterbegin",
        `<div class="warn"><strong class="bad">${gpuErrors.length} GPU error(s) — results below are meaningless:</strong>
         <div class="out">${gpuErrors.slice(0, 4).join("\n").replace(/</g, "&lt;")}</div></div>`);
      setStatus(`done, with ${gpuErrors.length} GPU error(s)`);
      return;
    }
    setStatus("done");
  } catch (err) {
    setStatus(`failed: ${err.message}`);
    console.error(err);
  } finally {
    runBtn.disabled = runOptBtn.disabled = document.getElementById("runGpu").disabled = false;
  }
}

const runGpuBtn = document.getElementById("runGpu");
runOptBtn.addEventListener("click", () => go(["optimized"]));
runGpuBtn.addEventListener("click", () => go(["original", "optimized"]));
runBtn.addEventListener("click", () => go(["original", "mixed", "optimized"]));
window.__runVariants = go;   // lets you try any mix from the console
setStatus(navigator.gpu ? "ready — WebGPU available" : "WebGPU unavailable in this browser");
