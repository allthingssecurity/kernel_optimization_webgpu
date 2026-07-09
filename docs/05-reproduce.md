# 5. Reproducing this

## Prerequisites

- Node 18+
- A WebGPU browser. **Chrome, Edge, or Brave.** Safari 18+ also works.
- ~1.5 GB of disk for model artifacts (not committed to this repo)

## Setup

```bash
npm install
npm run fetch:qwen            # downloads onnx-community/Qwen3-0.6B-ONNX (~543 MB)
npm run extract:qwen:runtime  # extracts 28-layer q4 weights -> public/qwen/runtime
npm run dev
```

Then open **`http://127.0.0.1:5173/ab.html`**.

`public/qwen/` and `.cache/` are gitignored — `embed_tokens.f16` alone is 297 MB and
GitHub blocks files over 100 MB.

## Using the A/B page

Three buttons:

| Button | Runs | Use when |
| --- | --- | --- |
| **Run optimized** | `optimized` | long generations; ~150 tok/s |
| **Run A/B** | `original`, `optimized` | the headline kernel comparison |
| **Run all three** | + `mixed` | attribute the speedup to each rewrite |

Tokens stream live with a running tokens/sec counter. The page reports both a
steady-state rate (excluding prefill and first token) and a prefill-inclusive rate, and
verifies that all variants emit identical token ids.

Any mix can be run from the console:

```js
window.__runVariants(["original", "mixed", "optimized"])
```

### Variants

All three keep the embedding table resident on the GPU.

| Variant | q4 matvec | LM head |
| --- | --- | --- |
| `original` | original | original shaders: f32 weights, 75 chunk round-trips, host argmax |
| `mixed` | optimized | original shaders (isolates the matvec's contribution) |
| `optimized` | optimized | resident f16 + two-stage GPU argmax, one submit |

## A caveat that will waste your afternoon

**WebGPU requires a secure context.** `navigator.gpu` is `undefined` on `about:blank`.
If you probe for WebGPU support on a blank page you will conclude the browser does not
support it. Test against `http://127.0.0.1` (localhost is a secure origin).

This cost real time here: it produced a false "Brave has WebGPU compiled out"
conclusion, which pushed the whole investigation onto Deno — which is exactly why the
Dawn-only `f32` literal bug survived as long as it did.

Also note: Playwright's bundled Chromium genuinely does *not* ship WebGPU. Use a real
Chrome/Brave, or Chrome for Testing.

## Headless (Deno)

Deno ships a native WebGPU implementation (wgpu), which is useful for scripted
benchmarking without a browser:

```bash
deno run --unstable-webgpu --allow-all your_harness.js
```

The harness used for this project loaded `src/bench/webgpu-runtime.js` directly by
rewriting only its `?raw` shader imports and shimming `fetch` to read from `public/` —
so the *real* module was under test, not a reimplementation.

**Do not trust Deno alone.** wgpu and Dawn disagree. See
[the portability section](02-webgpu-execution-model.md#portability-wgpu-is-not-dawn).

## Benchmarking rules learned the hard way

1. **Take the minimum of ≥5 repeats.** Single-shot GPU timings on small dispatches vary
   by 2x and will invent speedups that do not exist.
2. **Defeat the cache.** Dispatching the same buffer repeatedly measures L2, not DRAM.
   Rotate through enough distinct buffers to exceed cache.
3. **Amortize submit overhead.** Batch dispatches into one compute pass, or you measure
   a ~170 µs submit floor instead of your kernel.
4. **Warm up.** The first few tokens are dominated by pipeline creation and JIT.
5. **Attach `uncapturederror`.** A rejected shader silently produces all-zero output and
   a beautiful, meaningless speedup.
6. **Isolate with no-ops.** To learn what a kernel actually costs, replace it with a
   no-op of identical dispatch geometry and re-measure the difference.

## Verifying correctness, not just speed

The bar used here: **all variants must emit identical token ids.** Timing a kernel that
produces the wrong answer is worse than useless. The A/B page checks this on every run
and prints `Token ids identical: yes/NO`.

Beyond that, kernels were checked against float64 CPU references on the real weights
(see [Results → Correctness](04-results.md#correctness)).
