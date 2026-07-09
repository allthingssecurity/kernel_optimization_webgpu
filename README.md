# Optimizing WebGPU Kernels for Qwen3-0.6B

A worked, measured case study in GPU kernel optimization: making a working but slow
browser-side inference stack for [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
**6.5x faster**, kernel for kernel, on real hardware against the real model weights.

Both sides of that comparison run entirely on the GPU, share the same decode stack,
and emit byte-identical token ids. The measurement methodology is written down —
including the parts where the measurements initially lied to me.

---

## Results

Greedy decode, prompt `"What is the capital of France?"`, 24 generated tokens,
steady state (excluding prefill and first-token latency). Brave 146, Apple GPU
(metal-3). Each variant measured in its **own fresh page**, median of 3 runs.
**Every variant emits byte-identical token ids.**

| Variant | decode ms/token | tokens/sec | notes |
| --- | --- | --- | --- |
| Original kernels | 45.8 | 21.8 | baseline |
| + optimized q4 matvec | 38.0 | 26.3 | **1.21x** from the matvec rewrite |
| + optimized LM head & GPU argmax | 7.0 | 142.9 | **5.43x** from the LM head rewrite |

**6.5x, kernel for kernel.** An independent 4-run A/B of just the endpoints gave
6.20–6.46x, so the number is stable to within a few percent.

Every variant runs entirely on the GPU with the embedding table resident. Nothing here
is timed against a CPU fallback.

See [`docs/04-results.md`](docs/04-results.md) for the full tables, the
per-projection microbenchmarks, and the run-to-run variance.

---

## Documentation

Read in order:

1. **[What is kernel optimization?](docs/01-what-is-kernel-optimization.md)**
   Roofline thinking, arithmetic intensity, why "make the kernel faster" is usually
   the wrong first question.
2. **[The WebGPU execution model](docs/02-webgpu-execution-model.md)**
   Workgroups, lanes, barriers, dispatches, storage buffers, and the specific
   constraints (128 MiB bindings, 256 invocations/workgroup) that shape the design.
3. **[The investigation, step by step](docs/03-the-investigation.md)**
   What was actually slow, how it was found, the three optimizations, and the
   several confident conclusions that turned out to be measurement noise.
4. **[Results and methodology](docs/04-results.md)**
   Every number, how it was produced, and how to reproduce it.
5. **[Reproducing this](docs/05-reproduce.md)**
   Fetching weights, running the browser A/B page, running the headless harness.

---

## The three optimizations

### 1. The LM head kernel (5.43x)

Qwen3 ties its embedding and output projection. That table is `151936 x 1024` — at
f16, **311 MB, larger than all 28 transformer layers of q4 weights combined (275 MB)**.

Even with the table resident on the GPU, the original LM head:

- stored it as **f32**, doubling the bytes read per token
- split it into 75 chunks, each its own **submit + `mapAsync` readback** — 75 GPU
  round-trips per token
- scanned all 151,936 logits on the **CPU** to find the argmax

**Fix:** keep the table as f16 and unpack in-shader with `unpack2x16float`
([`lm_head_f16.wgsl`](src/kernels/lm_head_f16.wgsl)); reduce the argmax on the GPU
([`argmax_stage1.wgsl`](src/kernels/argmax_stage1.wgsl),
[`argmax_stage2.wgsl`](src/kernels/argmax_stage2.wgsl)) so **8 bytes** come back per
token instead of 151,936 floats; and issue the whole thing as **one submit**.

The LM head drops to ~1.3 ms/token — roughly 240 GB/s for a 311 MB read, near the
memory-bandwidth floor.

### 2. The q4 matvec was lane-starved (1.21x)

[`matmul_nbits_q4.wgsl`](src/kernels/matmul_nbits_q4.wgsl) parallelized over quantization
*blocks*. A 1024-wide projection has only 32 blocks per row, so **32 of 256 lanes did
work** and 224 idled through a 6-step barrier reduction.

| Projection | Shape | Old occupancy |
| --- | --- | --- |
| `gate`, `up` | 1024 → 3072 | 32/256 (12.5%) |
| `q` | 1024 → 2048 | 32/256 (12.5%) |
| `o` | 2048 → 1024 | 64/256 (25.0%) |
| `down` | 3072 → 1024 | 96/256 (37.5%) |

**Fix:** assign one packed `u32` (8 weights) per lane instead. Every lane carries work
for every projection shape, and adjacent lanes read adjacent `u32`s so the weight
loads coalesce.

### 3. Per-token host overhead

Rope position, KV-cache offset and GQA sequence length are identical across all 28
layers, but were written as 5 uniform buffers **per layer** — 140 `writeBuffer` calls
per token. They are now written once (4 calls). Prefill also queued a GPU sync after
every prompt token; it now enqueues and drains periodically.

---

## What this repository contains

```
src/kernels/            the WGSL kernels
  matmul_nbits_q4.wgsl    q4 MatMulNBits matvec (optimized)
  lm_head_f16.wgsl        resident f16 LM head
  argmax_stage1/2.wgsl    two-stage GPU argmax
  rms_norm, rope, gqa_decode, silu_mul, residual_add, cache_write, head_rms_norm
  baseline/               the ORIGINAL kernels, kept so the A/B page can measure them
src/bench/
  webgpu-runtime.js       the decode stack; `variant` selects which kernels run
  ab-page.js              the browser benchmark page
ab.html                 open this to run the A/B yourself
scripts/                ONNX fetch + weight extraction
docs/                   the writeup
```

## Quick start

```bash
npm install
npm run fetch:qwen          # downloads the Qwen3-0.6B ONNX artifact (~543 MB)
npm run extract:qwen:runtime # extracts 28-layer q4 weights into public/qwen
npm run dev
```

Open `http://127.0.0.1:5173/ab.html` and press **Run A/B**. Model weights are
not committed — `embed_tokens.f16` alone is 297 MB.

Full instructions, including the headless Deno harness, in
[`docs/05-reproduce.md`](docs/05-reproduce.md).

---

## Four lessons that cost real time

**Single-shot GPU timings lie.** Two separate workgroup-geometry sweeps reported
1.57x and 1.33x wins. Taking the minimum of 7 repeats, every geometry landed within
8% of every other. Both "wins" were noise. Always take a min over repeats.

**Test on more than one WebGPU implementation.** My argmax kernels declared
`const NEG_INF: f32 = -3.4028235e38`. That literal is *above* f32's maximum
(`3.40282346...e38`). Deno's wgpu silently rounds it; Dawn (Chrome/Brave) correctly
rejects the module — yielding an invalid pipeline, an invalid command buffer, and
then **silently all-zero output with no exception thrown**. The benchmark reported a
triumphant 267x speedup while generating nothing but token 0.

**Measure where the time is before optimizing.** Replacing the q4 matvec with a no-op
of identical dispatch geometry only moved the layer stack from 26.6 to 22.4 ms/token.
All seven matvecs across all 28 layers were ~15% of a token. I had already spent hours
tuning them. The LM head, which looks like a boring epilogue, was the real target.

**Isolate your variants.** Benchmarking all three kernel sets in one page leaves each
one's 311–622 MB resident table allocated and its pipelines warm, so the first variant
measured is penalised. That produced a physically impossible result (the mixed variant
appearing slower than the original it strictly improves on) and an inflated 9.4x
headline. Measured one variant per fresh page, median of 3, it is 6.5x.

---

## Honest limitations

- `gqa_decode.wgsl` reduces over a single 256-lane workgroup, so prompt + generation
  is capped at 256 tokens. Lifting it needs an online-softmax rewrite.
- Prefill replays the prompt one token at a time through the decode path. There is no
  batched prefill kernel.
- Sampling is pure greedy argmax. On a 0.6B base model with no chat template the
  output continues your prompt and eventually loops. That is the sampler, not the stack.
- An `unpack4xU8` + `dot` variant of the q4 matvec measured 1.24x faster cold, but is
  **not wired up**: it requires every producer of a matvec input to emit de-interleaved
  output, and 1.24x on ~15% of a token is ~3%.

## License

MIT
