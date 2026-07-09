# 4. Results and methodology

Every number here was produced against the real extracted Qwen3-0.6B q4 weights. No
synthetic tensors, no estimates. Where a number is uncertain or environment-specific,
it says so.

## Environment

| | |
| --- | --- |
| GPU | Apple Silicon, `metal-3` |
| Browser | Brave 146 (Chromium 146, Dawn) |
| Headless | Deno 2.8 (`--unstable-webgpu`, wgpu) |
| Model | `onnx-community/Qwen3-0.6B-ONNX`, q4f16 |
| Prompt | `"What is the capital of France?"` (7 tokens) |

"Steady state" = excludes prefill and first-token latency. That is what tokens/sec
conventionally means.

---

## Headline: kernel vs kernel

All three variants run **entirely on the GPU**, keep the embedding table resident, and
emit **identical token ids**. 24 generated tokens, Brave.

Each variant is measured in its **own fresh page**, median of 3 runs — see
[Measuring the variants in isolation](#measuring-the-variants-in-isolation).

| Variant | decode ms/token | tokens/sec |
| --- | --- | --- |
| original kernels | 45.8 | 21.8 |
| + optimized q4 matvec | 38.0 | 26.3 |
| + optimized LM head & GPU argmax | 7.0 | 142.9 |

- **q4 matvec rewrite: 1.21x**
- **LM head + argmax rewrite: 5.43x**
- **Combined: 6.54x**

An independent 4-run A/B of just the endpoints, fresh page each run, gave
6.20x / 6.35x / 6.20x / 6.46x — median **6.32x**. So 6.2–6.6x is the honest range.

### Measuring the variants in isolation

Running several variants inside one page is confounded: each allocates its own
resident embedding table (311 MB f16, or 622 MB f32 for the original LM head shader)
and never frees it, so later variants run under different memory pressure, and shader
pipelines are already warm. Doing that produced nonsense — one three-variant run had
`mixed` measuring *slower* than `original` (52.9 vs 42.6 ms/token), which is physically
impossible since `mixed` differs only by a strictly faster matvec.

An earlier version of this document reported **9.4x** from exactly that flawed setup.
It was an artifact of cold shader compilation inflating whichever variant ran first.
Each variant now gets a fresh page.

Nothing in this document is timed against a CPU fallback. Both sides of every
comparison execute on the device.

---

## q4 matvec: isolated microbenchmark

Real layer-0 weights, submit overhead amortized (all dispatches batched into one
pass), correctness checked against a float64 CPU reference each run.

| Projection | Shape | Old occupancy | Speedup | `maxAbsErr` |
| --- | --- | --- | --- | --- |
| `gate` | 1024 → 3072 | 32/256 (12.5%) | 1.85x | 1.6e-7 |
| `q` | 1024 → 2048 | 32/256 (12.5%) | 1.85x | 1.4e-7 |
| `down` | 3072 → 1024 | 96/256 (37.5%) | 1.21x | 1.2e-7 |
| `o` | 2048 → 1024 | 64/256 (25.0%) | 1.08x | 1.2e-7 |

Speedup tracks occupancy. `maxAbsErr` ~1e-7 is f32 rounding — the kernels agree.

### The kernel is ALU-bound, not bandwidth-bound

| Working set | µs/dispatch | GB/s |
| --- | --- | --- |
| same 1.97 MB buffer, 400x (cache-resident) | 54.7 | 35.9 |
| 40 distinct buffers, 79 MB (streams DRAM) | 59.6 | 33.0 |

Cold is only **1.09x** slower than hot. DRAM is not the limiter. The ~48 scalar ops per
`u32` of nibble unpacking are.

### Workgroup geometry barely matters

Min of 7 repeats × 300 iterations, summed over `gate`/`q`/`o`/`down`:

| Config | sum ms | GB/s |
| --- | --- | --- |
| current (64 lanes/row) | 0.264 | 24.8 |
| vec4, 2 lanes/row, wg=128 | 0.255 | 25.7 |
| vec4, 4 lanes/row, wg=128 | 0.255 | 25.7 |
| vec4, 4 lanes/row, wg=256 | **0.245** | 26.7 |
| vec4, 8 lanes/row, wg=256 | 0.262 | 25.0 |
| vec4, 16 lanes/row, wg=256 | 0.256 | 25.6 |

Everything is within 8%. **Single-shot runs of this same sweep reported 1.57x and
1.33x "wins" that do not exist.** Take the minimum over repeats.

### An ALU optimization that was not worth landing

Unpacking four nibbles at once with `unpack4xU8` + `dot`, against an `x` split into
even/odd lanes:

| Kernel | µs/dispatch (cold) | GB/s |
| --- | --- | --- |
| current (scalar nibble unpack) | 56.4 | 34.8 |
| `unpack4xU8` + `dot` | 45.4 | 43.3 |

**1.24x**, verified correct (`maxAbsErr` 2.8e-7). Not wired up: it requires
`rms_norm`, `silu_mul`, and `gqa_decode` to emit de-interleaved output, and 1.24x on
~15% of a token is ~3%.

---

## Where the time goes now

Swap `matmul_nbits_q4` for a no-op with **identical dispatch geometry** (Deno):

| Configuration | ms/token |
| --- | --- |
| full decode | 28.9 |
| layers only (no LM head) | 26.6 |
| layers only, matvec is a no-op | 22.4 |

- q4 matvec compute: **4.2 ms** (~15% of a token)
- LM head + argmax: **~1.3 ms**
- everything else — 532 serialized dispatches: **~22 ms**

### Dispatch cost, and why it is implementation-specific

| Measurement | Deno / wgpu |
| --- | --- |
| trivial dispatch, WAW dependency | 9.7 µs |
| trivial dispatch, no dependency | 8.7 µs |
| trivial dispatch, 3072 workgroups | 17.7 µs |
| CPU encode of 532 dispatches | 0.31 ms |

This motivated a "fuse 19 dispatches per layer down to ~9" recommendation. **The
browser then undercut it:** Dawn runs the same work at 6.5 ms/token where wgpu takes
~28 ms. The per-dispatch cost is largely a wgpu artifact, so dispatch fusion is worth
far less than the Deno numbers implied. Re-measure under Dawn before investing.

---

## Correctness

Every optimization was verified, not assumed.

| Check | Result |
| --- | --- |
| q4 nibble→column mapping vs original, float64, real weights | worst rel. diff `5e-14` |
| q4 GPU output vs float64 CPU reference | `maxAbsErr` 1.2–1.6e-7 (f32 rounding) |
| `unpack2x16float` packing vs direct f16 read | exact |
| LM head logit, resident f16 kernel vs float64 reference | rel. diff `0.0` |
| two-stage GPU argmax vs sequential CPU scan | matches on 6 adversarial cases incl. all-ties, winner-at-last-index, duplicate maxima |
| all variants, end-to-end | **identical token ids** |

Generated output, greedy, `"What is the capital of France?"` →
`" Also, what is the capital of the"`. Grammatical English: the whole stack
(embedding gather, RoPE, GQA, KV cache, q4 matvec, LM head, argmax) is semantically
correct, not merely finite. It continues rather than answers because this is a base
model with no chat template, and it eventually loops because sampling is pure greedy
argmax with no repetition penalty.

---

Next: [Reproducing this](05-reproduce.md)
