# 3. The investigation, step by step

The starting point: a complete, correct, custom WebGPU decode stack for Qwen3-0.6B —
embedding gather, RoPE, grouped-query attention with a KV cache, q4 `MatMulNBits`
projections, RMSNorm, SiLU, residuals, LM head, greedy sampler. All 28 layers. It
produced grammatical English. It was also slow.

Every number below is GPU-to-GPU. The embedding table is resident on the device in all
variants; the variants differ only in which kernels consume it.

---

## Step 0: Arithmetic before code

Before reading a single shader, count bytes. Qwen3-0.6B:

```
28 layers x 15.7M params x 0.5 bytes (q4)  =  220 MB
+ scales (f32, one per 32-weight block)    =   55 MB
                                    total  =  275 MB per token
```

Now the embedding table. Qwen3 **ties** its input embedding and output projection, so
one tensor serves both:

```
151936 x 1024 x 2 bytes (f16)  =  311 MB
```

**The output projection alone is bigger than the entire transformer.** That single fact
determines where the time goes, and it is available before you open an editor.

---

## Step 1: Find the actual bottleneck

The instinct is to open the matvec — it is the hot loop, it is where the FLOPs are, it
is what "kernel optimization" evokes. Resist it and measure instead.

The decisive technique: **replace a kernel with a no-op of identical dispatch geometry
and re-measure.** Whatever the time drops by is what the kernel actually cost.

| Configuration | ms/token |
| --- | --- |
| full decode | 28.9 |
| layers only (no LM head) | 26.6 |
| layers only, matvec replaced by a no-op | 22.4 |

Read that carefully. **All seven q4 matvecs, across all 28 layers, account for 4.2 ms of
a 28.9 ms token — about 15%.** Even an infinitely fast matvec buys 1.17x.

The LM head — which reads more bytes than the whole transformer, and which looks like a
boring epilogue — was the target all along.

---

## Optimization 1: The LM head kernel (5.43x)

The original LM head, with the table already resident on the GPU, still did three
expensive things per token:

1. **Read the table as f32.** Twice the bytes of f16, for weights that were f16 on disk.
2. **75 chunks, 75 round-trips.** Each chunk was its own `submit` +
   `onSubmittedWorkDone()` + `mapAsync` readback. A `mapAsync` is a full CPU↔GPU
   round-trip and it happened 75 times per token.
3. **Argmax on the host.** All 151,936 logits were copied back and scanned.

### The fix

**Store f16, unpack in-shader.** `unpack2x16float` decodes two f16 values from one
`u32`, halving the bytes read:

```wgsl
let pair = unpack2x16float(embed[row_word_offset + w]);
acc = acc + pair.x * hidden[col] + pair.y * hidden[col + 1u];
```

The table is chunked into 5 buffers of 64 MiB — under the 128 MiB default
`maxStorageBufferBindingSize` — and never touched again after setup.

**Argmax on the GPU.** Two stages ([`argmax_stage1.wgsl`](../src/kernels/argmax_stage1.wgsl),
[`argmax_stage2.wgsl`](../src/kernels/argmax_stage2.wgsl)): 256 workgroups each reduce a
strided slice to a `(value, index)` pair, then one workgroup reduces those 256 pairs to
one. Ties break on the lower index, matching a sequential scan.

**One submit.** All 5 chunk dispatches plus both argmax stages go into a single compute
pass. Exactly **8 bytes** come back per token, instead of 151,936 floats.

Result: the LM head drops to **~1.3 ms/token** — roughly 240 GB/s for a 311 MB read,
near the memory-bandwidth floor. Measured end-to-end contribution: **5.43x**
(38.0 → 7.0 ms/token).

---

## Optimization 2: Un-starve the q4 matvec (1.21x)

The original parallelized over quantization **blocks**:

```wgsl
@compute @workgroup_size(256)
for (var block = tid; block < dims.blocks_per_row; block = block + 256u) { ... }
```

A 1024-wide projection has `blocks_per_row = 32`. So **32 of 256 lanes** entered the
loop; the other 224 contributed `0.0` and then sat through a 6-step barrier reduction.

| Projection | Shape | Lanes active |
| --- | --- | --- |
| `gate`, `up` | 1024 → 3072 | 32/256 (12.5%) |
| `q` | 1024 → 2048 | 32/256 (12.5%) |
| `o` | 2048 → 1024 | 64/256 (25.0%) |
| `down` | 3072 → 1024 | 96/256 (37.5%) |

The rewrite parallelizes over **words**. Each row is `in_size / 8` packed `u32`s (128,
256, or 384 of them), so with 64 lanes every lane always has work, and adjacent lanes
read adjacent `u32`s — coalesced.

```wgsl
@compute @workgroup_size(64)
for (var w = tid; w < words_per_row; w = w + 64u) {
  let packed = weights[row_word_offset + w];
  let scale = scales[row_scale_offset + w / words_per_block];
  acc = acc + dot_word(packed, w * 8u) * scale;
}
```

The nibble→column mapping is the subtle part. In the packed format, nibble `i` of a
`u32` (bits `4i..4i+3`) holds the weight for column `base + i`. This was verified
against the original indexing on the real weights in float64: worst relative difference
`5e-14` across all four projection shapes — the same multiset of products, differing
only by floating-point reassociation.

Isolated per-projection speedups track occupancy almost exactly:

| Projection | Old occupancy | Speedup |
| --- | --- | --- |
| `gate` | 12.5% | 1.85x |
| `q` | 12.5% | 1.85x |
| `down` | 37.5% | 1.21x |
| `o` | 25.0% | 1.08x |

End-to-end contribution: **1.21x** (45.8 → 38.0 ms/token) in the browser. Isolated
against the layer stack alone under a different WebGPU implementation it measures
1.47x — the whole-token figure is lower because the matvec is only part of a token.

---

## Optimization 3: Per-token host overhead

Rope position, KV-cache offset, and GQA sequence length are the same for all 28 layers.
They were being written as 5 uniform buffers *per layer*: **140 `writeBuffer` calls per
token**. Now one set is shared and written 4 times.

Prefill also awaited `onSubmittedWorkDone()` after every prompt token. Since
`queue.writeBuffer` is ordered against `queue.submit`, tokens can be enqueued
back-to-back and drained periodically instead.

Small next to the other two, but free.

---

## Where the time goes now

From the no-op decomposition above, per 28.9 ms token:

- q4 matvec compute: **4.2 ms** (~15%)
- LM head + argmax: **~1.3 ms**
- everything else — 532 serialized dispatches (19 per layer): **~22 ms**

WebGPU inserts an implicit barrier between dispatches in a pass, so they serialize.
That suggests fusing dispatches (concatenate q/k/v into one matvec, gate/up into
another, fold `rope` into `cache_write`, fold the residual add into the matvec
epilogue) — 19 per layer down to roughly 9.

**But measure before you build it.** The per-dispatch cost is highly
implementation-specific: ~28 µs under wgpu, several times cheaper under Dawn. The same
decode stack runs at ~7 ms/token in a browser and ~28 ms under Deno. Most of the case
for fusion evaporates on Dawn.

---

## Five things I got wrong

Recorded because the mistakes are more instructive than the fixes.

### 1. Two "optimizations" that were noise

Sweeping workgroup geometry (lanes-per-row × rows-per-workgroup, 1×256 through 64×1)
reported a **1.57x** win, then on a re-run a **1.33x** win — for a *different* best
configuration. Both were noise. Single-shot dispatch timings on these kernels vary by
2x. Taking the **minimum of 7 repeats**, every geometry landed within 8% of every other,
and the current kernel measured 0.264 ms where the noisy run had said 0.496 ms.

I nearly landed a 1.57x optimization that did not exist.

### 2. Benchmarking the cache, not the memory

The microbenchmark dispatched the same 1.97 MB weight buffer 300 times. That is
cache-resident. Rerun with 40 distinct buffers (79 MB working set) to force DRAM
traffic: **1.09x slower**. So the kernel is not bandwidth-bound at all — it is ALU-bound
on nibble unpacking (~48 scalar ops per `u32`). That reframed the entire optimization
target.

### 3. Declaring the q4 rewrite worthless

An early end-to-end A/B showed 46.3 ms/token (optimized) vs 45.9 (original kernel) — no
difference. I was about to report it as a wash and revert. Those were 8-token runs
dominated by warmup. At 12 tokens in steady state the rewrite is worth 1.47x, and
layers-only isolates it at 43.9 → 27.6 ms.

### 4. Benchmarking three variants in one page

Each variant allocates its own resident embedding table — 311 MB as f16, or 622 MB as
f32 for the original LM head shader — and never frees it. Run them back to back in one
page and later variants execute under different memory pressure with already-warm
shader pipelines.

The tell was physically impossible output: one run measured `mixed` **slower** than
`original` (52.9 vs 42.6 ms/token), even though `mixed` differs from `original` only by
a strictly faster matvec. Cold shader compilation was inflating whichever variant ran
first.

An earlier version of this writeup reported **9.4x** on the strength of that setup. The
real figure, with each variant in its own fresh page and a median over 3 runs, is
**6.5x**. Isolate your variants.

### 5. A one-character bug no amount of single-implementation testing could find

```wgsl
const NEG_INF: f32 = -3.4028235e38;
```

`3.4028235e38` exceeds `f32::MAX` = `3.40282346638528859811e38`. wgpu rounds it and
runs. Dawn rejects the module. A rejected module ⇒ invalid pipeline ⇒ invalid command
buffer ⇒ **the LM head never runs, every logit is zero, argmax returns token 0, and
nothing throws.**

The first browser run reported a triumphant 267x speedup while generating `!!`.

The benchmark page now installs an `uncapturederror` handler and refuses to print
timings if any GPU error fired.

---

Next: [Results and methodology](04-results.md)
