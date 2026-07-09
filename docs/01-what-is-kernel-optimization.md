# 1. What is kernel optimization?

A **kernel** is a small program that runs on the GPU, executed simultaneously by
thousands of threads. In an LLM, essentially all the arithmetic lives in a handful of
kernels: matrix-vector multiplies, normalizations, attention, activation functions.

**Kernel optimization** is making those programs run closer to what the hardware is
physically capable of. It is not "writing cleverer math" — the math is fixed. It is
about feeding the machine efficiently.

## The only three things that limit a kernel

Every kernel is limited by one of:

1. **Memory bandwidth** — you cannot compute faster than you can read the operands.
2. **Compute (ALU) throughput** — you are doing too many arithmetic operations per byte.
3. **Latency / occupancy** — you are not keeping enough work in flight to hide the
   time it takes memory to arrive, so the GPU idles.

Optimizing without knowing *which one* you are hitting is guesswork. Most wasted
effort in GPU work comes from optimizing for the wrong limiter.

## Arithmetic intensity and the roofline

**Arithmetic intensity** = FLOPs performed ÷ bytes read from memory.

Single-token LLM decode has *terrible* arithmetic intensity. To compute one output
row of a matrix-vector product you read an entire row of weights and use each weight
exactly once. There is no reuse. Decode is therefore **memory-bandwidth bound** by
nature — the weights dominate, and the only real lever is reading fewer bytes or
reading them more efficiently.

This is why quantization exists. Qwen3-0.6B here stores weights as 4-bit integers
(`MatMulNBits`), so reading them costs one eighth of what f32 would.

Compare against the roofline. This project's 28 decoder layers read 275 MB of weights
and scales per token:

| Achieved | Time for 275 MB | Verdict |
| --- | --- | --- |
| 10 GB/s | 27.6 ms | 10–40x off the hardware |
| 100 GB/s | 2.75 ms | plausible target |
| 200 GB/s | 1.38 ms | good |

When we measured the LM head hitting ~240 GB/s (311 MB in 1.3 ms) on the *same GPU*
where the matvec managed 10 GB/s, that discrepancy was the entire diagnosis. Same
hardware, same memory: the matvec was badly shaped.

## What "badly shaped" means in practice

### Occupancy — are all your lanes working?

A GPU executes threads in lockstep groups. If your kernel assigns work such that only
32 of 256 threads in a workgroup have anything to do, you have thrown away 87.5% of
your parallelism and you still pay the full cost of the synchronization barriers.

This was the exact bug in the original `matmul_nbits_q4.wgsl`. See
[the investigation](03-the-investigation.md#optimization-2).

### Coalescing — are adjacent lanes reading adjacent memory?

Memory is delivered in wide cache lines. If lane 0 reads byte 0, lane 1 reads byte 4,
lane 2 reads byte 8 — one transaction serves everybody. If lane 0 reads byte 0 and
lane 1 reads byte 512, you issue a separate transaction per lane and waste most of
each one.

### Reduction depth — how much do you pay to combine results?

When N lanes cooperate on one output value, they must sum their partial results. A
tree reduction over N lanes costs `log2(N)` barriers. If each lane only did 2 loads,
the barriers dominate the loads and you are paying more to coordinate than to compute.

The fix is to give each lane more work (fewer lanes per output, more loads each) so
the fixed reduction cost amortizes.

### Dispatch count — how many times do you start the GPU?

Every `dispatchWorkgroups` call has a fixed cost, and WebGPU inserts an implicit
barrier between dispatches in a pass, serializing them. A layer built from 19 tiny
dispatches can spend more time starting and stopping than computing.

## The most important rule

**Profile first. Then optimize what is actually slow.**

This sounds obvious, and it is, and it was still the single biggest mistake made in
this project. The q4 matvec kernel — the thing that *looks* like the hot loop, the
thing any experienced GPU engineer would reach for first — turned out to be about 15%
of a token. The real cost sat in the LM head, which reads more bytes than the entire
transformer and looks like a boring epilogue.

The technique that settles this is simple: **replace the kernel with a no-op of
identical dispatch geometry and re-measure.** Whatever the time drops by is what the
kernel actually costs. Everything else is somewhere else.

---

Next: [The WebGPU execution model](02-webgpu-execution-model.md)
