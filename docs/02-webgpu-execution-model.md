# 2. The WebGPU execution model

WebGPU is the browser's GPU API. Compute shaders are written in **WGSL**. This chapter
covers only what you need to read the kernels in `src/kernels/`.

## The hierarchy

```
dispatchWorkgroups(N)        <- launches N workgroups
  └── workgroup              <- @workgroup_size(K) threads, shares workgroup memory
        └── invocation       <- one thread (a "lane")
```

- `@builtin(workgroup_id) wid` — which workgroup am I? (`wid.x` in `0..N-1`)
- `@builtin(local_invocation_id) lid` — which lane within my workgroup? (`0..K-1`)

The common pattern for a matrix-vector product is **one workgroup per output row**,
with the workgroup's lanes cooperating across that row and then summing their partial
results.

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;       // this workgroup owns output row `row`
  let tid = lid.x;       // this lane is one of 64
```

## Workgroup memory and barriers

`var<workgroup>` declares memory shared by all lanes in a workgroup. Lanes do not run
in perfect lockstep, so writes are not visible to other lanes until you synchronize:

```wgsl
var<workgroup> partial: array<f32, 64>;

partial[tid] = acc;
workgroupBarrier();          // now every lane can see every other lane's write
```

A **tree reduction** halves the active lanes each step, costing `log2(K)` barriers:

```wgsl
var stride = 32u;
loop {
  if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
  workgroupBarrier();
  if (stride == 1u) { break; }
  stride = stride / 2u;
}
if (tid == 0u) { y[row] = partial[0]; }
```

**Important:** every lane in the workgroup must reach a `workgroupBarrier()`. An early
`return` before a barrier is only safe if it is *uniform* across the workgroup — which
is why `if (row >= dims.out_size) { return; }` is legal (`row` is derived from
`workgroup_id`, so it is the same for all lanes) but `if (tid > 10) { return; }` before
a barrier is not.

## Buffers

| Kind | WGSL | Use |
| --- | --- | --- |
| storage, read | `var<storage, read>` | weights, inputs |
| storage, read_write | `var<storage, read_write>` | outputs |
| uniform | `var<uniform>` | small constants (dims, positions) |

Bindings are declared with `@group(0) @binding(N)`, and `layout: "auto"` derives the
bind group layout from whichever bindings the shader *statically uses*.

## Limits that shape the design

These are the WebGPU **defaults**. An adapter may support more, but a device only gets
the defaults unless you explicitly request higher limits.

| Limit | Default | Consequence here |
| --- | --- | --- |
| `maxStorageBufferBindingSize` | 128 MiB | the 311 MB embedding table must be **chunked**; we use 5 chunks of 64 MiB |
| `maxBufferSize` | 256 MiB | same |
| `maxComputeInvocationsPerWorkgroup` | 256 | `@workgroup_size` cannot exceed 256 lanes portably |
| `maxComputeWorkgroupsPerDimension` | 65535 | a 32768-workgroup dispatch is fine; 151936 would not be |

This is why `lm_head_f16.wgsl` is invoked once per chunk rather than once over the
whole vocabulary.

## Dispatches, ordering, and implicit barriers

Within a single compute pass, dispatches execute **in order**, and WebGPU inserts an
implicit memory barrier between them. So this is correct without any manual
synchronization:

```js
pass.setPipeline(lmHeadPipeline);
for (const chunk of chunks) { pass.setBindGroup(0, chunk.bg); pass.dispatchWorkgroups(chunk.rows); }
pass.setPipeline(argmaxPipeline);   // sees all logits written above
pass.setBindGroup(0, argmaxBg);
pass.dispatchWorkgroups(256);
```

It is also why dispatches **serialize**. You cannot overlap independent dispatches in
a pass; each one drains before the next begins. Reducing dispatch count is therefore a
real optimization, and its value depends heavily on the implementation:

| Implementation | Measured per-dispatch cost |
| --- | --- |
| Deno / wgpu (Metal) | ~28 µs |
| Chrome-Brave / Dawn (Metal) | far cheaper (~4x, inferred from end-to-end) |

**`queue.writeBuffer` is ordered against `queue.submit`.** A `writeBuffer` issued after
a submit executes after that submit's commands complete. This is what lets prefill
enqueue many tokens that share the same uniform buffers without a GPU sync between
them.

## Reading results back

`mapAsync` is a full CPU↔GPU round-trip. Doing it 75 times per token, as the original
LM head did, costs far more than the compute. The optimized path reduces on the GPU
and reads back **8 bytes**.

## Two WGSL builtins that matter here

- `unpack2x16float(u32) -> vec2<f32>` — unpacks two f16 values from one `u32`. This is
  what lets the embedding table stay resident as f16 and be decoded in-shader. The
  low 16 bits become `.x`.
- `unpack4xU8(u32) -> vec4<u32>` — unpacks four bytes. Used in the (unlanded)
  `dot`-based q4 variant.

## Portability: wgpu is not Dawn

They are different implementations of the same spec and they disagree at the edges.

The literal `-3.4028235e38` is **above** `f32::MAX` (`3.40282346638528859811e38`).
wgpu silently rounds it and runs. Dawn rejects the shader module — correctly, per
spec. And a rejected module produces an invalid pipeline, then an invalid command
buffer, and then **silently writes nothing**, with no exception thrown anywhere.

Always attach an error handler, and never trust a benchmark that has not:

```js
device.addEventListener("uncapturederror", (e) => { /* fail loudly */ });
```

---

Next: [The investigation](03-the-investigation.md)
