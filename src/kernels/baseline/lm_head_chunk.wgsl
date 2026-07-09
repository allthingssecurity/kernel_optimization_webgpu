struct Dims {
  hidden_size: u32,
  chunk_rows: u32,
  row_offset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> embed_chunk: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= dims.chunk_rows) {
    return;
  }

  var acc = 0.0;
  let row_base = row * dims.hidden_size;
  for (var i = tid; i < dims.hidden_size; i = i + 256u) {
    acc = acc + hidden[i] * embed_chunk[row_base + i];
  }
  partial[tid] = acc;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (tid < stride) {
      partial[tid] = partial[tid] + partial[tid + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (tid == 0u) {
    logits[row] = partial[0];
  }
}
