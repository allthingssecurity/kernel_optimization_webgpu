struct Dims {
  hidden_size: u32,
  rows: u32,
  row_offset: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
// Tied embedding rows, kept resident on the GPU as raw f16 pairs packed into
// u32 words. Two vocab columns per word.
@group(0) @binding(1) var<storage, read> embed: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const WG: u32 = 64u;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= dims.rows) {
    return;
  }

  let words_per_row = dims.hidden_size / 2u;
  let row_word_offset = row * words_per_row;

  var acc = 0.0;
  for (var w = tid; w < words_per_row; w = w + WG) {
    let pair = unpack2x16float(embed[row_word_offset + w]);
    let col = w * 2u;
    acc = acc + pair.x * hidden[col] + pair.y * hidden[col + 1u];
  }

  partial[tid] = acc;
  workgroupBarrier();

  var stride = WG / 2u;
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
    logits[dims.row_offset + row] = partial[0];
  }
}
