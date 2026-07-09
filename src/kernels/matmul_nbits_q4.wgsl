struct Dims {
  in_size: u32,
  out_size: u32,
  blocks_per_row: u32,
  block_size: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const WG: u32 = 64u;

var<workgroup> partial: array<f32, 64>;

// Each u32 packs 8 nibbles. Nibble i (bits 4i..4i+3) holds the weight for
// column base + i, matching the ONNX MatMulNBits layout.
fn dot_word(packed: u32, base: u32) -> f32 {
  var sum = 0.0;
  sum = sum + f32(i32((packed >> 0u) & 0xFu) - 8) * x[base + 0u];
  sum = sum + f32(i32((packed >> 4u) & 0xFu) - 8) * x[base + 1u];
  sum = sum + f32(i32((packed >> 8u) & 0xFu) - 8) * x[base + 2u];
  sum = sum + f32(i32((packed >> 12u) & 0xFu) - 8) * x[base + 3u];
  sum = sum + f32(i32((packed >> 16u) & 0xFu) - 8) * x[base + 4u];
  sum = sum + f32(i32((packed >> 20u) & 0xFu) - 8) * x[base + 5u];
  sum = sum + f32(i32((packed >> 24u) & 0xFu) - 8) * x[base + 6u];
  sum = sum + f32(i32((packed >> 28u) & 0xFu) - 8) * x[base + 7u];
  return sum;
}

fn dot_word_tail(packed: u32, base: u32) -> f32 {
  var sum = 0.0;
  for (var i = 0u; i < 8u; i = i + 1u) {
    if (base + i < dims.in_size) {
      sum = sum + f32(i32((packed >> (4u * i)) & 0xFu) - 8) * x[base + i];
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= dims.out_size) {
    return;
  }

  let words_per_block = dims.block_size / 8u;
  let words_per_row = dims.blocks_per_row * words_per_block;
  let row_word_offset = row * words_per_row;
  let row_scale_offset = row * dims.blocks_per_row;

  // One packed word per lane, strided across the row. Adjacent lanes read
  // adjacent u32s so weight loads coalesce, and every lane carries work for
  // every projection shape (previously only blocks_per_row lanes did).
  var acc = 0.0;
  for (var w = tid; w < words_per_row; w = w + WG) {
    let packed = weights[row_word_offset + w];
    let scale = scales[row_scale_offset + w / words_per_block];
    let col = w * 8u;
    if (col + 8u <= dims.in_size) {
      acc = acc + dot_word(packed, col) * scale;
    } else {
      acc = acc + dot_word_tail(packed, col) * scale;
    }
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
    y[row] = partial[0];
  }
}
