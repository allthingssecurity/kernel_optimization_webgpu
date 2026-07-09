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

var<workgroup> partial: array<f32, 256>;

fn unpack_u4(byte_value: u32, high: bool) -> f32 {
  let raw = select(byte_value & 0xFu, (byte_value >> 4u) & 0xFu, high);
  return f32(i32(raw) - 8);
}

fn byte_from_word(word: u32, byte_lane: u32) -> u32 {
  return (word >> (byte_lane * 8u)) & 0xffu;
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= dims.out_size) {
    return;
  }

  var acc = 0.0;
  let words_per_block = dims.block_size / 8u;
  let row_word_offset = row * dims.blocks_per_row * words_per_block;
  let row_scale_offset = row * dims.blocks_per_row;

  for (var block = tid; block < dims.blocks_per_row; block = block + 256u) {
    let scale = scales[row_scale_offset + block];
    let col_base = block * dims.block_size;
    let block_word_offset = row_word_offset + block * words_per_block;

    for (var word_idx = 0u; word_idx < words_per_block; word_idx = word_idx + 1u) {
      let packed_word = weights[block_word_offset + word_idx];
      for (var byte_lane = 0u; byte_lane < 4u; byte_lane = byte_lane + 1u) {
        let packed_byte = byte_from_word(packed_word, byte_lane);
        let pair_base = col_base + word_idx * 8u + byte_lane * 2u;
        if (pair_base < dims.in_size) {
          acc = acc + unpack_u4(packed_byte, false) * scale * x[pair_base];
        }
        if (pair_base + 1u < dims.in_size) {
          acc = acc + unpack_u4(packed_byte, true) * scale * x[pair_base + 1u];
        }
      }
    }
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
    y[row] = partial[0];
  }
}
