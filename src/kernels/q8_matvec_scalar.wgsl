struct Dims {
  in_size: u32,
  out_size: u32,
  words_per_row: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

fn unpack_i8(word: u32, lane: u32) -> f32 {
  let raw = (word >> (lane * 8u)) & 0xffu;
  let signed = i32(raw) - select(0, 256, raw >= 128u);
  return f32(signed);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= dims.out_size) {
    return;
  }

  var acc = 0.0;
  let row_offset = row * dims.words_per_row;
  let scale = scales[row];

  for (var word_idx = 0u; word_idx < dims.words_per_row; word_idx = word_idx + 1u) {
    let packed = weights[row_offset + word_idx];
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
      let col = word_idx * 4u + lane;
      if (col < dims.in_size) {
        acc = acc + unpack_i8(packed, lane) * scale * x[col];
      }
    }
  }

  y[row] = acc;
}
