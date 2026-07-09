struct Dims {
  in_size: u32,
  out_size: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= dims.out_size) {
    return;
  }

  var acc = 0.0;
  let row_offset = row * dims.in_size;
  for (var col = 0u; col < dims.in_size; col = col + 1u) {
    acc = acc + weights[row_offset + col] * x[col];
  }

  y[row] = acc;
}
