struct Dims {
  heads: u32,
  head_dim: u32,
  position: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = dims.heads * dims.head_dim;
  if (index >= total) {
    return;
  }

  let d = index % dims.head_dim;
  let half_dim = dims.head_dim / 2u;
  let pair_d = d % half_dim;
  let peer = select(index + half_dim, index - half_dim, d >= half_dim);
  let sign = select(-1.0, 1.0, d >= half_dim);
  let cache_offset = dims.position * dims.head_dim + pair_d;
  let c = cos_cache[cache_offset];
  let s = sin_cache[cache_offset];
  y[index] = x[index] * c + sign * x[peer] * s;
}
