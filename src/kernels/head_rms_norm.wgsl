struct Dims {
  head_dim: u32,
  heads: u32,
  epsilon_bits: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let head = wid.x;
  let tid = lid.x;
  if (head >= dims.heads) {
    return;
  }

  let base = head * dims.head_dim;
  var sum = 0.0;
  for (var i = tid; i < dims.head_dim; i = i + 256u) {
    let v = x[base + i];
    sum = sum + v * v;
  }
  partial[tid] = sum;
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

  let eps = bitcast<f32>(dims.epsilon_bits);
  let inv_rms = inverseSqrt(partial[0] / f32(dims.head_dim) + eps);
  for (var i = tid; i < dims.head_dim; i = i + 256u) {
    y[base + i] = x[base + i] * inv_rms * weight[i];
  }
}
