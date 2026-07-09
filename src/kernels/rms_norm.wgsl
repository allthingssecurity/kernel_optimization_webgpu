struct Dims {
  hidden_size: u32,
  epsilon_bits: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var sum = 0.0;

  for (var i = tid; i < dims.hidden_size; i = i + 256u) {
    let v = x[i];
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
  let inv_rms = inverseSqrt(partial[0] / f32(dims.hidden_size) + eps);
  for (var i = tid; i < dims.hidden_size; i = i + 256u) {
    y[i] = x[i] * inv_rms * weight[i];
  }
}
