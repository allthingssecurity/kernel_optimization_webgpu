struct Dims {
  seq_len: u32,
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> scores: array<f32, 256>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let q_head = wid.x;
  let tid = lid.x;
  if (q_head >= dims.q_heads || dims.seq_len > 256u) {
    return;
  }

  let heads_per_kv = dims.q_heads / dims.kv_heads;
  let kv_head = q_head / heads_per_kv;
  let scale = inverseSqrt(f32(dims.head_dim));

  if (tid < dims.seq_len) {
    var dot = 0.0;
    let q_offset = q_head * dims.head_dim;
    let k_offset = (tid * dims.kv_heads + kv_head) * dims.head_dim;
    for (var d = 0u; d < dims.head_dim; d = d + 1u) {
      dot = dot + q[q_offset + d] * k_cache[k_offset + d];
    }
    scores[tid] = dot * scale;
    scratch[tid] = scores[tid];
  } else {
    scratch[tid] = -3.402823e38;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (tid < stride) {
      scratch[tid] = max(scratch[tid], scratch[tid + stride]);
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }
  let max_score = scratch[0];

  if (tid < dims.seq_len) {
    let e = exp(scores[tid] - max_score);
    scores[tid] = e;
    scratch[tid] = e;
  } else {
    scratch[tid] = 0.0;
  }
  workgroupBarrier();

  stride = 128u;
  loop {
    if (tid < stride) {
      scratch[tid] = scratch[tid] + scratch[tid + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }
  let inv_sum = 1.0 / scratch[0];

  if (tid < dims.head_dim) {
    var acc = 0.0;
    for (var pos = 0u; pos < dims.seq_len; pos = pos + 1u) {
      let weight = scores[pos] * inv_sum;
      let v_offset = (pos * dims.kv_heads + kv_head) * dims.head_dim;
      acc = acc + weight * v_cache[v_offset + tid];
    }
    y[q_head * dims.head_dim + tid] = acc;
  }
}
