struct Dims {
  count: u32,
  groups: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> best_value: array<f32>;
@group(0) @binding(2) var<storage, read_write> best_index: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const WG: u32 = 256u;
// Must stay within f32 range: 3.4028235e38 rounds above f32::MAX and Dawn
// rejects the module outright (wgpu silently accepts it).
const NEG_INF: f32 = -3.4028234e38;

var<workgroup> shared_value: array<f32, 256>;
var<workgroup> shared_index: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let group = wid.x;
  let tid = lid.x;
  let stride_span = dims.groups * WG;

  var value = NEG_INF;
  var index = 0u;
  for (var i = group * WG + tid; i < dims.count; i = i + stride_span) {
    let candidate = logits[i];
    if (candidate > value) {
      value = candidate;
      index = i;
    }
  }

  shared_value[tid] = value;
  shared_index[tid] = index;
  workgroupBarrier();

  var stride = WG / 2u;
  loop {
    if (tid < stride) {
      let other_value = shared_value[tid + stride];
      let other_index = shared_index[tid + stride];
      // Lanes hold strided ranges, so a lower lane is not a lower vocab index.
      // Tie-break on index so the result matches a sequential CPU scan.
      let wins = other_value > shared_value[tid]
        || (other_value == shared_value[tid] && other_index < shared_index[tid]);
      if (wins) {
        shared_value[tid] = other_value;
        shared_index[tid] = other_index;
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (tid == 0u) {
    best_value[group] = shared_value[0];
    best_index[group] = shared_index[0];
  }
}
