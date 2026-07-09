struct Dims {
  groups: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> best_value: array<f32>;
@group(0) @binding(1) var<storage, read> best_index: array<u32>;
// result[0] = argmax token id, result[1] = f32 bits of the winning logit.
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const WG: u32 = 256u;
// Must stay within f32 range: 3.4028235e38 rounds above f32::MAX and Dawn
// rejects the module outright (wgpu silently accepts it).
const NEG_INF: f32 = -3.4028234e38;

var<workgroup> shared_value: array<f32, 256>;
var<workgroup> shared_index: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  if (tid < dims.groups) {
    shared_value[tid] = best_value[tid];
    shared_index[tid] = best_index[tid];
  } else {
    shared_value[tid] = NEG_INF;
    shared_index[tid] = 0xffffffffu;
  }
  workgroupBarrier();

  var stride = WG / 2u;
  loop {
    if (tid < stride) {
      let other_value = shared_value[tid + stride];
      let other_index = shared_index[tid + stride];
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
    result[0] = shared_index[0];
    result[1] = bitcast<u32>(shared_value[0]);
  }
}
