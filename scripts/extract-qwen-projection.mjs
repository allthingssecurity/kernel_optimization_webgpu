import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import onnxProto from "onnx-proto";

const input = process.argv[2] ?? ".cache/qwen3-0.6b-onnx/onnx/model_q4f16.onnx";
const prefix = process.argv[3] ?? "model.layers.0.mlp.gate_proj.MatMul";
const outDir = process.argv[4] ?? "public/qwen/layer0-gate-proj";

mkdirSync(outDir, { recursive: true });

const model = onnxProto.onnx.ModelProto.decode(readFileSync(input));
const initializers = new Map(model.graph.initializer.map((tensor) => [tensor.name, tensor]));
const weight = initializers.get(`${prefix}.weight_Q4`);
const scales = initializers.get(`${prefix}.weight_scales`);

if (!weight || !scales) {
  throw new Error(`Could not find ${prefix}.weight_Q4 and ${prefix}.weight_scales`);
}

const weightBytes = Buffer.from(weight.rawData);
const scaleF32 = float16TensorToFloat32(scales);
const manifest = {
  prefix,
  weight: {
    file: "weight_q4.bin",
    dims: weight.dims.map((x) => Number(x)),
    dataType: weight.dataType,
    bytes: weightBytes.byteLength,
  },
  scales: {
    file: "scales_f32.bin",
    dims: scales.dims.map((x) => Number(x)),
    sourceDataType: scales.dataType,
    dataType: "float32",
    bytes: scaleF32.byteLength,
  },
  matmulNBits: {
    outSize: Number(weight.dims[0]),
    blocksPerRow: Number(weight.dims[1]),
    bytesPerBlock: Number(weight.dims[2]),
    blockSize: Number(weight.dims[2]) * 2,
    inSize: Number(weight.dims[1]) * Number(weight.dims[2]) * 2,
  },
};

writeFileSync(join(outDir, "weight_q4.bin"), weightBytes);
writeFileSync(join(outDir, "scales_f32.bin"), Buffer.from(scaleF32.buffer));
writeFileSync(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`wrote ${outDir}`);
console.log(JSON.stringify(manifest, null, 2));

function float16TensorToFloat32(tensor) {
  if (tensor.dataType !== 10) {
    throw new Error(`Expected FLOAT16 tensor, got dataType=${tensor.dataType}`);
  }
  const bytes = Buffer.from(tensor.rawData);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = halfToFloat(bytes.readUInt16LE(i * 2));
  }
  return out;
}

function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) {
    return sign * 2 ** -14 * (frac / 1024);
  }
  if (exp === 0x1f) {
    return frac ? NaN : sign * Infinity;
  }
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}
