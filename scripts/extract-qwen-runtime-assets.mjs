import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import onnxProto from "onnx-proto";

const input = process.argv[2] ?? ".cache/qwen3-0.6b-onnx/onnx/model_q4f16.onnx";
const outDir = process.argv[3] ?? "public/qwen/runtime";
const layers = Number(process.argv[4] ?? 28);

mkdirSync(outDir, { recursive: true });

const model = onnxProto.onnx.ModelProto.decode(readFileSync(input));
const initializers = new Map(model.graph.initializer.map((tensor) => [tensor.name, tensor]));

const manifest = {
  source: input,
  dtype: "q4f16",
  hiddenSize: 1024,
  intermediateSize: 3072,
  attentionHeads: 16,
  kvHeads: 8,
  headDim: 128,
  layers: [],
  shared: {},
  finalNorm: null,
};

extractTensor("model.embed_tokens.weight", "shared/embed_tokens.f16", { convertFloat16ToFloat32: false });
extractTensor("cos_cache", "shared/cos_cache.f16", { convertFloat16ToFloat32: false });
extractTensor("sin_cache", "shared/sin_cache.f16", { convertFloat16ToFloat32: false });
manifest.shared.embedTokens = tensorEntry("model.embed_tokens.weight", "shared/embed_tokens.f16");
manifest.shared.cosCache = tensorEntry("cos_cache", "shared/cos_cache.f16");
manifest.shared.sinCache = tensorEntry("sin_cache", "shared/sin_cache.f16");
extractTensor("model.layers.28.final_norm_layernorm.weight", "shared/final_norm.f16", { convertFloat16ToFloat32: false });
manifest.finalNorm = tensorEntry("model.layers.28.final_norm_layernorm.weight", "shared/final_norm.f16");

for (let layer = 0; layer < layers; layer++) {
  const layerDir = `layers/${layer}`;
  const entry = {
    layer,
    norms: {},
    projections: {},
  };

  for (const [key, name] of [
    ["input", `model.layers.${layer}.input_layernorm.weight`],
    ["q", `model.layers.${layer}.attn.q_norm.layernorm.weight`],
    ["k", `model.layers.${layer}.attn.k_norm.layernorm.weight`],
    ["postAttention", `model.layers.${layer}.post_attention_layernorm.weight`],
  ]) {
    const file = `${layerDir}/norm_${key}.f16`;
    extractTensor(name, file, { convertFloat16ToFloat32: false });
    entry.norms[key] = tensorEntry(name, file);
  }

  for (const [key, prefix] of [
    ["q", `model.layers.${layer}.attn.q_proj.MatMul`],
    ["k", `model.layers.${layer}.attn.k_proj.MatMul`],
    ["v", `model.layers.${layer}.attn.v_proj.MatMul`],
    ["o", `model.layers.${layer}.attn.o_proj.MatMul`],
    ["gate", `model.layers.${layer}.mlp.gate_proj.MatMul`],
    ["up", `model.layers.${layer}.mlp.up_proj.MatMul`],
    ["down", `model.layers.${layer}.mlp.down_proj.MatMul`],
  ]) {
    entry.projections[key] = extractProjection(prefix, `${layerDir}/${key}`);
  }

  manifest.layers.push(entry);
}

writeFileSync(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`wrote ${outDir}/manifest.json`);
console.log(`layers=${manifest.layers.length}`);

function extractProjection(prefix, filePrefix) {
  const weightName = `${prefix}.weight_Q4`;
  const scaleName = `${prefix}.weight_scales`;
  const weight = requireTensor(weightName);
  const scales = requireTensor(scaleName);
  const weightFile = `${filePrefix}_weight_q4.bin`;
  const scalesFile = `${filePrefix}_scales_f32.bin`;

  writeFile(weightFile, Buffer.from(weight.rawData));
  writeFile(scalesFile, Buffer.from(float16TensorToFloat32(scales).buffer));

  return {
    prefix,
    weight: tensorEntry(weightName, weightFile),
    scales: {
      ...tensorEntry(scaleName, scalesFile),
      sourceDataType: scales.dataType,
      dataType: "float32",
      bytes: scales.dims.map(Number).reduce((a, b) => a * b, 1) * 4,
    },
    matmulNBits: {
      outSize: Number(weight.dims[0]),
      blocksPerRow: Number(weight.dims[1]),
      bytesPerBlock: Number(weight.dims[2]),
      blockSize: Number(weight.dims[2]) * 2,
      inSize: Number(weight.dims[1]) * Number(weight.dims[2]) * 2,
    },
  };
}

function extractTensor(name, file, { convertFloat16ToFloat32 }) {
  const tensor = requireTensor(name);
  const data = convertFloat16ToFloat32 ? Buffer.from(float16TensorToFloat32(tensor).buffer) : Buffer.from(tensor.rawData);
  writeFile(file, data);
}

function tensorEntry(name, file) {
  const tensor = requireTensor(name);
  return {
    name,
    file,
    dims: tensor.dims.map((x) => Number(x)),
    dataType: tensor.dataType,
    bytes: Buffer.from(tensor.rawData).byteLength,
  };
}

function requireTensor(name) {
  const tensor = initializers.get(name);
  if (!tensor) throw new Error(`Missing initializer ${name}`);
  return tensor;
}

function writeFile(relativePath, data) {
  const path = join(outDir, relativePath);
  mkdirSync(path.split("/").slice(0, -1).join("/"), { recursive: true });
  writeFileSync(path, data);
}

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
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}
