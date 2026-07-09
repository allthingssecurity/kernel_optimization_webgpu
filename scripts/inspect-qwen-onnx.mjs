import { readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import onnxProto from "onnx-proto";

const input = process.argv[2] ?? ".cache/qwen3-0.6b-onnx/onnx/model_q4f16.onnx";
const output = process.argv[3] ?? "reports/qwen-onnx-manifest.json";

const bytes = readFileSync(input);
const model = onnxProto.onnx.ModelProto.decode(bytes);
const graph = model.graph;

const opCounts = new Map();
const nodes = graph.node.map((node, index) => {
  const opType = node.opType;
  opCounts.set(opType, (opCounts.get(opType) ?? 0) + 1);
  return {
    index,
    name: node.name,
    opType,
    domain: node.domain,
    inputs: node.input,
    outputs: node.output,
    attributes: Object.fromEntries(node.attribute.map((attribute) => [attribute.name, attributeValue(attribute)])),
  };
});

const initializers = graph.initializer.map((tensor) => ({
  name: tensor.name,
  dataType: tensor.dataType,
  dims: tensor.dims.map((dim) => Number(dim)),
  rawBytes: tensor.rawData?.length ?? 0,
  externalData: tensor.externalData?.map((entry) => ({ key: entry.key, value: entry.value })) ?? [],
}));

const qwenNames = initializers
  .map((x) => x.name)
  .filter((name) => /embed|lm_head|self_attn|mlp|norm|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj/.test(name))
  .slice(0, 400);

const manifest = {
  source: basename(input),
  irVersion: Number(model.irVersion),
  producerName: model.producerName,
  graphName: graph.name,
  inputs: graph.input.map(valueInfo),
  outputs: graph.output.map(valueInfo),
  opCounts: Object.fromEntries([...opCounts.entries()].sort((a, b) => a[0].localeCompare(b[0]))),
  nodeCount: nodes.length,
  initializerCount: initializers.length,
  initializers,
  qwenInitializerNamesSample: qwenNames,
  nodes,
};

writeFileSync(output, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`wrote ${output}`);
console.log(`nodes=${manifest.nodeCount} initializers=${manifest.initializerCount}`);
console.log(JSON.stringify(manifest.opCounts, null, 2));

function valueInfo(info) {
  return {
    name: info.name,
    type: info.type?.tensorType
      ? {
          elemType: info.type.tensorType.elemType,
          shape: info.type.tensorType.shape?.dim?.map((dim) => ({
            value: dim.dimValue ? Number(dim.dimValue) : undefined,
            param: dim.dimParam || undefined,
          })),
        }
      : undefined,
  };
}

function attributeValue(attribute) {
  if (attribute.s) return Buffer.from(attribute.s).toString("utf8");
  if (attribute.i !== undefined && attribute.i !== null) return Number(attribute.i);
  if (attribute.f !== undefined && attribute.f !== null) return attribute.f;
  if (attribute.t) {
    return {
      tensor: true,
      name: attribute.t.name,
      dataType: attribute.t.dataType,
      dims: attribute.t.dims.map((x) => Number(x)),
      int64Data: attribute.t.int64Data?.map((x) => Number(x)) ?? [],
      rawBytes: attribute.t.rawData?.length ?? 0,
    };
  }
  if (attribute.ints?.length) return attribute.ints.map((x) => Number(x));
  if (attribute.floats?.length) return attribute.floats;
  if (attribute.strings?.length) return attribute.strings.map((x) => Buffer.from(x).toString("utf8"));
  return null;
}
