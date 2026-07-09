import { createWriteStream, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { finished } from "node:stream/promises";

const modelId = "onnx-community/Qwen3-0.6B-ONNX";
const revision = "main";
const outDir = ".cache/qwen3-0.6b-onnx";

const files = [
  "config.json",
  "generation_config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "onnx/model_q4f16.onnx",
];

mkdirSync(outDir, { recursive: true });

for (const file of files) {
  const out = join(outDir, file);
  if (existsSync(out)) {
    console.log(`exists ${out}`);
    continue;
  }

  mkdirSync(dirname(out), { recursive: true });
  const url = `https://huggingface.co/${modelId}/resolve/${revision}/${file}`;
  console.log(`download ${file}`);
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${file}: ${response.status} ${response.statusText}`);
  }

  const total = Number(response.headers.get("content-length") ?? 0);
  let loaded = 0;
  const progress = new TransformStream({
    transform(chunk, controller) {
      loaded += chunk.byteLength;
      if (total) {
        process.stdout.write(`\r${file} ${(loaded / total * 100).toFixed(1)}%`);
      }
      controller.enqueue(chunk);
    },
  });
  await finished(Readable.fromWeb(response.body.pipeThrough(progress)).pipe(createWriteStream(out)));
  if (total) process.stdout.write("\n");
}
