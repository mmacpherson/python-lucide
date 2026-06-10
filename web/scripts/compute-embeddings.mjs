/**
 * Compute icon embeddings using Transformers.js (Node.js).
 *
 * Reads icons.json, computes embeddings for each model, writes binary files.
 * Uses the same text construction as build_search.py:_embedding_text().
 */

import { readFileSync, writeFileSync } from "fs";
import { pipeline } from "@huggingface/transformers";

const DATA_DIR = new URL("../public/data/", import.meta.url).pathname;
const ICONS_JSON = `${DATA_DIR}icons.json`;

function embeddingText(icon, docPrefix) {
  const parts = [icon.name.replaceAll("-", " ")];
  if (icon.tags.length > 0) {
    parts.push(`Tags: ${icon.tags.join(", ")}`);
  }
  if (icon.categories.length > 0) {
    parts.push(`Categories: ${icon.categories.join(", ")}`);
  }
  parts.push(icon.description);
  return `${docPrefix}${parts.join(". ")}`;
}

function normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  }
  return vec;
}

async function computeForModel(icons, modelConfig) {
  console.log(
    `\nLoading model: ${modelConfig.hfModel} (${modelConfig.dim}d)...`
  );
  const extractor = await pipeline("feature-extraction", modelConfig.hfModel, {
    dtype: "fp32",
  });

  const texts = icons.map((icon) => embeddingText(icon, modelConfig.docPrefix));
  const batchSize = 32;
  const nIcons = icons.length;
  const dim = modelConfig.dim;
  const buffer = new Float32Array(nIcons * dim);

  for (let i = 0; i < nIcons; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const output = await extractor(batch, { pooling: "mean", normalize: true });
    const data = output.data;
    for (let j = 0; j < batch.length; j++) {
      const vec = new Float32Array(dim);
      for (let k = 0; k < dim; k++) {
        vec[k] = data[j * dim + k];
      }
      normalize(vec);
      buffer.set(vec, (i + j) * dim);
    }
    const done = Math.min(i + batchSize, nIcons);
    process.stderr.write(`\r  Embedded ${done}/${nIcons} icons`);
  }
  console.log();

  const outPath = `${DATA_DIR}${modelConfig.file}`;
  writeFileSync(outPath, Buffer.from(buffer.buffer));
  const sizeMb = (buffer.byteLength / 1024 / 1024).toFixed(1);
  console.log(`  Wrote ${outPath} (${sizeMb} MB)`);

  await extractor.dispose();
}

async function main() {
  const data = JSON.parse(readFileSync(ICONS_JSON, "utf-8"));
  console.log(`Loaded ${data.icons.length} icons (Lucide v${data.version})`);

  const only = process.argv[2];
  const models = only
    ? data.models.filter((m) => m.id === only)
    : data.models;

  if (only && models.length === 0) {
    console.error(`Unknown model id: ${only}`);
    process.exit(1);
  }

  for (const modelConfig of models) {
    await computeForModel(data.icons, modelConfig);
  }

  console.log("\nDone!");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
