// One-off helper: rank candidate example prompts against both embedding
// models so the UI only ships prompts that retrieve well on each.
// Usage: node scripts/eval-prompts.mjs
import { readFileSync } from "node:fs";
import { pipeline } from "@huggingface/transformers";

const DATA = new URL("../public/data/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("icons.json", DATA), "utf8"));

const MODELS = manifest.models;

const CANDIDATES = [
  "waiting for something to finish loading",
  "a person talking to customer support",
  "warning about something dangerous",
  "share this with a friend",
  "the weather is getting colder",
  "save my work to the cloud",
  "music is playing right now",
  "secure login with a password",
  "an idea just occurred to me",
  "travel somewhere by airplane",
  "money growing over time",
  "delete this permanently",
  "notifications are turned off",
  "connect two devices together",
  "celebrate a big achievement",
  "something went wrong with the code",
];

for (const model of MODELS) {
  const dim = model.dim;
  const buf = readFileSync(new URL(model.file, DATA));
  const matrix = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const extractor = await pipeline("feature-extraction", model.hfModel, { dtype: model.dtype });

  console.log(`\n=== ${model.id} ===`);
  for (const q of CANDIDATES) {
    const out = await extractor(`${model.queryPrefix}${q}`, { pooling: model.pooling, normalize: true });
    const v = out.data;
    const scores = [];
    for (let i = 0; i < manifest.icons.length; i++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += v[d] * matrix[i * dim + d];
      scores.push([dot, manifest.icons[i].name]);
    }
    scores.sort((a, b) => b[0] - a[0]);
    const top = scores.slice(0, 5).map(([s, n]) => `${n}:${(s * 100).toFixed(0)}`).join("  ");
    console.log(`  "${q}"\n    ${top}`);
  }
  await extractor.dispose();
}
