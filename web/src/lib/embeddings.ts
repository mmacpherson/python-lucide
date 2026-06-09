export async function loadEmbeddings(
  url: string,
  nIcons: number,
  dim: number,
): Promise<Float32Array> {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const embeddings = new Float32Array(buffer);
  if (embeddings.length !== nIcons * dim) {
    throw new Error(
      `Expected ${nIcons * dim} floats, got ${embeddings.length}`,
    );
  }
  return embeddings;
}

function normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  }
  return vec;
}

export function search(
  queryVec: Float32Array,
  matrix: Float32Array,
  dim: number,
  topK: number,
): { index: number; score: number }[] {
  const query = normalize(new Float32Array(queryVec));
  const nIcons = matrix.length / dim;
  const scores: { index: number; score: number }[] = [];

  for (let i = 0; i < nIcons; i++) {
    let dot = 0;
    const offset = i * dim;
    for (let j = 0; j < dim; j++) {
      dot += query[j] * matrix[offset + j];
    }
    scores.push({ index: i, score: dot });
  }

  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK);
}
