import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import type { ModelConfig } from "./types";

export interface LoadProgress {
  percent: number;
  loadedBytes: number;
  totalBytes: number;
}

export type ProgressCallback = (progress: LoadProgress) => void;

interface LoadedModel {
  config: ModelConfig;
  extractor: FeatureExtractionPipeline;
}

let currentModel: LoadedModel | null = null;

export async function loadModel(
  config: ModelConfig,
  onProgress?: ProgressCallback,
): Promise<void> {
  if (currentModel?.config.id === config.id) return;

  if (currentModel) {
    await currentModel.extractor.dispose();
    currentModel = null;
  }

  // transformers.js reports progress per file (config, tokenizer, onnx
  // weights); aggregate so the UI can show a single MB counter instead of
  // a bar that restarts for each file
  const files = new Map<string, { loaded: number; total: number }>();
  const extractor = await pipeline("feature-extraction", config.hfModel, {
    dtype: config.dtype,
    progress_callback: (event: {
      status: string;
      file?: string;
      loaded?: number;
      total?: number;
    }) => {
      if (event.status !== "progress" || !event.file || !onProgress) return;
      files.set(event.file, {
        loaded: event.loaded ?? 0,
        total: event.total ?? 0,
      });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      if (total > 0) {
        onProgress({
          percent: (loaded / total) * 100,
          loadedBytes: loaded,
          totalBytes: total,
        });
      }
    },
  });

  currentModel = { config, extractor };
}

export async function embed(
  text: string,
  config: ModelConfig,
): Promise<Float32Array> {
  if (!currentModel || currentModel.config.id !== config.id) {
    throw new Error(`Model ${config.id} not loaded`);
  }

  const input = `${config.queryPrefix}${text}`;
  const output = await currentModel.extractor(input, {
    pooling: config.pooling,
    normalize: true,
  });

  return new Float32Array(output.data as Float32Array);
}

export function isModelLoaded(configId: string): boolean {
  return currentModel?.config.id === configId;
}
