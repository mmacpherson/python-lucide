import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import type { ModelConfig } from "./types";

export type ProgressCallback = (progress: number) => void;

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

  const extractor = await pipeline("feature-extraction", config.hfModel, {
    dtype: "fp32",
    progress_callback: (event: { status: string; progress?: number }) => {
      if (event.status === "progress" && event.progress != null && onProgress) {
        onProgress(event.progress);
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
    pooling: "mean",
    normalize: true,
  });

  return new Float32Array(output.data as Float32Array);
}

export function isModelLoaded(configId: string): boolean {
  return currentModel?.config.id === configId;
}
