export interface ModelConfig {
  id: string;
  dim: number;
  file: string;
  hfModel: string;
  // Must match the pooling fastembed used to build the document vectors —
  // mean-pooling a CLS-pooled model still "works" but degrades ranking
  pooling: "mean" | "cls";
  dtype: "fp32" | "fp16" | "q8";
  queryPrefix: string;
  label: string;
  default: boolean;
  /** Model whose embeddings produced the UMAP layout and clusters */
  clusterSource: boolean;
}

export interface IconData {
  name: string;
  svg: string;
  description: string;
  tags: string[];
  categories: string[];
  cluster: string;
}

export interface IconsManifest {
  version: string;
  /** python-lucide package version that exported this data ("" if unknown) */
  packageVersion: string;
  models: ModelConfig[];
  icons: IconData[];
}

export interface SearchResult {
  icon: IconData;
  score: number;
  index: number;
}
