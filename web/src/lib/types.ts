export interface ModelConfig {
  id: string;
  dim: number;
  file: string;
  hfModel: string;
  queryPrefix: string;
  docPrefix: string;
  label: string;
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
  models: ModelConfig[];
  icons: IconData[];
}

export interface SearchResult {
  icon: IconData;
  score: number;
  index: number;
}
