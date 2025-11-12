export interface ModelPoint {
  model_id: string;
  x: number;
  y: number;
  library_name: string | null;
  pipeline_tag: string | null;
  downloads: number;
  likes: number;
  trending_score: number | null;
  tags: string | null;
}

export interface Stats {
  total_models: number;
  unique_libraries: number;
  unique_pipelines: number;
  avg_downloads: number;
  avg_likes: number;
}

