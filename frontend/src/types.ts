export interface ModelPoint {
  model_id: string;
  x: number;
  y: number;
  z: number;
  library_name: string | null;
  pipeline_tag: string | null;
  downloads: number;
  likes: number;
  trending_score: number | null;
  tags: string | null;
  parent_model: string | null;
  licenses: string | null;
  family_depth: number | null;  // Generation depth in family tree (0 = root)
  cluster_id: number | null;    // Cluster assignment for visualization
  created_at: string | null;    // ISO format date string
}

export interface FamilyTree {
  root_model: string;
  family: ModelPoint[];
  family_map: Record<string, ModelPoint & { children: string[] }>;
  root_models: string[];
}

export interface SearchResult {
  model_id: string;
  library_name: string | null;
  pipeline_tag: string | null;
  downloads: number;
  likes: number;
  parent_model: string | null;
  // Aliases for backward compatibility with VirtualSearchResults
  library?: string | null;
  pipeline?: string | null;
}

export interface SimilarModel {
  model_id: string;
  similarity: number;
  distance: number;
  library_name: string | null;
  pipeline_tag: string | null;
  downloads: number;
  likes: number;
}

export interface DistanceResult {
  model_1: string;
  model_2: string;
  cosine_similarity: number;
  cosine_distance: number;
  euclidean_distance: number;
}

export interface Stats {
  total_models: number;
  unique_libraries: number;
  unique_pipelines: number;  // Deprecated: use unique_task_types
  unique_task_types?: number;  // Number of distinct ML task types (e.g., text-classification, image-classification)
  unique_licenses?: number;
  licenses?: Record<string, number>;  // License name -> count mapping
  avg_downloads: number;
  avg_likes: number;
}

