"""
API routes for model data endpoints.
"""
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
import numpy as np
import pandas as pd
import pickle
import os
import logging

from umap import UMAP
from models.schemas import ModelPoint
from utils.family_tree import calculate_family_depths
from utils.dimensionality_reduction import DimensionReducer
from utils.cache import cached_response
from core.exceptions import DataNotLoadedError, EmbeddingsNotReadyError
import api.dependencies as deps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])

# Global cluster labels cache (shared across routes)
cluster_labels = None


def compute_clusters(reduced_embeddings: np.ndarray, n_clusters: int = 50) -> np.ndarray:
    from sklearn.cluster import KMeans
    
    n_samples = len(reduced_embeddings)
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples // 10)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(reduced_embeddings)


@router.get("/models")
async def get_models(
    min_downloads: int = Query(0),
    min_likes: int = Query(0),
    search_query: Optional[str] = Query(None),
    color_by: str = Query("library_name"),
    size_by: str = Query("downloads"),
    max_points: Optional[int] = Query(None),
    projection_method: str = Query("umap"),
    base_models_only: bool = Query(False),
    max_hierarchy_depth: Optional[int] = Query(None, ge=0, description="Filter to models at or below this hierarchy depth."),
    use_graph_embeddings: bool = Query(False, description="Use graph-aware embeddings that respect family tree structure")
):
    if deps.df is None:
        raise DataNotLoadedError()
    
    df = deps.df
    data_loader = deps.data_loader
    
    # Filter data
    filtered_df = data_loader.filter_data(
        df=df,
        min_downloads=min_downloads,
        min_likes=min_likes,
        search_query=search_query,
        libraries=None,
        pipeline_tags=None
    )
    
    if base_models_only:
        if 'parent_model' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['parent_model'].isna() | 
                (filtered_df['parent_model'].astype(str).str.strip() == '') |
                (filtered_df['parent_model'].astype(str) == 'nan')
            ]
    
    if max_hierarchy_depth is not None:
        family_depths = calculate_family_depths(df)
        filtered_df = filtered_df[
            filtered_df['model_id'].astype(str).map(lambda x: family_depths.get(x, 0) <= max_hierarchy_depth)
        ]
    
    filtered_count = len(filtered_df)
    
    if len(filtered_df) == 0:
        return {
            "models": [],
            "filtered_count": 0,
            "returned_count": 0
        }
    
    if max_points is not None and len(filtered_df) > max_points:
        if 'library_name' in filtered_df.columns and filtered_df['library_name'].notna().any():
            sampled_dfs = []
            for lib_name, group in filtered_df.groupby('library_name', group_keys=False):
                n_samples = max(1, int(max_points * len(group) / len(filtered_df)))
                sampled_dfs.append(group.sample(min(len(group), n_samples), random_state=42))
            filtered_df = pd.concat(sampled_dfs, ignore_index=True)
            if len(filtered_df) > max_points:
                filtered_df = filtered_df.sample(n=max_points, random_state=42).reset_index(drop=True)
            else:
                filtered_df = filtered_df.reset_index(drop=True)
        else:
            filtered_df = filtered_df.sample(n=max_points, random_state=42).reset_index(drop=True)
    
    # Determine which embeddings to use
    if use_graph_embeddings and deps.combined_embeddings is not None:
        current_embeddings = deps.combined_embeddings
        current_reduced = deps.reduced_embeddings_graph
        embedding_type = "graph-aware"
    else:
        if deps.embeddings is None:
            raise EmbeddingsNotReadyError()
        current_embeddings = deps.embeddings
        current_reduced = deps.reduced_embeddings
        embedding_type = "text-only"
    
    # Handle reduced embeddings loading/generation
    reducer = deps.reducer
    if current_reduced is None or (reducer and reducer.method != projection_method.lower()):
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root_dir = os.path.dirname(backend_dir)
        cache_dir = os.path.join(root_dir, "cache")
        cache_suffix = "_graph" if use_graph_embeddings and deps.combined_embeddings is not None else ""
        reduced_cache = os.path.join(cache_dir, f"reduced_{projection_method.lower()}_3d{cache_suffix}.pkl")
        reducer_cache = os.path.join(cache_dir, f"reducer_{projection_method.lower()}_3d{cache_suffix}.pkl")
        
        if os.path.exists(reduced_cache) and os.path.exists(reducer_cache):
            try:
                with open(reduced_cache, 'rb') as f:
                    current_reduced = pickle.load(f)
                if reducer is None or reducer.method != projection_method.lower():
                    reducer = DimensionReducer(method=projection_method.lower(), n_components=3)
                reducer.load_reducer(reducer_cache)
            except (IOError, pickle.UnpicklingError, EOFError) as e:
                logger.warning(f"Failed to load cached reduced embeddings: {e}")
                current_reduced = None
        
        if current_reduced is None:
            if reducer is None or reducer.method != projection_method.lower():
                reducer = DimensionReducer(method=projection_method.lower(), n_components=3)
                if projection_method.lower() == "umap":
                    reducer.reducer = UMAP(
                        n_components=3,
                        n_neighbors=30,
                        min_dist=0.3,
                        metric='cosine',
                        random_state=42,
                        n_jobs=-1,
                        low_memory=True,
                        spread=1.5
                    )
            current_reduced = reducer.fit_transform(current_embeddings)
            with open(reduced_cache, 'wb') as f:
                pickle.dump(current_reduced, f)
            reducer.save_reducer(reducer_cache)
            
            # Update global variable
            if use_graph_embeddings and deps.combined_embeddings is not None:
                deps.reduced_embeddings_graph = current_reduced
            else:
                deps.reduced_embeddings = current_reduced
    
    # Get indices for filtered data
    filtered_model_ids = filtered_df['model_id'].astype(str).values
    
    if df.index.name == 'model_id' or 'model_id' in df.index.names:
        filtered_indices = []
        for model_id in filtered_model_ids:
            try:
                pos = df.index.get_loc(model_id)
                if isinstance(pos, (int, np.integer)):
                    filtered_indices.append(int(pos))
                elif isinstance(pos, (slice, np.ndarray)):
                    if isinstance(pos, slice):
                        filtered_indices.append(int(pos.start))
                    else:
                        filtered_indices.append(int(pos[0]))
            except (KeyError, TypeError):
                continue
        filtered_indices = np.array(filtered_indices, dtype=np.int32)
    else:
        df_model_ids = df['model_id'].astype(str).values
        model_id_to_pos = {mid: pos for pos, mid in enumerate(df_model_ids)}
        filtered_indices = np.array([
            model_id_to_pos[mid] for mid in filtered_model_ids 
            if mid in model_id_to_pos
        ], dtype=np.int32)
    
    if len(filtered_indices) == 0:
        return {
            "models": [],
            "embedding_type": embedding_type,
            "filtered_count": filtered_count,
            "returned_count": 0
        }
    
    filtered_reduced = current_reduced[filtered_indices]
    family_depths = calculate_family_depths(df)
    
    global cluster_labels
    clustering_embeddings = current_reduced
    if cluster_labels is None or len(cluster_labels) != len(clustering_embeddings):
        cluster_labels = compute_clusters(clustering_embeddings, n_clusters=min(50, len(clustering_embeddings) // 100))
    
    filtered_clusters = cluster_labels[filtered_indices]
    
    model_ids = filtered_df['model_id'].astype(str).values
    library_names = filtered_df.get('library_name', pd.Series([None] * len(filtered_df))).values
    pipeline_tags = filtered_df.get('pipeline_tag', pd.Series([None] * len(filtered_df))).values
    downloads_arr = filtered_df.get('downloads', pd.Series([0] * len(filtered_df))).fillna(0).astype(int).values
    likes_arr = filtered_df.get('likes', pd.Series([0] * len(filtered_df))).fillna(0).astype(int).values
    trending_scores = filtered_df.get('trendingScore', pd.Series([None] * len(filtered_df))).values
    tags_arr = filtered_df.get('tags', pd.Series([None] * len(filtered_df))).values
    parent_models = filtered_df.get('parent_model', pd.Series([None] * len(filtered_df))).values
    licenses_arr = filtered_df.get('licenses', pd.Series([None] * len(filtered_df))).values
    created_at_arr = filtered_df.get('createdAt', pd.Series([None] * len(filtered_df))).values
    
    x_coords = filtered_reduced[:, 0].astype(float)
    y_coords = filtered_reduced[:, 1].astype(float)
    z_coords = filtered_reduced[:, 2].astype(float) if filtered_reduced.shape[1] > 2 else np.zeros(len(filtered_reduced), dtype=float)
    models = [
        ModelPoint(
            model_id=model_ids[idx],
            x=float(x_coords[idx]),
            y=float(y_coords[idx]),
            z=float(z_coords[idx]),
            library_name=library_names[idx] if pd.notna(library_names[idx]) else None,
            pipeline_tag=pipeline_tags[idx] if pd.notna(pipeline_tags[idx]) else None,
            downloads=int(downloads_arr[idx]),
            likes=int(likes_arr[idx]),
            trending_score=float(trending_scores[idx]) if idx < len(trending_scores) and pd.notna(trending_scores[idx]) else None,
            tags=tags_arr[idx] if idx < len(tags_arr) and pd.notna(tags_arr[idx]) else None,
            parent_model=parent_models[idx] if idx < len(parent_models) and pd.notna(parent_models[idx]) else None,
            licenses=licenses_arr[idx] if idx < len(licenses_arr) and pd.notna(licenses_arr[idx]) else None,
            family_depth=family_depths.get(model_ids[idx], None),
            cluster_id=int(filtered_clusters[idx]) if idx < len(filtered_clusters) else None,
            created_at=str(created_at_arr[idx]) if idx < len(created_at_arr) and pd.notna(created_at_arr[idx]) else None
        )
        for idx in range(len(filtered_df))
    ]
    
    return {
        "models": models,
        "embedding_type": embedding_type,
        "filtered_count": filtered_count,
        "returned_count": len(models)
    }


@router.get("/family/adoption")
@cached_response(ttl=3600, key_prefix="family_adoption")
async def get_family_adoption(
    family: str = Query(..., description="Family name (e.g., 'meta-llama', 'google', 'microsoft')"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of models to return")
):
    """
    Get adoption data for a specific family (S-curve data).
    Returns models sorted by creation date with their downloads.
    """
    if deps.df is None:
        raise DataNotLoadedError()
    
    df = deps.df
    
    # Filter by family name (check model_id prefix and tags)
    family_lower = family.lower()
    filtered_df = df[
        df['model_id'].astype(str).str.lower().str.contains(family_lower, regex=False, na=False) |
        df.get('tags', pd.Series([None] * len(df))).astype(str).str.lower().str.contains(family_lower, regex=False, na=False)
    ]
    
    if len(filtered_df) == 0:
        return {
            "family": family,
            "models": [],
            "total_models": 0
        }
    
    # Sort by downloads and limit
    filtered_df = filtered_df.nlargest(limit, 'downloads', keep='first')
    
    # Extract required fields
    model_ids = filtered_df['model_id'].astype(str).values
    downloads_arr = filtered_df.get('downloads', pd.Series([0] * len(filtered_df))).fillna(0).astype(int).values
    created_at_arr = filtered_df.get('createdAt', pd.Series([None] * len(filtered_df))).values
    
    # Parse dates efficiently
    dates = pd.to_datetime(created_at_arr, errors='coerce', utc=True)
    
    # Build response
    adoption_data = []
    for idx in range(len(filtered_df)):
        date_val = dates.iloc[idx] if isinstance(dates, pd.Series) else dates[idx]
        if pd.notna(date_val):
            adoption_data.append({
                "model_id": model_ids[idx],
                "downloads": int(downloads_arr[idx]),
                "created_at": date_val.isoformat()
            })
    
    # Sort by date
    adoption_data.sort(key=lambda x: x['created_at'] if x['created_at'] else '')
    
    return {
        "family": family,
        "models": adoption_data,
        "total_models": len(adoption_data)
    }

