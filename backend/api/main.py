import sys
import os
import pickle
import tempfile
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from umap import UMAP

from utils.data_loader import ModelDataLoader
from utils.embeddings import ModelEmbedder
from utils.dimensionality_reduction import DimensionReducer
from utils.network_analysis import ModelNetworkBuilder
from utils.graph_embeddings import GraphEmbedder
from services.model_tracker import get_tracker
from services.arxiv_api import extract_arxiv_ids, fetch_arxiv_papers
from core.config import settings
from core.exceptions import DataNotLoadedError, EmbeddingsNotReadyError
from models.schemas import ModelPoint
from utils.family_tree import calculate_family_depths
from utils.cache import cache, cached_response
from utils.response_encoder import FastJSONResponse, MessagePackResponse, encode_models_msgpack
import api.dependencies as deps
from api.routes import models, stats, clusters

# Create aliases for backward compatibility with existing routes
# Note: These are set at module load time and may be None initially
# Functions should access via deps.* to get current values
data_loader = deps.data_loader

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

logger = logging.getLogger(__name__)

app = FastAPI(title="HF Model Ecosystem API", version="2.0.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=CORS_HEADERS,
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=CORS_HEADERS,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
        headers=CORS_HEADERS,
    )

if settings.ALLOW_ALL_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", settings.FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Include routers
app.include_router(models.router)
app.include_router(stats.router)
app.include_router(clusters.router)

@app.on_event("startup")
async def startup_event():
    """
    Fast startup using pre-computed data.
    Falls back to traditional loading if pre-computed data not available.
    """
    import time
    startup_start = time.time()
    
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(backend_dir)
    
    # Try to load pre-computed data first (instant startup!)
    from utils.precomputed_loader import get_precomputed_loader
    
    precomputed_loader = get_precomputed_loader(version="v1")
    
    if precomputed_loader:
        logger.info("=" * 60)
        logger.info("LOADING PRE-COMPUTED DATA (Fast Startup Mode)")
        logger.info("=" * 60)
        
        try:
            # Load everything in seconds
            deps.df, deps.embeddings, metadata = precomputed_loader.load_all()
            
            # Extract 3D coordinates from dataframe
            deps.reduced_embeddings = np.column_stack([
                deps.df['x_3d'].values,
                deps.df['y_3d'].values,
                deps.df['z_3d'].values
            ])
            
            # Initialize embedder (without loading/generating embeddings)
            deps.embedder = ModelEmbedder()
            
            # Initialize reducer (already fitted)
            deps.reducer = DimensionReducer(method="umap", n_components=3)
            
            # No graph embeddings in fast mode (optional feature)
            deps.graph_embedder = None
            deps.graph_embeddings_dict = None
            deps.combined_embeddings = None
            deps.reduced_embeddings_graph = None
            
            startup_time = time.time() - startup_start
            logger.info("=" * 60)
            logger.info(f"STARTUP COMPLETE in {startup_time:.2f} seconds!")
            logger.info(f"Loaded {len(deps.df):,} models with pre-computed coordinates")
            logger.info(f"Unique libraries: {metadata.get('unique_libraries')}")
            logger.info(f"Unique pipelines: {metadata.get('unique_pipelines')}")
            logger.info("=" * 60)
            
            # Update module-level aliases
            df = deps.df
            embedder = deps.embedder
            reducer = deps.reducer
            embeddings = deps.embeddings
            reduced_embeddings = deps.reduced_embeddings
            
            return
        
        except Exception as e:
            logger.warning(f"Failed to load pre-computed data: {e}")
            logger.info("Falling back to traditional loading...")
    
    else:
        logger.info("=" * 60)
        logger.info("Pre-computed data not found.")
        logger.info("To enable fast startup, run:")
        logger.info("  cd backend && python scripts/precompute_data.py --sample-size 150000")
        logger.info("=" * 60)
        logger.info("Falling back to traditional loading (may take 1-8 hours)...")
    
    # Traditional loading (slow path)
    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    embeddings_cache = os.path.join(cache_dir, "embeddings.pkl")
    graph_embeddings_cache = os.path.join(cache_dir, "graph_embeddings.pkl")
    combined_embeddings_cache = os.path.join(cache_dir, "combined_embeddings.pkl")
    reduced_cache_umap = os.path.join(cache_dir, "reduced_umap_3d.pkl")
    reduced_cache_umap_graph = os.path.join(cache_dir, "reduced_umap_3d_graph.pkl")
    reducer_cache_umap = os.path.join(cache_dir, "reducer_umap_3d.pkl")
    reducer_cache_umap_graph = os.path.join(cache_dir, "reducer_umap_3d_graph.pkl")
    
    # Load dataset with sample (for reasonable startup time)
    sample_size = settings.SAMPLE_SIZE or settings.get_sample_size() or 5000
    logger.info(f"Loading dataset (sample_size={sample_size}, prioritizing base models)...")
    
    deps.df = deps.data_loader.load_data(sample_size=sample_size, prioritize_base_models=True)
    deps.df = deps.data_loader.preprocess_for_embedding(deps.df)
    
    if 'model_id' in deps.df.columns:
        deps.df.set_index('model_id', drop=False, inplace=True)
    for col in ['downloads', 'likes']:
        if col in deps.df.columns:
            deps.df[col] = pd.to_numeric(deps.df[col], errors='coerce').fillna(0).astype(int)
    
    deps.embedder = ModelEmbedder()
    
    # Load or generate text embeddings
    if os.path.exists(embeddings_cache):
        try:
            deps.embeddings = deps.embedder.load_embeddings(embeddings_cache)
        except (IOError, pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            deps.embeddings = None
    
    if deps.embeddings is None:
        texts = deps.df['combined_text'].tolist()
        deps.embeddings = deps.embedder.generate_embeddings(texts, batch_size=128)
        deps.embedder.save_embeddings(deps.embeddings, embeddings_cache)
    
    # Skip graph embeddings in fallback mode (too slow)
    deps.graph_embedder = None
    deps.graph_embeddings_dict = None
    deps.combined_embeddings = None
    
    # Initialize reducer for text embeddings
    deps.reducer = DimensionReducer(method="umap", n_components=3)
    
    # Pre-compute clusters for faster requests
    logger.info("Pre-computing clusters...")
    
    if os.path.exists(reduced_cache_umap) and os.path.exists(reducer_cache_umap):
        try:
            with open(reduced_cache_umap, 'rb') as f:
                deps.reduced_embeddings = pickle.load(f)
            deps.reducer.load_reducer(reducer_cache_umap)
        except (IOError, pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Failed to load cached reduced embeddings: {e}")
            deps.reduced_embeddings = None
    
    if deps.reduced_embeddings is None:
        deps.reducer.reducer = UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.3,
            metric='cosine',
            random_state=42,
            n_jobs=-1,
            low_memory=True,
            spread=1.5
        )
        deps.reduced_embeddings = deps.reducer.fit_transform(deps.embeddings)
        with open(reduced_cache_umap, 'wb') as f:
            pickle.dump(deps.reduced_embeddings, f)
        deps.reducer.save_reducer(reducer_cache_umap)
    
    # No graph embeddings in fallback mode
    deps.reduced_embeddings_graph = None
    
    # Pre-compute clusters now instead of on first request
    if deps.reduced_embeddings is not None and len(deps.reduced_embeddings) > 0:
        models.cluster_labels = compute_clusters(
            deps.reduced_embeddings, 
            n_clusters=min(50, len(deps.reduced_embeddings) // 100)
        )
        logger.info(f"Pre-computed {len(set(models.cluster_labels))} clusters")
    
    startup_time = time.time() - startup_start
    logger.info(f"Startup complete in {startup_time:.2f} seconds")
    
    # Update module-level aliases
    df = deps.df
    embedder = deps.embedder
    graph_embedder = deps.graph_embedder
    reducer = deps.reducer
    embeddings = deps.embeddings
    graph_embeddings_dict = deps.graph_embeddings_dict
    combined_embeddings = deps.combined_embeddings
    reduced_embeddings = deps.reduced_embeddings
    reduced_embeddings_graph = deps.reduced_embeddings_graph


from utils.family_tree import calculate_family_depths


def compute_clusters(reduced_embeddings: np.ndarray, n_clusters: int = 50) -> np.ndarray:
    from sklearn.cluster import KMeans
    
    n_samples = len(reduced_embeddings)
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples // 10)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(reduced_embeddings)


@app.get("/")
async def root():
    # Check if frontend build exists and serve it
    _backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _frontend_build_path = os.path.join(os.path.dirname(_backend_dir), "frontend", "build")
    index_path = os.path.join(_frontend_build_path, "index.html")
    
    if os.path.exists(index_path):
        from starlette.responses import FileResponse as StarletteFileResponse
        return StarletteFileResponse(index_path)
    
    # Fallback to API status when no frontend build
    return {"message": "HF Model Ecosystem API", "status": "running"}


@app.get("/api/models")
async def get_models(
    min_downloads: int = Query(0),
    min_likes: int = Query(0),
    search_query: Optional[str] = Query(None),
    color_by: str = Query("library_name"),
    size_by: str = Query("downloads"),
    max_points: Optional[int] = Query(10000),  # REDUCED from None (was 50k default in frontend)
    projection_method: str = Query("umap"),
    base_models_only: bool = Query(False),
    max_hierarchy_depth: Optional[int] = Query(None, ge=0, description="Filter to models at or below this hierarchy depth."),
    use_graph_embeddings: bool = Query(False, description="Use graph-aware embeddings that respect family tree structure"),
    format: str = Query("json", regex="^(json|msgpack)$", description="Response format: json or msgpack")
):
    if deps.df is None:
        raise DataNotLoadedError()
    
    df = deps.df
    
    # Filter data
    filtered_df = data_loader.filter_data(
        df=df,
        min_downloads=min_downloads,
        min_likes=min_likes,
        search_query=search_query,
        libraries=None,  # Can be added as query params
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
    
    # Handle max_points: None means no limit, very large number also means no limit
    effective_max_points = None if max_points is None or max_points >= 1000000 else max_points
    
    if effective_max_points is not None and len(filtered_df) > effective_max_points:
        if 'library_name' in filtered_df.columns and filtered_df['library_name'].notna().any():
            # Sample proportionally by library, preserving all columns
            sampled_dfs = []
            for lib_name, group in filtered_df.groupby('library_name', group_keys=False):
                n_samples = max(1, int(effective_max_points * len(group) / len(filtered_df)))
                sampled_dfs.append(group.sample(min(len(group), n_samples), random_state=42))
            filtered_df = pd.concat(sampled_dfs, ignore_index=True)
            if len(filtered_df) > effective_max_points:
                filtered_df = filtered_df.sample(n=effective_max_points, random_state=42).reset_index(drop=True)
            else:
                filtered_df = filtered_df.reset_index(drop=True)
        else:
            filtered_df = filtered_df.sample(n=effective_max_points, random_state=42).reset_index(drop=True)
    
    # Determine which embeddings to use
    if use_graph_embeddings and combined_embeddings is not None:
        current_embeddings = combined_embeddings
        current_reduced = reduced_embeddings_graph
        embedding_type = "graph-aware"
    else:
        if embeddings is None:
            raise EmbeddingsNotReadyError()
        current_embeddings = embeddings
        current_reduced = reduced_embeddings
        embedding_type = "text-only"
    
    # Handle reduced embeddings loading/generation
    if current_reduced is None or (reducer and reducer.method != projection_method.lower()):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_dir = os.path.dirname(backend_dir)
        cache_dir = os.path.join(root_dir, "cache")
        cache_suffix = "_graph" if use_graph_embeddings and combined_embeddings is not None else ""
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
    # Use model_id column to map between filtered_df and original df
    # This is safer than using index positions which can change after filtering
    filtered_model_ids = filtered_df['model_id'].astype(str).values
    
    # Map model_ids to positions in original df
    if df.index.name == 'model_id' or 'model_id' in df.index.names:
        # When df is indexed by model_id, use get_loc directly
        filtered_indices = []
        for model_id in filtered_model_ids:
            try:
                pos = df.index.get_loc(model_id)
                # Handle both single position and array of positions
                if isinstance(pos, (int, np.integer)):
                    filtered_indices.append(int(pos))
                elif isinstance(pos, (slice, np.ndarray)):
                    # If multiple matches, take first
                    if isinstance(pos, slice):
                        filtered_indices.append(int(pos.start))
                    else:
                        filtered_indices.append(int(pos[0]))
            except (KeyError, TypeError):
                continue
        filtered_indices = np.array(filtered_indices, dtype=np.int32)
    else:
        # When df is not indexed by model_id, find positions by matching model_id column
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
    
    # Use appropriate embeddings for clustering
    clustering_embeddings = current_reduced
    # Compute clusters if not already computed or if size changed
    if models.cluster_labels is None or len(models.cluster_labels) != len(clustering_embeddings):
        models.cluster_labels = compute_clusters(clustering_embeddings, n_clusters=min(50, len(clustering_embeddings) // 100))
    
    # Handle case where cluster_labels might not match filtered data yet
    if models.cluster_labels is not None and len(models.cluster_labels) > 0:
        if len(filtered_indices) <= len(models.cluster_labels):
            filtered_clusters = models.cluster_labels[filtered_indices]
        else:
            # Fallback: use first cluster for all if indices don't match
            filtered_clusters = np.zeros(len(filtered_indices), dtype=int)
    else:
        filtered_clusters = np.zeros(len(filtered_indices), dtype=int)
    
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
    
    # Return models with metadata about embedding type
    response_data = {
        "models": models,
        "embedding_type": embedding_type,
        "filtered_count": filtered_count,
        "returned_count": len(models)
    }
    
    # Return in requested format with caching headers
    if format == "msgpack":
        try:
            binary_data = encode_models_msgpack([m.dict() for m in models])
            return Response(
                content=binary_data,
                media_type="application/msgpack",
                headers={
                    "Cache-Control": "public, max-age=300",
                    "X-Content-Type-Options": "nosniff",
                    "Access-Control-Expose-Headers": "Cache-Control"
                }
            )
        except Exception as e:
            logger.warning(f"MessagePack encoding failed, falling back to JSON: {e}")
    
    # Return JSON with caching headers
    return FastJSONResponse(
        content=response_data,
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Content-Type-Options": "nosniff",
            "Access-Control-Expose-Headers": "Cache-Control"
        }
    )


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics."""
    if df is None:
        raise DataNotLoadedError()
    
    total_models = len(df.index) if hasattr(df, 'index') else len(df)
    
    # Get unique licenses with counts
    licenses = {}
    if 'license' in df.columns:
        license_counts = df['license'].value_counts().to_dict()
        licenses = {str(k): int(v) for k, v in license_counts.items() if pd.notna(k) and str(k) != 'nan'}
    
    return {
        "total_models": total_models,
        "unique_libraries": int(df['library_name'].nunique()) if 'library_name' in df.columns else 0,
        "unique_pipelines": int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,
        "unique_task_types": int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,  # Alias for clarity
        "unique_licenses": len(licenses),
        "licenses": licenses,  # License name -> count mapping
        "avg_downloads": float(df['downloads'].mean()) if 'downloads' in df.columns else 0,
        "avg_likes": float(df['likes'].mean()) if 'likes' in df.columns else 0
    }


@app.get("/api/model/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed information about a specific model."""
    if df is None:
        raise DataNotLoadedError()
    
    model = df[df.get('model_id', '') == model_id]
    if len(model) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = model.iloc[0]
    
    tags_str = str(model.get('tags', '')) if pd.notna(model.get('tags')) else ''
    arxiv_ids = extract_arxiv_ids(tags_str)
    
    papers = []
    if arxiv_ids:
        papers = await fetch_arxiv_papers(arxiv_ids[:5])  # Limit to 5 papers
    
    return {
        "model_id": model.get('model_id'),
        "library_name": model.get('library_name'),
        "pipeline_tag": model.get('pipeline_tag'),
        "downloads": int(model.get('downloads', 0)),
        "likes": int(model.get('likes', 0)),
        "trending_score": float(model.get('trendingScore', 0)) if pd.notna(model.get('trendingScore')) else None,
        "tags": model.get('tags') if pd.notna(model.get('tags')) else None,
        "licenses": model.get('licenses') if pd.notna(model.get('licenses')) else None,
        "parent_model": model.get('parent_model') if pd.notna(model.get('parent_model')) else None,
        "arxiv_papers": papers,
        "arxiv_ids": arxiv_ids
    }


# Clusters endpoint is handled by routes/clusters.py router

@app.get("/api/family/stats")
async def get_family_stats():
    """
    Get aggregate statistics about family trees for paper visualizations.
    Returns family size distribution, depth statistics, model card length by depth, etc.
    """
    if df is None:
        raise DataNotLoadedError()
    
    family_sizes = {}
    root_models = set()
    
    for idx, row in df.iterrows():
        model_id = str(row.get('model_id', ''))
        parent_id = row.get('parent_model')
        
        if pd.isna(parent_id) or str(parent_id) == 'nan' or str(parent_id) == '':
            root_models.add(model_id)
            if model_id not in family_sizes:
                family_sizes[model_id] = 0
        else:
            parent_id_str = str(parent_id)
            root = parent_id_str
            visited = set()
            while root in df.index and pd.notna(df.loc[root].get('parent_model')):
                parent = df.loc[root].get('parent_model')
                if pd.isna(parent) or str(parent) == 'nan' or str(parent) == '':
                    break
                if str(parent) in visited:
                    break
                visited.add(root)
                root = str(parent)
            
            if root not in family_sizes:
                family_sizes[root] = 0
            family_sizes[root] += 1
    
    size_distribution = {}
    for root, size in family_sizes.items():
        size_distribution[size] = size_distribution.get(size, 0) + 1
    
    depths = calculate_family_depths(df)
    depth_counts = {}
    for depth in depths.values():
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    
    model_card_lengths_by_depth = {}
    if 'modelCard' in df.columns:
        for idx, row in df.iterrows():
            model_id = str(row.get('model_id', ''))
            depth = depths.get(model_id, 0)
            model_card = row.get('modelCard', '')
            if pd.notna(model_card):
                card_length = len(str(model_card))
                if depth not in model_card_lengths_by_depth:
                    model_card_lengths_by_depth[depth] = []
                model_card_lengths_by_depth[depth].append(card_length)
    
    model_card_stats = {}
    for depth, lengths in model_card_lengths_by_depth.items():
        if lengths:
            model_card_stats[depth] = {
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "q1": float(np.percentile(lengths, 25)),
                "q3": float(np.percentile(lengths, 75)),
                "min": float(np.min(lengths)),
                "max": float(np.max(lengths)),
                "count": len(lengths)
            }
    
    return {
        "total_families": len(root_models),
        "family_size_distribution": size_distribution,
        "depth_distribution": depth_counts,
        "max_family_size": max(family_sizes.values()) if family_sizes else 0,
        "max_depth": max(depths.values()) if depths else 0,
        "avg_family_size": sum(family_sizes.values()) / len(family_sizes) if family_sizes else 0,
        "model_card_length_by_depth": model_card_stats
    }


@app.get("/api/family/top")
async def get_top_families(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of families to return"),
    min_size: int = Query(2, ge=1, description="Minimum family size to include")
):
    """
    Get top families by total lineage count (sum of all descendants).
    Calculates the actual family tree size by traversing parent-child relationships.
    """
    if deps.df is None:
        raise DataNotLoadedError()
    
    df = deps.df
    
    # Build parent -> children mapping
    children_map = {}
    root_models = set()
    
    for idx, row in df.iterrows():
        model_id = str(row.get('model_id', ''))
        parent_id = row.get('parent_model')
        
        if pd.isna(parent_id) or str(parent_id) == 'nan' or str(parent_id) == '':
            root_models.add(model_id)
        else:
            parent_str = str(parent_id)
            if parent_str not in children_map:
                children_map[parent_str] = []
            children_map[parent_str].append(model_id)
    
    # For each root, count all descendants
    def count_descendants(model_id: str, visited: set) -> int:
        if model_id in visited:
            return 0
        visited.add(model_id)
        count = 1  # Count self
        for child in children_map.get(model_id, []):
            count += count_descendants(child, visited)
        return count
    
    # Calculate family sizes
    family_data = []
    for root in root_models:
        visited = set()
        total_count = count_descendants(root, visited)
        if total_count >= min_size:
            # Get organization from model_id
            org = root.split('/')[0] if '/' in root else root
            family_data.append({
                "root_model": root,
                "organization": org,
                "total_models": total_count,
                "depth_count": len(visited)  # Same as total for tree traversal
            })
    
    # Sort by total count descending
    family_data.sort(key=lambda x: x['total_models'], reverse=True)
    
    # Also aggregate by organization (sum all families under same org)
    org_totals = {}
    for fam in family_data:
        org = fam['organization']
        if org not in org_totals:
            org_totals[org] = {
                "organization": org,
                "total_models": 0,
                "family_count": 0,
                "root_models": []
            }
        org_totals[org]['total_models'] += fam['total_models']
        org_totals[org]['family_count'] += 1
        if len(org_totals[org]['root_models']) < 5:  # Keep top 5 root models
            org_totals[org]['root_models'].append(fam['root_model'])
    
    # Sort organizations by total models
    top_orgs = sorted(org_totals.values(), key=lambda x: x['total_models'], reverse=True)[:limit]
    
    return {
        "families": family_data[:limit],
        "organizations": top_orgs,
        "total_families": len(family_data),
        "total_root_models": len(root_models)
    }


@app.get("/api/family/path/{model_id}")
async def get_family_path(
    model_id: str,
    target_id: Optional[str] = Query(None, description="Target model ID. If None, returns path to root.")
):
    """
    Get path from model to root or to target model.
    Returns list of model IDs representing the path.
    """
    if df is None:
        raise DataNotLoadedError()
    
    model_id_str = str(model_id)
    
    if df.index.name == 'model_id':
        if model_id_str not in df.index:
            raise HTTPException(status_code=404, detail="Model not found")
    else:
        model_rows = df[df.get('model_id', '') == model_id_str]
        if len(model_rows) == 0:
            raise HTTPException(status_code=404, detail="Model not found")
    
    path = [model_id_str]
    visited = set([model_id_str])
    current = model_id_str
    
    if target_id:
        target_str = str(target_id)
        if df.index.name == 'model_id':
            if target_str not in df.index:
                raise HTTPException(status_code=404, detail="Target model not found")
        
        while current != target_str and current not in visited:
            try:
                if df.index.name == 'model_id':
                    row = df.loc[current]
                else:
                    rows = df[df.get('model_id', '') == current]
                    if len(rows) == 0:
                        break
                    row = rows.iloc[0]
                
                parent_id = row.get('parent_model')
                if parent_id and pd.notna(parent_id):
                    parent_str = str(parent_id)
                    if parent_str == target_str:
                        path.append(parent_str)
                        break
                    if parent_str not in visited:
                        path.append(parent_str)
                        visited.add(parent_str)
                        current = parent_str
                    else:
                        break
                else:
                    break
            except (KeyError, IndexError):
                break
    else:
        while True:
            try:
                if df.index.name == 'model_id':
                    row = df.loc[current]
                else:
                    rows = df[df.get('model_id', '') == current]
                    if len(rows) == 0:
                        break
                    row = rows.iloc[0]
                
                parent_id = row.get('parent_model')
                if parent_id and pd.notna(parent_id):
                    parent_str = str(parent_id)
                    if parent_str not in visited:
                        path.append(parent_str)
                        visited.add(parent_str)
                        current = parent_str
                    else:
                        break
                else:
                    break
            except (KeyError, IndexError):
                break
    
    return {
        "path": path,
        "source": model_id_str,
        "target": target_id if target_id else "root",
        "path_length": len(path) - 1
    }


@app.get("/api/family/{model_id}")
async def get_family_tree(
    model_id: str, 
    max_depth: Optional[int] = Query(None, ge=1, le=100, description="Maximum depth to traverse. If None, traverses entire tree without limit."),
    max_depth_filter: Optional[int] = Query(None, ge=0, description="Filter results to models at or below this hierarchy depth.")
):
    """
    Get family tree for a model (ancestors and descendants).
    Returns the model, its parent chain, and all children.
    
    If max_depth is None, traverses the entire family tree without depth limits.
    """
    if df is None:
        raise DataNotLoadedError()
    
    if reduced_embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not ready")
    
    model_id_str = str(model_id)
    
    if df.index.name == 'model_id':
        if model_id_str not in df.index:
            raise HTTPException(status_code=404, detail="Model not found")
        model_lookup = df.loc
    else:
        model_rows = df[df.get('model_id', '') == model_id_str]
        if len(model_rows) == 0:
            raise HTTPException(status_code=404, detail="Model not found")
        model_lookup = lambda x: df[df.get('model_id', '') == x]
    
    from utils.network_analysis import _get_all_parents, _parse_parent_list
    
    children_index: Dict[str, List[str]] = {}
    parent_columns = ['parent_model', 'finetune_parent', 'quantized_parent', 'adapter_parent', 'merge_parent']
    
    for idx, row in df.iterrows():
        model_id_from_row = str(row.get('model_id', idx))
        all_parents = _get_all_parents(row)
        
        for rel_type, parent_list in all_parents.items():
            for parent_str in parent_list:
                if parent_str not in children_index:
                    children_index[parent_str] = []
                children_index[parent_str].append(model_id_from_row)
    
    visited = set()
    
    def get_ancestors(current_id: str, depth: Optional[int]):
        if current_id in visited:
            return
        if depth is not None and depth <= 0:
            return
        visited.add(current_id)
        
        try:
            if df.index.name == 'model_id':
                row = df.loc[current_id]
            else:
                rows = model_lookup(current_id)
                if len(rows) == 0:
                    return
                row = rows.iloc[0]
            
            all_parents = _get_all_parents(row)
            for rel_type, parent_list in all_parents.items():
                for parent_str in parent_list:
                    if parent_str != 'nan' and parent_str != '':
                        next_depth = depth - 1 if depth is not None else None
                        get_ancestors(parent_str, next_depth)
        except (KeyError, IndexError):
            return
    
    def get_descendants(current_id: str, depth: Optional[int]):
        if current_id in visited:
            return
        if depth is not None and depth <= 0:
            return
        visited.add(current_id)
        
        children = children_index.get(current_id, [])
        for child_id in children:
            if child_id not in visited:
                next_depth = depth - 1 if depth is not None else None
                get_descendants(child_id, next_depth)
    
    get_ancestors(model_id_str, max_depth)
    visited = set()
    get_descendants(model_id_str, max_depth)
    visited.add(model_id_str)
    
    if df.index.name == 'model_id':
        try:
            family_df = df.loc[list(visited)]
        except KeyError:
            missing = [v for v in visited if v not in df.index]
            if missing:
                logger.warning(f"Some family members not found in index: {missing}")
            family_df = df.loc[[v for v in visited if v in df.index]]
    else:
        family_df = df[df.get('model_id', '').isin(visited)]
    
    if len(family_df) == 0:
        raise HTTPException(status_code=404, detail="Family tree data not available")
    
    family_indices = family_df.index.values
    if len(family_indices) > len(reduced_embeddings):
        raise HTTPException(status_code=503, detail="Embedding indices mismatch")
    
    family_reduced = reduced_embeddings[family_indices]
    
    family_map = {}
    for idx, (i, row) in enumerate(family_df.iterrows()):
        model_id_val = str(row.get('model_id', i))
        parent_id = row.get('parent_model')
        parent_id_str = str(parent_id) if parent_id and pd.notna(parent_id) else None
        
        depths = calculate_family_depths(df)
        model_depth = depths.get(model_id_val, 0)
        
        if max_depth_filter is not None and model_depth > max_depth_filter:
            continue
        
        family_map[model_id_val] = {
            "model_id": model_id_val,
            "x": float(family_reduced[idx, 0]),
            "y": float(family_reduced[idx, 1]),
            "z": float(family_reduced[idx, 2]) if family_reduced.shape[1] > 2 else 0.0,
            "library_name": str(row.get('library_name')) if pd.notna(row.get('library_name')) else None,
            "pipeline_tag": str(row.get('pipeline_tag')) if pd.notna(row.get('pipeline_tag')) else None,
            "downloads": int(row.get('downloads', 0)) if pd.notna(row.get('downloads')) else 0,
            "likes": int(row.get('likes', 0)) if pd.notna(row.get('likes')) else 0,
            "parent_model": parent_id_str,
            "licenses": str(row.get('licenses')) if pd.notna(row.get('licenses')) else None,
            "family_depth": model_depth,
            "children": []
        }
    
    root_models = []
    for model_id_val, model_data in family_map.items():
        parent_id = model_data["parent_model"]
        if parent_id and parent_id in family_map:
            family_map[parent_id]["children"].append(model_id_val)
        else:
            root_models.append(model_id_val)
    
    return {
        "root_model": model_id_str,
        "family": list(family_map.values()),
        "family_map": family_map,
        "root_models": root_models
    }


@app.get("/api/search")
async def search_models(
    q: str = Query(..., min_length=1, alias="query"),
    query: str = Query(None, min_length=1),
    limit: int = Query(20, ge=1, le=100),
    graph_aware: bool = Query(False),
    include_neighbors: bool = Query(True)
):
    """
    Search for models by name (for autocomplete and family tree lookup).
    Enhanced with graph-aware search option that includes network relationships.
    """
    if df is None:
        raise DataNotLoadedError()
    
    # Support both 'q' and 'query' parameters
    search_query = query or q
    
    if graph_aware:
        try:
            network_builder = ModelNetworkBuilder(df)
            top_models = network_builder.get_top_models_by_field(n=1000)
            model_ids = [mid for mid, _ in top_models]
            graph = network_builder.build_cooccurrence_network(model_ids, cooccurrence_method='combined')
            
            results = network_builder.search_graph_aware(
                query=search_query,
                graph=graph,
                max_results=limit,
                include_neighbors=include_neighbors
            )
            
            return {"results": results, "search_type": "graph_aware", "query": search_query}
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Graph-aware search failed, falling back to basic search: {e}")
    
    query_lower = search_query.lower()
    
    # Enhanced search: search model_id, org, tags, library, pipeline
    model_id_col = df.get('model_id', '').astype(str).str.lower()
    library_col = df.get('library_name', '').astype(str).str.lower()
    pipeline_col = df.get('pipeline_tag', '').astype(str).str.lower()
    tags_col = df.get('tags', '').astype(str).str.lower()
    license_col = df.get('license', '').astype(str).str.lower()
    
    # Extract org from model_id
    org_col = model_id_col.str.split('/').str[0]
    
    # Multi-field search
    mask = (
        model_id_col.str.contains(query_lower, na=False) |
        org_col.str.contains(query_lower, na=False) |
        library_col.str.contains(query_lower, na=False) |
        pipeline_col.str.contains(query_lower, na=False) |
        tags_col.str.contains(query_lower, na=False) |
        license_col.str.contains(query_lower, na=False)
    )
    
    matches = df[mask].head(limit)
    
    results = []
    for _, row in matches.iterrows():
        model_id = str(row.get('model_id', ''))
        org = model_id.split('/')[0] if '/' in model_id else ''
        
        # Get coordinates if available
        x = float(row.get('x', 0.0)) if 'x' in row else None
        y = float(row.get('y', 0.0)) if 'y' in row else None
        z = float(row.get('z', 0.0)) if 'z' in row else None
        
        results.append({
            "model_id": model_id,
            "x": x,
            "y": y,
            "z": z,
            "org": org,
            "library": row.get('library_name'),
            "pipeline": row.get('pipeline_tag'),
            "license": row.get('license') if pd.notna(row.get('license')) else None,
            "downloads": int(row.get('downloads', 0)),
            "likes": int(row.get('likes', 0)),
            "parent_model": row.get('parent_model') if pd.notna(row.get('parent_model')) else None,
            "match_type": "direct"
        })
    
    return {"results": results, "search_type": "basic", "query": search_query}


@app.get("/api/search/fuzzy")
async def fuzzy_search_models(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    threshold: int = Query(60, ge=0, le=100, description="Minimum fuzzy match score (0-100)"),
):
    """
    Fuzzy search for models using rapidfuzz.
    Handles typos and partial matches across model names, libraries, and pipelines.
    Returns results sorted by relevance score.
    """
    if deps.df is None:
        raise DataNotLoadedError()
    
    df = deps.df
    
    try:
        from rapidfuzz import fuzz, process
        from rapidfuzz.utils import default_process
        
        query_lower = q.lower().strip()
        
        # Prepare choices - combine model_id, library, and pipeline for searching
        # Create a searchable string for each model
        model_ids = df['model_id'].astype(str).tolist()
        libraries = df.get('library_name', pd.Series([''] * len(df))).fillna('').astype(str).tolist()
        pipelines = df.get('pipeline_tag', pd.Series([''] * len(df))).fillna('').astype(str).tolist()
        
        # Create search strings - just model_id for better fuzzy matching
        # Library and pipeline are used for secondary filtering
        search_strings = [m.lower() for m in model_ids]
        
        # Use rapidfuzz to find best matches
        # WRatio is best for general fuzzy matching with typo tolerance
        # It handles transpositions, insertions, deletions well
        
        # extract returns list of (match, score, index)
        matches = process.extract(
            query_lower,
            search_strings,
            scorer=fuzz.WRatio,
            limit=limit * 3,  # Get extra to filter by threshold and dedupe
            score_cutoff=threshold,
            processor=default_process
        )
        
        # Also try partial matching for substring searches
        if len(matches) < limit:
            partial_matches = process.extract(
                query_lower,
                search_strings,
                scorer=fuzz.partial_ratio,
                limit=limit * 2,
                score_cutoff=threshold + 10,  # Higher threshold for partial
                processor=default_process
            )
            # Add unique partial matches
            seen_indices = {m[2] for m in matches}
            for m in partial_matches:
                if m[2] not in seen_indices:
                    matches.append(m)
                    seen_indices.add(m[2])
        
        results = []
        seen_ids = set()
        
        for match_str, score, idx in matches:
            if len(results) >= limit:
                break
                
            model_id = model_ids[idx]
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            
            row = df.iloc[idx]
            
            # Get coordinates
            x = float(row.get('x', 0.0)) if 'x' in row else None
            y = float(row.get('y', 0.0)) if 'y' in row else None
            z = float(row.get('z', 0.0)) if 'z' in row else None
            
            results.append({
                "model_id": model_id,
                "x": x,
                "y": y,
                "z": z,
                "score": round(score, 1),
                "library": row.get('library_name') if pd.notna(row.get('library_name')) else None,
                "pipeline": row.get('pipeline_tag') if pd.notna(row.get('pipeline_tag')) else None,
                "downloads": int(row.get('downloads', 0)),
                "likes": int(row.get('likes', 0)),
                "family_depth": int(row.get('family_depth', 0)) if pd.notna(row.get('family_depth')) else None,
            })
        
        # Sort by score descending, then by downloads for tie-breaking
        results.sort(key=lambda x: (-x['score'], -x['downloads']))
        
        return {
            "results": results,
            "query": q,
            "total_matches": len(matches),
            "threshold": threshold
        }
        
    except ImportError:
        raise HTTPException(status_code=500, detail="rapidfuzz not installed")
    except Exception as e:
        logger.exception(f"Fuzzy search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/similar/{model_id}")
async def get_similar_models(model_id: str, k: int = Query(10, ge=1, le=50)):
    """
    Get k-nearest neighbors of a model based on embedding similarity.
    Returns similar models with distance scores.
    """
    if deps.df is None or deps.embeddings is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = deps.df
    embeddings = deps.embeddings
    
    if 'model_id' in df.index.names or df.index.name == 'model_id':
        try:
            model_row = df.loc[[model_id]]
            model_idx = model_row.index[0]
        except KeyError:
            raise HTTPException(status_code=404, detail="Model not found")
    else:
        model_row = df[df.get('model_id', '') == model_id]
        if len(model_row) == 0:
            raise HTTPException(status_code=404, detail="Model not found")
        model_idx = model_row.index[0]
    model_embedding = embeddings[model_idx]
    
    from sklearn.metrics.pairwise import cosine_similarity
    model_embedding_2d = model_embedding.reshape(1, -1)
    similarities = cosine_similarity(model_embedding_2d, embeddings)[0]
    
    top_k_indices = np.argpartition(similarities, -k-1)[-k-1:-1]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]
    
    similar_models = []
    for idx in top_k_indices:
        if idx == model_idx:
            continue
        row = df.iloc[idx]
        similar_models.append({
            "model_id": row.get('model_id', 'Unknown'),
            "similarity": float(similarities[idx]),
            "distance": float(1 - similarities[idx]),
            "library_name": row.get('library_name'),
            "pipeline_tag": row.get('pipeline_tag'),
            "downloads": int(row.get('downloads', 0)),
            "likes": int(row.get('likes', 0)),
        })
    
    return {
        "query_model": model_id,
        "similar_models": similar_models
    }


@app.get("/api/models/semantic-similarity")
async def get_models_by_semantic_similarity(
    query_model_id: str = Query(...),
    k: int = Query(100, ge=1, le=1000),
    min_downloads: int = Query(0),
    min_likes: int = Query(0),
    projection_method: str = Query("umap")
):
    """
    Get models sorted by semantic similarity to a query model.
    Returns models with their similarity scores and coordinates.
    Useful for exploring the embedding space around a specific model.
    """
    if deps.df is None or deps.embeddings is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = deps.df
    embeddings = deps.embeddings
    
    # Find the query model
    if 'model_id' in df.index.names or df.index.name == 'model_id':
        try:
            model_row = df.loc[[query_model_id]]
            model_idx = model_row.index[0]
        except KeyError:
            raise HTTPException(status_code=404, detail="Query model not found")
    else:
        model_row = df[df.get('model_id', '') == query_model_id]
        if len(model_row) == 0:
            raise HTTPException(status_code=404, detail="Query model not found")
        model_idx = model_row.index[0]
    
    query_embedding = embeddings[model_idx]
    
    filtered_df = data_loader.filter_data(
        df=df,
        min_downloads=min_downloads,
        min_likes=min_likes,
        search_query=None,
        libraries=None,
        pipeline_tags=None
    )
    
    if df.index.name == 'model_id' or 'model_id' in df.index.names:
        filtered_indices = [df.index.get_loc(idx) for idx in filtered_df.index]
        filtered_indices = np.array(filtered_indices, dtype=int)
    else:
        filtered_indices = filtered_df.index.values.astype(int)
    
    filtered_embeddings = embeddings[filtered_indices]
    from sklearn.metrics.pairwise import cosine_similarity
    query_embedding_2d = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_2d, filtered_embeddings)[0]
    
    top_k_local_indices = np.argpartition(similarities, -k)[-k:]
    top_k_local_indices = top_k_local_indices[np.argsort(similarities[top_k_local_indices])][::-1]
    
    if reduced_embeddings is None:
        raise HTTPException(status_code=503, detail="Reduced embeddings not ready")
    
    top_k_original_indices = filtered_indices[top_k_local_indices]
    top_k_reduced = reduced_embeddings[top_k_original_indices]
    
    similar_models = []
    for i, orig_idx in enumerate(top_k_original_indices):
        row = df.iloc[orig_idx]
        local_idx = top_k_local_indices[i]
        similar_models.append({
            "model_id": str(row.get('model_id', 'Unknown')),
            "x": float(top_k_reduced[i, 0]),
            "y": float(top_k_reduced[i, 1]),
            "z": float(top_k_reduced[i, 2]) if top_k_reduced.shape[1] > 2 else 0.0,
            "similarity": float(similarities[local_idx]),
            "distance": float(1 - similarities[local_idx]),
            "library_name": str(row.get('library_name')) if pd.notna(row.get('library_name')) else None,
            "pipeline_tag": str(row.get('pipeline_tag')) if pd.notna(row.get('pipeline_tag')) else None,
            "downloads": int(row.get('downloads', 0)),
            "likes": int(row.get('likes', 0)),
            "trending_score": float(row.get('trendingScore')) if pd.notna(row.get('trendingScore')) else None,
            "tags": str(row.get('tags')) if pd.notna(row.get('tags')) else None,
            "parent_model": str(row.get('parent_model')) if pd.notna(row.get('parent_model')) else None,
            "licenses": str(row.get('licenses')) if pd.notna(row.get('licenses')) else None,
        })
    
    return {
        "query_model": query_model_id,
        "models": similar_models,
        "count": len(similar_models)
    }


@app.get("/api/distance")
async def get_distance(
    model_id_1: str = Query(...),
    model_id_2: str = Query(...)
):
    """
    Calculate distance/similarity between two models.
    """
    if deps.df is None or deps.embeddings is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = deps.df
    embeddings = deps.embeddings
    
    # Find both models - optimized with index lookup
    if 'model_id' in df.index.names or df.index.name == 'model_id':
        try:
            model1_row = df.loc[[model_id_1]]
            model2_row = df.loc[[model_id_2]]
            idx1 = model1_row.index[0]
            idx2 = model2_row.index[0]
        except KeyError:
            raise HTTPException(status_code=404, detail="One or both models not found")
    else:
        model1_row = df[df.get('model_id', '') == model_id_1]
        model2_row = df[df.get('model_id', '') == model_id_2]
        if len(model1_row) == 0 or len(model2_row) == 0:
            raise HTTPException(status_code=404, detail="One or both models not found")
        idx1 = model1_row.index[0]
        idx2 = model2_row.index[0]
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0][0]
    distance = 1 - similarity
    
    return {
        "model_1": model_id_1,
        "model_2": model_id_2,
        "cosine_similarity": float(similarity),
        "cosine_distance": float(distance),
        "euclidean_distance": float(np.linalg.norm(embeddings[idx1] - embeddings[idx2]))
    }


@app.post("/api/export")
async def export_models(model_ids: List[str]):
    """
    Export selected models as JSON with full metadata.
    """
    if df is None:
        raise DataNotLoadedError()
    
    # Optimized export with index lookup
    if 'model_id' in df.index.names or df.index.name == 'model_id':
        try:
            exported = df.loc[model_ids]
        except KeyError:
            # Fallback if some IDs not in index
            exported = df[df.get('model_id', '').isin(model_ids)]
    else:
        exported = df[df.get('model_id', '').isin(model_ids)]
    
    if len(exported) == 0:
        return {"models": []}
    
    models = [
        {
            "model_id": str(row.get('model_id', '')),
            "library_name": str(row.get('library_name')) if pd.notna(row.get('library_name')) else None,
            "pipeline_tag": str(row.get('pipeline_tag')) if pd.notna(row.get('pipeline_tag')) else None,
            "downloads": int(row.get('downloads', 0)) if pd.notna(row.get('downloads')) else 0,
            "likes": int(row.get('likes', 0)) if pd.notna(row.get('likes')) else 0,
            "trending_score": float(row.get('trendingScore', 0)) if pd.notna(row.get('trendingScore')) else None,
            "tags": str(row.get('tags')) if pd.notna(row.get('tags')) else None,
            "licenses": str(row.get('licenses')) if pd.notna(row.get('licenses')) else None,
            "parent_model": str(row.get('parent_model')) if pd.notna(row.get('parent_model')) else None,
        }
        for _, row in exported.iterrows()
    ]
    
    return {
        "count": len(models),
        "models": models
    }


@app.get("/api/network/cooccurrence")
async def get_cooccurrence_network(
    library: Optional[str] = Query(None),
    pipeline_tag: Optional[str] = Query(None),
    min_downloads: int = Query(0),
    min_likes: int = Query(0),
    n: int = Query(100, ge=1, le=1000),
    cooccurrence_method: str = Query("combined", regex="^(parent_family|library|pipeline|tags|combined)$")
):
    """
    Build co-occurrence network for models (inspired by Open Syllabus Project).
    Connects models that appear together in same contexts (parent family, library, pipeline, tags).
    
    Returns network graph data suitable for visualization.
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        network_builder = ModelNetworkBuilder(df)
        top_models = network_builder.get_top_models_by_field(
            library=library,
            pipeline_tag=pipeline_tag,
            min_downloads=min_downloads,
            min_likes=min_likes,
            n=n
        )
        
        if not top_models:
            return {
                "nodes": [],
                "links": [],
                "statistics": {}
            }
        
        model_ids = [mid for mid, _ in top_models]
        graph = network_builder.build_cooccurrence_network(
            model_ids=model_ids,
            cooccurrence_method=cooccurrence_method
        )
        
        nodes = []
        for node_id, attrs in graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "title": attrs.get('title', node_id),
                "author": attrs.get('author', ''),
                "freq": attrs.get('freq', 0),
                "likes": attrs.get('likes', 0),
                "library": attrs.get('library', ''),
                "pipeline": attrs.get('pipeline', '')
            })
        
        links = []
        for source, target, attrs in graph.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "weight": attrs.get('weight', 1)
            })
        
        stats = network_builder.get_network_statistics(graph)
        
        return {
            "nodes": nodes,
            "links": links,
            "statistics": stats
        }
    except (ValueError, KeyError, AttributeError) as e:
        logger.error(f"Error building network: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error building network: {str(e)}")


@app.get("/api/network/family/{model_id}")
async def get_family_network(
    model_id: str,
    max_depth: Optional[int] = Query(None, ge=1, le=100, description="Maximum depth to traverse. If None, traverses entire tree without limit."),
    edge_types: Optional[str] = Query(None, description="Comma-separated list of edge types to include (finetune,quantized,adapter,merge,parent). If None, includes all types."),
    include_edge_attributes: bool = Query(True, description="Whether to include edge attributes (change in likes, downloads, etc.)")
):
    """
    Build family tree network for a model (directed graph).
    Returns network graph data showing parent-child relationships with multiple relationship types.
    Supports filtering by edge type (finetune, quantized, adapter, merge, parent).
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        filter_types = None
        if edge_types:
            filter_types = [t.strip() for t in edge_types.split(',') if t.strip()]
        
        network_builder = ModelNetworkBuilder(df)
        graph = network_builder.build_family_tree_network(
            root_model_id=model_id,
            max_depth=max_depth,
            include_edge_attributes=include_edge_attributes,
            filter_edge_types=filter_types
        )
        
        nodes = []
        for node_id, attrs in graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "title": attrs.get('title', node_id),
                "freq": attrs.get('freq', 0),
                "likes": attrs.get('likes', 0),
                "downloads": attrs.get('downloads', 0),
                "library": attrs.get('library', ''),
                "pipeline": attrs.get('pipeline', '')
            })
        
        links = []
        for source, target, edge_attrs in graph.edges(data=True):
            link_data = {
                "source": source,
                "target": target,
                "edge_type": edge_attrs.get('edge_type'),
                "edge_types": edge_attrs.get('edge_types', [])
            }
            
            if include_edge_attributes:
                link_data.update({
                    "change_in_likes": edge_attrs.get('change_in_likes'),
                    "percentage_change_in_likes": edge_attrs.get('percentage_change_in_likes'),
                    "change_in_downloads": edge_attrs.get('change_in_downloads'),
                    "percentage_change_in_downloads": edge_attrs.get('percentage_change_in_downloads'),
                    "change_in_createdAt_days": edge_attrs.get('change_in_createdAt_days')
                })
            
            links.append(link_data)
        
        stats = network_builder.get_network_statistics(graph)
        
        return {
            "nodes": nodes,
            "links": links,
            "statistics": stats,
            "root_model": model_id
        }
    except (ValueError, KeyError, AttributeError) as e:
        logger.error(f"Error building family network: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error building family network: {str(e)}")


@app.get("/api/network/full-derivatives")
@cached_response(ttl=3600, key_prefix="full_derivatives_network")
async def get_full_derivative_network(
    edge_types: Optional[str] = Query(None, description="Comma-separated list of edge types to include (finetune,quantized,adapter,merge,parent). If None, includes all types."),
    include_edge_attributes: bool = Query(False, description="Whether to include edge attributes (change in likes, downloads, etc.). Default False for performance.")
):
    """
    Build full derivative relationship network for ALL models in the database.
    Returns a non-embedding based force-directed graph where edges represent derivative types.
    This computes over every single model in the database.
    
    Note: Edge attributes are disabled by default for performance with large datasets.
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        import time
        start_time = time.time()
        logger.info(f"Building full derivative network for {len(df):,} models...")
        
        filter_types = None
        if edge_types:
            filter_types = [t.strip() for t in edge_types.split(',') if t.strip()]
        
        network_builder = ModelNetworkBuilder(df)
        logger.info("Calling build_full_derivative_network...")
        
        # Disable edge attributes for very large graphs to improve performance
        # They can be slow to compute for 100k+ edges
        graph = network_builder.build_full_derivative_network(
            include_edge_attributes=include_edge_attributes,
            filter_edge_types=filter_types
        )
        
        build_time = time.time() - start_time
        logger.info(f"Graph built in {build_time:.2f}s: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
        
        # Build nodes list
        nodes = []
        for node_id, attrs in graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "title": attrs.get('title', node_id),
                "freq": attrs.get('freq', 0),
                "likes": attrs.get('likes', 0),
                "downloads": attrs.get('downloads', 0),
                "library": attrs.get('library', ''),
                "pipeline": attrs.get('pipeline', '')
            })
        
        logger.info(f"Processed {len(nodes):,} nodes")
        
        # Build links list
        links = []
        edge_count = 0
        for source, target, edge_attrs in graph.edges(data=True):
            link_data = {
                "source": source,
                "target": target,
                "edge_type": edge_attrs.get('edge_type'),
                "edge_types": edge_attrs.get('edge_types', [])
            }
            
            if include_edge_attributes:
                link_data.update({
                    "change_in_likes": edge_attrs.get('change_in_likes'),
                    "percentage_change_in_likes": edge_attrs.get('percentage_change_in_likes'),
                    "change_in_downloads": edge_attrs.get('change_in_downloads'),
                    "percentage_change_in_downloads": edge_attrs.get('percentage_change_in_downloads'),
                    "change_in_createdAt_days": edge_attrs.get('change_in_createdAt_days')
                })
            
            links.append(link_data)
            edge_count += 1
            if edge_count % 10000 == 0:
                logger.info(f"Processed {edge_count:,} edges...")
        
        logger.info(f"Processed {len(links):,} links")
        
        stats = network_builder.get_network_statistics(graph)
        total_time = time.time() - start_time
        logger.info(f"Full derivative network built successfully in {total_time:.2f}s")
        
        return {
            "nodes": nodes,
            "links": links,
            "statistics": stats
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error building full derivative network: {e}\n{error_trace}")
        error_detail = f"Error building full derivative network: {str(e)}"
        if isinstance(e, (ValueError, KeyError, AttributeError)):
            error_detail += f" (Type: {type(e).__name__})"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/search/neighbors/{model_id}")
async def get_model_neighbors(
    model_id: str,
    max_neighbors: int = Query(50, ge=1, le=200),
    min_weight: float = Query(0.0, ge=0.0)
):
    """
    Find neighbors of a model in the co-occurrence network (graph-based search).
    Similar to graph database queries for finding connected nodes.
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        network_builder = ModelNetworkBuilder(df)
        top_models = network_builder.get_top_models_by_field(n=1000)
        model_ids = [mid for mid, _ in top_models]
        graph = network_builder.build_cooccurrence_network(model_ids, cooccurrence_method='combined')
        
        neighbors = network_builder.find_neighbors(
            model_id=model_id,
            graph=graph,
            max_neighbors=max_neighbors,
            min_weight=min_weight
        )
        
        return {
            "model_id": model_id,
            "neighbors": neighbors,
            "count": len(neighbors)
        }
    except (ValueError, KeyError, AttributeError) as e:
        logger.error(f"Error finding neighbors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error finding neighbors: {str(e)}")


@app.get("/api/search/path")
async def find_path_between_models(
    source_id: str = Query(...),
    target_id: str = Query(...),
    max_path_length: int = Query(5, ge=1, le=10)
):
    """
    Find shortest path between two models (graph-based search).
    Similar to graph database path queries.
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        network_builder = ModelNetworkBuilder(df)
        # Build network for top models (for performance)
        top_models = network_builder.get_top_models_by_field(n=1000)
        model_ids = [mid for mid, _ in top_models]
        graph = network_builder.build_cooccurrence_network(model_ids, cooccurrence_method='combined')
        
        path = network_builder.find_path(
            source_id=source_id,
            target_id=target_id,
            graph=graph,
            max_path_length=max_path_length
        )
        
        if path is None:
            return {
                "source_id": source_id,
                "target_id": target_id,
                "path": None,
                "path_length": None,
                "found": False
            }
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "path": path,
            "path_length": len(path) - 1,
            "found": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding path: {str(e)}")


@app.get("/api/search/cooccurrence/{model_id}")
async def search_by_cooccurrence(
    model_id: str,
    max_results: int = Query(20, ge=1, le=100),
    min_weight: float = Query(1.0, ge=0.0)
):
    """
    Search for models that co-occur with a query model.
    Similar to graph database queries for co-assignment patterns.
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        network_builder = ModelNetworkBuilder(df)
        # Build network for top models (for performance)
        top_models = network_builder.get_top_models_by_field(n=1000)
        model_ids = [mid for mid, _ in top_models]
        graph = network_builder.build_cooccurrence_network(model_ids, cooccurrence_method='combined')
        
        results = network_builder.search_by_cooccurrence(
            query_model_id=model_id,
            graph=graph,
            max_results=max_results,
            min_weight=min_weight
        )
        
        return {
            "query_model": model_id,
            "cooccurring_models": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching by co-occurrence: {str(e)}")


@app.get("/api/search/relationships/{model_id}")
async def get_model_relationships(
    model_id: str,
    relationship_type: str = Query("all", regex="^(family|library|pipeline|tags|all)$"),
    max_results: int = Query(50, ge=1, le=200)
):
    """
    Find models by specific relationship types (family, library, pipeline, tags).
    Similar to graph database relationship queries.
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        network_builder = ModelNetworkBuilder(df)
        related_models = network_builder.find_models_by_relationship(
            model_id=model_id,
            relationship_type=relationship_type,
            max_results=max_results
        )
        
        return {
            "model_id": model_id,
            "relationship_type": relationship_type,
            "related_models": related_models,
            "count": len(related_models)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding relationships: {str(e)}")


@app.get("/api/model-count/current")
async def get_current_model_count(
    use_cache: bool = Query(True),
    force_refresh: bool = Query(False),
    use_dataset_snapshot: bool = Query(False),
    use_models_page: bool = Query(True)
):
    """
    Get the current number of models on Hugging Face Hub.
    Uses multiple strategies: models page scraping (fastest), dataset snapshot, or API.
    
    Query Parameters:
        use_cache: Use cached results if available (default: True)
        force_refresh: Force refresh even if cache is valid (default: False)
        use_dataset_snapshot: Use dataset snapshot for breakdowns (default: False)
        use_models_page: Try to get count from HF models page first (default: True)
    """
    try:
        tracker = get_tracker()
        
        if use_dataset_snapshot:
            count_data = tracker.get_count_from_models_page()
            if count_data is None:
                count_data = tracker.get_current_model_count(use_models_page=False)
            else:
                try:
                    from utils.data_loader import ModelDataLoader
                    data_loader = ModelDataLoader()
                    df = data_loader.load_data(sample_size=10000, prioritize_base_models=True)
                    library_counts = {}
                    pipeline_counts = {}
                    
                    for _, row in df.iterrows():
                        if pd.notna(row.get('library_name')):
                            lib = str(row.get('library_name'))
                            library_counts[lib] = library_counts.get(lib, 0) + 1
                        if pd.notna(row.get('pipeline_tag')):
                            pipeline = str(row.get('pipeline_tag'))
                            pipeline_counts[pipeline] = pipeline_counts.get(pipeline, 0) + 1
                    
                    if len(df) > 0 and count_data["total_models"] > len(df):
                        scale_factor = count_data["total_models"] / len(df)
                        library_counts = {k: int(v * scale_factor) for k, v in library_counts.items()}
                        pipeline_counts = {k: int(v * scale_factor) for k, v in pipeline_counts.items()}
                    
                    count_data["models_by_library"] = library_counts
                    count_data["models_by_pipeline"] = pipeline_counts
                except Exception as e:
                    logger.warning(f"Could not get breakdowns from dataset: {e}")
        else:
            count_data = tracker.get_current_model_count(use_models_page=use_models_page)
        
        return count_data
    except Exception as e:
        logger.error(f"Error fetching model count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching model count: {str(e)}")


@app.get("/api/model-count/historical")
async def get_historical_model_counts(
    days: int = Query(30, ge=1, le=365),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """
    Get historical model counts.
    
    Args:
        days: Number of days to look back (if start_date not provided)
        start_date: Start date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        end_date: End date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    """
    try:
        from datetime import datetime
        
        tracker = get_tracker()
        
        start = None
        end = None
        
        if start_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start is None:
            from datetime import timedelta
            start = datetime.utcnow() - timedelta(days=days)
        
        historical = tracker.get_historical_counts(start, end)
        
        return {
            "counts": historical,
            "count": len(historical),
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical counts: {str(e)}")


@app.get("/api/model-count/latest")
async def get_latest_model_count():
    """Get the most recently recorded model count from database."""
    try:
        tracker = get_tracker()
        latest = tracker.get_latest_count()
        if latest is None:
            raise HTTPException(status_code=404, detail="No model counts recorded yet")
        return latest
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching latest count: {str(e)}")


@app.post("/api/model-count/record")
async def record_model_count(
    background_tasks: BackgroundTasks,
    use_dataset_snapshot: bool = Query(False, description="Use dataset snapshot instead of API (faster)")
):
    """
    Record the current model count to the database.
    This can be called periodically (e.g., via cron job) to track growth over time.
    
    Query Parameters:
        use_dataset_snapshot: Use dataset snapshot instead of API (faster, default: False)
    """
    try:
        tracker = get_tracker()
        
        def record():
            if use_dataset_snapshot:
                count_data = tracker.get_count_from_dataset_snapshot()
                if count_data:
                    tracker.record_count(count_data, source="dataset_snapshot")
                else:
                    count_data = tracker.get_current_model_count(use_cache=False)
                    tracker.record_count(count_data, source="api")
            else:
                count_data = tracker.get_current_model_count(use_cache=False)
                tracker.record_count(count_data, source="api")
        
        background_tasks.add_task(record)
        
        return {
            "status": "recording",
            "message": "Model count recording started in background",
            "source": "dataset_snapshot" if use_dataset_snapshot else "api"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording model count: {str(e)}")


@app.get("/api/model-count/growth")
async def get_growth_stats(days: int = Query(7, ge=1, le=365)):
    """
    Get growth statistics over the specified period.
    
    Args:
        days: Number of days to analyze
    """
    try:
        tracker = get_tracker()
        stats = tracker.get_growth_stats(days)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating growth stats: {str(e)}")


@app.get("/api/network/export/graphml")
async def export_network_graphml(
    background_tasks: BackgroundTasks,
    library: Optional[str] = Query(None),
    pipeline_tag: Optional[str] = Query(None),
    min_downloads: int = Query(0),
    min_likes: int = Query(0),
    n: int = Query(100, ge=1, le=1000),
    cooccurrence_method: str = Query("combined", regex="^(parent_family|library|pipeline|tags|combined)$")
):
    """
    Export co-occurrence network as GraphML file (for import into Gephi, Cytoscape, etc.).
    Similar to Open Syllabus graph export functionality.
    """
    if df is None:
        raise DataNotLoadedError()
    
    try:
        network_builder = ModelNetworkBuilder(df)
        
        top_models = network_builder.get_top_models_by_field(
            library=library,
            pipeline_tag=pipeline_tag,
            min_downloads=min_downloads,
            min_likes=min_likes,
            n=n
        )
        
        if not top_models:
            raise HTTPException(status_code=404, detail="No models found matching criteria")
        
        model_ids = [mid for mid, _ in top_models]
        graph = network_builder.build_cooccurrence_network(
            model_ids=model_ids,
            cooccurrence_method=cooccurrence_method
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            network_builder.export_graphml(graph, tmp_path)
        
        background_tasks.add_task(os.unlink, tmp_path)
        
        return FileResponse(
            tmp_path,
            media_type='application/xml',
            filename=f'network_{cooccurrence_method}_{n}_models.graphml'
        )
    except (ValueError, KeyError, AttributeError, IOError) as e:
        logger.error(f"Error exporting network: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error exporting network: {str(e)}")


@app.get("/api/model/{model_id}/papers")
async def get_model_papers(model_id: str):
    """
    Get arXiv papers associated with a model.
    Extracts arXiv IDs from model tags and fetches paper information.
    """
    if df is None:
        raise DataNotLoadedError()
    
    model = df[df.get('model_id', '') == model_id]
    if len(model) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = model.iloc[0]
    
    # Extract arXiv IDs from tags
    tags_str = str(model.get('tags', '')) if pd.notna(model.get('tags')) else ''
    arxiv_ids = extract_arxiv_ids(tags_str)
    
    if not arxiv_ids:
        return {
            "model_id": model_id,
            "arxiv_ids": [],
            "papers": []
        }
    
    # Fetch papers
    papers = await fetch_arxiv_papers(arxiv_ids[:10])  # Limit to 10 papers
    
    return {
        "model_id": model_id,
        "arxiv_ids": arxiv_ids,
        "papers": papers
    }


@app.get("/api/models/minimal.bin")
async def get_minimal_binary():
    """
    Serve the binary minimal dataset file.
    This is optimized for fast client-side loading.
    """
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(backend_dir)
    binary_path = os.path.join(root_dir, "cache", "binary", "embeddings.bin")
    
    if not os.path.exists(binary_path):
        raise HTTPException(status_code=404, detail="Binary dataset not found. Run export_binary.py first.")
    
    return FileResponse(
        binary_path,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=embeddings.bin",
            "Cache-Control": "public, max-age=3600"
        }
    )


@app.get("/api/models/model_ids.json")
async def get_model_ids_json():
    """Serve the model IDs JSON file."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(backend_dir)
    json_path = os.path.join(root_dir, "cache", "binary", "model_ids.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Model IDs file not found.")
    
    return FileResponse(
        json_path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"}
    )


@app.get("/api/models/metadata.json")
async def get_metadata_json():
    """Serve the metadata JSON file with lookup tables."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(backend_dir)
    json_path = os.path.join(root_dir, "cache", "binary", "metadata.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Metadata file not found.")
    
    return FileResponse(
        json_path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"}
    )


@app.get("/api/model/{model_id}/files")
async def get_model_files(model_id: str, branch: str = Query("main")):
    """
    Get file tree for a model from Hugging Face.
    Proxies the request to avoid CORS issues.
    Returns a flat list of files with path and size information.
    """
    if not model_id or not model_id.strip():
        raise HTTPException(status_code=400, detail="Invalid model ID")
    
    branches_to_try = [branch, "main", "master"] if branch not in ["main", "master"] else [branch, "main" if branch == "master" else "master"]
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for branch_name in branches_to_try:
                try:
                    url = f"https://huggingface.co/api/models/{model_id}/tree/{branch_name}"
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Ensure we return an array
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict) and 'tree' in data:
                            return data['tree']
                        else:
                            return []
                    
                    elif response.status_code == 404:
                        # Try next branch
                        continue
                    else:
                        logger.warning(f"Unexpected status {response.status_code} for {url}")
                        continue
                        
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        continue  # Try next branch
                    logger.warning(f"HTTP error for branch {branch_name}: {e}")
                    continue
                except httpx.HTTPError as e:
                    logger.warning(f"HTTP error for branch {branch_name}: {e}")
                    continue
            
            # All branches failed
            raise HTTPException(
                status_code=404, 
                detail=f"File tree not found for model '{model_id}'. The model may not exist or may not have any files."
            )
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, 
            detail="Request to Hugging Face timed out. Please try again later."
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error fetching file tree: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching file tree: {str(e)}"
        )


# =============================================================================
# BACKGROUND COMPUTATION ENDPOINTS
# =============================================================================

import subprocess
import threading

# Store for background process
_background_process = None
_background_lock = threading.Lock()


class ComputeRequest(BaseModel):
    sample_size: Optional[int] = None
    all_models: bool = False


@app.get("/api/compute/status")
async def get_compute_status():
    """Get the status of background pre-computation."""
    from pathlib import Path
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    status_file = Path(root_dir) / "precomputed_data" / "background_status_v1.json"
    
    if status_file.exists():
        import json
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        # Check if process is still running
        global _background_process
        with _background_lock:
            if _background_process is not None:
                poll = _background_process.poll()
                if poll is None:
                    status['process_running'] = True
                else:
                    status['process_running'] = False
                    status['process_exit_code'] = poll
            else:
                status['process_running'] = False
        
        return status
    
    # Check for existing precomputed data
    metadata_file = Path(root_dir) / "precomputed_data" / "metadata_v1.json"
    models_file = Path(root_dir) / "precomputed_data" / "models_v1.parquet"
    
    if metadata_file.exists() and models_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return {
            'status': 'completed',
            'total_models': metadata.get('total_models', 0),
            'created_at': metadata.get('created_at'),
            'process_running': False
        }
    
    return {
        'status': 'not_started',
        'total_models': 0,
        'process_running': False
    }


@app.post("/api/compute/start")
async def start_background_compute(request: ComputeRequest, background_tasks: BackgroundTasks):
    """Start background pre-computation of model embeddings."""
    global _background_process
    
    with _background_lock:
        if _background_process is not None and _background_process.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="Background computation is already running"
            )
    
    # Prepare command
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_path = os.path.join(root_dir, "backend", "scripts", "precompute_background.py")
    venv_python = os.path.join(root_dir, "venv", "bin", "python")
    
    cmd = [venv_python, script_path]
    
    if request.all_models:
        cmd.append("--all")
    elif request.sample_size:
        cmd.extend(["--sample-size", str(request.sample_size)])
    else:
        cmd.extend(["--sample-size", "150000"])  # Default
    
    cmd.extend(["--output-dir", os.path.join(root_dir, "precomputed_data")])
    
    # Start process in background
    log_file = os.path.join(root_dir, "precompute_background.log")
    
    def run_computation():
        global _background_process
        with open(log_file, 'w') as f:
            with _background_lock:
                _background_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.join(root_dir, "backend")
                )
            _background_process.wait()
    
    thread = threading.Thread(target=run_computation, daemon=True)
    thread.start()
    
    sample_desc = "all models" if request.all_models else f"{request.sample_size or 150000:,} models"
    
    return {
        "message": f"Background computation started for {sample_desc}",
        "status": "starting",
        "log_file": log_file
    }


@app.post("/api/compute/stop")
async def stop_background_compute():
    """Stop the running background computation."""
    global _background_process
    
    with _background_lock:
        if _background_process is None or _background_process.poll() is not None:
            return {"message": "No computation is running"}
        
        _background_process.terminate()
        try:
            _background_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _background_process.kill()
        
        return {"message": "Background computation stopped"}


@app.get("/api/data/info")
async def get_data_info():
    """Get information about currently loaded data."""
    df = deps.df
    
    if df is None:
        return {
            "loaded": False,
            "message": "No data loaded"
        }
    
    return {
        "loaded": True,
        "total_models": len(df),
        "columns": list(df.columns),
        "unique_libraries": int(df['library_name'].nunique()) if 'library_name' in df.columns else 0,
        "unique_pipelines": int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,
        "has_3d_coords": all(col in df.columns for col in ['x_3d', 'y_3d', 'z_3d']),
        "has_2d_coords": all(col in df.columns for col in ['x_2d', 'y_2d'])
    }


# =============================================================================
# STATIC FILE SERVING (for HF Spaces full-stack deployment)
# =============================================================================

from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse as StarletteFileResponse

# Check if frontend build exists (for HF Spaces deployment)
frontend_build_path = os.path.join(os.path.dirname(backend_dir), "frontend", "build")
if os.path.exists(frontend_build_path):
    # Serve static files from React build
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_build_path, "static")), name="static")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve React frontend for non-API routes."""
        # Don't serve frontend for API routes
        if full_path.startswith("api/") or full_path == "docs" or full_path == "openapi.json":
            raise HTTPException(status_code=404, detail="Not found")
        
        # Try to serve the requested file
        file_path = os.path.join(frontend_build_path, full_path)
        if os.path.isfile(file_path):
            return StarletteFileResponse(file_path)
        
        # Fall back to index.html for SPA routing
        index_path = os.path.join(frontend_build_path, "index.html")
        if os.path.exists(index_path):
            return StarletteFileResponse(index_path)
        
        raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

