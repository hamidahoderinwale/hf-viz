"""
FastAPI backend for serving model data to React/Visx frontend.
"""
import sys
import os
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from pydantic import BaseModel
from umap import UMAP
import tempfile
import traceback
import httpx

from utils.data_loader import ModelDataLoader
from utils.embeddings import ModelEmbedder
from utils.dimensionality_reduction import DimensionReducer
from utils.network_analysis import ModelNetworkBuilder
from services.model_tracker import get_tracker
from services.model_tracker_improved import get_improved_tracker
from services.arxiv_api import extract_arxiv_ids, fetch_arxiv_papers

app = FastAPI(title="HF Model Ecosystem API")

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that ensures CORS headers are included even on errors."""
    import traceback
    error_detail = str(exc)
    traceback_str = traceback.format_exc()
    import sys
    sys.stderr.write(f"Unhandled exception: {error_detail}\n{traceback_str}\n")
    return JSONResponse(
        status_code=500,
        content={"detail": error_detail, "error": "Internal server error"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP exception handler with CORS headers."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Validation exception handler with CORS headers."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# CORS middleware for React frontend
# Update allow_origins with your Netlify URL in production
# Note: Add your specific Netlify URL after deployment
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
# Allow all origins for development (restrict in production)
ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "true").lower() == "true"
if ALLOW_ALL_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=False,  # Must be False when allow_origins is ["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Local development
            FRONTEND_URL,  # Production frontend URL
            # Add your Netlify URL here after deployment, e.g.:
            # "https://your-app-name.netlify.app",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

data_loader = ModelDataLoader()
embedder: Optional[ModelEmbedder] = None
reducer: Optional[DimensionReducer] = None
df: Optional[pd.DataFrame] = None
embeddings: Optional[np.ndarray] = None
reduced_embeddings: Optional[np.ndarray] = None
cluster_labels: Optional[np.ndarray] = None  # Cached cluster assignments


class FilterParams(BaseModel):
    min_downloads: int = 0
    min_likes: int = 0
    search_query: Optional[str] = None
    libraries: Optional[List[str]] = None
    pipeline_tags: Optional[List[str]] = None


class ModelPoint(BaseModel):
    model_id: str
    x: float
    y: float
    z: float  # 3D coordinate
    library_name: Optional[str]
    pipeline_tag: Optional[str]
    downloads: int
    likes: int
    trending_score: Optional[float]
    tags: Optional[str]
    parent_model: Optional[str] = None
    licenses: Optional[str] = None
    family_depth: Optional[int] = None  # Generation depth in family tree (0 = root)
    cluster_id: Optional[int] = None    # Cluster assignment for visualization


@app.on_event("startup")
async def startup_event():
    """Initialize data and models on startup with caching."""
    global df, embedder, reducer, embeddings, reduced_embeddings
    
    import os
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(backend_dir)
    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    embeddings_cache = os.path.join(cache_dir, "embeddings.pkl")
    reduced_cache_umap = os.path.join(cache_dir, "reduced_umap_3d.pkl")
    reducer_cache_umap = os.path.join(cache_dir, "reducer_umap_3d.pkl")
    
    sample_size_env = os.getenv("SAMPLE_SIZE")
    if sample_size_env is None:
        sample_size = None
    else:
        sample_size = int(sample_size_env)
        if sample_size == 0:
            sample_size = None
    df = data_loader.load_data(sample_size=sample_size)
    df = data_loader.preprocess_for_embedding(df)
    
    if 'model_id' in df.columns:
        df.set_index('model_id', drop=False, inplace=True)
    for col in ['downloads', 'likes']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    embedder = ModelEmbedder()
    
    if os.path.exists(embeddings_cache):
        try:
            embeddings = embedder.load_embeddings(embeddings_cache)
        except Exception as e:
            embeddings = None
    
    if embeddings is None:
        texts = df['combined_text'].tolist()
        embeddings = embedder.generate_embeddings(texts, batch_size=128)
        embedder.save_embeddings(embeddings, embeddings_cache)
    
    reducer = DimensionReducer(method="umap", n_components=3)
    
    if os.path.exists(reduced_cache_umap) and os.path.exists(reducer_cache_umap):
        try:
            import pickle
            with open(reduced_cache_umap, 'rb') as f:
                reduced_embeddings = pickle.load(f)
            reducer.load_reducer(reducer_cache_umap)
        except Exception as e:
            reduced_embeddings = None
    
    if reduced_embeddings is None:
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
        reduced_embeddings = reducer.fit_transform(embeddings)
        import pickle
        with open(reduced_cache_umap, 'wb') as f:
            pickle.dump(reduced_embeddings, f)
        reducer.save_reducer(reducer_cache_umap)


def calculate_family_depths(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate family tree depth for each model.
    Returns a dictionary mapping model_id to depth (0 = root, 1 = first generation, etc.)
    """
    depths = {}
    visited = set()
    
    def get_depth(model_id: str) -> int:
        if model_id in depths:
            return depths[model_id]
        if model_id in visited:
            # Circular reference, treat as root
            depths[model_id] = 0
            return 0
        
        visited.add(model_id)
        
        if model_id not in df.index:
            depths[model_id] = 0
            return 0
        
        parent_id = df.loc[model_id].get('parent_model')
        if parent_id and pd.notna(parent_id) and str(parent_id) != 'nan' and str(parent_id) != '':
            parent_id_str = str(parent_id)
            if parent_id_str in df.index:
                depth = get_depth(parent_id_str) + 1
            else:
                depth = 0  # Parent not in dataset, treat as root
        else:
            depth = 0  # No parent, this is a root
        
        depths[model_id] = depth
        return depth
    
    for model_id in df.index:
        if model_id not in depths:
            visited = set()  # Reset for each tree
            get_depth(model_id)
    
    return depths


def compute_clusters(reduced_embeddings: np.ndarray, n_clusters: int = 50) -> np.ndarray:
    """
    Compute clusters using KMeans on reduced embeddings.
    Returns cluster labels for each point.
    """
    from sklearn.cluster import KMeans
    
    n_samples = len(reduced_embeddings)
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples // 10)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    return cluster_labels


@app.get("/")
async def root():
    return {"message": "HF Model Ecosystem API", "status": "running"}


@app.get("/api/models", response_model=List[ModelPoint])
async def get_models(
    min_downloads: int = Query(0),
    min_likes: int = Query(0),
    search_query: Optional[str] = Query(None),
    color_by: str = Query("library_name"),
    size_by: str = Query("downloads"),
    max_points: Optional[int] = Query(None),  # Optional limit (None = all points)
    projection_method: str = Query("umap"),  # umap or tsne
    base_models_only: bool = Query(False)  # Only show root models (no parent)
):
    """
    Get filtered models with 3D coordinates for visualization.
    Supports multiple projection methods: UMAP or t-SNE.
    If base_models_only=True, only returns root models (models without a parent_model).
    """
    global df, embedder, reducer, embeddings, reduced_embeddings
    
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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
    
    if len(filtered_df) == 0:
        return []
    
    if max_points is not None and len(filtered_df) > max_points:
        # Use stratified sampling to preserve distribution of important attributes
        # Sample proportionally from different libraries/pipelines for better representation
        if 'library_name' in filtered_df.columns and filtered_df['library_name'].notna().any():
            # Stratified sampling by library
            filtered_df = filtered_df.groupby('library_name', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(max_points * len(x) / len(filtered_df)))), random_state=42)
            ).reset_index(drop=True)
            # If still too many, random sample the rest
            if len(filtered_df) > max_points:
                filtered_df = filtered_df.sample(n=max_points, random_state=42)
        else:
            filtered_df = filtered_df.sample(n=max_points, random_state=42)
    
    if embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")
    
    if reduced_embeddings is None or (reducer and reducer.method != projection_method.lower()):
        import os
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_dir = os.path.dirname(backend_dir)
        cache_dir = os.path.join(root_dir, "cache")
        reduced_cache = os.path.join(cache_dir, f"reduced_{projection_method.lower()}_3d.pkl")
        reducer_cache = os.path.join(cache_dir, f"reducer_{projection_method.lower()}_3d.pkl")
        
        if os.path.exists(reduced_cache) and os.path.exists(reducer_cache):
            try:
                import pickle
                with open(reduced_cache, 'rb') as f:
                    reduced_embeddings = pickle.load(f)
                if reducer is None or reducer.method != projection_method.lower():
                    reducer = DimensionReducer(method=projection_method.lower(), n_components=3)
                reducer.load_reducer(reducer_cache)
            except Exception as e:
                reduced_embeddings = None
        
        if reduced_embeddings is None:
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
            reduced_embeddings = reducer.fit_transform(embeddings)
            import pickle
            with open(reduced_cache, 'wb') as f:
                pickle.dump(reduced_embeddings, f)
            reducer.save_reducer(reducer_cache)
    
    # Get coordinates for filtered data - optimized vectorized approach
    # Map filtered dataframe indices to original dataframe integer positions
    # Since df is indexed by model_id, we need to get the integer positions
    if df.index.name == 'model_id' or 'model_id' in df.index.names:
        # Get integer positions of filtered rows in original dataframe
        filtered_indices = [df.index.get_loc(idx) for idx in filtered_df.index]
        filtered_indices = np.array(filtered_indices, dtype=int)
    else:
        # If using integer index, use directly
        filtered_indices = filtered_df.index.values.astype(int)
    filtered_reduced = reduced_embeddings[filtered_indices]
    
    family_depths = calculate_family_depths(df)
    
    global cluster_labels
    if cluster_labels is None or len(cluster_labels) != len(reduced_embeddings):
        cluster_labels = compute_clusters(reduced_embeddings, n_clusters=min(50, len(reduced_embeddings) // 100))
    
    filtered_clusters = cluster_labels[filtered_indices]
    
    models = [
        ModelPoint(
            model_id=str(row['model_id']) if pd.notna(row.get('model_id')) else 'Unknown',
            x=float(filtered_reduced[idx, 0]),
            y=float(filtered_reduced[idx, 1]),
            z=float(filtered_reduced[idx, 2]) if filtered_reduced.shape[1] > 2 else 0.0,
            library_name=str(row['library_name']) if pd.notna(row.get('library_name')) else None,
            pipeline_tag=str(row['pipeline_tag']) if pd.notna(row.get('pipeline_tag')) else None,
            downloads=int(row.get('downloads', 0)) if pd.notna(row.get('downloads')) else 0,
            likes=int(row.get('likes', 0)) if pd.notna(row.get('likes')) else 0,
            trending_score=float(row['trendingScore']) if pd.notna(row.get('trendingScore')) else None,
            tags=str(row['tags']) if pd.notna(row.get('tags')) else None,
            parent_model=str(row['parent_model']) if pd.notna(row.get('parent_model')) else None,
            licenses=str(row['licenses']) if pd.notna(row.get('licenses')) else None,
            family_depth=family_depths.get(str(row['model_id']), None),
            cluster_id=int(filtered_clusters[idx]) if idx < len(filtered_clusters) else None
        )
        for idx, (i, row) in enumerate(filtered_df.iterrows())
    ]
    
    return models


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics."""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Use len(df.index) to handle both regular and indexed DataFrames correctly
    total_models = len(df.index) if hasattr(df, 'index') else len(df)
    
    return {
        "total_models": total_models,
        "unique_libraries": int(df['library_name'].nunique()) if 'library_name' in df.columns else 0,
        "unique_pipelines": int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,
        "unique_task_types": int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,  # Alias for clarity
        "avg_downloads": float(df['downloads'].mean()) if 'downloads' in df.columns else 0,
        "avg_likes": float(df['likes'].mean()) if 'likes' in df.columns else 0
    }


@app.get("/api/model/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed information about a specific model."""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    model = df[df.get('model_id', '') == model_id]
    if len(model) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = model.iloc[0]
    
    # Extract arXiv IDs from tags
    tags_str = str(model.get('tags', '')) if pd.notna(model.get('tags')) else ''
    arxiv_ids = extract_arxiv_ids(tags_str)
    
    # Fetch arXiv papers if any IDs found
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


@app.get("/api/family/stats")
async def get_family_stats():
    """
    Get aggregate statistics about family trees for paper visualizations.
    Returns family size distribution, depth statistics, model card length by depth, etc.
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Calculate family sizes
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
            # Find root of this family
            root = parent_id_str
            visited = set()
            while root in df.index and pd.notna(df.loc[root].get('parent_model')):
                parent = df.loc[root].get('parent_model')
                if pd.isna(parent) or str(parent) == 'nan' or str(parent) == '':
                    break
                if str(parent) in visited:  # Circular reference
                    break
                visited.add(root)
                root = str(parent)
            
            if root not in family_sizes:
                family_sizes[root] = 0
            family_sizes[root] += 1
    
    # Count family sizes
    size_distribution = {}
    for root, size in family_sizes.items():
        size_distribution[size] = size_distribution.get(size, 0) + 1
    
    # Calculate depth statistics
    depths = calculate_family_depths(df)
    depth_counts = {}
    for depth in depths.values():
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    
    # Calculate model card length by depth
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
    
    # Calculate statistics for each depth
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


@app.get("/api/family/{model_id}")
async def get_family_tree(model_id: str, max_depth: int = Query(5, ge=1, le=10)):
    """
    Get family tree for a model (ancestors and descendants).
    Returns the model, its parent chain, and all children.
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Find the model
    model_row = df[df.get('model_id', '') == model_id]
    if len(model_row) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    
    family_models = []
    visited = set()
    
    # Get coordinates for family members
    if reduced_embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not ready")
    
    # Optimize: create parent_model index for faster lookups
    if 'parent_model' not in df.index.names and 'parent_model' in df.columns:
        # Create a reverse index for faster parent lookups
        parent_index = df[df['parent_model'].notna()].set_index('parent_model', drop=False, append=True)
    
    def get_ancestors(current_id: str, depth: int):
        """Recursively get parent chain - optimized with index lookup."""
        if depth <= 0 or current_id in visited:
            return
        visited.add(current_id)
        
        # Use index lookup if available, otherwise fallback to query
        if 'model_id' in df.index.names or df.index.name == 'model_id':
            try:
                model = df.loc[[current_id]]
            except KeyError:
                return
        else:
            model = df[df.get('model_id', '') == current_id]
            if len(model) == 0:
                return
            model = model.iloc[[0]]
        
        parent_id = model.iloc[0].get('parent_model')
        
        if parent_id and pd.notna(parent_id) and str(parent_id) != 'nan':
            get_ancestors(str(parent_id), depth - 1)
    
    def get_descendants(current_id: str, depth: int):
        """Recursively get all children - optimized with index lookup."""
        if depth <= 0 or current_id in visited:
            return
        visited.add(current_id)
        
        # Use optimized parent lookup
        if 'parent_model' in df.columns:
            children = df[df['parent_model'] == current_id]
            # Use vectorized iteration
            child_ids = children['model_id'].dropna().astype(str).unique()
            for child_id in child_ids:
                if child_id not in visited:
                    get_descendants(child_id, depth - 1)
    
    # Get ancestors (parents)
    get_ancestors(model_id, max_depth)
    
    # Get descendants (children)
    visited = set()  # Reset for descendants
    get_descendants(model_id, max_depth)
    
    # Add the root model
    visited.add(model_id)
    
    # Get all family members with coordinates - optimized
    if 'model_id' in df.index.names or df.index.name == 'model_id':
        # Use index lookup if available
        try:
            family_df = df.loc[list(visited)]
        except KeyError:
            # Fallback to isin if some IDs not in index
            family_df = df[df.get('model_id', '').isin(visited)]
    else:
        family_df = df[df.get('model_id', '').isin(visited)]
    
    family_indices = family_df.index.values  # Use values instead of tolist() for speed
    family_reduced = reduced_embeddings[family_indices]
    
    # Build family tree structure - optimized with vectorized operations
    family_map = {}
    for idx, (i, row) in enumerate(family_df.iterrows()):
        model_id_val = str(row.get('model_id', 'Unknown'))
        parent_id = row.get('parent_model') if pd.notna(row.get('parent_model')) else None
        
        family_map[model_id_val] = {
            "model_id": model_id_val,
            "x": float(family_reduced[idx, 0]),
            "y": float(family_reduced[idx, 1]),
            "z": float(family_reduced[idx, 2]) if family_reduced.shape[1] > 2 else 0.0,
            "library_name": str(row.get('library_name')) if pd.notna(row.get('library_name')) else None,
            "pipeline_tag": str(row.get('pipeline_tag')) if pd.notna(row.get('pipeline_tag')) else None,
            "downloads": int(row.get('downloads', 0)) if pd.notna(row.get('downloads')) else 0,
            "likes": int(row.get('likes', 0)) if pd.notna(row.get('likes')) else 0,
            "parent_model": str(parent_id) if parent_id else None,
            "licenses": str(row.get('licenses')) if pd.notna(row.get('licenses')) else None,
            "children": []
        }
    
    # Build tree structure
    root_models = []
    for model_id_val, model_data in family_map.items():
        parent_id = model_data["parent_model"]
        if parent_id and parent_id in family_map:
            family_map[parent_id]["children"].append(model_id_val)
        else:
            root_models.append(model_id_val)
    
    return {
        "root_model": model_id,
        "family": list(family_map.values()),
        "family_map": family_map,
        "root_models": root_models
    }


@app.get("/api/search")
async def search_models(
    query: str = Query(..., min_length=1),
    graph_aware: bool = Query(False),
    include_neighbors: bool = Query(True)
):
    """
    Search for models by name (for autocomplete and family tree lookup).
    Enhanced with graph-aware search option that includes network relationships.
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if graph_aware:
        # Use graph-aware search
        try:
            network_builder = ModelNetworkBuilder(df)
            # Build network for top models (for performance)
            top_models = network_builder.get_top_models_by_field(n=1000)
            model_ids = [mid for mid, _ in top_models]
            graph = network_builder.build_cooccurrence_network(model_ids, cooccurrence_method='combined')
            
            results = network_builder.search_graph_aware(
                query=query,
                graph=graph,
                max_results=20,
                include_neighbors=include_neighbors
            )
            
            return {"results": results, "search_type": "graph_aware"}
        except Exception as e:
            pass
    
    query_lower = query.lower()
    matches = df[
        df.get('model_id', '').astype(str).str.lower().str.contains(query_lower, na=False)
    ].head(20)  # Limit to 20 results
    
    results = []
    for _, row in matches.iterrows():
        results.append({
            "model_id": row.get('model_id'),
            "title": row.get('model_id', '').split('/')[-1] if '/' in str(row.get('model_id', '')) else str(row.get('model_id', '')),
            "library_name": row.get('library_name'),
            "pipeline_tag": row.get('pipeline_tag'),
            "downloads": int(row.get('downloads', 0)),
            "likes": int(row.get('likes', 0)),
            "parent_model": row.get('parent_model') if pd.notna(row.get('parent_model')) else None,
            "match_type": "direct"
        })
    
    return {"results": results, "search_type": "basic"}


@app.get("/api/similar/{model_id}")
async def get_similar_models(model_id: str, k: int = Query(10, ge=1, le=50)):
    """
    Get k-nearest neighbors of a model based on embedding similarity.
    Returns similar models with distance scores.
    """
    global df, embedder, embeddings, reduced_embeddings
    
    if df is None or embeddings is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Find the model - optimized with index lookup
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
    
    # Calculate cosine similarity to all other models - optimized
    from sklearn.metrics.pairwise import cosine_similarity
    # Use vectorized operations for better performance
    model_embedding_2d = model_embedding.reshape(1, -1)
    similarities = cosine_similarity(model_embedding_2d, embeddings)[0]
    
    # Get top k similar models (excluding itself) - use argpartition for speed
    # argpartition is faster than full sort for top-k
    top_k_indices = np.argpartition(similarities, -k-1)[-k-1:-1]
    # Sort only the top k (much faster than sorting all)
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]
    
    similar_models = []
    for idx in top_k_indices:
        if idx == model_idx:
            continue
        row = df.iloc[idx]
        similar_models.append({
            "model_id": row.get('model_id', 'Unknown'),
            "similarity": float(similarities[idx]),
            "distance": float(1 - similarities[idx]),  # Convert similarity to distance
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
    global df, embedder, embeddings, reduced_embeddings
    
    if df is None or embeddings is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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
    
    # Filter by downloads/likes first for performance
    filtered_df = data_loader.filter_data(
        df=df,
        min_downloads=min_downloads,
        min_likes=min_likes,
        search_query=None,
        libraries=None,
        pipeline_tags=None
    )
    
    # Get indices of filtered models
    if df.index.name == 'model_id' or 'model_id' in df.index.names:
        filtered_indices = [df.index.get_loc(idx) for idx in filtered_df.index]
        filtered_indices = np.array(filtered_indices, dtype=int)
    else:
        filtered_indices = filtered_df.index.values.astype(int)
    
    # Calculate similarities only for filtered models
    filtered_embeddings = embeddings[filtered_indices]
    from sklearn.metrics.pairwise import cosine_similarity
    query_embedding_2d = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_2d, filtered_embeddings)[0]
    
    # Get top k similar models
    top_k_local_indices = np.argpartition(similarities, -k)[-k:]
    top_k_local_indices = top_k_local_indices[np.argsort(similarities[top_k_local_indices])][::-1]
    
    # Get reduced embeddings for visualization
    if reduced_embeddings is None:
        raise HTTPException(status_code=503, detail="Reduced embeddings not ready")
    
    # Map back to original indices
    top_k_original_indices = filtered_indices[top_k_local_indices]
    top_k_reduced = reduced_embeddings[top_k_original_indices]
    
    # Build response
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
    global df, embedder, embeddings
    
    if df is None or embeddings is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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
    
    # Use list comprehension for faster building
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
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        network_builder = ModelNetworkBuilder(df)
        
        # Get top models by field
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
        
        # Build co-occurrence network
        graph = network_builder.build_cooccurrence_network(
            model_ids=model_ids,
            cooccurrence_method=cooccurrence_method
        )
        
        # Convert to JSON-serializable format
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building network: {str(e)}")


@app.get("/api/network/family/{model_id}")
async def get_family_network(
    model_id: str,
    max_depth: int = Query(5, ge=1, le=10)
):
    """
    Build family tree network for a model (directed graph).
    Returns network graph data showing parent-child relationships.
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        network_builder = ModelNetworkBuilder(df)
        graph = network_builder.build_family_tree_network(
            root_model_id=model_id,
            max_depth=max_depth
        )
        
        # Convert to JSON-serializable format
        nodes = []
        for node_id, attrs in graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "title": attrs.get('title', node_id),
                "freq": attrs.get('freq', 0)
            })
        
        links = []
        for source, target in graph.edges():
            links.append({
                "source": source,
                "target": target
            })
        
        stats = network_builder.get_network_statistics(graph)
        
        return {
            "nodes": nodes,
            "links": links,
            "statistics": stats,
            "root_model": model_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building family network: {str(e)}")


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
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        network_builder = ModelNetworkBuilder(df)
        # Build network for top models (for performance)
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
    
    except Exception as e:
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
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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
    use_dataset_snapshot: bool = Query(False)
):
    """
    Get the current number of models on Hugging Face Hub.
    Fetches live data from the Hub API or uses dataset snapshot (faster but may be outdated).
    
    Query Parameters:
        use_cache: Use cached results if available (default: True)
        force_refresh: Force refresh even if cache is valid (default: False)
        use_dataset_snapshot: Use dataset snapshot instead of API (faster, default: False)
    """
    try:
        if use_dataset_snapshot:
            # Use improved tracker with dataset snapshot (like ai-ecosystem repo)
            tracker = get_improved_tracker()
            count_data = tracker.get_count_from_dataset_snapshot()
            if count_data is None:
                # Fallback to API if dataset unavailable
                count_data = tracker.get_current_model_count(use_cache=use_cache, force_refresh=force_refresh)
        else:
            # Use improved tracker with API (has caching)
            tracker = get_improved_tracker()
            count_data = tracker.get_current_model_count(use_cache=use_cache, force_refresh=force_refresh)
        
        return count_data
    except Exception as e:
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
        
        tracker = get_improved_tracker()
        
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
        tracker = get_improved_tracker()
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
        tracker = get_improved_tracker()
        
        # Fetch and record in background to avoid blocking
        def record():
            if use_dataset_snapshot:
                count_data = tracker.get_count_from_dataset_snapshot()
                if count_data:
                    tracker.record_count(count_data, source="dataset_snapshot")
                else:
                    # Fallback to API
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
        tracker = get_improved_tracker()
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
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        network_builder = ModelNetworkBuilder(df)
        
        # Get top models by field
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
        
        # Build co-occurrence network
        graph = network_builder.build_cooccurrence_network(
            model_ids=model_ids,
            cooccurrence_method=cooccurrence_method
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            network_builder.export_graphml(graph, tmp_path)
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(os.unlink, tmp_path)
        
        # Return file for download
        return FileResponse(
            tmp_path,
            media_type='application/xml',
            filename=f'network_{cooccurrence_method}_{n}_models.graphml'
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting network: {str(e)}")


@app.get("/api/model/{model_id}/papers")
async def get_model_papers(model_id: str):
    """
    Get arXiv papers associated with a model.
    Extracts arXiv IDs from model tags and fetches paper information.
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
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


@app.get("/api/model/{model_id}/files")
async def get_model_files(model_id: str, branch: str = Query("main")):
    """
    Get file tree for a model from Hugging Face.
    Proxies the request to avoid CORS issues.
    """
    try:
        # Try main branch first, then master
        branches_to_try = [branch, "main", "master"] if branch not in ["main", "master"] else [branch, "main" if branch == "master" else "master"]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for branch_name in branches_to_try:
                try:
                    url = f"https://huggingface.co/api/models/{model_id}/tree/{branch_name}"
                    response = await client.get(url)
                    if response.status_code == 200:
                        return response.json()
                except Exception:
                    continue
            
            raise HTTPException(status_code=404, detail="File tree not found for this model")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Hugging Face timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching file tree: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for cloud platforms (Railway, Render, Heroku)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

