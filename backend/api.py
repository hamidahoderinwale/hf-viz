"""
FastAPI backend for serving model data to React/Visx frontend.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import numpy as np
import os
from pydantic import BaseModel

from data_loader import ModelDataLoader
from embeddings import ModelEmbedder
from dimensionality_reduction import DimensionReducer

app = FastAPI(title="HF Model Ecosystem API")

# CORS middleware for React frontend
# Update allow_origins with your Netlify URL in production
# Note: Add your specific Netlify URL after deployment
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
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

# Global state
data_loader = ModelDataLoader()
embedder: Optional[ModelEmbedder] = None
reducer: Optional[DimensionReducer] = None
df: Optional[pd.DataFrame] = None
embeddings: Optional[np.ndarray] = None
reduced_embeddings: Optional[np.ndarray] = None


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
    library_name: Optional[str]
    pipeline_tag: Optional[str]
    downloads: int
    likes: int
    trending_score: Optional[float]
    tags: Optional[str]


@app.on_event("startup")
async def startup_event():
    """Initialize data and models on startup."""
    global df, embedder, reducer
    
    print("Loading data...")
    df = data_loader.load_data(sample_size=10000)
    df = data_loader.preprocess_for_embedding(df)
    
    print("Initializing embedder...")
    embedder = ModelEmbedder()
    
    print("Initializing reducer...")
    reducer = DimensionReducer(method="umap", n_components=2)
    
    print("API ready!")


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
    max_points: int = Query(5000)
):
    """
    Get filtered models with 2D coordinates for visualization.
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
    
    if len(filtered_df) == 0:
        return []
    
    # Limit points
    if len(filtered_df) > max_points:
        filtered_df = filtered_df.sample(n=max_points, random_state=42)
    
    # Generate embeddings if needed
    if embeddings is None:
        texts = df['combined_text'].tolist()
        embeddings = embedder.generate_embeddings(texts)
    
    # Reduce dimensions if needed
    if reduced_embeddings is None:
        reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Get coordinates for filtered data
    filtered_indices = filtered_df.index.tolist()
    filtered_reduced = reduced_embeddings[filtered_indices]
    
    # Prepare response
    models = []
    for idx, (i, row) in enumerate(filtered_df.iterrows()):
        models.append(ModelPoint(
            model_id=row.get('model_id', 'Unknown'),
            x=float(filtered_reduced[idx, 0]),
            y=float(filtered_reduced[idx, 1]),
            library_name=row.get('library_name'),
            pipeline_tag=row.get('pipeline_tag'),
            downloads=int(row.get('downloads', 0)),
            likes=int(row.get('likes', 0)),
            trending_score=float(row.get('trendingScore', 0)) if pd.notna(row.get('trendingScore')) else None,
            tags=row.get('tags') if pd.notna(row.get('tags')) else None
        ))
    
    return models


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics."""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "total_models": len(df),
        "unique_libraries": df['library_name'].nunique() if 'library_name' in df.columns else 0,
        "unique_pipelines": df['pipeline_tag'].nunique() if 'pipeline_tag' in df.columns else 0,
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
    return {
        "model_id": model.get('model_id'),
        "library_name": model.get('library_name'),
        "pipeline_tag": model.get('pipeline_tag'),
        "downloads": int(model.get('downloads', 0)),
        "likes": int(model.get('likes', 0)),
        "trending_score": float(model.get('trendingScore', 0)) if pd.notna(model.get('trendingScore')) else None,
        "tags": model.get('tags') if pd.notna(model.get('tags')) else None,
        "licenses": model.get('licenses') if pd.notna(model.get('licenses')) else None,
        "parent_model": model.get('parent_model') if pd.notna(model.get('parent_model')) else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

