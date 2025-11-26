"""
API routes for statistics endpoints.
"""
from fastapi import APIRouter
from core.exceptions import DataNotLoadedError
import api.dependencies as deps

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
async def get_stats():
    """Get dataset statistics."""
    if deps.df is None:
        raise DataNotLoadedError()
    
    df = deps.df
    total_models = len(df.index) if hasattr(df, 'index') else len(df)
    
    # Get unique licenses with counts
    licenses = {}
    if 'license' in df.columns:
        import pandas as pd
        license_counts = df['license'].value_counts().to_dict()
        licenses = {str(k): int(v) for k, v in license_counts.items() if pd.notna(k) and str(k) != 'nan'}
    
    return {
        "total_models": total_models,
        "unique_libraries": int(df['library_name'].nunique()) if 'library_name' in df.columns else 0,
        "unique_pipelines": int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,
        "unique_task_types": int(df['pipeline_tag'].nunique()) if 'pipeline_tag' in df.columns else 0,
        "unique_licenses": len(licenses),
        "licenses": licenses,
        "avg_downloads": float(df['downloads'].mean()) if 'downloads' in df.columns else 0,
        "avg_likes": float(df['likes'].mean()) if 'likes' in df.columns else 0
    }

