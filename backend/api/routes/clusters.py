"""
API routes for cluster endpoints.
"""
from fastapi import APIRouter
import numpy as np
import pandas as pd
from core.exceptions import DataNotLoadedError
import api.dependencies as deps

router = APIRouter(prefix="/api", tags=["clusters"])


@router.get("/clusters")
async def get_clusters():
    """Get all clusters with metadata and hierarchical labels."""
    if deps.df is None:
        raise DataNotLoadedError()
    
    # Import cluster_labels from models route
    from api.routes.models import cluster_labels
    
    # If clusters haven't been computed yet, return empty list instead of error
    # This allows the frontend to work while data is still loading
    if cluster_labels is None:
        return {"clusters": []}
    
    df = deps.df
    
    # Generate hierarchical labels for clusters
    clusters = []
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_models = df[cluster_mask]
        
        if len(cluster_models) == 0:
            continue
        
        # Generate hierarchical label
        library_counts = cluster_models['library_name'].value_counts()
        pipeline_counts = cluster_models['pipeline_tag'].value_counts()
        
        # Determine primary domain/library
        if len(library_counts) > 0:
            primary_lib = library_counts.index[0]
            if primary_lib and pd.notna(primary_lib):
                if 'transformers' in str(primary_lib).lower():
                    domain = "NLP"
                elif 'diffusers' in str(primary_lib).lower():
                    domain = "Multimodal"
                elif 'timm' in str(primary_lib).lower():
                    domain = "Computer Vision"
                else:
                    domain = str(primary_lib).replace('_', ' ').title()
            else:
                domain = "Other"
        else:
            domain = "Other"
        
        # Determine subdomain from pipeline
        if len(pipeline_counts) > 0:
            primary_pipeline = pipeline_counts.index[0]
            if primary_pipeline and pd.notna(primary_pipeline):
                subdomain = str(primary_pipeline).replace('-', ' ').replace('_', ' ').title()
            else:
                subdomain = "General"
        else:
            subdomain = "General"
        
        # Determine characteristics
        characteristics = []
        model_ids_lower = cluster_models['model_id'].astype(str).str.lower()
        if model_ids_lower.str.contains('gpt', na=False).any():
            characteristics.append("GPT-based")
        if cluster_models['parent_model'].notna().any():
            characteristics.append("Fine-tuned")
        if not characteristics:
            characteristics.append("Base Models")
        
        char_str = "; ".join(characteristics)
        label = f"{domain} — {subdomain} ({char_str})"
        
        # Generate color (use consistent colors based on cluster_id)
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        color = colors[cluster_id % len(colors)]
        
        clusters.append({
            "cluster_id": int(cluster_id),
            "cluster_label": label,
            "count": int(len(cluster_models)),
            "color": color
        })
    
    # Sort by count descending
    clusters.sort(key=lambda x: x["count"], reverse=True)
    
    return {"clusters": clusters}

