"""
Netlify Serverless Function for model data API.
This is a simplified version that works with Netlify Functions.
"""
import json
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import ModelDataLoader
from embeddings import ModelEmbedder
from dimensionality_reduction import DimensionReducer
import pandas as pd
import numpy as np

# Global state (persists across invocations in serverless)
data_loader = ModelDataLoader()
embedder = None
reducer = None
df = None
embeddings = None
reduced_embeddings = None


def handler(event, context):
    """
    Netlify serverless function handler.
    """
    global embedder, reducer, df, embeddings, reduced_embeddings
    
    # Parse query parameters
    query_params = event.get('queryStringParameters') or {}
    path = event.get('path', '')
    
    # CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Content-Type': 'application/json',
    }
    
    # Handle OPTIONS (CORS preflight)
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    # Initialize data on first request
    if df is None:
        try:
            print("Loading data...")
            df = data_loader.load_data(sample_size=10000)
            df = data_loader.preprocess_for_embedding(df)
            print(f"Loaded {len(df)} models")
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': f'Failed to load data: {str(e)}'})
            }
    
    # Route requests
    if path.endswith('/api/models') or '/api/models' in path:
        return get_models(query_params, headers)
    elif path.endswith('/api/stats') or '/api/stats' in path:
        return get_stats(headers)
    else:
        return {
            'statusCode': 404,
            'headers': headers,
            'body': json.dumps({'error': 'Not found'})
        }


def get_models(query_params, headers):
    """Get filtered models."""
    global df, embedder, reducer, embeddings, reduced_embeddings
    
    try:
        min_downloads = int(query_params.get('min_downloads', 0))
        min_likes = int(query_params.get('min_likes', 0))
        search_query = query_params.get('search_query')
        max_points = int(query_params.get('max_points', 5000))
        
        # Filter data
        filtered_df = data_loader.filter_data(
            df=df,
            min_downloads=min_downloads,
            min_likes=min_likes,
            search_query=search_query
        )
        
        if len(filtered_df) == 0:
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps([])
            }
        
        # Limit points
        if len(filtered_df) > max_points:
            filtered_df = filtered_df.sample(n=max_points, random_state=42)
        
        # Generate embeddings if needed
        if embedder is None:
            embedder = ModelEmbedder()
        
        if embeddings is None:
            texts = df['combined_text'].tolist()
            embeddings = embedder.generate_embeddings(texts)
        
        # Reduce dimensions if needed
        if reducer is None:
            reducer = DimensionReducer(method="umap", n_components=2)
        
        if reduced_embeddings is None:
            reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Get coordinates
        filtered_indices = filtered_df.index.tolist()
        filtered_reduced = reduced_embeddings[filtered_indices]
        
        # Prepare response
        models = []
        for idx, (i, row) in enumerate(filtered_df.iterrows()):
            models.append({
                'model_id': row.get('model_id', 'Unknown'),
                'x': float(filtered_reduced[idx, 0]),
                'y': float(filtered_reduced[idx, 1]),
                'library_name': row.get('library_name'),
                'pipeline_tag': row.get('pipeline_tag'),
                'downloads': int(row.get('downloads', 0)),
                'likes': int(row.get('likes', 0)),
                'trending_score': float(row.get('trendingScore', 0)) if pd.notna(row.get('trendingScore')) else None,
                'tags': row.get('tags') if pd.notna(row.get('tags')) else None
            })
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(models)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }


def get_stats(headers):
    """Get dataset statistics."""
    global df
    
    if df is None:
        return {
            'statusCode': 503,
            'headers': headers,
            'body': json.dumps({'error': 'Data not loaded'})
        }
    
    stats = {
        'total_models': len(df),
        'unique_libraries': df['library_name'].nunique() if 'library_name' in df.columns else 0,
        'unique_pipelines': df['pipeline_tag'].nunique() if 'pipeline_tag' in df.columns else 0,
        'avg_downloads': float(df['downloads'].mean()) if 'downloads' in df.columns else 0,
        'avg_likes': float(df['likes'].mean()) if 'likes' in df.columns else 0
    }
    
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps(stats)
    }

