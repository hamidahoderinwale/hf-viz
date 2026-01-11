---
title: HF Model Ecosystem Visualizer
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Anatomy of a Machine Learning Ecosystem: 2 Million Models on Hugging Face

**Authors:** Benjamin Laufer, Hamidah Oderinwale, Jon Kleinberg

**Research Paper**: [arXiv:2508.06811](https://arxiv.org/abs/2508.06811)

## About This Tool

This interactive visualization explores ~1.86M models from the Hugging Face ecosystem, visualizing them in a 3D embedding space where similar models appear closer together. The tool uses **chunked embeddings** for fast startup and efficient memory usage.

## Features

- **Fast Startup**: 2-5 seconds (uses chunked embeddings)
- **Low Memory**: ~100MB idle (vs 2.8GB without chunking)
- **Scalable**: Handles millions of models efficiently
- **Interactive**: Filter, search, and explore model relationships
- **Family Trees**: Visualize parent-child relationships between models

## How It Works

The system uses:
1. **Chunked Embeddings**: Pre-computed embeddings stored in chunks (50k models per chunk)
2. **On-Demand Loading**: Only loads embeddings for filtered models
3. **Pre-computed Coordinates**: UMAP coordinates stored with model metadata
4. **Fast API**: FastAPI backend with efficient data loading

## Data Source

- **Dataset**: [modelbiome/ai_ecosystem](https://huggingface.co/datasets/modelbiome/ai_ecosystem)
- **Pre-computed Data**: Automatically downloaded from `modelbiome/hf-viz-precomputed` on startup

## Deployment

This Space automatically:
1. Downloads pre-computed chunked data from Hugging Face Hub
2. Starts the FastAPI backend
3. Serves the React frontend
4. Uses chunked loading for efficient memory usage

## Performance

- **Startup**: 2-5 seconds
- **Memory**: ~100MB idle, ~200-500MB active
- **API Response**: <1s for filtered queries
- **Scales To**: Unlimited models

## Usage

1. **Filter Models**: Use the sidebar to filter by downloads, likes, search query
2. **Explore**: Zoom and pan to explore the embedding space
3. **Search**: Search for specific models or tags
4. **View Details**: Click on models to see detailed information

## Technical Details

- **Backend**: FastAPI (Python)
- **Frontend**: React + TypeScript
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Visualization**: UMAP (3D coordinates)
- **Storage**: Parquet files with chunked embeddings

## Resources

- **GitHub**: [bendlaufer/ai-ecosystem](https://github.com/bendlaufer/ai-ecosystem)
- **Paper**: [arXiv:2508.06811](https://arxiv.org/abs/2508.06811)
- **Dataset**: [modelbiome/ai_ecosystem](https://huggingface.co/datasets/modelbiome/ai_ecosystem)

