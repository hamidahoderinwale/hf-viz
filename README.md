# Anatomy of a Machine Learning Ecosystem: 2 Million Models on Hugging Face

**Authors:** Benjamin Laufer, Hamidah Oderinwale, Jon Kleinberg

**Research Paper**: [arXiv:2508.06811](https://arxiv.org/abs/2508.06811)

## Abstract

Many have observed that the development and deployment of generative machine learning (ML) and artificial intelligence (AI) models follow a distinctive pattern in which pre-trained models are adapted and fine-tuned for specific downstream tasks. However, there is limited empirical work that examines the structure of these interactions. This paper analyzes 1.86 million models on Hugging Face, a leading peer production platform for model development. Our study of model family trees -- networks that connect fine-tuned models to their base or parent -- reveals sprawling fine-tuning lineages that vary widely in size and structure. Using an evolutionary biology lens to study ML models, we use model metadata and model cards to measure the genetic similarity and mutation of traits over model families. We find that models tend to exhibit a family resemblance, meaning their genetic markers and traits exhibit more overlap when they belong to the same model family. However, these similarities depart in certain ways from standard models of asexual reproduction, because mutations are fast and directed, such that two `sibling' models tend to exhibit more similarity than parent/child pairs. Further analysis of the directional drifts of these mutations reveals qualitative insights about the open machine learning ecosystem: Licenses counter-intuitively drift from restrictive, commercial licenses towards permissive or copyleft licenses, often in violation of upstream license's terms; models evolve from multi-lingual compatibility towards english-only compatibility; and model cards reduce in length and standardize by turning, more often, to templates and automatically generated text. Overall, this work takes a step toward an empirically grounded understanding of model fine-tuning and suggests that ecological models and methods can yield novel scientific insights.

## About This Tool

This interactive latent space navigator visualizes ~1.84M models from the [modelbiome/ai_ecosystem_withmodelcards](https://huggingface.co/datasets/modelbiome/ai_ecosystem_withmodelcards) dataset in a 2D space where similar models appear closer together, allowing you to explore the relationships and family structures described in the paper.

**Resources:**
- **GitHub Repository**: [bendlaufer/ai-ecosystem](https://github.com/bendlaufer/ai-ecosystem) - Original research repository with analysis notebooks and datasets
- **Hugging Face Project**: [modelbiome](https://huggingface.co/modelbiome) - Dataset and project page on Hugging Face Hub

## Project Structure

```
hf_viz/
├── backend/              # FastAPI backend
│   ├── api/             # API routes (main.py)
│   ├── services/        # External services (arXiv, model tracking, scheduler)
│   ├── utils/           # Utility modules (data loading, embeddings, etc.)
│   ├── config/          # Configuration files
│   └── cache/           # Backend cache directory
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── utils/       # Frontend utilities
│   │   └── workers/     # Web Workers
│   └── public/          # Static assets
├── cache/               # Shared cache directory
├── deploy/              # Deployment configuration files
└── netlify-functions/   # Netlify serverless functions
```

## Features

### 3D Latent Space Visualization

- **Interactive 3D Scatter Plot** (Three.js/React Three Fiber):
  - Navigate 1.84M+ models in 3D space
  - Spatial sparsity filtering for better navigability
  - Frustum culling and adaptive sampling for performance
  - Instanced rendering for large datasets
  - Family tree visualization with connecting edges
  - Multiple color encoding options (library, pipeline, cluster, family depth, popularity)
  - Dynamic size encoding based on downloads/likes
  - Smooth camera animations
  - UV projection minimap for navigation

### 2D Visualizations (D3.js)

- **Enhanced Scatter Plot**: 
  - Brush selection for multi-model selection
  - Real-time tooltips with model details
  - Dynamic color and size encoding
  - Interactive zoom and pan
  - Click to view model details modal

- **Network Graph**: 
  - Force-directed layout showing model relationships
  - Connectivity based on latent space similarity
  - Draggable nodes
  - Color-coded by library
  - Node size based on popularity

- **Histograms**: 
  - Distribution analysis of downloads, likes, trending scores
  - Interactive bars with hover details
  - Dynamic attribute selection

- **UV Projection Minimap**:
  - 2D projection of 3D latent space (XY plane)
  - Click to navigate 3D view to specific regions
  - Shows current view center

### Advanced Features

- **Semantic Similarity Search**: Find models similar to a query model using embeddings
- **Base Models Filter**: View only root models (no parent) to see the base of family trees
- **Family Tree Visualization**: Click any model to see its family tree with parent-child relationships
- **Clustering**: Automatic K-means clustering reveals semantic groups
- **Model Details Modal**: 
  - Comprehensive model information
  - File tree browser
  - Color-coded tags and licenses
  - Links to Hugging Face Hub

### Model Tracking & Analytics

- **Live Model Count Tracking**: Track the number of models on Hugging Face Hub over time
- **Growth Statistics**: Calculate growth rates, daily averages, and trends
- **Historical Data**: Query historical model counts with breakdowns by library and pipeline
- **API Endpoints**: RESTful API for accessing tracking data

### Performance Optimizations

- **Real-time Updates**: 
  - Debounced search (300ms)
  - Instant filter updates
  - Dynamic visualization switching
- **Client-side Caching**: IndexedDB caching for API responses
- **Request Cancellation**: Prevents race conditions with concurrent requests
- **Adaptive Rendering**: Quality adjusts based on user interaction
- **Spatial Indexing**: Octree for efficient nearest neighbor queries

## Quick Start

**Start Backend:**
```bash
cd backend
pip install -r requirements.txt
python api.py
```

**Start Frontend:**
```bash
cd frontend
npm install
npm start
```

Opens at `http://localhost:3000` with full D3.js interactivity.

## Installation

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

## Usage

### Local Development

**Start Backend:**
```bash
cd backend
python api.py
```

The backend will:
1. Load a sample of 10,000 models from the dataset
2. Generate embeddings (first run takes ~2-3 minutes)
3. Reduce dimensions using UMAP
4. Serve the API at `http://localhost:8000`

**Start Frontend:**
```bash
cd frontend
npm start
```

The frontend will open at `http://localhost:3000`

### Using the Interface

1. **Filters**: Use the left sidebar to filter models by:
   - Search query (model ID or tags)
   - Minimum downloads
   - Minimum likes
   - Color mapping (library, pipeline, popularity)
   - Size mapping (downloads, likes, trending score)

2. **Exploration**: 
   - Hover over points to see model information
   - Zoom and pan to explore different regions
   - Use the legend to understand color coding

3. **Understanding the Space**:
   - Models closer together are more similar
   - Similarity is based on tags, pipeline type, library, and model card content

## Deployment

### Netlify (React Frontend)

1. Deploy frontend to Netlify (set base directory to `frontend`)
2. Deploy backend to Railway/Render (set root directory to `backend`)
3. Set `REACT_APP_API_URL` environment variable in Netlify to your backend URL
4. Update CORS in backend to include your Netlify URL

## Architecture

- **Backend** (`backend/api.py`): FastAPI server serving model data
- **Frontend** (`frontend/`): React app with D3.js visualizations
  - **Enhanced Scatter Plot**: D3.js scatter with brush selection, real-time tooltips
  - **Network Graph**: Force-directed graph showing model relationships and connectivity
  - **Histograms**: Distribution analysis of downloads, likes, trending scores
  - **Real-time Updates**: Debounced filtering, dynamic visualizations
  - **Interactive Features**: Click, brush, drag, zoom, pan
- **Data Loading** (`data_loader.py`): Loads dataset from Hugging Face Hub, handles filtering and preprocessing
- **Embedding Generation** (`embeddings.py`): Creates embeddings from model metadata using sentence transformers
- **Dimensionality Reduction** (`dimensionality_reduction.py`): Uses UMAP to reduce to 2D for visualization
- **Clustering** (`clustering.py`): K-Means clustering with automatic optimization for model grouping

### Comparison with Hugging Face Dataset Viewer

This project uses a different approach than Hugging Face's built-in dataset viewer:

- **HF Dataset Viewer**: Tabular browser for exploring dataset rows (see [dataset-viewer](https://github.com/huggingface/dataset-viewer))
- **This Project**: Latent space visualization showing semantic relationships between models

The HF viewer is optimized for browsing data structure, while this tool focuses on understanding model relationships through embeddings and spatial visualization.

## Design Decisions

The application uses:
- **3D visualization** for immersive exploration of latent space with **2D fallbacks** for accessibility
- **UMAP** for dimensionality reduction (better global structure than t-SNE, optimized parameters for structure preservation)
- **Sentence transformers** for efficient embedding generation
- **Smart sampling** with spatial sparsity to maintain interactivity with large datasets
- **Multi-level caching** (disk + IndexedDB) to avoid recomputation on filter changes
- **Adaptive rendering** with frustum culling and level-of-detail for smooth performance
- **Instanced rendering** for efficient GPU utilization with large point clouds

## Performance Notes

- **Full Dataset**: Loads all ~1.86 million models from the dataset
- **Visualization Limit**: Maximum 50,000 points for smooth interaction (configurable via `max_points` API parameter)
- **Level-of-Detail Rendering**: Frontend automatically samples to 10,000 points for 3D visualization while preserving family tree members
- **Embedding Model**: `all-MiniLM-L6-v2` (good balance of quality and speed)
- **Caching**: Embeddings and reduced dimensions are cached to disk for fast startup
- **Optimizations**: Index-based lookups, vectorized operations, response compression, and optimized top-k queries

## Requirements

- Python 3.8+
- ~2-4GB RAM for 10K models
- Internet connection for dataset download
- Optional: GPU for faster embedding generation (not required)

## Citation

If you use this tool or dataset, please cite:

```bibtex
@article{laufer2025anatomy,
  title={Anatomy of a Machine Learning Ecosystem: 2 Million Models on Hugging Face},
  author={Laufer, Benjamin and Oderinwale, Hamidah and Kleinberg, Jon},
  journal={arXiv preprint arXiv:2508.06811},
  year={2025},
  url={https://arxiv.org/abs/2508.06811}
}
```

**Paper**: [arXiv:2508.06811](https://arxiv.org/abs/2508.06811)
