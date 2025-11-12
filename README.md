# Anatomy of a Machine Learning Ecosystem: 2 Million Models on Hugging Face

**Authors:** Benjamin Laufer, Hamidah Oderinwale, Jon Kleinberg

**Research Paper**: [arXiv:2508.06811](https://arxiv.org/abs/2508.06811)

## Abstract

Many have observed that the development and deployment of generative machine learning (ML) and artificial intelligence (AI) models follow a distinctive pattern in which pre-trained models are adapted and fine-tuned for specific downstream tasks. However, there is limited empirical work that examines the structure of these interactions. This paper analyzes 1.86 million models on Hugging Face, a leading peer production platform for model development. Our study of model family trees -- networks that connect fine-tuned models to their base or parent -- reveals sprawling fine-tuning lineages that vary widely in size and structure. Using an evolutionary biology lens to study ML models, we use model metadata and model cards to measure the genetic similarity and mutation of traits over model families. We find that models tend to exhibit a family resemblance, meaning their genetic markers and traits exhibit more overlap when they belong to the same model family. However, these similarities depart in certain ways from standard models of asexual reproduction, because mutations are fast and directed, such that two `sibling' models tend to exhibit more similarity than parent/child pairs. Further analysis of the directional drifts of these mutations reveals qualitative insights about the open machine learning ecosystem: Licenses counter-intuitively drift from restrictive, commercial licenses towards permissive or copyleft licenses, often in violation of upstream license's terms; models evolve from multi-lingual compatibility towards english-only compatibility; and model cards reduce in length and standardize by turning, more often, to templates and automatically generated text. Overall, this work takes a step toward an empirically grounded understanding of model fine-tuning and suggests that ecological models and methods can yield novel scientific insights.

## About This Tool

This interactive latent space navigator visualizes ~1.84M models from the [modelbiome/ai_ecosystem_withmodelcards](https://huggingface.co/datasets/modelbiome/ai_ecosystem_withmodelcards) dataset in a 2D space where similar models appear closer together, allowing you to explore the relationships and family structures described in the paper.

## Features

- **Latent Space Visualization**: 2D embedding visualization showing model relationships
- **Interactive Exploration**: Hover, click, and zoom to explore models
- **Smart Filtering**: Filter by library, pipeline tag, popularity, and more
- **Color & Size Encoding**: Visualize different attributes through color and size
- **Caching**: Efficient caching of embeddings and reduced dimensions
- **Performance Optimized**: Handles large datasets through smart sampling

## Quick Start

### Option 1: Plotly + Gradio (Hugging Face Spaces)

```bash
pip install -r requirements.txt
python app.py
```

### Option 2: Visx + React (Netlify Deployment)

For Netlify deployment, deploy the frontend to Netlify and the backend to Railway or Render. Set the `REACT_APP_API_URL` environment variable to your backend URL.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local Development

```bash
python app.py
```

Or use the test script:

```bash
python test_local.py
```

The app will:
1. Load a sample of 10,000 models from the dataset
2. Generate embeddings (first run takes ~2-3 minutes)
3. Reduce dimensions using UMAP
4. Launch a Gradio interface at `http://localhost:7860`

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

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Push this repository to the Space
3. Ensure `requirements.txt` and `app.py` are in the root
4. The app will automatically:
   - Load the dataset from Hugging Face Hub
   - Generate embeddings on first run (cached afterwards)
   - Serve the interface via Gradio

**Note**: First load may take 2-3 minutes for embedding generation. Subsequent loads will be faster due to caching.

### Netlify (React Frontend)

1. Deploy frontend to Netlify (set base directory to `frontend`)
2. Deploy backend to Railway/Render (set root directory to `backend`)
3. Set `REACT_APP_API_URL` environment variable in Netlify to your backend URL
4. Update CORS in backend to include your Netlify URL

## Architecture

### Current Implementation (Plotly + Gradio)

- **Data Loading** (`data_loader.py`): Loads dataset from Hugging Face Hub, handles filtering and preprocessing
- **Embedding Generation** (`embeddings.py`): Creates embeddings from model metadata using sentence transformers
- **Dimensionality Reduction** (`dimensionality_reduction.py`): Uses UMAP to reduce to 2D for visualization
- **Main App** (`app.py`): Gradio interface with Plotly visualizations

### Alternative Implementation (Visx + React)

For better performance and customization, see the `frontend/` and `backend/` directories for a React + Visx implementation:

- **Backend** (`backend/api.py`): FastAPI server serving model data
- **Frontend** (`frontend/`): React app with Visx visualizations

### Comparison with Hugging Face Dataset Viewer

This project uses a different approach than Hugging Face's built-in dataset viewer:

- **HF Dataset Viewer**: Tabular browser for exploring dataset rows (see [dataset-viewer](https://github.com/huggingface/dataset-viewer))
- **This Project**: Latent space visualization showing semantic relationships between models

The HF viewer is optimized for browsing data structure, while this tool focuses on understanding model relationships through embeddings and spatial visualization.

## Design Decisions

The application uses:
- **2D visualization** for simplicity and accessibility
- **UMAP** for dimensionality reduction (better global structure than t-SNE)
- **Sentence transformers** for efficient embedding generation
- **Smart sampling** to maintain interactivity with large datasets
- **Caching** to avoid recomputation on filter changes

## Performance Notes

- **Initial Sample**: 10,000 models (configurable in `app.py`)
- **Visualization Limit**: Maximum 5,000 points for smooth interaction
- **Embedding Model**: `all-MiniLM-L6-v2` (good balance of quality and speed)
- **Caching**: Embeddings and reduced dimensions are cached to disk

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
