# Backend API

FastAPI backend for serving model data to the React frontend.

## Structure

- `api/` - API routes and main application
- `services/` - External service integrations (arXiv, model tracking, scheduling)
- `utils/` - Utility modules (data loading, embeddings, dimensionality reduction, clustering, network analysis)
- `config/` - Configuration files (requirements.txt, etc.)
- `cache/` - Cached data (embeddings, reduced dimensions)

## Running

```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

- `SAMPLE_SIZE` - Limit number of models to load (for development). Set to 0 or leave unset to load all models.


