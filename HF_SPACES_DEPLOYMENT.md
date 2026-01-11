# Hugging Face Spaces Deployment Guide

## Overview

This guide explains how to deploy the HF Model Ecosystem Visualizer to Hugging Face Spaces with chunked embeddings support.

## Prerequisites

1. Hugging Face account
2. A Space created on Hugging Face
3. Pre-computed chunked data uploaded to a Hugging Face Dataset

## Step 1: Prepare Pre-computed Data

### Upload Chunked Data to HF Dataset

The chunked embeddings need to be uploaded to a Hugging Face Dataset. The system will automatically download them on startup.

**Dataset Structure:**
```
modelbiome/hf-viz-precomputed/
├── metadata_v1.json
├── models_v1.parquet
├── chunk_index_v1.parquet
├── embeddings_chunk_000_v1.parquet
├── embeddings_chunk_001_v1.parquet
├── ...
└── embeddings_chunk_036_v1.parquet
```

**Upload Script:**
```python
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
dataset_id = "modelbiome/hf-viz-precomputed"

# Upload files
data_dir = Path("precomputed_data")
files = [
    "metadata_v1.json",
    "models_v1.parquet",
    "chunk_index_v1.parquet",
] + [f"embeddings_chunk_{i:03d}_v1.parquet" for i in range(37)]

for filename in files:
    filepath = data_dir / filename
    if filepath.exists():
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=dataset_id,
            repo_type="dataset"
        )
        print(f"Uploaded {filename}")
```

## Step 2: Deploy to Space

### Option A: Git Push (Recommended)

1. **Initialize Git Repository:**
   ```bash
   cd hf-viz
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```

2. **Add Required Files:**
   ```bash
   git add app.py
   git add requirements.txt
   git add Dockerfile
   git add README_SPACE.md
   git add backend/
   git add frontend/
   git add precomputed_data/.gitkeep  # Keep directory structure
   ```

3. **Commit and Push:**
   ```bash
   git commit -m "Deploy to HF Spaces with chunked embeddings"
   git push origin main
   ```

### Option B: Web Interface

1. Go to your Space on Hugging Face
2. Click "Files and versions"
3. Upload files:
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README_SPACE.md` (rename to `README.md`)
   - `backend/` directory
   - `frontend/` directory

## Step 3: Configure Environment Variables

In your Space settings, add:

- `HF_PRECOMPUTED_DATASET`: `modelbiome/hf-viz-precomputed` (or your dataset)
- `PORT`: `7860` (default, usually not needed)
- `SAMPLE_SIZE`: Leave empty (uses all models from precomputed data)

## Step 4: Verify Deployment

1. **Check Build Logs:**
   - Go to your Space
   - Click "Logs" tab
   - Look for: "Downloaded chunk index" and "Downloaded X embedding chunks"

2. **Test the API:**
   - Visit: `https://YOUR_SPACE.hf.space/api/models?max_points=10`
   - Should return JSON with models

3. **Check Startup Time:**
   - Should be 2-5 seconds
   - Look for: "STARTUP COMPLETE in X seconds"

## File Structure for HF Spaces

```
your-space/
├── app.py                    # Entry point (required)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── README.md                # Space description (from README_SPACE.md)
├── backend/                  # Backend code
│   ├── api/
│   ├── utils/
│   └── ...
├── frontend/                 # Frontend source (will be built)
│   ├── src/
│   └── package.json
└── precomputed_data/         # Empty directory (data downloaded from HF Hub)
    └── .gitkeep
```

## How It Works

1. **Build Time:**
   - Dockerfile builds React frontend
   - Installs Python dependencies
   - Copies code

2. **Startup:**
   - `app.py` is executed
   - Downloads precomputed data from HF Hub
   - Loads chunked embeddings
   - Starts FastAPI server

3. **Runtime:**
   - API requests load embeddings on-demand
   - Only loads chunks containing requested models
   - Efficient memory usage

## Troubleshooting

### Issue: Data Not Downloading

**Solution:**
1. Check `HF_PRECOMPUTED_DATASET` environment variable
2. Verify dataset exists: https://huggingface.co/datasets/modelbiome/hf-viz-precomputed
3. Check logs for download errors

### Issue: Out of Memory

**Solution:**
1. Ensure chunked data is being used (check logs)
2. Reduce `SAMPLE_SIZE` if needed
3. Upgrade Space hardware if available

### Issue: Slow Startup

**Solution:**
1. Verify chunked data is downloading correctly
2. Check network connectivity in logs
3. Ensure metadata file exists in dataset

### Issue: API Not Responding

**Solution:**
1. Check if server started successfully (logs)
2. Verify port 7860 is exposed
3. Check CORS settings in `api/main.py`

## Performance Optimization

1. **Use Chunked Data**: Always use chunked embeddings (default)
2. **Pre-compute Coordinates**: Coordinates are stored in `models_v1.parquet`
3. **Cache Chunks**: Chunked loader caches recently used chunks
4. **Filter First**: API filters before loading embeddings

## Updating Data

When you need to update the precomputed data:

1. **Regenerate Locally:**
   ```bash
   python backend/scripts/precompute_data.py --sample-size 0 --chunked
   ```

2. **Upload to Dataset:**
   ```bash
   # Use the upload script above
   ```

3. **Redeploy Space:**
   - Data will be automatically downloaded on next startup
   - Or trigger a rebuild in Space settings

## Monitoring

- **Logs**: Check Space logs for startup and runtime info
- **Metrics**: Monitor memory usage in Space dashboard
- **API**: Test endpoints via `/docs` (Swagger UI)

## Success Indicators

✅ **Startup**: <5 seconds  
✅ **Memory**: <500MB idle  
✅ **API**: Responds in <1s  
✅ **Data**: Chunked files downloaded successfully

---

**Note**: The Space will automatically download chunked data from the Hugging Face Dataset on startup. No manual data upload to the Space repository is needed!

