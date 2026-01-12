# Deploy to Hugging Face Spaces - Quick Guide

## ✅ What's Ready

All files are configured for HF Spaces deployment:
- ✅ `app.py` - Entry point
- ✅ `Dockerfile` - Docker configuration  
- ✅ `requirements.txt` - Dependencies
- ✅ `README_SPACE.md` - Space description
- ✅ Chunked data download - Automatic from HF Hub

## 🚀 Quick Deployment Steps

### Step 1: Upload Precomputed Data to HF Dataset

**Option A: Use the upload script (after precompute completes)**
```bash
cd hf-viz
python upload_to_hf_dataset.py --dataset-id modelbiome/hf-viz-precomputed
```

**Option B: Manual upload**
1. Go to https://huggingface.co/datasets/modelbiome/hf-viz-precomputed
2. Upload files:
   - `metadata_v1.json`
   - `models_v1.parquet`
   - `chunk_index_v1.parquet`
   - `embeddings_chunk_000_v1.parquet` through `embeddings_chunk_036_v1.parquet`

### Step 2: Create/Configure HF Space

1. **Create Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `hf-viz` (or your choice)
   - SDK: **Docker**
   - Visibility: Public/Private

2. **Clone Space:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   ```

### Step 3: Copy Files to Space

```bash
# From hf-viz directory
cp app.py YOUR_SPACE_NAME/
cp requirements.txt YOUR_SPACE_NAME/
cp Dockerfile YOUR_SPACE_NAME/
cp README_SPACE.md YOUR_SPACE_NAME/README.md
cp -r backend YOUR_SPACE_NAME/
cp -r frontend YOUR_SPACE_NAME/
mkdir -p YOUR_SPACE_NAME/precomputed_data
touch YOUR_SPACE_NAME/precomputed_data/.gitkeep
```

### Step 4: Push to Space

```bash
cd YOUR_SPACE_NAME
git add .
git commit -m "Deploy HF Model Ecosystem Visualizer with chunked embeddings"
git push
```

### Step 5: Configure Environment Variables

In Space settings → Variables:
- `HF_PRECOMPUTED_DATASET`: `modelbiome/hf-viz-precomputed`
- (Optional) `SAMPLE_SIZE`: Leave empty

### Step 6: Wait for Build

- HF Spaces will build the Docker image
- Check logs for: "Downloaded chunk index" and "Downloaded X embedding chunks"
- Startup should complete in 2-5 seconds

## 📋 File Checklist

Ensure these files are in your Space:
- [ ] `app.py`
- [ ] `requirements.txt`
- [ ] `Dockerfile`
- [ ] `README.md` (from `README_SPACE.md`)
- [ ] `backend/` directory
- [ ] `frontend/` directory
- [ ] `precomputed_data/.gitkeep`

## 🔍 Verify Deployment

1. **Check Logs:**
   - Should see: "Downloaded chunk index"
   - Should see: "Downloaded X embedding chunks"
   - Should see: "STARTUP COMPLETE in X seconds"

2. **Test API:**
   - Visit: `https://YOUR_SPACE.hf.space/api/models?max_points=10`
   - Should return JSON

3. **Test Frontend:**
   - Visit: `https://YOUR_SPACE.hf.space/`
   - Should load the visualization

## 🐛 Troubleshooting

### Build Fails
- Check Dockerfile syntax
- Verify all files are present
- Check logs for specific errors

### Data Not Downloading
- Verify `HF_PRECOMPUTED_DATASET` environment variable
- Check dataset exists and is public
- Verify files are uploaded to dataset

### Out of Memory
- Ensure chunked data is being used
- Check logs for "Chunked embeddings detected"
- Consider upgrading Space hardware

### Slow Startup
- Check if data is downloading (logs)
- Verify chunked files exist in dataset
- Check network connectivity

## 📊 Expected Performance

- **Build Time**: 5-10 minutes (first time)
- **Startup Time**: 2-5 seconds
- **Memory**: ~100-200MB idle
- **API Response**: <1s

## 🔄 Updating

When you update the code:
```bash
cd YOUR_SPACE_NAME
git pull  # Get latest
# Make changes
git add .
git commit -m "Update"
git push
```

When you update data:
1. Regenerate locally
2. Upload to dataset (using `upload_to_hf_dataset.py`)
3. Space will auto-download on next startup

## 📚 Documentation

- `HF_SPACES_DEPLOYMENT.md` - Detailed deployment guide
- `README_SPACE.md` - Space description
- `PRODUCTION_DEPLOYMENT.md` - Local deployment guide

---

**Note**: The Space automatically downloads chunked data from the Hugging Face Dataset. No need to include data files in the Space repository!


