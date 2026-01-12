# Production Deployment Guide: Chunked Embeddings

## ✅ What Was Implemented

All necessary code changes have been made to support chunked embeddings in production:

### 1. **Chunked Loader Utility** (`backend/utils/chunked_loader.py`)
   - ✅ Created `ChunkedEmbeddingLoader` class
   - ✅ Loads embeddings in chunks (50k models per chunk)
   - ✅ Only loads chunks containing requested models
   - ✅ Caches recently used chunks

### 2. **Precomputed Loader Updates** (`backend/utils/precomputed_loader.py`)
   - ✅ Added `is_chunked()` method to detect chunked data
   - ✅ Added `get_chunked_loader()` method
   - ✅ Updated `load_all()` to skip embedding load when chunked

### 3. **Dependencies Updates** (`backend/api/dependencies.py`)
   - ✅ Added `chunked_embedding_loader` to global state
   - ✅ Imported `ChunkedEmbeddingLoader`

### 4. **Startup Updates** (`backend/api/main.py`)
   - ✅ Detects chunked data automatically
   - ✅ Initializes chunked loader when available
   - ✅ Skips embedding load at startup (fast startup)
   - ✅ Falls back to full load if chunked loader unavailable

### 5. **API Route Updates** (`backend/api/routes/models.py`)
   - ✅ Uses chunked loader when embeddings not loaded
   - ✅ Loads embeddings only for filtered models
   - ✅ Uses pre-computed coordinates from dataframe
   - ✅ Maintains backward compatibility

### 6. **Precompute Script Updates** (`backend/scripts/precompute_data.py`)
   - ✅ Added `--chunked` flag
   - ✅ Added `--chunk-size` parameter
   - ✅ Creates chunk index automatically

## 🚀 Deployment Steps

### Step 1: Generate Chunked Data

Generate chunked embeddings for all models:

```bash
cd backend
python scripts/precompute_data.py \
  --sample-size 0 \          # 0 = all models
  --chunked \                # Enable chunked storage
  --chunk-size 50000 \       # 50k models per chunk
  --output-dir ../precomputed_data \
  --version v1
```

This will create:
- `chunk_index_v1.parquet` - Maps model_id → chunk_id
- `embeddings_chunk_000_v1.parquet` - First 50k models
- `embeddings_chunk_001_v1.parquet` - Next 50k models
- ... (one file per chunk)
- `models_v1.parquet` - All model metadata + coordinates

**Note**: This process may take several hours for large datasets. Consider running it in the background or on a powerful machine.

### Step 2: Verify Chunked Data

Check that chunked data was created:

```bash
ls -lh precomputed_data/embeddings_chunk_*_v1.parquet
ls -lh precomputed_data/chunk_index_v1.parquet
```

### Step 3: Deploy Code

The code is already updated! Just ensure:
- ✅ `backend/utils/chunked_loader.py` exists
- ✅ All updated files are deployed
- ✅ Dependencies are installed

### Step 4: Test Startup

Start the server and verify fast startup:

```bash
cd backend
python -m uvicorn api.main:app --reload
```

Expected output:
```
LOADING PRE-COMPUTED DATA (Fast Startup Mode)
============================================================
Loaded metadata for version v1
  Created: 2024-...
  Total models: 1,860,411
  Embedding dim: 384
Loading pre-computed models from .../models_v1.parquet...
Loaded 1,860,411 models with pre-computed coordinates
Chunked embeddings detected - skipping full embedding load for fast startup
Embeddings will be loaded on-demand using chunked loader
Chunked embedding loader initialized - embeddings will be loaded on-demand
============================================================
STARTUP COMPLETE in 2.45 seconds!
Loaded 1,860,411 models with pre-computed coordinates
Using chunked embeddings - fast startup mode enabled
============================================================
```

### Step 5: Test API

Test the API endpoint:

```bash
curl "http://localhost:8000/api/models?max_points=1000&min_downloads=1000"
```

Expected behavior:
- ✅ Fast response (<1s)
- ✅ Only loads embeddings for filtered models
- ✅ Uses pre-computed coordinates

## 📊 Performance Expectations

| Metric | Before | After (Chunked) |
|--------|--------|-----------------|
| Startup Time | 10-30s | **2-5s** |
| Memory (Idle) | ~500MB | **~100MB** |
| Memory (Active) | ~500MB | **~200-500MB** |
| API Response | 1-3s | **<1s** (filtered) |
| Scales To | 150k models | **Millions** |

## 🔍 Monitoring

### Check Memory Usage

```bash
# Monitor memory usage
ps aux | grep uvicorn
```

Expected: ~100-200MB idle, ~200-500MB when processing requests

### Check Logs

Look for these log messages:
- ✅ "Chunked embeddings detected"
- ✅ "Loading embeddings for X filtered models using chunked loader"
- ✅ "Using pre-computed coordinates from dataframe"

### Verify Chunked Loading

Add logging to see chunk loading:

```python
# In routes/models.py, the logger.debug will show:
# "Loading embeddings for X filtered models using chunked loader"
# "Loaded embeddings for Y models"
```

## 🐛 Troubleshooting

### Issue: "Embeddings not loaded and chunked loader not available"

**Cause**: Chunked data not found or chunked loader failed to initialize

**Solution**:
1. Verify chunked data exists: `ls precomputed_data/chunk_index_v1.parquet`
2. Check logs for initialization errors
3. Ensure `chunked_loader.py` is in the correct location

### Issue: Slow API responses

**Cause**: Loading too many chunks or inefficient filtering

**Solution**:
1. Check filter effectiveness (should filter before loading embeddings)
2. Reduce `max_points` parameter
3. Check chunk cache size (default: 10 chunks)

### Issue: High memory usage

**Cause**: Too many chunks cached or loading all embeddings

**Solution**:
1. Reduce chunk cache size in `ChunkedEmbeddingLoader._max_cache_size`
2. Clear cache periodically: `loader.clear_cache()`
3. Verify embeddings aren't being loaded at startup

### Issue: Missing coordinates

**Cause**: Pre-computed coordinates not in dataframe

**Solution**:
1. Regenerate pre-computed data with coordinates
2. Verify `x_3d`, `y_3d`, `z_3d` columns exist in `models_v1.parquet`

## 🔄 Rollback Plan

If issues occur, you can rollback by:

1. **Disable chunked mode**: Remove or rename `chunk_index_v1.parquet`
2. **Use full embeddings**: Ensure `embeddings_v1.parquet` exists
3. **Restart server**: Will fall back to full embedding load

The code maintains backward compatibility, so existing non-chunked data will still work.

## 📝 Next Steps

After successful deployment:

1. ✅ Monitor performance metrics
2. ✅ Collect user feedback
3. ✅ Optimize chunk size if needed
4. ✅ Consider additional optimizations (PCA, incremental UMAP, etc.)

## 📚 Additional Resources

- `SCALING_EMBEDDINGS_STRATEGY.md` - Complete strategy document
- `SCALING_QUICKSTART.md` - Quick start guide
- `SCALING_SUMMARY.md` - Implementation summary


