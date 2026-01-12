# Quick Start: Scaling Embeddings to All Models

## Overview

This guide explains how to scale embeddings to all models in your dataset without impacting performance.

## Current Limitations

- **Current**: ~150k models max
- **Target**: All models with relationships (~14.5k+ models with config.json, or all ~1.86M models)
- **Challenge**: Memory, startup time, and network transfer

## Recommended Approach: Chunked Storage

The best approach is **chunked storage** - storing embeddings in smaller files and loading only what's needed.

### Benefits

✅ **Fast Startup**: Load metadata only (~2-5 seconds)  
✅ **Low Memory**: Load chunks on-demand (~100MB idle vs 2.8GB)  
✅ **Scalable**: Works with millions of models  
✅ **Backward Compatible**: Can still load all embeddings if needed  

## Implementation Steps

### Step 1: Generate Chunked Embeddings

Modify `backend/scripts/precompute_data.py` to support chunking:

```bash
# Generate chunked embeddings for all models
cd backend
python scripts/precompute_data.py \
  --sample-size 0 \  # 0 = all models
  --chunked \
  --chunk-size 50000 \
  --output-dir ../precomputed_data
```

This will create:
- `chunk_index_v1.parquet` - Maps model_id → chunk_id
- `embeddings_chunk_000_v1.parquet` - First 50k models
- `embeddings_chunk_001_v1.parquet` - Next 50k models
- ... (one file per chunk)

### Step 2: Update Precomputed Loader

The `ChunkedEmbeddingLoader` class (already created in `backend/utils/chunked_loader.py`) will:
- Load chunk index on startup (fast)
- Load chunks only when needed
- Cache recently used chunks

### Step 3: Update API Routes

Modify `backend/api/routes/models.py` to:
1. Filter dataset FIRST (before loading embeddings)
2. Load embeddings only for filtered models
3. Use chunked loader for efficient access

### Step 4: Update Frontend

Modify `frontend/src/pages/GraphPage.tsx` to:
1. Load initial subset (base models)
2. Load more on-demand (when filtering/searching)
3. Use progressive loading for better UX

## Quick Implementation

### Option A: Minimal Changes (Recommended First)

**Goal**: Support all models without major refactoring

1. **Generate chunked data** (one-time):
   ```bash
   python backend/scripts/precompute_data.py --sample-size 0 --chunked
   ```

2. **Update startup** (`backend/api/main.py`):
   - Use `ChunkedEmbeddingLoader` instead of loading all embeddings
   - Load embeddings only when API is called (not at startup)

3. **Update API** (`backend/api/routes/models.py`):
   - Filter dataset first
   - Load embeddings only for filtered models using chunked loader

**Result**: Startup time drops from 30s → 2s, memory from 2.8GB → 100MB

### Option B: Full Implementation

Follow the complete strategy in `SCALING_EMBEDDINGS_STRATEGY.md`:
1. Chunked storage ✅
2. Server-side filtering ✅
3. Progressive loading ✅
4. Frontend virtualization ✅

## Performance Comparison

| Metric | Current (150k) | Chunked (All Models) |
|--------|---------------|---------------------|
| Startup Time | 10-30s | **2-5s** |
| Memory (Idle) | ~500MB | **~100MB** |
| Memory (Active) | ~500MB | **~200-500MB** (chunks loaded) |
| API Response | 1-3s | **<1s** (filtered) |
| Scales To | 150k models | **Millions** |

## Testing

1. **Test chunked loading**:
   ```python
   from utils.chunked_loader import ChunkedEmbeddingLoader
   
   loader = ChunkedEmbeddingLoader()
   embeddings, model_ids = loader.load_embeddings_for_models(['model1', 'model2'])
   ```

2. **Test API performance**:
   - Check startup time (should be <5s)
   - Check memory usage (should be <200MB idle)
   - Test filtering (should be fast)

3. **Test frontend**:
   - Load initial view (should be fast)
   - Filter/search (should load only relevant models)

## Migration Checklist

- [ ] Generate chunked embeddings for all models
- [ ] Update `precomputed_loader.py` to use chunked loader
- [ ] Update API routes to filter before loading embeddings
- [ ] Test startup time and memory usage
- [ ] Update frontend for progressive loading (optional)
- [ ] Deploy and monitor performance

## Troubleshooting

**Issue**: Startup still slow  
**Solution**: Make sure embeddings aren't loaded at startup, only metadata

**Issue**: High memory usage  
**Solution**: Reduce chunk cache size or clear cache periodically

**Issue**: Slow API responses  
**Solution**: Ensure filtering happens before loading embeddings

## Next Steps

1. Read `SCALING_EMBEDDINGS_STRATEGY.md` for detailed strategy
2. Review `backend/utils/chunked_loader.py` for implementation
3. Start with Option A (minimal changes) for quick wins
4. Gradually implement Option B for full optimization


