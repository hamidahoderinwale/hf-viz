# Scaling Embeddings to All Models: Strategy & Implementation Plan

## Current State

- **Dataset**: ~1.86M models total, ~14.5k models with config.json
- **Current Limit**: 150k models (sample_size parameter)
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2), 384 dimensions
- **Storage**: Parquet files (models + embeddings + UMAP coordinates)
- **Memory**: ~2.8GB for 1.86M embeddings (384 dims × 4 bytes × 1.86M)

## Challenges

1. **Memory**: Loading all embeddings into RAM (~2.8GB+)
2. **Startup Time**: Generating embeddings takes hours
3. **UMAP Computation**: Very slow on large datasets (hours)
4. **Network Transfer**: Sending millions of points to frontend
5. **Frontend Rendering**: Browser can't efficiently render millions of points

## Solution Architecture

### Phase 1: Chunked Storage & Lazy Loading (Recommended First Step)

**Goal**: Store embeddings in chunks, load only what's needed

#### 1.1 Chunked Embedding Storage

```python
# Store embeddings in chunks by model_id hash or library
# Structure: embeddings_<chunk_id>.parquet
# Each chunk: 10k-50k models
```

**Implementation**:
- Modify `precompute_data.py` to save embeddings in chunks
- Create index file mapping model_id → chunk_id
- Load chunks on-demand based on filters

**Benefits**:
- Fast startup (load metadata only)
- Memory efficient (load chunks as needed)
- Scales to millions of models

#### 1.2 Lazy Embedding Generation

**Implementation**:
- Generate embeddings on-demand for filtered subsets
- Cache generated embeddings per chunk
- Background pre-computation for popular models

**Benefits**:
- No upfront computation cost
- Only compute what's needed

### Phase 2: Progressive Loading & Server-Side Filtering

**Goal**: Load initial subset, then progressively load more

#### 2.1 Hierarchical Loading Strategy

1. **Initial Load**: Base models + popular models (~10k-50k)
2. **On-Demand**: Load child models when parent is selected
3. **Background**: Pre-load popular families

#### 2.2 Server-Side Filtering Before Embedding

**Implementation**:
- Filter dataset BEFORE generating embeddings
- Only embed models matching current filters
- Cache filtered embeddings per filter combination

**Benefits**:
- Faster response times
- Lower memory usage
- Better user experience

### Phase 3: Approximate Methods & Optimization

#### 3.1 Incremental UMAP

**Implementation**:
- Use incremental UMAP (umap-learn's `fit_transform` with `transform`)
- Pre-compute UMAP on base set
- Transform new models into existing space

**Benefits**:
- Fast projection for new models
- Consistent coordinate space
- No full recomputation needed

#### 3.2 PCA Preprocessing

**Implementation**:
- Reduce embedding dimensions with PCA (384 → 128)
- Store both full and reduced embeddings
- Use reduced for visualization, full for search

**Benefits**:
- 3x memory reduction
- Faster UMAP computation
- Minimal quality loss

#### 3.3 Frontend Virtualization

**Implementation**:
- Use `react-window` or `react-virtualized`
- Only render visible points
- Progressive rendering as user zooms/pans

**Benefits**:
- Smooth rendering with millions of points
- Lower memory usage in browser
- Better performance

### Phase 4: CDN & Static Hosting

#### 4.1 Static File Hosting

**Implementation**:
- Host pre-computed parquet files on CDN
- Frontend loads directly from CDN
- Backend only handles dynamic queries

**Benefits**:
- Faster loading
- Reduced server load
- Better scalability

## Recommended Implementation Order

### Step 1: Chunked Storage (High Impact, Medium Effort)

**Files to Modify**:
- `backend/scripts/precompute_data.py`
- `backend/utils/precomputed_loader.py`
- `backend/api/routes/models.py`

**Changes**:
1. Add chunking logic to `precompute_data.py`
2. Create chunk index file
3. Modify loader to load chunks on-demand
4. Update API to load chunks based on filters

**Estimated Impact**:
- Startup time: 10s → 2s (load metadata only)
- Memory: 2.8GB → ~100MB (load chunks as needed)
- Scales to millions of models

### Step 2: Server-Side Filtering (High Impact, Low Effort)

**Files to Modify**:
- `backend/api/routes/models.py`
- `backend/utils/data_loader.py`

**Changes**:
1. Filter dataset BEFORE loading embeddings
2. Only load embeddings for filtered models
3. Cache filtered embeddings

**Estimated Impact**:
- Response time: 50% faster
- Memory: 50-90% reduction (depending on filters)

### Step 3: Progressive Loading (Medium Impact, Medium Effort)

**Files to Modify**:
- `frontend/src/pages/GraphPage.tsx`
- `frontend/src/App.tsx`
- `backend/api/routes/models.py`

**Changes**:
1. Load initial subset (base models)
2. Load more on scroll/zoom
3. Background loading for popular models

**Estimated Impact**:
- Initial load: 80% faster
- Better perceived performance

### Step 4: Frontend Virtualization (Medium Impact, High Effort)

**Files to Modify**:
- `frontend/src/components/visualizations/EmbeddingSpace.tsx`
- Add virtualization library

**Changes**:
1. Integrate `react-window` or similar
2. Only render visible points
3. Progressive rendering

**Estimated Impact**:
- Rendering: Smooth with millions of points
- Memory: 70% reduction in browser

## Implementation Details

### Chunked Storage Format

```
precomputed_data/
├── metadata_v1.json
├── chunk_index.parquet          # model_id → chunk_id mapping
├── embeddings_chunk_000.parquet # 0-49k models
├── embeddings_chunk_001.parquet # 50k-99k models
├── ...
└── models_v1.parquet            # All model metadata (with coordinates)
```

### Chunk Index Schema

```python
chunk_index = pd.DataFrame({
    'model_id': [...],
    'chunk_id': [...],  # Which chunk file contains this model
    'chunk_offset': [...],  # Position within chunk
})
```

### Lazy Loading Logic

```python
def load_embeddings_for_models(model_ids: List[str]) -> np.ndarray:
    """Load embeddings only for requested model IDs."""
    # 1. Look up chunk IDs for each model_id
    # 2. Load only needed chunks
    # 3. Extract embeddings for requested models
    # 4. Return combined array
```

### API Changes

```python
@router.get("/api/models")
async def get_models(
    # ... existing params ...
    load_embeddings: bool = Query(True),  # New: control embedding loading
):
    # Filter first
    filtered_df = filter_data(...)
    
    if load_embeddings:
        # Load embeddings only for filtered models
        model_ids = filtered_df['model_id'].tolist()
        embeddings = load_embeddings_for_models(model_ids)
        # ... rest of logic
    else:
        # Return metadata only (coordinates pre-computed)
        # Frontend can load embeddings on-demand if needed
        ...
```

## Performance Targets

| Metric | Current (150k) | Target (All Models) |
|--------|---------------|---------------------|
| Startup Time | 10-30s | <5s |
| Memory Usage | ~500MB | <200MB (idle) |
| API Response | 1-3s | <1s (filtered) |
| Frontend Load | 2-5s | <2s (initial) |
| Rendering FPS | 30-60 | 60 (with virtualization) |

## Testing Strategy

1. **Unit Tests**: Chunk loading, filtering logic
2. **Integration Tests**: End-to-end API with chunked data
3. **Performance Tests**: Memory usage, response times
4. **Load Tests**: Simulate concurrent users

## Migration Path

1. **Phase 1**: Implement chunked storage, keep old system as fallback
2. **Phase 2**: Enable chunked loading for new deployments
3. **Phase 3**: Migrate existing pre-computed data to chunks
4. **Phase 4**: Remove old system once stable

## Monitoring

- Track memory usage per chunk load
- Monitor API response times
- Track frontend rendering performance
- Alert on memory spikes or slow responses

## Future Enhancements

1. **Distributed Storage**: Store chunks on S3/Cloud Storage
2. **Caching Layer**: Redis cache for frequently accessed chunks
3. **Background Jobs**: Pre-compute embeddings for new models
4. **Compression**: Use better compression (zstd) for parquet files
5. **Quantization**: Use int8 embeddings (50% memory reduction)


