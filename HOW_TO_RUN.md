# How to Run the Server

## Quick Start

### 1. Start the Server

```bash
cd hf-viz/backend
source venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the convenience script:
```bash
cd hf-viz
./start_server.sh
```

### 2. Verify Server is Running

Open a new terminal and check:
```bash
curl http://localhost:8000/
```

Expected response:
```json
{"message": "HF Model Ecosystem API", "status": "running"}
```

### 3. Test the API

```bash
# Get 10 models
curl "http://localhost:8000/api/models?max_points=10"

# Get models with filters
curl "http://localhost:8000/api/models?max_points=100&min_downloads=1000"

# Search for specific models
curl "http://localhost:8000/api/models?max_points=50&search_query=bert"
```

### 4. Check Server Logs

The server will show startup logs:
```
LOADING PRE-COMPUTED DATA (Fast Startup Mode)
============================================================
Loaded metadata for version v1_test
Chunked embeddings detected - skipping full embedding load for fast startup
Chunked embedding loader initialized - embeddings will be loaded on-demand
STARTUP COMPLETE in 2.45 seconds!
```

## Troubleshooting

### Server Won't Start

1. **Check if port is in use:**
   ```bash
   lsof -ti:8000
   # If something is running, kill it:
   kill $(lsof -ti:8000)
   ```

2. **Check virtual environment:**
   ```bash
   cd hf-viz/backend
   source venv/bin/activate
   which python  # Should show venv path
   ```

3. **Install missing dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### No Data Found

1. **Check if precomputed data exists:**
   ```bash
   ls -lh hf-viz/precomputed_data/*v1_test*
   ```

2. **Verify chunked files:**
   ```bash
   ls -lh hf-viz/precomputed_data/chunk_index_v1_test.parquet
   ```

### Server Starts But API Fails

1. **Check server logs** for error messages
2. **Verify data files** are readable
3. **Test with smaller max_points** (e.g., `max_points=5`)

## Expected Performance

- **Startup time**: 2-5 seconds
- **Memory usage**: ~100MB idle
- **API response**: <1s for filtered queries
- **First request**: May take 1-2s (loading chunks)

## Access from Browser

Once running, open:
- **API Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/
- **Models Endpoint**: http://localhost:8000/api/models?max_points=10

## Stop the Server

Press `Ctrl+C` in the terminal where the server is running, or:
```bash
pkill -f "uvicorn api.main:app"
```


