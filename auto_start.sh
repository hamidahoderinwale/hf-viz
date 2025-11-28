#!/bin/bash
# Auto-start backend once pre-computation completes

echo "🔍 Monitoring pre-computation process..."
echo "📊 Log file: /Users/hamidaho/hf_viz/precompute_fast.log"
echo ""

# Wait for pre-computation to complete
while ps aux | grep -q "[p]recompute_fast.py"; do
    # Show latest progress
    LATEST=$(tail -1 /Users/hamidaho/hf_viz/precompute_fast.log 2>/dev/null | grep -E "Batches:|Step|INFO")
    if [ ! -z "$LATEST" ]; then
        echo -ne "\r⏳ $LATEST                    "
    fi
    sleep 5
done

echo ""
echo ""
echo "✅ Pre-computation complete!"
echo ""

# Check if files were created successfully
if [ -f "/Users/hamidaho/hf_viz/precomputed_data/models_v1.parquet" ]; then
    echo "✅ Found: models_v1.parquet"
    ls -lh /Users/hamidaho/hf_viz/precomputed_data/models_v1.parquet
else
    echo "❌ ERROR: models_v1.parquet not found"
    exit 1
fi

if [ -f "/Users/hamidaho/hf_viz/precomputed_data/embeddings_v1.parquet" ]; then
    echo "✅ Found: embeddings_v1.parquet"
    ls -lh /Users/hamidaho/hf_viz/precomputed_data/embeddings_v1.parquet
else
    echo "❌ ERROR: embeddings_v1.parquet not found"
    exit 1
fi

if [ -f "/Users/hamidaho/hf_viz/precomputed_data/metadata_v1.json" ]; then
    echo "✅ Found: metadata_v1.json"
    cat /Users/hamidaho/hf_viz/precomputed_data/metadata_v1.json | python3 -m json.tool 2>/dev/null | grep -E "total_models|unique_libraries|unique_pipelines" | head -3
else
    echo "❌ ERROR: metadata_v1.json not found"
    exit 1
fi

echo ""
echo "🚀 Starting backend server..."
echo ""

cd /Users/hamidaho/hf_viz/backend

# Kill any existing backend processes
pkill -f "uvicorn.*api.main:app" 2>/dev/null

# Start backend
source /Users/hamidaho/hf_viz/venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

echo ""
echo "Backend started on http://localhost:8000"
echo "Frontend should be running on http://localhost:3000"
echo ""
echo "📊 Open your browser and refresh the page!"

