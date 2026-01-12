#!/bin/bash
# Check precompute status and deploy when ready

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Checking precompute status..."

# Check if precompute process is running
if ps aux | grep -q "[p]recompute_data.py"; then
    echo "⏳ Precompute is still running..."
    echo ""
    echo "Progress:"
    tail -1 precompute_full.log 2>/dev/null | grep -o "Batches:.*" || echo "   Check precompute_full.log for details"
    echo ""
    echo "Estimated time remaining: 2-3 hours"
    echo ""
    echo "To monitor: tail -f precompute_full.log"
else
    echo "✅ Precompute process not running"
    echo ""
    
    # Check if files exist
    if [ -f "precomputed_data/models_v1.parquet" ] && [ -f "precomputed_data/chunk_index_v1.parquet" ]; then
        echo "✅ Precomputed files found!"
        echo ""
        echo "Files ready:"
        ls -lh precomputed_data/models_v1.parquet
        ls -lh precomputed_data/chunk_index_v1.parquet
        ls -lh precomputed_data/embeddings_chunk_*_v1.parquet 2>/dev/null | wc -l | xargs echo "   Chunk files:"
        echo ""
        echo "🚀 Ready to deploy!"
        echo ""
        echo "Next steps:"
        echo "  1. Upload data: python upload_to_hf_dataset.py"
        echo "  2. Deploy to Space: ./auto_deploy.sh"
    else
        echo "⚠️  Precomputed files not found"
        echo "   Precompute may have failed or is still in progress"
        echo "   Check: tail -50 precompute_full.log"
    fi
fi


