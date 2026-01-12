#!/bin/bash
# Automated deployment script for Hugging Face Spaces
# This script checks precompute status and handles deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     HF Spaces Auto-Deployment Script                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if precompute is complete
check_precompute() {
    if [ -f "precomputed_data/models_v1.parquet" ] && [ -f "precomputed_data/chunk_index_v1.parquet" ]; then
        echo "✅ Precomputed data files found"
        return 0
    else
        echo "⏳ Precomputed data not ready yet"
        return 1
    fi
}

# Upload data to HF Dataset
upload_data() {
    echo ""
    echo "📤 Uploading chunked data to Hugging Face Dataset..."
    echo ""
    
    cd backend
    source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
    pip install -q huggingface-hub tqdm 2>/dev/null
    
    cd ..
    python upload_to_hf_dataset.py --dataset-id modelbiome/hf-viz-precomputed --version v1
    
    echo ""
    echo "✅ Data upload complete!"
}

# Prepare Space files
prepare_space() {
    SPACE_DIR="${1:-hf-viz-space}"
    
    echo ""
    echo "📦 Preparing files for HF Space..."
    echo ""
    
    mkdir -p "$SPACE_DIR"
    
    # Copy required files
    cp app.py "$SPACE_DIR/"
    cp requirements.txt "$SPACE_DIR/"
    cp Dockerfile "$SPACE_DIR/"
    cp README_SPACE.md "$SPACE_DIR/README.md"
    cp -r backend "$SPACE_DIR/"
    cp -r frontend "$SPACE_DIR/"
    mkdir -p "$SPACE_DIR/precomputed_data"
    touch "$SPACE_DIR/precomputed_data/.gitkeep"
    
    echo "✅ Files prepared in: $SPACE_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. cd $SPACE_DIR"
    echo "  2. git init"
    echo "  3. git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME"
    echo "  4. git add ."
    echo "  5. git commit -m 'Deploy HF Model Ecosystem Visualizer'"
    echo "  6. git push"
}

# Main execution
main() {
    if check_precompute; then
        echo ""
        read -p "Precompute complete! Upload data to HF Dataset? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            upload_data
        fi
        
        echo ""
        read -p "Prepare files for HF Space deployment? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            prepare_space
        fi
    else
        echo ""
        echo "⏳ Waiting for precompute to complete..."
        echo "   Check progress: tail -f precompute_full.log"
        echo "   Or run this script again when precompute is done"
        echo ""
        echo "Current status:"
        ps aux | grep "[p]recompute_data.py" && echo "   Precompute is running" || echo "   Precompute not running"
    fi
}

main "$@"


