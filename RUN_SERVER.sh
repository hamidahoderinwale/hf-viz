#!/bin/bash
echo "Starting HF Model Ecosystem API Server..."
echo "=========================================="
cd backend
source venv/bin/activate
echo "✓ Virtual environment activated"
echo "✓ Starting server on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
