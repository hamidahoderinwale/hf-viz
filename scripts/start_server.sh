#!/bin/bash
cd backend
source venv/bin/activate
echo "Starting server with chunked embeddings..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
