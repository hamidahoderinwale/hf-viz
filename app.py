#!/usr/bin/env python3
"""
Hugging Face Spaces Entry Point
This file serves as the entry point for Hugging Face Spaces deployment.
It wraps the FastAPI backend and serves the frontend.
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Import the FastAPI app from backend
from api.main import app

# The app is already configured in api/main.py
# Hugging Face Spaces will automatically detect and serve it

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


