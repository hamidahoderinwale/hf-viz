# Hugging Face Spaces - Full Stack Deployment
# Serves both React frontend and FastAPI backend

FROM node:18-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --legacy-peer-deps
COPY frontend/ ./
ENV CI=false
RUN npm run build

# Production image
FROM python:3.11-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY --chown=user backend/ /app/backend/

# Copy frontend build
COPY --from=frontend-builder --chown=user /frontend/build /app/frontend/build

# Create directories for runtime data
RUN mkdir -p /app/precomputed_data /app/cache && chown -R user:user /app/precomputed_data /app/cache

# Copy precomputed data if available (metadata only in repo)
COPY --chown=user precomputed_data/ /app/precomputed_data/

# Switch to non-root user
USER user

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV ALLOW_ALL_ORIGINS=true
ENV SAMPLE_SIZE=50000
ENV HF_PRECOMPUTED_DATASET=modelbiome/hf-viz-precomputed

WORKDIR /app/backend
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
