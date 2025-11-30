# Hugging Face Spaces Docker deployment
FROM python:3.11-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=user backend/ /app/backend/

# Bundle precomputed data for instant startup
COPY --chown=user precomputed_data/ /app/precomputed_data/
COPY --chown=user cache/ /app/cache/

# Switch to non-root user
USER user

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV ALLOW_ALL_ORIGINS=true

WORKDIR /app/backend
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
