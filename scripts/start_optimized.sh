#!/bin/bash
# Start HF Viz with all optimizations enabled

# Start Redis if available
if command -v redis-server &> /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes --port 6379 --maxmemory 512mb --maxmemory-policy allkeys-lru
    sleep 1
    REDIS_ENABLED=true
else
    echo "Redis not available, using in-memory cache"
    REDIS_ENABLED=false
fi

# Start backend
echo "Starting backend with optimizations..."
cd backend
source ../venv/bin/activate

SAMPLE_SIZE=5000 \
REDIS_ENABLED=$REDIS_ENABLED \
REDIS_HOST=localhost \
REDIS_PORT=6379 \
REDIS_TTL=300 \
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &

BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
for i in {1..60}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✓ Backend is ready!"
        break
    fi
    sleep 1
done

# Start frontend
echo "Starting frontend..."
cd ../frontend
npm start &

FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "🎉 HF Viz is running with all optimizations!"
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Wait for user interrupt
trap 'kill $BACKEND_PID $FRONTEND_PID; exit' INT
wait
