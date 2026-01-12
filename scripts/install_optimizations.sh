#!/bin/bash
# Complete installation script for all speed optimizations

set -e

echo "🚀 Installing Speed Optimizations for HF Viz..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get project root
PROJECT_ROOT="/Users/hamidaho/hf_viz"
cd "$PROJECT_ROOT"

# 1. Install Backend Dependencies
echo -e "${BLUE}📦 Installing backend dependencies...${NC}"
source venv/bin/activate
pip install -q -r backend/config/requirements.txt
echo -e "${GREEN}✓${NC} Backend dependencies installed"

# 2. Install Frontend Dependencies
echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
cd frontend
npm install --silent
cd ..
echo -e "${GREEN}✓${NC} Frontend dependencies installed"

# 3. Check for Redis
echo -e "${BLUE}🔍 Checking for Redis...${NC}"
if command -v redis-server &> /dev/null; then
    echo -e "${GREEN}✓${NC} Redis is installed"
    REDIS_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC}  Redis not found. Install with:"
    echo "    brew install redis  (macOS)"
    echo "    apt-get install redis  (Ubuntu)"
    echo "    or use Docker: docker run -d -p 6379:6379 redis:7-alpine"
    echo ""
    echo "Backend will use in-memory cache as fallback"
    REDIS_AVAILABLE=false
fi

# 4. Check for Docker
echo -e "${BLUE}🔍 Checking for Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC}  Docker not found. Install from https://docker.com"
    DOCKER_AVAILABLE=false
fi

# 5. Create startup script
echo -e "${BLUE}📝 Creating startup scripts...${NC}"

cat > start_optimized.sh << 'EOF'
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
EOF

chmod +x start_optimized.sh

# Create Docker startup script
cat > start_docker.sh << 'EOF'
#!/bin/bash
# Start HF Viz with Docker Compose (includes Redis, Nginx)

echo "🐳 Starting HF Viz with Docker Compose..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 5

echo ""
echo "🎉 HF Viz is running!"
echo ""
echo "Backend: http://localhost:8000 (via nginx: http://localhost/api/)"
echo "API Docs: http://localhost:8000/docs"
echo "Frontend: Build and deploy separately or add to docker-compose.yml"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
echo "Cache stats:"
echo "  docker-compose exec redis redis-cli INFO stats"
EOF

chmod +x start_docker.sh

echo -e "${GREEN}✓${NC} Startup scripts created"

# 6. Create test script
cat > test_optimizations.sh << 'EOF'
#!/bin/bash
# Test that all optimizations are working

echo "🧪 Testing Speed Optimizations..."
echo ""

API_BASE="http://localhost:8000"

# Test 1: Basic connectivity
echo "Test 1: Backend connectivity..."
if curl -sf "$API_BASE/" > /dev/null; then
    echo "✓ Backend is responding"
else
    echo "✗ Backend not responding. Start it first with ./start_optimized.sh"
    exit 1
fi

# Test 2: Check SAMPLE_SIZE is applied
echo ""
echo "Test 2: SAMPLE_SIZE configuration..."
STATS=$(curl -s "$API_BASE/api/stats")
TOTAL=$(echo $STATS | grep -o '"total_models":[0-9]*' | grep -o '[0-9]*')
if [ "$TOTAL" -le 10000 ]; then
    echo "✓ SAMPLE_SIZE is working (loaded $TOTAL models)"
else
    echo "⚠ Warning: Loaded $TOTAL models (expected ≤10000)"
fi

# Test 3: Test caching headers
echo ""
echo "Test 3: HTTP caching headers..."
HEADERS=$(curl -sI "$API_BASE/api/stats")
if echo "$HEADERS" | grep -q "Cache-Control"; then
    echo "✓ Cache-Control headers present"
else
    echo "⚠ Cache-Control headers missing"
fi

# Test 4: Test Redis connection
echo ""
echo "Test 4: Redis connectivity..."
if command -v redis-cli &> /dev/null; then
    if redis-cli -h localhost -p 6379 PING > /dev/null 2>&1; then
        echo "✓ Redis is running"
        KEYS=$(redis-cli -h localhost -p 6379 KEYS "hfviz:*" | wc -l)
        echo "  Cached keys: $KEYS"
    else
        echo "⚠ Redis not responding (using in-memory cache)"
    fi
else
    echo "⚠ redis-cli not installed"
fi

# Test 5: Test MessagePack support
echo ""
echo "Test 5: MessagePack binary format..."
MSGPACK_RESPONSE=$(curl -s -w "%{http_code}" -H "Accept: application/msgpack" "$API_BASE/api/models?max_points=10" -o /tmp/test.msgpack)
if [ "$MSGPACK_RESPONSE" = "200" ]; then
    MSGPACK_SIZE=$(stat -f%z /tmp/test.msgpack 2>/dev/null || stat -c%s /tmp/test.msgpack 2>/dev/null)
    echo "✓ MessagePack endpoint working (${MSGPACK_SIZE} bytes)"
else
    echo "⚠ MessagePack endpoint returned $MSGPACK_RESPONSE"
fi

# Test 6: Compare JSON vs MessagePack size
echo ""
echo "Test 6: Payload size comparison..."
JSON_SIZE=$(curl -s "$API_BASE/api/models?max_points=100" | wc -c)
MSGPACK_SIZE=$(curl -s -H "Accept: application/msgpack" "$API_BASE/api/models?max_points=100&format=msgpack" | wc -c)
REDUCTION=$(echo "scale=1; 100 - ($MSGPACK_SIZE * 100 / $JSON_SIZE)" | bc)
echo "  JSON: ${JSON_SIZE} bytes"
echo "  MessagePack: ${MSGPACK_SIZE} bytes"
echo "  Reduction: ${REDUCTION}%"

# Test 7: Test response time (with cache)
echo ""
echo "Test 7: Response time (cache test)..."
# First request (cold)
START=$(date +%s%N)
curl -s "$API_BASE/api/stats" > /dev/null
END=$(date +%s%N)
COLD_TIME=$(echo "scale=0; ($END - $START) / 1000000" | bc)

# Second request (should be cached)
START=$(date +%s%N)
curl -s "$API_BASE/api/stats" > /dev/null
END=$(date +%s%N)
WARM_TIME=$(echo "scale=0; ($END - $START) / 1000000" | bc)

echo "  Cold request: ${COLD_TIME}ms"
echo "  Warm request: ${WARM_TIME}ms"

if [ "$WARM_TIME" -lt "$COLD_TIME" ]; then
    SPEEDUP=$(echo "scale=1; $COLD_TIME / $WARM_TIME" | bc)
    echo "  ✓ Cache working (${SPEEDUP}x faster)"
else
    echo "  ⚠ Cache may not be working"
fi

echo ""
echo "🎉 Testing complete!"
EOF

chmod +x test_optimizations.sh

echo -e "${GREEN}✓${NC} Test script created"

# 7. Summary
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📚 Created files:"
echo "  ./start_optimized.sh     - Start with optimizations (local)"
echo "  ./start_docker.sh        - Start with Docker Compose"
echo "  ./test_optimizations.sh  - Test all optimizations"
echo ""
echo "🚀 Quick Start:"
echo ""
if [ "$REDIS_AVAILABLE" = true ] && [ "$DOCKER_AVAILABLE" = true ]; then
    echo "  Option 1 (Recommended): Docker with Redis + Nginx"
    echo "    ./start_docker.sh"
    echo ""
    echo "  Option 2: Local development"
    echo "    ./start_optimized.sh"
elif [ "$REDIS_AVAILABLE" = true ]; then
    echo "  ./start_optimized.sh"
elif [ "$DOCKER_AVAILABLE" = true ]; then
    echo "  ./start_docker.sh"
else
    echo "  ./start_optimized.sh  (will use in-memory cache)"
fi
echo ""
echo "🧪 After starting, test with:"
echo "  ./test_optimizations.sh"
echo ""
echo "📖 Full documentation:"
echo "  cat SPEED_OPTIMIZATIONS_COMPLETE.md"
echo ""



