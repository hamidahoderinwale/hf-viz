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
