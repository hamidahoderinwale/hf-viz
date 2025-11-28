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
