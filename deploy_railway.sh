#!/bin/bash
# Quick deployment script for Railway

echo "🚂 Deploying HF Viz Backend to Railway..."
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway
echo "📝 Please login to Railway..."
railway login

# Initialize project if needed
if [ ! -f "railway.json" ]; then
    echo "🎬 Initializing Railway project..."
    railway init
fi

# Deploy
echo "🚀 Deploying..."
railway up

# Set environment variables
echo "⚙️  Setting environment variables..."
railway variables set SAMPLE_SIZE=5000
railway variables set PORT=8000

# Generate domain
echo "🌐 Setting up domain..."
railway domain

# Get the URL
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the URL shown above"
echo "2. Update frontend/src/config/api.ts with this URL"
echo "3. Redeploy frontend to Netlify"
echo ""
echo "🔍 Check logs with: railway logs"
echo "🌐 Open dashboard: railway open"




