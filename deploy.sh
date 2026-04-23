#!/bin/bash

# ============================================================
# CodeSense - Azure Linux Deployment Script
# ============================================================
# This script deploys CodeSense using Docker Compose
# ============================================================

set -e  # Exit on error

echo "🚀 CodeSense Deployment Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo "Please copy .env.production to .env and configure your API keys:"
    echo "  cp .env.production .env"
    echo "  nano .env  # Edit and add your API keys"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker is not installed${NC}"
    echo "Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}❌ Error: Docker Compose is not installed${NC}"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if API keys are configured
if grep -q "your_github_token_here" .env && grep -q "your_google_api_key_here" .env; then
    echo -e "${YELLOW}⚠️  Warning: API keys not configured in .env file${NC}"
    echo "Please edit .env and add at least one API key (GITHUB_TOKEN or GOOGLE_API_KEY)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "📦 Building Docker images..."
docker compose build

echo ""
echo "🔄 Starting services..."
docker compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check backend health
echo "Checking backend health..."
for i in {1..30}; do
    if curl -f http://localhost:5000/health &> /dev/null; then
        echo -e "${GREEN}✅ Backend is healthy${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend health check failed${NC}"
        echo "Check logs: docker compose logs backend"
        exit 1
    fi
    sleep 2
done

# Check frontend health
echo "Checking frontend health..."
for i in {1..10}; do
    if curl -f http://localhost:80 &> /dev/null; then
        echo -e "${GREEN}✅ Frontend is healthy${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${RED}❌ Frontend health check failed${NC}"
        echo "Check logs: docker compose logs frontend"
        exit 1
    fi
    sleep 2
done

echo ""
echo -e "${GREEN}✅ Deployment successful!${NC}"
echo ""
echo "📊 Service Status:"
docker compose ps
echo ""
echo "🌐 Access CodeSense:"
echo "   Frontend: http://localhost"
echo "   Backend API: http://localhost:5000"
echo ""
echo "📝 Useful commands:"
echo "   View logs:        docker compose logs -f"
echo "   Stop services:    docker compose down"
echo "   Restart:          docker compose restart"
echo "   Update & restart: docker compose up -d --build"
echo ""
