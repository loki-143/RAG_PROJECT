#!/bin/bash

# ============================================================
# CodeSense - Quick Deployment Commands
# ============================================================
# Use these commands to deploy/redeploy your application
# ============================================================

echo "🚀 CodeSense Deployment Commands"
echo "=================================="
echo ""

# Function to show command with description
show_command() {
    echo "📌 $1"
    echo "   Command: $2"
    echo ""
}

show_command "Stop all services" \
    "docker compose down"

show_command "Rebuild everything (no cache)" \
    "docker compose build --no-cache"

show_command "Start services" \
    "docker compose up -d"

show_command "View logs (all services)" \
    "docker compose logs -f"

show_command "View backend logs only" \
    "docker compose logs -f backend"

show_command "View frontend logs only" \
    "docker compose logs -f frontend"

show_command "Check service status" \
    "docker compose ps"

show_command "Test backend health" \
    "curl http://localhost:5000/health"

show_command "Test frontend health" \
    "curl http://localhost:80/health"

show_command "Test API proxy" \
    "curl http://localhost:80/api/health"

show_command "Restart specific service" \
    "docker compose restart backend"

show_command "View nginx config" \
    "docker compose exec frontend cat /etc/nginx/conf.d/default.conf"

show_command "Test nginx config" \
    "docker compose exec frontend nginx -t"

show_command "Check if backend is reachable from frontend" \
    "docker compose exec frontend wget -O- http://backend:5000/health"

echo "=================================="
echo "🎯 Quick Deploy (One Command)"
echo "=================================="
echo ""
echo "docker compose down && docker compose build --no-cache && docker compose up -d && docker compose logs -f"
echo ""
echo "=================================="
echo "✅ Verification Steps"
echo "=================================="
echo ""
echo "1. Check services: docker compose ps"
echo "2. Check backend: curl http://localhost:5000/health"
echo "3. Check API proxy: curl http://localhost:80/api/health"
echo "4. Open browser: http://your-vm-ip"
echo "5. Check browser console (F12) - no errors"
echo ""
