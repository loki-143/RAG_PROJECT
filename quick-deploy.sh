#!/bin/bash

# ============================================================
# CodeSense - One-Line Quick Deploy Script for Azure Linux
# ============================================================
# Usage: curl -fsSL <script-url> | bash
# ============================================================

set -e

echo "🚀 CodeSense Quick Deploy"
echo "=========================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "⚠️  Please don't run as root. Run as regular user."
    exit 1
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed"
    echo "⚠️  Please log out and log back in, then run this script again"
    exit 0
fi

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
    echo "✅ Docker Compose installed"
fi

# Clone repository
REPO_URL="${CODESENSE_REPO_URL:-https://github.com/YOUR_USERNAME/YOUR_REPO.git}"
INSTALL_DIR="${CODESENSE_INSTALL_DIR:-$HOME/codesense}"

if [ -d "$INSTALL_DIR" ]; then
    echo "📁 Directory $INSTALL_DIR already exists"
    read -p "Remove and reinstall? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$INSTALL_DIR"
    else
        exit 1
    fi
fi

echo "📥 Cloning repository..."
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Setup environment
echo "⚙️  Setting up environment..."
cp .env.production .env

# Prompt for API keys
echo ""
echo "🔑 API Key Configuration"
echo "You need at least one API key to use CodeSense"
echo ""

read -p "Enter GitHub Token (or press Enter to skip): " github_token
if [ ! -z "$github_token" ]; then
    sed -i "s/your_github_token_here/$github_token/" .env
fi

read -p "Enter Google API Key (or press Enter to skip): " google_key
if [ ! -z "$google_key" ]; then
    sed -i "s/your_google_api_key_here/$google_key/" .env
fi

# Check if at least one key was provided
if [ -z "$github_token" ] && [ -z "$google_key" ]; then
    echo "⚠️  No API keys provided. You'll need to edit .env manually:"
    echo "   nano $INSTALL_DIR/.env"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Deploy
echo ""
echo "🚀 Deploying CodeSense..."
chmod +x deploy.sh
./deploy.sh

echo ""
echo "✅ Quick deploy complete!"
echo ""
echo "📍 Installation directory: $INSTALL_DIR"
echo "🌐 Access CodeSense at: http://$(hostname -I | awk '{print $1}')"
echo ""
echo "📝 Useful commands:"
echo "   cd $INSTALL_DIR"
echo "   docker compose logs -f    # View logs"
echo "   docker compose restart    # Restart services"
echo "   docker compose down       # Stop services"
echo ""
