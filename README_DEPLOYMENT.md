# CodeSense - Docker Deployment Guide

## 🚀 Quick Deploy on Azure Linux

### Single Command Deployment
```bash
git clone <your-repo-url> codesense && cd codesense && cp .env.production .env && nano .env && chmod +x deploy.sh && ./deploy.sh
```

That's it! Your CodeSense instance will be running at `http://your-server-ip`

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [AZURE_DEPLOYMENT_SUMMARY.md](AZURE_DEPLOYMENT_SUMMARY.md) | **START HERE** - Quick overview and commands |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Complete deployment guide with troubleshooting |
| [QUICKSTART_AZURE.md](QUICKSTART_AZURE.md) | 5-minute quick start guide |

## 🎯 What's Included

### Deployment Scripts
- **`deploy.sh`** - Automated deployment with health checks
- **`quick-deploy.sh`** - One-line installation script
- **`.env.production`** - Production environment template

### Docker Configuration
- **`docker-compose.yml`** - Production-ready orchestration
- **`backend/Dockerfile`** - Optimized backend container
- **`frontend/Dockerfile`** - Multi-stage frontend build
- **`frontend/nginx.conf`** - Production nginx configuration

### Features
- ✅ One-command deployment
- ✅ Automatic health checks
- ✅ Persistent data volumes
- ✅ Pre-downloaded embedding models
- ✅ Optimized for production
- ✅ Security headers configured
- ✅ Gzip compression enabled
- ✅ Static asset caching

## 🔧 Configuration

### Required
- At least one API key:
  - `GITHUB_TOKEN` (recommended) - Get from https://github.com/settings/tokens
  - OR `GOOGLE_API_KEY` - Get from https://aistudio.google.com/apikey

### Optional Performance Tuning
```env
# For low-end systems (2-4 cores)
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=10000

# For high-end systems (8+ cores)
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=2000
```

## 📊 System Requirements

### Minimum
- 2 CPU cores
- 4GB RAM
- 10GB disk space
- Ubuntu 20.04+ or similar

### Recommended
- 4+ CPU cores
- 8GB+ RAM
- 20GB+ disk space
- SSD storage

## 🎛️ Management

```bash
# View logs
docker compose logs -f

# Restart services
docker compose restart

# Stop services
docker compose down

# Update application
git pull && docker compose up -d --build

# Check status
docker compose ps
```

## 🔐 Security

- Environment variables for sensitive data
- No hardcoded credentials
- Security headers configured
- Health check endpoints
- Isolated Docker network

**For production:** Add HTTPS using Certbot (see DEPLOYMENT.md)

## 🐛 Troubleshooting

### Backend won't start
```bash
docker compose logs backend
# Check API keys in .env
```

### Slow performance
```bash
# Edit .env:
USE_PARALLEL_EMBEDDINGS=false
# Restart:
docker compose restart backend
```

### Out of disk space
```bash
docker system prune -a --volumes
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete troubleshooting guide.

## 📈 Performance

### Optimizations Included
- Parallel file chunking (2-5x faster)
- Optional parallel embeddings (2-4x faster)
- Lazy model loading (10x faster startup)
- Optimized file scanning (10-30x faster)
- Embedding model caching
- Static asset caching

### Benchmarks
- **Startup time:** 2-3 seconds (vs 30s before)
- **File scanning:** 5-15 seconds for 10K files (vs 5 minutes)
- **Chunking:** 10K files in ~7 seconds with parallel processing
- **Embeddings:** Configurable based on system resources

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         Azure Linux VM              │
│                                     │
│  ┌───────────────────────────────┐ │
│  │     Docker Compose            │ │
│  │                               │ │
│  │  ┌──────────┐  ┌───────────┐ │ │
│  │  │ Frontend │  │  Backend  │ │ │
│  │  │  Nginx   │→ │  FastAPI  │ │ │
│  │  │  :80     │  │  :5000    │ │ │
│  │  └──────────┘  └───────────┘ │ │
│  │                               │ │
│  │  Volumes:                     │ │
│  │  • indexes/                   │ │
│  │  • histories/                 │ │
│  │  • model_cache/               │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```

## 🎓 Learn More

- [Backend Optimizations](backend/PARALLEL_PROCESSING.md)
- [Embedding Optimization](backend/EMBEDDING_OPTIMIZATION.md)
- [Startup Optimization](backend/STARTUP_OPTIMIZATION.md)
- [CPU Lag Fixes](backend/CPU_LAG_FIX.md)

## 📝 License

[Your License Here]

---

**Ready to deploy?** Start with [AZURE_DEPLOYMENT_SUMMARY.md](AZURE_DEPLOYMENT_SUMMARY.md)
