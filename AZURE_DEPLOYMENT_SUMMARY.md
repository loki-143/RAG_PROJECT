# CodeSense - Azure Linux Deployment Summary

## 🎯 Deployment Options

### Option 1: Super Quick (Recommended for Testing)
```bash
git clone <your-repo> codesense && cd codesense && cp .env.production .env && nano .env && chmod +x deploy.sh && ./deploy.sh
```

### Option 2: Step-by-Step (Recommended for Production)
See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guide.

### Option 3: One-Line Auto-Install (Coming Soon)
```bash
curl -fsSL <script-url> | bash
```

## 📋 Prerequisites Checklist

- [ ] Azure Linux VM (Ubuntu 20.04+)
- [ ] Docker installed
- [ ] Docker Compose installed
- [ ] At least one API key (GitHub or Google)
- [ ] Ports 80 and 5000 available
- [ ] 2GB+ RAM, 2+ CPU cores

## 🚀 Quick Start Commands

### 1. Install Docker (if needed)
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Clone Repository
```bash
git clone <your-repo-url> codesense
cd codesense
```

### 3. Configure
```bash
cp .env.production .env
nano .env  # Add your API keys
```

### 4. Deploy
```bash
chmod +x deploy.sh
./deploy.sh
```

### 5. Access
```
Frontend: http://your-server-ip
Backend:  http://your-server-ip:5000
```

## 📁 Files Created

| File | Purpose |
|------|---------|
| `DEPLOYMENT.md` | Complete deployment guide with troubleshooting |
| `QUICKSTART_AZURE.md` | Quick start guide for Azure |
| `deploy.sh` | Automated deployment script |
| `quick-deploy.sh` | One-line installation script |
| `.env.production` | Production environment template |
| `AZURE_DEPLOYMENT_SUMMARY.md` | This file |

## 🔧 Configuration Files

### `.env.production` (Template)
- Copy to `.env` and fill in your values
- Required: At least one API key (GITHUB_TOKEN or GOOGLE_API_KEY)
- Optional: Performance tuning parameters

### `docker-compose.yml`
- Production-ready configuration
- Health checks enabled
- Persistent volumes for data
- Optimized networking

### `backend/Dockerfile`
- Pre-downloads embedding model
- Health check included
- Optimized for production

### `frontend/Dockerfile`
- Multi-stage build
- Nginx with production settings
- Health check included

### `frontend/nginx.conf`
- Reverse proxy to backend
- Security headers
- Gzip compression
- Static asset caching
- SSE support for streaming

## 🎛️ Performance Settings

### Low-End Systems (2-4 cores, 4GB RAM)
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=2
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=10000
MAX_WORKERS=4
```

### Mid-Range Systems (4-8 cores, 8GB RAM)
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=0
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=5000
MAX_WORKERS=8
```

### High-End Systems (8+ cores, 16GB+ RAM)
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=0
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=2000
MAX_WORKERS=16
```

## 📊 Architecture

```
Internet
   ↓
Azure Linux VM
   ↓
Port 80 → Frontend (Nginx)
   ↓
   └─→ /api/* → Backend (FastAPI) :5000
                    ↓
                    ├─→ Indexes (persistent)
                    ├─→ Histories (persistent)
                    └─→ Model Cache (persistent)
```

## 🔐 Security Features

- ✅ Security headers (X-Frame-Options, X-Content-Type-Options, X-XSS-Protection)
- ✅ Gzip compression
- ✅ Static asset caching
- ✅ Health check endpoints
- ✅ Environment variable isolation
- ✅ No hardcoded credentials
- ⚠️ HTTPS not configured (add manually for production)

## 📝 Management Commands

```bash
# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f backend
docker compose logs -f frontend

# Restart services
docker compose restart

# Stop services
docker compose down

# Update and restart
git pull
docker compose up -d --build

# Check status
docker compose ps

# Check health
curl http://localhost:5000/health
curl http://localhost:80/health
```

## 🐛 Common Issues & Solutions

### Issue: Backend won't start
```bash
# Check logs
docker compose logs backend

# Common causes:
# 1. Missing API keys → Edit .env
# 2. Port 5000 in use → Stop conflicting service
# 3. Out of memory → Increase VM RAM
```

### Issue: Frontend can't connect to backend
```bash
# Verify backend is running
docker compose ps backend

# Check network
docker compose exec frontend ping backend

# Restart services
docker compose restart
```

### Issue: Slow performance
```bash
# Edit .env:
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=10000

# Restart
docker compose restart backend
```

### Issue: Out of disk space
```bash
# Clean Docker
docker system prune -a --volumes

# Remove old indexes
rm -rf backend/indexes/*
```

## 🎯 Next Steps

1. **Test the deployment** - Index a small repository
2. **Configure firewall** - `sudo ufw allow 80/tcp`
3. **Set up HTTPS** - Use Certbot for SSL
4. **Configure backups** - See DEPLOYMENT.md
5. **Monitor resources** - `docker stats`
6. **Set up monitoring** - Consider Prometheus/Grafana

## 📚 Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [QUICKSTART_AZURE.md](QUICKSTART_AZURE.md) - Quick start guide
- [backend/PARALLEL_PROCESSING.md](backend/PARALLEL_PROCESSING.md) - Parallel processing details
- [backend/EMBEDDING_OPTIMIZATION.md](backend/EMBEDDING_OPTIMIZATION.md) - Embedding optimization
- [backend/STARTUP_OPTIMIZATION.md](backend/STARTUP_OPTIMIZATION.md) - Startup optimization
- [backend/CPU_LAG_FIX.md](backend/CPU_LAG_FIX.md) - CPU lag fixes

## 🆘 Support

If you encounter issues:
1. Check logs: `docker compose logs -f`
2. Review troubleshooting in DEPLOYMENT.md
3. Verify API keys are valid
4. Check system resources: `docker stats`
5. Review GitHub issues

## ✅ Deployment Checklist

- [ ] Docker installed
- [ ] Repository cloned
- [ ] .env configured with API keys
- [ ] deploy.sh executed successfully
- [ ] Frontend accessible at http://your-ip
- [ ] Backend health check passes
- [ ] Test repository indexing works
- [ ] Firewall configured (optional)
- [ ] HTTPS configured (optional)
- [ ] Backups configured (optional)

## 🎉 Success Criteria

Your deployment is successful when:
- ✅ `docker compose ps` shows all services as "healthy"
- ✅ Frontend loads in browser
- ✅ You can index a test repository
- ✅ You can query the indexed repository
- ✅ No errors in logs

---

**Ready to deploy?** Run: `chmod +x deploy.sh && ./deploy.sh`
