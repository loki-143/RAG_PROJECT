# CodeSense - Azure Linux Deployment Guide

Complete guide for deploying CodeSense on Azure Linux using Docker Compose.

## Prerequisites

- Azure Linux VM (Ubuntu 20.04+ or similar)
- Docker and Docker Compose installed
- At least 2GB RAM, 2 CPU cores
- Ports 80 and 5000 available

## Quick Start (Single Command)

```bash
# Clone repository, configure, and deploy
git clone <your-repo-url> codesense && cd codesense && cp .env.production .env && nano .env && chmod +x deploy.sh && ./deploy.sh
```

## Step-by-Step Deployment

### 1. Install Docker (if not installed)

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo apt-get install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version
```

### 2. Clone Repository

```bash
git clone <your-repo-url> codesense
cd codesense
```

### 3. Configure Environment

```bash
# Copy production template
cp .env.production .env

# Edit configuration
nano .env
```

**Required Configuration:**
- Set `GITHUB_TOKEN` (get from https://github.com/settings/tokens)
- OR set `GOOGLE_API_KEY` (get from https://aistudio.google.com/apikey)
- At least one API key is required

**Optional Configuration:**
- Adjust `USE_PARALLEL_EMBEDDINGS` based on your CPU (false for 2-4 cores)
- Modify `INDEXING_MAX_WORKERS` if needed (0 = auto-detect)

### 4. Deploy

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

The script will:
- ✅ Validate configuration
- 📦 Build Docker images
- 🚀 Start services
- 🏥 Run health checks
- ✅ Confirm successful deployment

### 5. Access Application

- **Frontend:** http://your-server-ip
- **Backend API:** http://your-server-ip:5000

## Manual Deployment (Alternative)

If you prefer manual control:

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

## Management Commands

### View Logs
```bash
# All services
docker compose logs -f

# Backend only
docker compose logs -f backend

# Frontend only
docker compose logs -f frontend
```

### Restart Services
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart backend
```

### Stop Services
```bash
# Stop (keeps data)
docker compose down

# Stop and remove volumes (deletes data)
docker compose down -v
```

### Update Application
```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose up -d --build
```

### Check Service Health
```bash
# Backend health
curl http://localhost:5000/health

# Frontend health
curl http://localhost:80
```

## Troubleshooting

### Backend Won't Start

```bash
# Check logs
docker compose logs backend

# Common issues:
# 1. Missing API keys - check .env file
# 2. Port 5000 in use - stop conflicting service
# 3. Out of memory - increase VM RAM
```

### Frontend Can't Connect to Backend

```bash
# Check nginx configuration
docker compose exec frontend cat /etc/nginx/conf.d/default.conf

# Verify backend is running
docker compose ps backend

# Check network connectivity
docker compose exec frontend ping backend
```

### Slow Performance

```bash
# Check resource usage
docker stats

# Adjust workers in .env:
# - Set USE_PARALLEL_EMBEDDINGS=false for low-end systems
# - Reduce MAX_WORKERS if CPU is maxed out
# - Increase PARALLEL_EMBEDDING_THRESHOLD to 10000

# Restart after changes
docker compose restart backend
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a --volumes

# Remove old indexes (if needed)
rm -rf backend/indexes/*
```

## Production Optimizations

### Enable HTTPS (Recommended)

1. Install Certbot:
```bash
sudo apt-get install certbot python3-certbot-nginx -y
```

2. Update nginx.conf to add SSL configuration

3. Get certificate:
```bash
sudo certbot --nginx -d your-domain.com
```

### Configure Firewall

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Block direct backend access (optional)
sudo ufw deny 5000/tcp

# Enable firewall
sudo ufw enable
```

### Set Up Automatic Backups

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/codesense"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/codesense_$DATE.tar.gz backend/indexes backend/histories
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x backup.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /path/to/backup.sh") | crontab -
```

### Monitor with Docker Stats

```bash
# Real-time monitoring
docker stats

# Or install monitoring tools
docker run -d --name=cadvisor -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  google/cadvisor:latest
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              Azure Linux VM                  │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │         Docker Compose                 │ │
│  │                                        │ │
│  │  ┌──────────────┐  ┌───────────────┐ │ │
│  │  │   Frontend   │  │    Backend    │ │ │
│  │  │   (Nginx)    │  │   (FastAPI)   │ │ │
│  │  │   Port 80    │  │   Port 5000   │ │ │
│  │  └──────┬───────┘  └───────┬───────┘ │ │
│  │         │                   │         │ │
│  │         └───────────────────┘         │ │
│  │           codesense_network           │ │
│  │                                        │ │
│  │  Volumes:                              │ │
│  │  - indexes/  (persistent)              │ │
│  │  - histories/ (persistent)             │ │
│  │  - model_cache/ (persistent)           │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Performance Tuning

### For Low-End Systems (2-4 cores, 4GB RAM)
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=2
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=10000
MAX_WORKERS=4
```

### For Mid-Range Systems (4-8 cores, 8GB RAM)
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=0
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=5000
MAX_WORKERS=8
```

### For High-End Systems (8+ cores, 16GB+ RAM)
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=0
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=2000
MAX_WORKERS=16
```

## Security Best Practices

1. **Never commit .env with real credentials**
2. **Use strong API keys** with minimal required permissions
3. **Enable firewall** to restrict access
4. **Set up HTTPS** for production
5. **Regular backups** of indexes and histories
6. **Keep Docker updated**: `sudo apt-get update && sudo apt-get upgrade docker-ce`
7. **Monitor logs** for suspicious activity

## Support

For issues or questions:
1. Check logs: `docker compose logs -f`
2. Review troubleshooting section above
3. Check GitHub issues
4. Verify API keys are valid

## License

[Your License Here]
