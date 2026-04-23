# CodeSense - Command Reference

Quick reference for common Docker Compose commands.

## 🚀 Deployment

```bash
# Initial deployment
chmod +x deploy.sh && ./deploy.sh

# Manual deployment
docker compose up -d --build
```

## 📊 Status & Monitoring

```bash
# Check service status
docker compose ps

# View all logs (follow mode)
docker compose logs -f

# View backend logs only
docker compose logs -f backend

# View frontend logs only
docker compose logs -f frontend

# View last 100 lines
docker compose logs --tail=100

# Check resource usage
docker stats

# Check health
curl http://localhost:5000/health
curl http://localhost:80/health
```

## 🔄 Service Management

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart backend
docker compose restart frontend

# Stop all services (keeps data)
docker compose down

# Stop and remove volumes (deletes data)
docker compose down -v

# Start services
docker compose up -d

# Rebuild and restart
docker compose up -d --build
```

## 🔧 Configuration

```bash
# Edit environment variables
nano .env

# Restart after config change
docker compose restart backend

# View current configuration
docker compose config
```

## 🗄️ Data Management

```bash
# Backup indexes and histories
tar -czf backup_$(date +%Y%m%d).tar.gz backend/indexes backend/histories

# Restore from backup
tar -xzf backup_20260424.tar.gz

# Clear all indexes (careful!)
rm -rf backend/indexes/*
rm -rf backend/histories/*
docker compose restart backend
```

## 🧹 Cleanup

```bash
# Remove unused Docker resources
docker system prune

# Remove all unused images, containers, volumes
docker system prune -a --volumes

# Remove specific service
docker compose rm -f backend

# Clean up old logs
docker compose logs --tail=0 > /dev/null
```

## 🔍 Debugging

```bash
# Enter backend container
docker compose exec backend bash

# Enter frontend container
docker compose exec frontend sh

# Check backend environment variables
docker compose exec backend env

# Test backend API directly
curl http://localhost:5000/health
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'

# Check nginx configuration
docker compose exec frontend cat /etc/nginx/conf.d/default.conf

# Test nginx configuration
docker compose exec frontend nginx -t

# Check network connectivity
docker compose exec frontend ping backend
```

## 📦 Updates

```bash
# Pull latest code
git pull

# Rebuild images
docker compose build

# Restart with new code
docker compose up -d --build

# View what changed
git log --oneline -10
git diff HEAD~1
```

## 🔐 Security

```bash
# View environment variables (check for leaks)
docker compose config

# Check exposed ports
docker compose ps
sudo netstat -tlnp | grep -E '(80|5000)'

# Update base images
docker compose pull
docker compose up -d --build
```

## 📈 Performance

```bash
# Monitor resource usage
docker stats

# Check disk usage
df -h
docker system df

# View container processes
docker compose top

# Check backend performance
time curl http://localhost:5000/health
```

## 🆘 Emergency

```bash
# Force stop everything
docker compose kill

# Remove everything and start fresh
docker compose down -v
docker compose up -d --build

# Check Docker daemon
sudo systemctl status docker

# Restart Docker daemon
sudo systemctl restart docker

# View Docker daemon logs
sudo journalctl -u docker -n 100
```

## 🔄 Common Workflows

### Update Application
```bash
git pull
docker compose up -d --build
docker compose logs -f
```

### Change Configuration
```bash
nano .env
docker compose restart backend
docker compose logs -f backend
```

### Troubleshoot Backend Issues
```bash
docker compose logs backend
docker compose exec backend bash
# Inside container:
python -c "import os; print(os.environ.get('GITHUB_TOKEN'))"
exit
```

### Troubleshoot Frontend Issues
```bash
docker compose logs frontend
docker compose exec frontend sh
# Inside container:
cat /etc/nginx/conf.d/default.conf
nginx -t
exit
```

### Performance Tuning
```bash
nano .env
# Adjust: USE_PARALLEL_EMBEDDINGS, MAX_WORKERS, etc.
docker compose restart backend
docker stats
```

### Clean Slate Restart
```bash
docker compose down
rm -rf backend/indexes/* backend/histories/*
docker compose up -d
docker compose logs -f
```

## 📱 One-Liners

```bash
# Quick status check
docker compose ps && docker compose logs --tail=10

# Restart and follow logs
docker compose restart && docker compose logs -f

# Update and restart
git pull && docker compose up -d --build && docker compose logs -f

# Clean and restart
docker compose down && docker system prune -f && docker compose up -d

# Backup before update
tar -czf backup_$(date +%Y%m%d).tar.gz backend/indexes backend/histories && git pull && docker compose up -d --build
```

## 🎯 Quick Diagnostics

```bash
# Full system check
echo "=== Services ===" && docker compose ps && \
echo "=== Health ===" && curl -s http://localhost:5000/health && \
echo "=== Resources ===" && docker stats --no-stream && \
echo "=== Disk ===" && df -h && \
echo "=== Recent Logs ===" && docker compose logs --tail=20
```

---

**Tip:** Bookmark this file for quick reference!
