# CodeSense - Azure Linux Deployment Checklist

## ✅ Pre-Deployment

- [ ] Azure Linux VM provisioned (Ubuntu 20.04+)
- [ ] SSH access to VM configured
- [ ] At least 2 CPU cores, 4GB RAM available
- [ ] Ports 80 and 5000 available
- [ ] GitHub Token OR Google API Key ready

## 📦 Installation Steps

### Step 1: Install Docker
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```
- [ ] Docker installed
- [ ] Docker Compose available
- [ ] User added to docker group

### Step 2: Clone Repository
```bash
git clone <your-repo-url> codesense
cd codesense
```
- [ ] Repository cloned
- [ ] Changed to project directory

### Step 3: Configure Environment
```bash
cp .env.production .env
nano .env
```
- [ ] .env file created from template
- [ ] GITHUB_TOKEN or GOOGLE_API_KEY added
- [ ] Performance settings reviewed (optional)

### Step 4: Deploy
```bash
chmod +x deploy.sh
./deploy.sh
```
- [ ] Deploy script made executable
- [ ] Deployment script executed
- [ ] No errors during deployment

## ✅ Post-Deployment Verification

### Health Checks
```bash
# Check services are running
docker compose ps
```
- [ ] Backend status: healthy
- [ ] Frontend status: healthy

```bash
# Check backend health
curl http://localhost:5000/health
```
- [ ] Backend responds with "healthy"

```bash
# Check frontend health
curl http://localhost:80
```
- [ ] Frontend loads successfully

### Functional Testing
- [ ] Open browser to `http://your-server-ip`
- [ ] Frontend loads without errors
- [ ] Can access "Index Repository" feature
- [ ] Can index a test repository (e.g., small GitHub repo)
- [ ] Indexing completes successfully
- [ ] Can query the indexed repository
- [ ] Receives relevant answers

### Log Verification
```bash
docker compose logs backend | tail -50
```
- [ ] No error messages in backend logs
- [ ] Backend started successfully
- [ ] Model loaded correctly

```bash
docker compose logs frontend | tail -20
```
- [ ] No error messages in frontend logs
- [ ] Nginx started successfully

## 🔧 Optional Configuration

### Firewall Setup
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```
- [ ] Firewall configured
- [ ] HTTP port open
- [ ] HTTPS port open (for future SSL)

### HTTPS Setup (Production)
```bash
sudo apt-get install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```
- [ ] Certbot installed
- [ ] SSL certificate obtained
- [ ] HTTPS configured
- [ ] Auto-renewal enabled

### Backup Configuration
```bash
# Create backup script
cat > ~/backup-codesense.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/codesense"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
cd ~/codesense
tar -czf $BACKUP_DIR/codesense_$DATE.tar.gz backend/indexes backend/histories
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x ~/backup-codesense.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * ~/backup-codesense.sh") | crontab -
```
- [ ] Backup script created
- [ ] Backup script tested
- [ ] Cron job configured

### Monitoring Setup
```bash
# Install monitoring tools (optional)
docker run -d --name=cadvisor -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  google/cadvisor:latest
```
- [ ] Monitoring tools installed
- [ ] Can access monitoring dashboard

## 🎯 Performance Tuning

### For Your System Type

**Low-End (2-4 cores, 4GB RAM):**
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=2
USE_PARALLEL_EMBEDDINGS=false
PARALLEL_EMBEDDING_THRESHOLD=10000
MAX_WORKERS=4
```
- [ ] Settings configured for low-end system

**Mid-Range (4-8 cores, 8GB RAM):**
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=0
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=5000
MAX_WORKERS=8
```
- [ ] Settings configured for mid-range system

**High-End (8+ cores, 16GB+ RAM):**
```env
USE_PARALLEL_INDEXING=true
INDEXING_MAX_WORKERS=0
USE_PARALLEL_EMBEDDINGS=true
PARALLEL_EMBEDDING_THRESHOLD=2000
MAX_WORKERS=16
```
- [ ] Settings configured for high-end system

## 📊 Monitoring & Maintenance

### Daily Checks
- [ ] Check service status: `docker compose ps`
- [ ] Review logs for errors: `docker compose logs --tail=100`
- [ ] Check disk space: `df -h`
- [ ] Monitor resource usage: `docker stats`

### Weekly Checks
- [ ] Review backup logs
- [ ] Check for updates: `git fetch`
- [ ] Review system logs: `journalctl -xe`
- [ ] Clean up old Docker images: `docker system prune`

### Monthly Checks
- [ ] Update system packages: `sudo apt-get update && sudo apt-get upgrade`
- [ ] Update Docker: `sudo apt-get upgrade docker-ce`
- [ ] Review and optimize performance settings
- [ ] Test disaster recovery procedure

## 🆘 Troubleshooting Reference

### Issue: Services won't start
```bash
docker compose logs
docker compose down
docker compose up -d
```

### Issue: Out of memory
```bash
# Check memory usage
free -h
docker stats

# Adjust settings in .env
USE_PARALLEL_EMBEDDINGS=false
docker compose restart backend
```

### Issue: Slow performance
```bash
# Check CPU usage
top

# Adjust workers in .env
INDEXING_MAX_WORKERS=2
MAX_WORKERS=4
docker compose restart backend
```

### Issue: Can't access from internet
```bash
# Check firewall
sudo ufw status

# Check if services are listening
sudo netstat -tlnp | grep -E '(80|5000)'

# Check Docker network
docker network inspect codesense_codesense_network
```

## 📚 Documentation Reference

- [AZURE_DEPLOYMENT_SUMMARY.md](AZURE_DEPLOYMENT_SUMMARY.md) - Quick reference
- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete guide
- [QUICKSTART_AZURE.md](QUICKSTART_AZURE.md) - Quick start
- [README_DEPLOYMENT.md](README_DEPLOYMENT.md) - Overview

## ✅ Final Verification

- [ ] All services running and healthy
- [ ] Can access frontend from browser
- [ ] Can index repositories successfully
- [ ] Can query indexed repositories
- [ ] Logs show no errors
- [ ] Performance is acceptable
- [ ] Backups configured (optional)
- [ ] HTTPS configured (optional)
- [ ] Monitoring configured (optional)

## 🎉 Deployment Complete!

Your CodeSense instance is now running and ready to use.

**Access:** http://your-server-ip

**Next Steps:**
1. Index your first repository
2. Test querying functionality
3. Configure additional optimizations
4. Set up monitoring and backups
5. Share with your team!

---

**Need help?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for troubleshooting.
