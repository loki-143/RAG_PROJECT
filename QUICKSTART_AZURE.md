# CodeSense - Azure Linux Quick Start

Deploy CodeSense on Azure Linux in under 5 minutes.

## One-Line Deployment

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/quick-deploy.sh | bash
```

## Manual Quick Start

### 1. Install Docker (if needed)

```bash
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER && newgrp docker
```

### 2. Clone & Configure

```bash
git clone YOUR_REPO_URL codesense && cd codesense && cp .env.production .env
```

### 3. Edit API Keys

```bash
nano .env
```

Add your API key (at least one required):
- `GITHUB_TOKEN` - Get from https://github.com/settings/tokens
- OR `GOOGLE_API_KEY` - Get from https://aistudio.google.com/apikey

Save and exit (Ctrl+X, Y, Enter)

### 4. Deploy

```bash
chmod +x deploy.sh && ./deploy.sh
```

### 5. Access

Open browser: `http://YOUR_SERVER_IP`

## That's It! 🎉

Your CodeSense instance is now running.

## Next Steps

- **View logs:** `docker compose logs -f`
- **Stop:** `docker compose down`
- **Restart:** `docker compose restart`
- **Update:** `git pull && docker compose up -d --build`

## Troubleshooting

**Backend won't start?**
```bash
docker compose logs backend
# Check if API keys are set correctly in .env
```

**Can't access frontend?**
```bash
# Check if port 80 is open
sudo ufw allow 80/tcp
```

**Slow performance?**
```bash
# Edit .env and set:
USE_PARALLEL_EMBEDDINGS=false
# Then restart:
docker compose restart backend
```

## Full Documentation

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guide.
