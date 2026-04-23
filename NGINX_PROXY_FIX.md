# Nginx Reverse Proxy Fix - Production Grade Solution

## Problem Summary

**Issue:** Frontend cannot connect to backend in Docker deployment

**Root Cause:** Frontend build hardcoded `http://localhost:5000` which points to user's local machine, not the Docker backend container.

**Browser behavior:**
```
❌ Frontend tries: http://localhost:5000 → User's local machine (fails)
✅ Should use: /api → Nginx → http://backend:5000 (Docker network)
```

## Solution: Nginx Reverse Proxy (Industry Standard)

Instead of relying on Vite environment variables during build, we use **Nginx reverse proxy** to route API calls from frontend to backend.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser                            │
│                                                      │
│  Frontend loads from: http://your-vm-ip             │
│  API calls go to: http://your-vm-ip/api/*           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────┐
│              Nginx (Frontend Container)              │
│                                                      │
│  location / → Serve React app                       │
│  location /api/ → proxy_pass http://backend:5000/   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────┐
│            FastAPI (Backend Container)               │
│                                                      │
│  Receives requests on: http://backend:5000          │
│  (Docker internal network)                          │
└─────────────────────────────────────────────────────┘
```

## Changes Made

### 1. Frontend API Configuration (`frontend/src/services/api.js`)

**Before:**
```javascript
const API_BASE_URL = window.location.hostname.includes('loca.lt') 
  ? 'https://rag-backend-api.loca.lt' 
  : (import.meta.env.VITE_API_URL || 'http://localhost:5000');
```

**After:**
```javascript
// API Configuration - Uses Nginx reverse proxy
// All API calls go through /api which Nginx proxies to backend container
const API_BASE_URL = '/api';
```

**Benefits:**
- ✅ No hardcoded URLs
- ✅ Works in any environment (local, VM, cloud)
- ✅ No build-time configuration needed
- ✅ Browser uses relative URLs

### 2. System Ready Notification (`frontend/src/components/SystemReadyNotification.jsx`)

**Before:**
```javascript
const API_URL = import.meta.env.VITE_API_URL || '/api';
```

**After:**
```javascript
// Use Nginx reverse proxy - no need for env variable
const API_URL = '/api';
```

### 3. Nginx Configuration (`frontend/nginx.conf`)

Already configured correctly:

```nginx
# Reverse proxy backend API calls to the backend container
location /api/ {
    proxy_pass http://backend:5000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # Disable buffering for streaming LLM responses (SSE)
    proxy_buffering off;
    proxy_cache off;
    chunked_transfer_encoding off;
    proxy_read_timeout 3600;
    proxy_connect_timeout 3600;
    proxy_send_timeout 3600;
}
```

**Key points:**
- `location /api/` - Matches all requests to `/api/*`
- `proxy_pass http://backend:5000/` - Forwards to backend container
- `http://backend:5000` - Uses Docker network name (not localhost!)
- Trailing `/` strips `/api` prefix before forwarding

### 4. Frontend Dockerfile (`frontend/Dockerfile`)

**Before:**
```dockerfile
# Build argument for API URL
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

# Build the application
RUN npm run build
```

**After:**
```dockerfile
# Build the application (no need for VITE_API_URL - using Nginx proxy)
RUN npm run build
```

**Benefits:**
- ✅ No build args needed
- ✅ Simpler Dockerfile
- ✅ Same build works everywhere

### 5. Docker Compose (`docker-compose.yml`)

**Before:**
```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      - VITE_API_URL=${VITE_API_URL:-http://localhost:5000}
```

**After:**
```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
```

**Benefits:**
- ✅ No environment variables needed
- ✅ Simpler configuration
- ✅ Less error-prone

## How It Works

### Request Flow

1. **Browser makes API call:**
   ```javascript
   fetch('/api/health')
   ```

2. **Browser resolves to:**
   ```
   http://your-vm-ip/api/health
   ```

3. **Nginx receives request:**
   ```
   GET /api/health
   ```

4. **Nginx proxies to backend:**
   ```
   GET http://backend:5000/health
   ```
   (Note: `/api` prefix is stripped)

5. **Backend responds:**
   ```json
   {"status": "healthy", "ready": true}
   ```

6. **Nginx forwards response to browser**

### URL Mapping

| Browser Request | Nginx Receives | Backend Receives |
|----------------|----------------|------------------|
| `/api/health` | `/api/health` | `/health` |
| `/api/index` | `/api/index` | `/index` |
| `/api/chat` | `/api/chat` | `/chat` |
| `/api/indexes` | `/api/indexes` | `/indexes` |

## Deployment Steps

### 1. Stop Current Containers

```bash
docker compose down
```

### 2. Rebuild with No Cache

```bash
docker compose build --no-cache
```

**Why `--no-cache`?**
- Ensures old build artifacts are removed
- Frontend gets rebuilt with new API configuration
- No cached layers with old URLs

### 3. Start Services

```bash
docker compose up -d
```

### 4. Verify Deployment

```bash
# Check services are running
docker compose ps

# Check backend health
curl http://localhost:5000/health

# Check frontend health
curl http://localhost:80/health

# Check API proxy (from inside VM)
curl http://localhost:80/api/health
```

### 5. Test from Browser

1. Open browser: `http://your-vm-ip`
2. Open browser console (F12)
3. Check Network tab
4. All API calls should go to: `http://your-vm-ip/api/*`
5. No errors about "Unable to connect to backend"

## Verification Checklist

- [ ] No `localhost:5000` in frontend code
- [ ] No `VITE_API_URL` in frontend code
- [ ] All API calls use `/api` prefix
- [ ] Nginx config has `location /api/` block
- [ ] Nginx proxies to `http://backend:5000/`
- [ ] Docker compose has no build args for frontend
- [ ] Containers are on same Docker network
- [ ] Backend is accessible at `http://backend:5000` (inside Docker)
- [ ] Frontend is accessible at `http://your-vm-ip`
- [ ] API calls work: `http://your-vm-ip/api/health`

## Troubleshooting

### Issue: Still getting "Unable to connect to backend"

**Check 1: Verify Nginx config**
```bash
docker compose exec frontend cat /etc/nginx/conf.d/default.conf
# Should have: location /api/ { proxy_pass http://backend:5000/; }
```

**Check 2: Verify backend is reachable**
```bash
docker compose exec frontend wget -O- http://backend:5000/health
# Should return: {"status": "healthy", ...}
```

**Check 3: Check browser console**
```
F12 → Network tab
Look for failed requests
Should see: /api/health, /api/indexes, etc.
Should NOT see: http://localhost:5000
```

**Check 4: Verify frontend code**
```bash
docker compose exec frontend grep -r "localhost:5000" /usr/share/nginx/html/
# Should return: nothing (no matches)
```

### Issue: 502 Bad Gateway

**Cause:** Backend container not running or not reachable

**Solution:**
```bash
# Check backend status
docker compose ps backend

# Check backend logs
docker compose logs backend

# Restart backend
docker compose restart backend
```

### Issue: 404 Not Found on /api/*

**Cause:** Nginx config not loaded correctly

**Solution:**
```bash
# Verify nginx config
docker compose exec frontend nginx -t

# Reload nginx
docker compose exec frontend nginx -s reload

# Or restart frontend
docker compose restart frontend
```

### Issue: CORS errors

**Cause:** Backend not allowing frontend origin

**Solution:**
Backend already configured with CORS middleware. If issues persist:

```python
# In backend/fastapi_app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

## Testing

### Test 1: Health Check

```bash
# From VM
curl http://localhost:80/api/health

# Expected:
{"status": "healthy", "ready": true, "message": "System ready"}
```

### Test 2: List Indexes

```bash
# From VM
curl http://localhost:80/api/indexes

# Expected:
{"indexes": [...]}
```

### Test 3: Browser Console

```javascript
// In browser console (F12)
fetch('/api/health')
  .then(r => r.json())
  .then(console.log)

// Expected:
{status: "healthy", ready: true, message: "System ready"}
```

### Test 4: Index Repository

```bash
# From VM
curl -X POST http://localhost:80/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'

# Should start indexing
```

## Benefits of This Approach

### 1. Production Ready
- ✅ Industry standard pattern
- ✅ Used by major companies
- ✅ Battle-tested solution

### 2. Environment Agnostic
- ✅ Works on localhost
- ✅ Works on VM
- ✅ Works on cloud
- ✅ No configuration changes needed

### 3. Security
- ✅ Backend not exposed directly
- ✅ All traffic goes through Nginx
- ✅ Can add authentication at proxy level
- ✅ Can add rate limiting at proxy level

### 4. Flexibility
- ✅ Easy to add SSL/HTTPS
- ✅ Easy to add caching
- ✅ Easy to add load balancing
- ✅ Easy to add multiple backends

### 5. Simplicity
- ✅ No build-time configuration
- ✅ No environment variables
- ✅ Same build works everywhere
- ✅ Less error-prone

## Comparison: Before vs After

| Aspect | Before (Vite Env) | After (Nginx Proxy) |
|--------|-------------------|---------------------|
| API URL | `http://localhost:5000` | `/api` |
| Configuration | Build-time | Runtime |
| Environment | Specific | Agnostic |
| Complexity | High | Low |
| Error-prone | Yes | No |
| Production ready | No | Yes |
| Industry standard | No | Yes |

## Additional Nginx Features

### Add SSL/HTTPS

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # ... rest of config
}
```

### Add Caching

```nginx
location /api/ {
    proxy_pass http://backend:5000/;
    
    # Cache GET requests
    proxy_cache my_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_methods GET HEAD;
}
```

### Add Rate Limiting

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

location /api/ {
    limit_req zone=api_limit burst=20;
    proxy_pass http://backend:5000/;
}
```

### Add Authentication

```nginx
location /api/ {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://backend:5000/;
}
```

## Summary

✅ **Frontend code updated** - All API calls use `/api`  
✅ **Nginx configured** - Proxies `/api/*` to `http://backend:5000/`  
✅ **Dockerfile simplified** - No build args needed  
✅ **Docker Compose simplified** - No environment variables needed  
✅ **Production ready** - Industry standard pattern  

**Result:** Stable, production-grade setup where frontend never directly calls backend via absolute URL.

## Quick Reference

```bash
# Rebuild and deploy
docker compose down
docker compose build --no-cache
docker compose up -d

# Verify
curl http://localhost:80/api/health

# Check logs
docker compose logs -f

# Test in browser
# Open: http://your-vm-ip
# All API calls should work
```

---

**🎉 Your frontend now communicates with backend through Nginx reverse proxy!**
