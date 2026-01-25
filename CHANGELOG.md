# RAG Project - Changelog

All notable changes to this project will be documented in this file.

---

## [2.1.0] - 2026-01-24

### 🗑️ Auto-Cleanup Feature

#### Added
- **Storage Cleanup Utility** ([cleanup.py](rag_agent/cleanup.py))
  - Auto-deletes oldest indexes when storage limit is reached
  - Configurable via environment variables
  - Tracks last access time for each index
  - Deletes old chat histories automatically

- **New API Endpoints**
  - `GET /storage` - Get storage usage statistics
  - `GET /storage/recommendation` - Get cleanup recommendations
  - `POST /storage/cleanup` - Trigger manual cleanup

- **Auto-Cleanup on Index** ([fastapi_app.py](rag_agent/fastapi_app.py))
  - Automatically cleans up before indexing new repos
  - Ensures storage never exceeds configured limits

#### New Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_STORAGE_MB` | 500 | Maximum total storage in MB |
| `MAX_INDEXES` | 50 | Maximum number of indexes to keep |
| `MAX_HISTORY_AGE_DAYS` | 30 | Delete histories older than X days |

---

## [2.0.0] - 2026-01-24

### 🔐 Security Enhancements

#### Added
- **Rate Limiting** ([security.py](rag_agent/security.py))
  - In-memory rate limiter with sliding window algorithm
  - Default: 60 requests per minute per IP
  - Configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW` env vars
  - Returns `429 Too Many Requests` with `Retry-After` header

- **API Key Authentication** ([security.py](rag_agent/security.py))
  - Optional API key validation via `X-API-Key` header
  - Enable by setting `RAG_API_KEY` environment variable
  - Uses constant-time comparison to prevent timing attacks
  - Returns `401 Unauthorized` for invalid/missing keys

- **Input Validation** ([security.py](rag_agent/security.py))
  - Question length limit (default: 5000 chars)
  - Repository URL allowlist (default: github.com, gitlab.com, bitbucket.org)
  - Request size limiting (default: 10MB)
  - Pydantic validators on all request models

- **Thread-Safe Storage** ([storage_safe.py](rag_agent/storage_safe.py))
  - File locking for concurrent access to indexes and histories
  - Cross-process safety using `fcntl` (Unix) / `msvcrt` (Windows)
  - Atomic file writes to prevent corruption

- **Error Sanitization** ([security.py](rag_agent/security.py))
  - Removes sensitive data from error messages (paths, API keys, etc.)
  - Production mode returns generic error messages

- **Security Headers** ([security.py](rag_agent/security.py))
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: DENY
  - X-XSS-Protection: 1; mode=block
  - Referrer-Policy: strict-origin-when-cross-origin

- **Audit Logging** ([security.py](rag_agent/security.py))
  - Logs all API requests with IP and user info
  - Logs authentication failures
  - Logs rate limit events
  - Logs index operations (create/delete)

#### Changed
- **[fastapi_app.py](rag_agent/fastapi_app.py)** - Complete rewrite with security middleware
  - Added all security middleware layers
  - Added input validation to all Pydantic models
  - Protected all routes with API key dependency
  - Added `/health` endpoint for load balancers
  - Improved error handling (no stack traces in production)

#### New Files
| File | Description |
|------|-------------|
| `rag_agent/security.py` | Security middleware, rate limiting, auth, validation |
| `rag_agent/storage_safe.py` | Thread-safe storage with file locking |
| `rag_agent/fastapi_app_secure.py` | Backup of secure app (now merged into main) |

---

### 🚀 Deployment Files

#### Added
- **[Procfile](Procfile)** - For Railway/Heroku deployment
- **[render.yaml](render.yaml)** - For Render.com deployment
- **[runtime.txt](runtime.txt)** - Python version specification
- **[.env.example](.env.example)** - Environment variable template
- **[docker-compose.prod.yml](docker-compose.prod.yml)** - Production Docker setup
- **[Dockerfile.prod](Dockerfile.prod)** - Production Dockerfile with:
  - Non-root user for security
  - Gunicorn with Uvicorn workers
  - Health checks
- **[nginx/nginx.conf](nginx/nginx.conf)** - Nginx reverse proxy with:
  - SSL/TLS termination
  - Additional rate limiting
  - Security headers
  - Upstream load balancing
- **[generate-ssl.sh](generate-ssl.sh)** - SSL certificate generation script

#### Changed
- **[.gitignore](.gitignore)** - Updated with more patterns
- **[requirements.txt](requirements.txt)** - Added gunicorn, uvicorn[standard], pydantic>=2.0

---

### 📝 Configuration

#### New Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_API_KEY` | None | API key for authentication (optional) |
| `ENVIRONMENT` | development | Set to `production` for prod mode |
| `LOG_LEVEL` | INFO | Logging level |
| `MAX_WORKERS` | 8 | Thread pool workers |
| `RATE_LIMIT_REQUESTS` | 60 | Requests per window |
| `RATE_LIMIT_WINDOW` | 60 | Window in seconds |
| `MAX_QUESTION_LENGTH` | 5000 | Max question characters |
| `MAX_REQUEST_SIZE_MB` | 10 | Max request body size |
| `ALLOWED_DOMAINS` | github.com,gitlab.com,bitbucket.org | Allowed repo domains |
| `ALLOWED_ORIGINS` | * | CORS allowed origins |
| `ENABLE_DOCS` | true | Enable /docs endpoint |

---

## [1.0.0] - Previous Version

### Original Features
- RAG Agent with Gemini LLM
- Hybrid retrieval (BM25 + FAISS)
- Repository indexing
- Chat history management
- Basic FastAPI endpoints

---

## How to Migrate from 1.0.0 to 2.0.0

1. **Update your `.env` file:**
   ```env
   # Add these new variables (optional but recommended)
   RAG_API_KEY=your_secure_key_here
   ENVIRONMENT=production
   ```

2. **Update dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **If using Docker, use the new production files:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **Update client code to include API key:**
   ```python
   headers = {"X-API-Key": "your_api_key"}
   requests.post(url, headers=headers, json=data)
   ```
