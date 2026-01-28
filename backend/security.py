"""Security middleware and utilities for RAG API."""

import os
import re
import time
import logging
import hashlib
import secrets
from typing import Optional, List, Callable
from collections import defaultdict
from functools import wraps
from urllib.parse import urlparse

from fastapi import Request, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("security")

# =============================================================================
# CONFIGURATION
# =============================================================================

class SecurityConfig:
    """Security configuration from environment variables."""
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))  # per minute
    RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # API Key
    API_KEY = os.environ.get("RAG_API_KEY")  # If None, auth is disabled
    API_KEY_HEADER = "X-API-Key"
    
    # Input validation
    MAX_QUESTION_LENGTH = int(os.environ.get("MAX_QUESTION_LENGTH", "5000"))
    MAX_REQUEST_SIZE_MB = int(os.environ.get("MAX_REQUEST_SIZE_MB", "10"))
    
    # URL allowlist (comma-separated domains, or * for all)
    ALLOWED_DOMAINS = os.environ.get("ALLOWED_DOMAINS", "github.com,gitlab.com,bitbucket.org")
    
    # CORS
    ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    
    @classmethod
    def get_allowed_domains(cls) -> List[str]:
        if cls.ALLOWED_DOMAINS == "*":
            return []  # Empty means allow all
        return [d.strip().lower() for d in cls.ALLOWED_DOMAINS.split(",") if d.strip()]


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """In-memory rate limiter with sliding window."""
    
    def __init__(self, requests_per_window: int = 60, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._cleanup_interval = 300  # Clean old entries every 5 minutes
        self._last_cleanup = time.time()
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier from request."""
        # Use X-Forwarded-For if behind proxy, else client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Also consider API key if present
        api_key = request.headers.get(SecurityConfig.API_KEY_HEADER, "")
        if api_key:
            return f"{client_ip}:{hashlib.md5(api_key.encode()).hexdigest()[:8]}"
        return client_ip
    
    def _cleanup_old_entries(self):
        """Remove old request timestamps to prevent memory leak."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        cutoff = now - self.window_seconds
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [t for t in self.requests[client_id] if t > cutoff]
            if not self.requests[client_id]:
                del self.requests[client_id]
        
        self._last_cleanup = now
    
    def is_allowed(self, request: Request) -> tuple[bool, int, int]:
        """
        Check if request is allowed.
        
        Returns:
            tuple: (is_allowed, remaining_requests, retry_after_seconds)
        """
        self._cleanup_old_entries()
        
        client_id = self._get_client_id(request)
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Clean old timestamps for this client
        self.requests[client_id] = [t for t in self.requests[client_id] if t > cutoff]
        
        current_count = len(self.requests[client_id])
        remaining = max(0, self.requests_per_window - current_count)
        
        if current_count >= self.requests_per_window:
            # Calculate retry-after
            oldest = min(self.requests[client_id]) if self.requests[client_id] else now
            retry_after = int(oldest + self.window_seconds - now) + 1
            return False, 0, retry_after
        
        # Record this request
        self.requests[client_id].append(now)
        return True, remaining - 1, 0


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_window=SecurityConfig.RATE_LIMIT_REQUESTS,
    window_seconds=SecurityConfig.RATE_LIMIT_WINDOW
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        is_allowed, remaining, retry_after = rate_limiter.is_allowed(request)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {request.client.host}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(SecurityConfig.RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(SecurityConfig.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

api_key_header = APIKeyHeader(name=SecurityConfig.API_KEY_HEADER, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> Optional[str]:
    """
    Verify API key from header.
    
    If RAG_API_KEY is not set, authentication is disabled (dev mode).
    """
    configured_key = SecurityConfig.API_KEY
    
    # If no API key configured, allow all requests (dev mode)
    if not configured_key:
        return None
    
    if not api_key:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, configured_key):
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return api_key


# =============================================================================
# INPUT VALIDATION
# =============================================================================

class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_question(question: str) -> str:
        """Validate and sanitize question input."""
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty."
            )
        
        question = question.strip()
        
        if len(question) > SecurityConfig.MAX_QUESTION_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Question too long. Maximum {SecurityConfig.MAX_QUESTION_LENGTH} characters."
            )
        
        return question
    
    @staticmethod
    def validate_repo_url(url: str) -> str:
        """Validate repository URL against allowlist."""
        try:
            parsed = urlparse(str(url))
            
            if parsed.scheme not in ["http", "https"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid URL scheme. Only HTTP/HTTPS allowed."
                )
            
            domain = parsed.netloc.lower()
            
            # Remove port if present
            if ":" in domain:
                domain = domain.split(":")[0]
            
            allowed_domains = SecurityConfig.get_allowed_domains()
            
            # If allowlist is empty, allow all
            if not allowed_domains:
                return str(url)
            
            # Check if domain is in allowlist
            if not any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Domain not allowed. Allowed domains: {', '.join(allowed_domains)}"
                )
            
            return str(url)
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid repository URL."
            )
    
    @staticmethod
    def validate_top_k(top_k: int) -> int:
        """Validate top_k parameter."""
        if top_k < 1:
            return 1
        if top_k > 50:
            return 50
        return top_k


# =============================================================================
# REQUEST SIZE LIMITING
# =============================================================================

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size."""
    
    def __init__(self, app, max_size_mb: int = 10):
        super().__init__(app)
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        
        if content_length and int(content_length) > self.max_size_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request too large. Maximum size: {self.max_size_bytes // (1024*1024)}MB"}
            )
        
        return await call_next(request)


# =============================================================================
# ERROR SANITIZATION
# =============================================================================

class ErrorSanitizer:
    """Sanitize error messages for production."""
    
    # Patterns to remove from error messages
    SENSITIVE_PATTERNS = [
        r'/home/[^/\s]+',  # Home directory paths
        r'/Users/[^/\s]+',  # macOS paths
        r'[A-Z]:\\Users\\[^\\s]+',  # Windows paths
        r'api[_-]?key[=:]\s*\S+',  # API keys
        r'password[=:]\s*\S+',  # Passwords
        r'token[=:]\s*\S+',  # Tokens
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
    ]
    
    @classmethod
    def sanitize(cls, message: str) -> str:
        """Remove sensitive information from error messages."""
        sanitized = message
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
        return sanitized


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handler that sanitizes errors in production."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            logger.exception("Unhandled exception")
            
            # In production, don't expose internal errors
            is_production = os.environ.get("ENVIRONMENT", "development") == "production"
            
            if is_production:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "An internal error occurred. Please try again later."}
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={"detail": ErrorSanitizer.sanitize(str(e))}
                )


# =============================================================================
# SECURITY HEADERS
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        
        return response


# =============================================================================
# AUDIT LOGGING
# =============================================================================

class AuditLogger:
    """Log security-relevant events."""
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
    
    def log_request(self, request: Request, user_id: Optional[str] = None):
        """Log incoming request."""
        self.logger.info(
            f"REQUEST | {request.method} {request.url.path} | "
            f"IP: {request.client.host if request.client else 'unknown'} | "
            f"User: {user_id or 'anonymous'}"
        )
    
    def log_auth_failure(self, request: Request, reason: str):
        """Log authentication failure."""
        self.logger.warning(
            f"AUTH_FAILURE | {request.url.path} | "
            f"IP: {request.client.host if request.client else 'unknown'} | "
            f"Reason: {reason}"
        )
    
    def log_rate_limit(self, request: Request):
        """Log rate limit event."""
        self.logger.warning(
            f"RATE_LIMIT | {request.url.path} | "
            f"IP: {request.client.host if request.client else 'unknown'}"
        )
    
    def log_index_operation(self, operation: str, repo_url: str, user_id: Optional[str] = None):
        """Log indexing operation."""
        self.logger.info(
            f"INDEX_OP | {operation} | Repo: {repo_url} | User: {user_id or 'anonymous'}"
        )


audit_logger = AuditLogger()
