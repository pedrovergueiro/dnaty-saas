import redis
import os
from fastapi import HTTPException, Request
from functools import wraps
from typing import Callable
import time

class RateLimiter:
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

    def is_allowed(self, key: str, limit: int = 100, window: int = 3600) -> bool:
        """
        Check if request is allowed.
        limit: requests per window
        window: time window in seconds (default: 1 hour)
        """
        current = self.redis_client.get(key)
        if current is None:
            self.redis_client.setex(key, window, 1)
            return True

        current_count = int(current)
        if current_count < limit:
            self.redis_client.incr(key)
            return True

        return False

    def get_remaining(self, key: str, limit: int = 100) -> int:
        """Get remaining requests for this key"""
        current = self.redis_client.get(key)
        if current is None:
            return limit
        return max(0, limit - int(current))

    def get_reset_time(self, key: str) -> int:
        """Get Unix timestamp when rate limit resets"""
        ttl = self.redis_client.ttl(key)
        if ttl == -1:
            return 0
        return int(time.time()) + ttl

async def rate_limit_middleware(
    request: Request,
    limiter: RateLimiter,
    limit: int = 100,
    window: int = 3600,
):
    """Middleware for rate limiting"""
    # Use user ID or IP address as key
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        key = f"rate_limit:user:{user_id}"
    else:
        key = f"rate_limit:ip:{request.client.host}"

    if not limiter.is_allowed(key, limit=limit, window=window):
        remaining = limiter.get_remaining(key, limit=limit)
        reset_time = limiter.get_reset_time(key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Reset in {limiter.redis_client.ttl(key)}s",
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time),
            },
        )

    # Add rate limit headers to response
    request.state.ratelimit_remaining = limiter.get_remaining(key, limit=limit)
    request.state.ratelimit_reset = limiter.get_reset_time(key)
