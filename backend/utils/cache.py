"""
Redis-based caching for API responses.
"""
import json
import hashlib
import logging
from typing import Optional, Any
from functools import wraps

try:
    import redis
    import msgpack
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import settings

logger = logging.getLogger(__name__)


class ResponseCache:
    """Redis-based response cache with fallback to in-memory."""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.enabled = settings.REDIS_ENABLED and REDIS_AVAILABLE
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=0,
                    decode_responses=False,  # We'll use msgpack
                    socket_timeout=2,
                    socket_connect_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Redis cache enabled at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            except Exception as e:
                logger.warning(f"⚠️  Redis connection failed, using in-memory cache: {e}")
                self.enabled = False
                self.redis_client = None
        else:
            logger.info("📝 Using in-memory cache (Redis disabled)")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return f"hfviz:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.enabled and self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return msgpack.unpackb(data, raw=False)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL."""
        try:
            ttl = ttl or settings.REDIS_TTL
            
            if self.enabled and self.redis_client:
                packed_data = msgpack.packb(value, use_bin_type=True)
                self.redis_client.setex(key, ttl, packed_data)
                return True
            else:
                # In-memory cache with simple TTL tracking
                self.memory_cache[key] = value
                # Limit in-memory cache size
                if len(self.memory_cache) > 100:
                    # Remove oldest entry
                    self.memory_cache.pop(next(iter(self.memory_cache)))
                return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if self.enabled and self.redis_client:
                self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    def clear(self, pattern: str = "hfviz:*") -> bool:
        """Clear all cache keys matching pattern."""
        try:
            if self.enabled and self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                self.memory_cache.clear()
            return True
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return False


# Global cache instance
cache = ResponseCache()


def cached_response(ttl: int = 300, key_prefix: str = "api"):
    """
    Decorator for caching API responses.
    
    Usage:
        @cached_response(ttl=600, key_prefix="models")
        async def get_models(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function args
            cache_key = cache._generate_key(key_prefix, func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache HIT: {cache_key[:20]}...")
                return cached_data
            
            # Execute function
            logger.debug(f"Cache MISS: {cache_key[:20]}...")
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator



