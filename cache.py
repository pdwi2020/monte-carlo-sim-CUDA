"""
Caching Module for Monte Carlo Pricing

This module provides caching capabilities for pricing results:
- In-memory LRU cache
- Optional Redis backend for distributed caching
- Cache key generation from pricing parameters
- TTL (time-to-live) support
"""

import hashlib
import json
import time
from functools import wraps
from typing import Optional, Dict, Any, Callable, List
from collections import OrderedDict
import threading
import pickle

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


# =============================================================================
# Cache Key Generation
# =============================================================================

def generate_cache_key(
    func_name: str,
    args: tuple,
    kwargs: dict,
    precision: int = 6
) -> str:
    """
    Generate a deterministic cache key from function arguments.

    Args:
        func_name: Name of the cached function
        args: Positional arguments
        kwargs: Keyword arguments
        precision: Decimal precision for floating point rounding

    Returns:
        SHA256 hash as cache key
    """
    def normalize_value(v):
        """Normalize values for consistent hashing."""
        if isinstance(v, float):
            return round(v, precision)
        elif isinstance(v, (list, tuple)):
            return [normalize_value(x) for x in v]
        elif isinstance(v, dict):
            return {k: normalize_value(val) for k, val in sorted(v.items())}
        elif hasattr(v, '__dict__'):
            return {k: normalize_value(val) for k, val in sorted(v.__dict__.items())}
        elif hasattr(v, 'value'):
            return v.value
        return v

    key_data = {
        'func': func_name,
        'args': [normalize_value(a) for a in args],
        'kwargs': {k: normalize_value(v) for k, v in sorted(kwargs.items())}
    }

    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


# =============================================================================
# In-Memory LRU Cache
# =============================================================================

class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, maxsize: int = 1000, ttl: Optional[int] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            if self.ttl is not None:
                if time.time() - self._timestamps[key] > self.ttl:
                    del self._cache[key]
                    del self._timestamps[key]
                    self._misses += 1
                    return None

            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    del self._timestamps[oldest_key]

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'maxsize': self.maxsize,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'ttl': self.ttl
        }


# =============================================================================
# Redis Cache Backend
# =============================================================================

class RedisCache:
    """Redis-backed cache for distributed caching."""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = 'mc_pricer:',
        ttl: Optional[int] = 3600
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for Redis caching")

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False
        )
        self.prefix = prefix
        self.ttl = ttl
        self._hits = 0
        self._misses = 0

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        full_key = self._full_key(key)
        data = self.client.get(full_key)

        if data is None:
            self._misses += 1
            return None

        self._hits += 1
        return pickle.loads(data)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        full_key = self._full_key(key)
        data = pickle.dumps(value)

        ex = ttl if ttl is not None else self.ttl
        if ex:
            self.client.setex(full_key, ex, data)
        else:
            self.client.set(full_key, data)

    def delete(self, key: str) -> bool:
        full_key = self._full_key(key)
        return self.client.delete(full_key) > 0

    def clear(self, pattern: str = '*') -> int:
        full_pattern = self._full_key(pattern)
        keys = self.client.keys(full_pattern)
        if keys:
            return self.client.delete(*keys)
        return 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'ttl': self.ttl
        }


# =============================================================================
# Unified Pricing Cache
# =============================================================================

class PricingCache:
    """Unified caching system for Monte Carlo pricing."""

    def __init__(
        self,
        backend: str = 'memory',
        maxsize: int = 1000,
        ttl: Optional[int] = 3600,
        redis_config: Optional[Dict[str, Any]] = None
    ):
        self.backend_type = backend

        if backend == 'memory':
            self._backend = LRUCache(maxsize=maxsize, ttl=ttl)
        elif backend == 'redis':
            config = redis_config or {}
            config.setdefault('ttl', ttl)
            self._backend = RedisCache(**config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get(self, key: str) -> Optional[Any]:
        return self._backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if ttl is not None and isinstance(self._backend, RedisCache):
            self._backend.set(key, value, ttl)
        else:
            self._backend.set(key, value)

    def delete(self, key: str) -> bool:
        return self._backend.delete(key)

    def clear(self) -> None:
        self._backend.clear()

    def make_key(self, name: str, **kwargs) -> str:
        return generate_cache_key(name, (), kwargs)

    def cached(self, ttl: Optional[int] = None, key_prefix: Optional[str] = None) -> Callable:
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                prefix = key_prefix or func.__name__
                key = generate_cache_key(prefix, args, kwargs)

                result = self.get(key)
                if result is not None:
                    return result

                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result

            wrapper.cache_clear = lambda: self.clear()
            wrapper.cache_info = lambda: self._backend.stats()
            return wrapper
        return decorator

    def stats(self) -> Dict[str, Any]:
        return {'backend': self.backend_type, **self._backend.stats()}


# Global cache instance
_global_cache: Optional[PricingCache] = None


def get_cache() -> PricingCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PricingCache(backend='memory', maxsize=1000, ttl=3600)
    return _global_cache


def configure_cache(
    backend: str = 'memory',
    maxsize: int = 1000,
    ttl: int = 3600,
    redis_config: Optional[Dict[str, Any]] = None
) -> PricingCache:
    """Configure global cache instance."""
    global _global_cache
    _global_cache = PricingCache(backend=backend, maxsize=maxsize, ttl=ttl, redis_config=redis_config)
    return _global_cache


def cached_pricing(ttl: Optional[int] = None):
    """Decorator for caching pricing functions using global cache."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            key = generate_cache_key(func.__name__, args, kwargs)

            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result

        wrapper.cache = get_cache
        return wrapper
    return decorator


if __name__ == "__main__":
    print("=" * 60)
    print("Caching Module - Demo")
    print("=" * 60)

    cache = PricingCache(backend='memory', maxsize=100, ttl=60)

    key = cache.make_key('test_option', S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
    print(f"\nGenerated key: {key[:16]}...")

    cache.set(key, {'price': 10.45, 'std_error': 0.05})
    result = cache.get(key)
    print(f"Cached result: {result}")

    @cache.cached(ttl=30)
    def mock_pricing(S0, K, r, sigma, T):
        print("(Computing...)")
        return {'price': S0 * 0.1, 'computed': True}

    print("\nFirst call:")
    result1 = mock_pricing(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
    print(f"Result: {result1}")

    print("\nSecond call (cached):")
    result2 = mock_pricing(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
    print(f"Result: {result2}")

    print(f"\nCache stats: {cache.stats()}")
