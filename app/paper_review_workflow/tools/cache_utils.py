"""Caching utilities for paper review tools.

This module provides backward compatibility wrappers around CacheManager.
New code should use CacheManager directly instead of these utility functions.
"""

import hashlib
from typing import Any

from loguru import logger

from app.paper_review_workflow.tools.cache_manager import CacheManager
from app.paper_review_workflow.constants import DEFAULT_CACHE_TTL_HOURS, CACHE_DIR_NAME


# Global CacheManager instance (for compatibility)
_default_cache_manager = CacheManager(
    cache_dir=CACHE_DIR_NAME,
    ttl_hours=DEFAULT_CACHE_TTL_HOURS,
)


def get_cache_key(*args: Any) -> str:
    """Generate cache key.
    
    Args:
    ----
        *args: Values used to generate cache key
        
    Returns:
    -------
        Hashed cache key
        
    Note:
    ----
        This function is kept for backward compatibility.
        New code should use CacheManager directly.
    """
    key_string = "_".join(str(arg) for arg in args if arg is not None)
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cached_data(cache_key: str, cache_type: str = "papers") -> dict[str, Any] | None:
    """Get data from cache.
    
    Args:
    ----
        cache_key: Cache key
        cache_type: Cache type ("papers", "metadata")
        
    Returns:
    -------
        Cached data, None if not found
        
    Note:
    ----
        This function is kept for backward compatibility.
        New code should use CacheManager directly.
    """
    import json
    result = _default_cache_manager.get(prefix=cache_type, cache_key=cache_key)
    if result:
        return json.loads(result)
    return None


def save_to_cache(
    cache_key: str,
    data: Any,
    cache_type: str = "papers",
) -> None:
    """Save data to cache.
    
    Args:
    ----
        cache_key: Cache key
        data: Data to save
        cache_type: Cache type ("papers", "metadata")
        
    Note:
    ----
        This function is kept for backward compatibility.
        New code should use CacheManager directly.
    """
    import json
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
    
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    _default_cache_manager.set(json_str, prefix=cache_type, cache_key=cache_key)


def clear_cache(cache_type: str | None = None) -> None:
    """Clear cache.
    
    Args:
    ----
        cache_type: Type of cache to clear. Clear all if None
        
    Note:
    ----
        This function is kept for backward compatibility.
        New code should use CacheManager directly.
    """
    if cache_type:
        deleted = _default_cache_manager.clear(prefix=cache_type)
        logger.info(f"Cleared {cache_type} cache ({deleted} files)")
    else:
        deleted = _default_cache_manager.clear()
        logger.info(f"Cleared all caches ({deleted} files)")





