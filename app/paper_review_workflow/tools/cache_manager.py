"""Cache manager for OpenReview API responses."""

import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

from loguru import logger


class CacheManager:
    """Manages cache for OpenReview API responses."""
    
    def __init__(
        self,
        cache_dir: str = "storage/cache",
        ttl_hours: int = 24,
    ) -> None:
        """Initialize CacheManager.
        
        Args:
        ----
            cache_dir: Path to cache directory
            ttl_hours: Cache validity period (in hours)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(self, **kwargs: Any) -> str:
        """Generate cache key.
        
        Args:
        ----
            **kwargs: Parameters used to generate the key
            
        Returns:
        -------
            Hashed cache key
        """
        # Sort dictionary to generate consistent string
        key_str = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, prefix: str = "") -> Path:
        """Get cache file path.
        
        Args:
        ----
            cache_key: Cache key
            prefix: Filename prefix
            
        Returns:
        -------
            Cache file path
        """
        if prefix:
            filename = f"{prefix}_{cache_key}.json"
        else:
            filename = f"{cache_key}.json"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is valid.
        
        Args:
        ----
            cache_path: Cache file path
            
        Returns:
        -------
            True if cache is valid
        """
        if not cache_path.exists():
            return False
        
        # Get file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        
        return age < timedelta(hours=self.ttl_hours)
    
    def get(self, prefix: str = "", **kwargs: Any) -> str | None:
        """Get data from cache.
        
        Args:
        ----
            prefix: Cache file prefix
            **kwargs: Parameters used to generate cache key
            
        Returns:
        -------
            Cached data (JSON string), None if not found
        """
        cache_key = self._generate_cache_key(**kwargs)
        cache_path = self._get_cache_path(cache_key, prefix)
        
        if self._is_cache_valid(cache_path):
            logger.debug(f"Cache hit: {cache_path.name}")
            return cache_path.read_text(encoding="utf-8")
        
        logger.debug(f"Cache miss: {cache_path.name}")
        return None
    
    def set(self, data: str, prefix: str = "", **kwargs: Any) -> None:
        """Save data to cache.
        
        Args:
        ----
            data: Data to save (JSON string)
            prefix: Cache file prefix
            **kwargs: Parameters used to generate cache key
        """
        cache_key = self._generate_cache_key(**kwargs)
        cache_path = self._get_cache_path(cache_key, prefix)
        
        cache_path.write_text(data, encoding="utf-8")
        logger.debug(f"Cache saved: {cache_path.name}")
    
    def clear(self, prefix: str = "") -> int:
        """Clear cache.
        
        Args:
        ----
            prefix: Prefix of cache files to delete (delete all if not specified)
            
        Returns:
        -------
            Number of files deleted
        """
        if prefix:
            pattern = f"{prefix}_*.json"
        else:
            pattern = "*.json"
        
        deleted = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            deleted += 1
        
        logger.info(f"Cleared {deleted} cache file(s)")
        return deleted
    
    def get_cache_info(self) -> dict[str, Any]:
        """Get cache information.
        
        Returns:
        -------
            Dictionary of cache information
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        valid_files = [f for f in cache_files if self._is_cache_valid(f)]
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "total_files": len(cache_files),
            "valid_files": len(valid_files),
            "expired_files": len(cache_files) - len(valid_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl_hours,
        }

