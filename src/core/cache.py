"""Simple caching system for processed results."""

import hashlib
import json
import logging
import sqlite3
import threading
from typing import Any, Optional
from pathlib import Path

from src.core.schema import CharacterAttributes
from config.settings import CACHE_DB_PATH, ENABLE_CACHING

logger = logging.getLogger(__name__)


class SimpleCache:
    """SQLite-based cache for character attributes."""
    
    def __init__(self, cache_path: Path = CACHE_DB_PATH, enabled: bool = ENABLE_CACHING):
        self.cache_path = cache_path
        self.enabled = enabled
        self.lock = threading.Lock()
        
        if self.enabled:
            self._init_db()
    
    def _init_db(self):
        """Initialize the cache database."""
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        hash_key TEXT PRIMARY KEY,
                        attributes TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            logger.info(f"Cache initialized at {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self.enabled = False
    
    def _get_hash(self, data: str) -> str:
        """Generate hash for input data."""
        return hashlib.md5(data.encode('utf-8')).hexdigest()
    
    def get(self, key: str) -> Optional[CharacterAttributes]:
        """Get cached result."""
        if not self.enabled:
            return None
        
        hash_key = self._get_hash(key)
        
        try:
            with self.lock:
                with sqlite3.connect(str(self.cache_path)) as conn:
                    cursor = conn.execute(
                        "SELECT attributes FROM cache WHERE hash_key = ?",
                        (hash_key,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        attr_dict = json.loads(result[0])
                        return CharacterAttributes.from_dict(attr_dict)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        
        return None
    
    def set(self, key: str, attributes: CharacterAttributes):
        """Cache result."""
        if not self.enabled:
            return
        
        hash_key = self._get_hash(key)
        attr_json = json.dumps(attributes.to_dict())
        
        try:
            with self.lock:
                with sqlite3.connect(str(self.cache_path)) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (hash_key, attributes) VALUES (?, ?)",
                        (hash_key, attr_json)
                    )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        if not self.enabled:
            return
        
        try:
            with self.lock:
                with sqlite3.connect(str(self.cache_path)) as conn:
                    conn.execute("DELETE FROM cache")
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM cache")
                timestamps = cursor.fetchone()
                
                return {
                    "enabled": True,
                    "entries": count,
                    "oldest": timestamps[0],
                    "newest": timestamps[1]
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}