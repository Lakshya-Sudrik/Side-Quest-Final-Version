"""
SideQuest Offline Module
Provides offline functionality through service workers, IndexedDB, and caching
"""

from .cache_manager import OfflineCacheManager, get_cache_manager

__all__ = ['OfflineCacheManager', 'get_cache_manager']
