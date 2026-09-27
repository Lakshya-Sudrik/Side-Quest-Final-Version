"""
SideQuest Offline Cache Manager
Handles IndexedDB operations for offline functionality
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OfflineCacheManager:
    """
    Manages offline caching using IndexedDB (via JavaScript bridge)
    and local file storage as fallback.
    """
    
    def __init__(self, cache_dir: str = f"{_SQ_ROOT}/offline/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.places_cache_file = self.cache_dir / "cached_places.json"
        self.recommendations_cache_file = self.cache_dir / "cached_recommendations.json"
        self.user_data_cache_file = self.cache_dir / "user_data.json"
        self.pending_sync_file = self.cache_dir / "pending_sync.json"
        
        # Initialize caches
        self._init_caches()
    
    def _init_caches(self):
        """Initialize cache files if they don't exist"""
        caches = {
            self.places_cache_file: [],
            self.recommendations_cache_file: [],
            self.user_data_cache_file: {},
            self.pending_sync_file: {"reviews": [], "checkins": [], "favorites": []}
        }
        
        for file_path, default_data in caches.items():
            if not file_path.exists():
                self._save_json(file_path, default_data)
    
    def _load_json(self, file_path: Path) -> Any:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _save_json(self, file_path: Path, data: Any):
        """Save JSON data to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
    
    # Places caching
    def cache_place(self, place: Dict):
        """Cache a single place"""
        places = self._load_json(self.places_cache_file) or []
        
        # Check if place already exists
        existing_idx = next(
            (i for i, p in enumerate(places) if p.get('id') == place.get('id')),
            None
        )
        
        if existing_idx is not None:
            places[existing_idx] = place
        else:
            places.append(place)
        
        # Keep only last 100 places (to manage storage)
        if len(places) > 100:
            places = places[-100:]
        
        self._save_json(self.places_cache_file, places)
        logger.info(f"Cached place: {place.get('name', 'Unknown')}")
    
    def cache_places(self, places: List[Dict]):
        """Cache multiple places"""
        existing_places = self._load_json(self.places_cache_file) or []
        
        # Merge with existing, avoiding duplicates
        existing_ids = {p.get('id') for p in existing_places}
        new_places = [p for p in places if p.get('id') not in existing_ids]
        
        all_places = existing_places + new_places
        
        # Keep only last 100 places
        if len(all_places) > 100:
            all_places = all_places[-100:]
        
        self._save_json(self.places_cache_file, all_places)
        logger.info(f"Cached {len(new_places)} new places")
    
    def get_cached_places(self, city: str = None, category: str = None) -> List[Dict]:
        """Get cached places, optionally filtered"""
        places = self._load_json(self.places_cache_file) or []
        
        if city:
            places = [p for p in places if p.get('city', '').lower() == city.lower()]
        
        if category:
            places = [
                p for p in places 
                if category.lower() in [c.lower() for c in p.get('categories', [])]
            ]
        
        return places
    
    def get_random_cached_places(self, count: int = 5) -> List[Dict]:
        """Get random cached places for offline browsing"""
        import random
        places = self._load_json(self.places_cache_file) or []
        return random.sample(places, min(count, len(places)))
    
    # Recommendations caching
    def cache_recommendation(self, city: str, recommendation: Dict):
        """Cache a recommendation"""
        recommendations = self._load_json(self.recommendations_cache_file) or []
        
        # Add timestamp
        recommendation['cached_at'] = datetime.now().isoformat()
        recommendation['city'] = city
        
        recommendations.append(recommendation)
        
        # Keep only last 50 recommendations
        if len(recommendations) > 50:
            recommendations = recommendations[-50:]
        
        self._save_json(self.recommendations_cache_file, recommendations)
        logger.info(f"Cached recommendation for {city}")
    
    def get_cached_recommendations(self, city: str = None, max_age_hours: int = 24) -> List[Dict]:
        """Get cached recommendations, optionally filtered by city and age"""
        recommendations = self._load_json(self.recommendations_cache_file) or []
        
        if city:
            recommendations = [r for r in recommendations if r.get('city') == city]
        
        # Filter by age
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        fresh_recommendations = []
        
        for rec in recommendations:
            cached_at = rec.get('cached_at')
            if cached_at:
                try:
                    cached_time = datetime.fromisoformat(cached_at)
                    if cached_time > cutoff_time:
                        fresh_recommendations.append(rec)
                except:
                    fresh_recommendations.append(rec)
            else:
                fresh_recommendations.append(rec)
        
        return fresh_recommendations
    
    # User data caching
    def cache_user_data(self, key: str, value: Any):
        """Cache user-specific data"""
        user_data = self._load_json(self.user_data_cache_file) or {}
        user_data[key] = value
        self._save_json(self.user_data_cache_file, user_data)
    
    def get_cached_user_data(self, key: str) -> Any:
        """Get cached user data"""
        user_data = self._load_json(self.user_data_cache_file) or {}
        return user_data.get(key)
    
    def clear_user_data(self, key: str = None):
        """Clear user data cache"""
        if key:
            user_data = self._load_json(self.user_data_cache_file) or {}
            user_data.pop(key, None)
            self._save_json(self.user_data_cache_file, user_data)
        else:
            self._save_json(self.user_data_cache_file, {})
    
    # Pending sync operations
    def add_pending_review(self, review: Dict):
        """Add a review to be synced when online"""
        pending = self._load_json(self.pending_sync_file) or {"reviews": [], "checkins": [], "favorites": []}
        
        review['pending_id'] = f"review_{datetime.now().timestamp()}"
        review['created_at'] = datetime.now().isoformat()
        
        pending['reviews'].append(review)
        self._save_json(self.pending_sync_file, pending)
        logger.info(f"Added pending review for {review.get('place_name', 'Unknown')}")
    
    def add_pending_checkin(self, checkin: Dict):
        """Add a check-in to be synced when online"""
        pending = self._load_json(self.pending_sync_file) or {"reviews": [], "checkins": [], "favorites": []}
        
        checkin['pending_id'] = f"checkin_{datetime.now().timestamp()}"
        checkin['timestamp'] = datetime.now().isoformat()
        
        pending['checkins'].append(checkin)
        self._save_json(self.pending_sync_file, pending)
        logger.info(f"Added pending check-in for {checkin.get('place_name', 'Unknown')}")
    
    def add_pending_favorite(self, place_id: str):
        """Add a favorite to be synced when online"""
        pending = self._load_json(self.pending_sync_file) or {"reviews": [], "checkins": [], "favorites": []}
        
        if place_id not in pending['favorites']:
            pending['favorites'].append({
                'place_id': place_id,
                'added_at': datetime.now().isoformat()
            })
            
            self._save_json(self.pending_sync_file, pending)
            logger.info(f"Added pending favorite: {place_id}")
    
    def get_pending_sync(self) -> Dict:
        """Get all pending sync operations"""
        return self._load_json(self.pending_sync_file) or {"reviews": [], "checkins": [], "favorites": []}
    
    def clear_pending_sync(self, sync_type: str = None):
        """Clear pending sync operations"""
        if sync_type:
            pending = self._load_json(self.pending_sync_file) or {"reviews": [], "checkins": [], "favorites": []}
            pending[sync_type] = []
            self._save_json(self.pending_sync_file, pending)
        else:
            self._save_json(self.pending_sync_file, {"reviews": [], "checkins": [], "favorites": []})
    
    def clear_all_pending_after_sync(self):
        """Clear all pending sync after successful sync"""
        self.clear_pending_sync()
        logger.info("All pending sync operations cleared")
    
    # Cache management
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached data"""
        places = self._load_json(self.places_cache_file) or []
        recommendations = self._load_json(self.recommendations_cache_file) or []
        pending = self.get_pending_sync()
        
        return {
            "cached_places": len(places),
            "cached_recommendations": len(recommendations),
            "pending_reviews": len(pending.get('reviews', [])),
            "pending_checkins": len(pending.get('checkins', [])),
            "pending_favorites": len(pending.get('favorites', [])),
            "cache_size_mb": self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB"""
        total_size = 0
        
        for file_path in self.cache_dir.rglob('*.json'):
            total_size += file_path.stat().st_size
        
        return round(total_size / (1024 * 1024), 2)
    
    def clear_cache(self, cache_type: str = None):
        """Clear specific cache or all caches"""
        if cache_type == 'places':
            self._save_json(self.places_cache_file, [])
        elif cache_type == 'recommendations':
            self._save_json(self.recommendations_cache_file, [])
        elif cache_type == 'user_data':
            self._save_json(self.user_data_cache_file, {})
        elif cache_type == 'pending':
            self.clear_pending_sync()
        else:
            # Clear all
            self._init_caches()
        
        logger.info(f"Cleared cache: {cache_type or 'all'}")
    
    def export_cache(self, export_path: str) -> bool:
        """Export cache data to a single JSON file"""
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "places": self._load_json(self.places_cache_file) or [],
                "recommendations": self._load_json(self.recommendations_cache_file) or [],
                "user_data": self._load_json(self.user_data_cache_file) or {},
                "pending_sync": self.get_pending_sync()
            }
            
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cache exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_cache(self, import_path: str) -> bool:
        """Import cache data from a JSON file"""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Import each section
            if 'places' in import_data:
                self.cache_places(import_data['places'])
            
            if 'recommendations' in import_data:
                for rec in import_data['recommendations']:
                    city = rec.get('city', 'unknown')
                    self.cache_recommendation(city, rec)
            
            if 'user_data' in import_data:
                for key, value in import_data['user_data'].items():
                    self.cache_user_data(key, value)
            
            if 'pending_sync' in import_data:
                pending = self.get_pending_sync()
                for review in import_data['pending_sync'].get('reviews', []):
                    self.add_pending_review(review)
                for checkin in import_data['pending_sync'].get('checkins', []):
                    self.add_pending_checkin(checkin)
                for fav in import_data['pending_sync'].get('favorites', []):
                    self.add_pending_favorite(fav.get('place_id'))
            
            logger.info(f"Cache imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False


# Singleton instance
cache_manager = OfflineCacheManager()


def get_cache_manager() -> OfflineCacheManager:
    """Get the cache manager singleton"""
    return cache_manager


# Example usage
if __name__ == "__main__":
    # Test cache manager
    manager = OfflineCacheManager()
    
    # Cache a place
    test_place = {
        "id": "place_1",
        "name": "Test Café",
        "city": "Delhi",
        "categories": ["Cafe"],
        "rating": 4.2,
        "review_count": 50
    }
    
    manager.cache_place(test_place)
    print("Cached place:", test_place['name'])
    
    # Get cached places
    delhi_places = manager.get_cached_places(city="Delhi")
    print(f"Found {len(delhi_places)} places in Delhi")
    
    # Add pending review
    test_review = {
        "place_id": "place_1",
        "place_name": "Test Café",
        "rating": 5,
        "text": "Great place!"
    }
    manager.add_pending_review(test_review)
    print("Added pending review")
    
    # Get stats
    stats = manager.get_cache_stats()
    print("Cache stats:", stats)
