"""
SideQuest Model Configuration
Central configuration for all modules
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Base paths
BASE_DIR = Path(_SQ_ROOT)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCRAPED_DATA_DIR = DATA_DIR / "scraped"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

@dataclass
class ModelConfig:
    """Unified model configuration"""
    # Model parameters
    n_estimators: int = 500
    max_depth: int = 15
    learning_rate: float = 0.1
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Training
    test_size: float = 0.25
    random_state: int = 42
    cv_folds: int = 5
    
    # Feature engineering
    max_features: int = 50
    embedding_dim: int = 128
    
    # Multi-task weights
    task_weights: dict = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'hidden_gem': 0.30,
                'solo_matching': 0.20,
                'fake_detection': 0.20,
                'safety_score': 0.15,
                'collaboration_auth': 0.15
            }

@dataclass
class ScrapingConfig:
    """Data scraping configuration"""
    # Yelp
    yelp_api_key: Optional[str] = os.getenv("YELP_API_KEY")
    yelp_base_url: str = "https://api.yelp.com/v3"
    
    # Google Places
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    google_places_url: str = "https://maps.googleapis.com/maps/api/place"
    
    # Reddit
    reddit_client_id: Optional[str] = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = "SideQuest/1.0"
    
    # Scraping limits
    max_results_per_query: int = 100
    request_delay: float = 1.0  # seconds between requests
    
    # Target cities
    target_cities: list = None
    
    def __post_init__(self):
        if self.target_cities is None:
            self.target_cities = [
                "Jaipur", "Delhi", "Mumbai", "Bangalore",
                "Goa", "Pune", "Hyderabad", "Chennai",
                "Kolkata", "Varanasi"
            ]

@dataclass
class RAGConfig:
    """RAG Pipeline configuration"""
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Vector DB
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
    collection_name: str = "sidequest_places"
    
    # LLM
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    max_tokens: int = 500
    
    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.7

@dataclass
class OfflineConfig:
    """Offline functionality configuration"""
    # Cache settings
    max_cache_size_mb: int = 100
    cache_expiry_hours: int = 24
    
    # PWA settings
    service_worker_path: str = str(BASE_DIR / "offline" / "sw.js")
    manifest_path: str = str(BASE_DIR / "offline" / "manifest.json")
    
    # Offline data
    offline_places_per_city: int = 50
    offline_map_radius_km: float = 5.0

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    
    # CORS
    allowed_origins: list = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]

# Initialize configs
model_config = ModelConfig()
scraping_config = ScrapingConfig()
rag_config = RAGConfig()
offline_config = OfflineConfig()
api_config = APIConfig()
