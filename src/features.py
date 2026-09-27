"""
SideQuest Feature Engineering
Transforms raw data into model features
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Text features
    max_text_length: int = 500
    min_word_count: int = 3
    
    # Rating features
    rating_bins: List[float] = None
    
    # Location features
    location_grid_size: float = 0.01  # ~1km grid
    
    def __post_init__(self):
        if self.rating_bins is None:
            self.rating_bins = [0, 2, 3, 3.5, 4, 4.5, 5]

class TextFeatureExtractor:
    """Extract features from text data"""
    
    def __init__(self):
        self.fake_indicators = [
            r'!!!+',  # Multiple exclamation marks
            r'(?i)best\s+ever',  # Superlatives
            r'(?i)must\s+visit',
            r'(?i)highly\s+recommend',
            r'(?i)amazing',
            r'(?i)perfect'
        ]
        
        self.real_indicators = [
            r'(?i)hidden\s+gem',
            r'(?i)local\s+favorite',
            r'(?i)authentic',
            r'(?i)not\s+many\s+tourists',
            r'(?i)worth\s+the',
            r'(?i)coming\s+back'
        ]
    
    def extract_review_features(self, text: str) -> Dict:
        """Extract features from a single review"""
        if not text or not isinstance(text, str):
            return self._empty_text_features()
        
        text = text.strip()
        words = text.split()
        
        features = {
            # Basic text stats
            "text_length": len(text),
            "word_count": len(words),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            
            # Punctuation features
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "comma_count": text.count(','),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            
            # Fake detection indicators
            "fake_indicators_count": sum(
                1 for pattern in self.fake_indicators 
                if re.search(pattern, text)
            ),
            "real_indicators_count": sum(
                1 for pattern in self.real_indicators 
                if re.search(pattern, text)
            ),
            
            # Sentiment (simple lexicon-based)
            "positive_words": self._count_sentiment_words(text, "positive"),
            "negative_words": self._count_sentiment_words(text, "negative"),
            "neutral_words": self._count_sentiment_words(text, "neutral"),
            
            # Uniqueness
            "unique_word_ratio": len(set(words)) / max(len(words), 1),
            "has_numbers": bool(re.search(r'\d', text)),
            "has_emojis": bool(re.search(r'[\U0001F600-\U0001F64F]', text)),
        }
        
        return features
    
    def _count_sentiment_words(self, text: str, sentiment: str) -> int:
        """Count words with specific sentiment"""
        positive = ["good", "great", "excellent", "amazing", "wonderful", 
                   "love", "best", "perfect", "beautiful", "nice"]
        negative = ["bad", "terrible", "awful", "worst", "hate", 
                   "poor", "disappointing", "horrible", "waste"]
        neutral = ["okay", "average", "fine", "decent", "普通"]
        
        words = text.lower().split()
        
        if sentiment == "positive":
            return sum(1 for w in words if w in positive)
        elif sentiment == "negative":
            return sum(1 for w in words if w in negative)
        else:
            return sum(1 for w in words if w in neutral)
    
    def _empty_text_features(self) -> Dict:
        """Return empty features for missing text"""
        return {
            "text_length": 0, "word_count": 0, "avg_word_length": 0,
            "sentence_count": 0, "exclamation_count": 0, "question_count": 0,
            "comma_count": 0, "uppercase_ratio": 0, "fake_indicators_count": 0,
            "real_indicators_count": 0, "positive_words": 0, "negative_words": 0,
            "neutral_words": 0, "unique_word_ratio": 0, "has_numbers": False,
            "has_emojis": False
        }

class PlaceFeatureExtractor:
    """Extract features from place data"""
    
    def __init__(self):
        self.text_extractor = TextFeatureExtractor()
    
    def extract_place_features(self, place: Dict) -> Dict:
        """Extract all features from a place"""
        features = {}
        
        # Rating features
        rating = place.get("rating", 0)
        features["rating"] = rating
        features["rating_squared"] = rating ** 2
        features["is_sweet_spot"] = 1 if 4.0 <= rating <= 4.5 else 0
        features["is_tourist_trap"] = 1 if rating >= 4.8 else 0
        
        # Review count features
        review_count = place.get("review_count", 0)
        features["review_count"] = review_count
        features["review_count_log"] = np.log1p(review_count)
        features["is_hidden_gem_reviews"] = 1 if review_count < 200 else 0
        features["is_mainstream"] = 1 if review_count > 1000 else 0
        
        # Price features
        price = place.get("price_level", 0)
        features["price_level"] = price
        features["is_budget_friendly"] = 1 if price <= 2 else 0
        
        # Category features
        categories = place.get("categories", [])
        features["category_count"] = len(categories)
        features["is_food"] = 1 if any("food" in c.lower() or "restaurant" in c.lower() 
                                       for c in categories) else 0
        features["is_cafe"] = 1 if any("cafe" in c.lower() for c in categories) else 0
        features["is_nightlife"] = 1 if any("bar" in c.lower() or "nightlife" in c.lower() 
                                            for c in categories) else 0
        features["is_cultural"] = 1 if any("museum" in c.lower() or "art" in c.lower() 
                                           for c in categories) else 0
        
        # Photo features
        photos = place.get("photos", [])
        features["photo_count"] = len(photos)
        features["has_photos"] = 1 if photos else 0
        
        # Location features (simple grid-based)
        lat = place.get("latitude", 0)
        lng = place.get("longitude", 0)
        features["lat_grid"] = round(lat / 0.01) * 0.01
        features["lng_grid"] = round(lng / 0.01) * 0.01
        
        # Review text features (aggregate if available)
        reviews = place.get("reviews", [])
        if reviews:
            review_texts = [r.get("text", "") for r in reviews if r.get("text")]
            if review_texts:
                combined_text = " ".join(review_texts)
                text_features = self.text_extractor.extract_review_features(combined_text)
                for key, value in text_features.items():
                    features[f"review_{key}"] = value
        
        return features
    
    def extract_solo_matching_features(self, user: Dict, place: Dict) -> Dict:
        """Extract features for solo matching"""
        features = {}
        
        # User features
        features["user_age"] = user.get("age", 0)
        features["user_rating"] = user.get("rating", 0)
        features["user_verified"] = 1 if user.get("verified", False) else 0
        
        # Interest overlap
        user_interests = set(user.get("interests", []))
        place_categories = set(place.get("categories", []))
        features["interest_overlap"] = len(user_interests & place_categories)
        
        # City match
        features["same_city"] = 1 if user.get("city") == place.get("city") else 0
        
        # Language features
        user_languages = set(user.get("languages", []))
        features["language_count"] = len(user_languages)
        
        # Availability features
        available_dates = user.get("available_dates", [])
        features["availability_days"] = len(available_dates)
        
        return features
    
    def extract_safety_features(self, place: Dict, nearby_facilities: Dict = None) -> Dict:
        """Extract safety-related features"""
        features = {}
        
        # Basic safety features
        features["has_reviews"] = 1 if place.get("review_count", 0) > 0 else 0
        features["has_photos"] = 1 if place.get("photos", []) else 0
        features["is_open"] = 1 if place.get("is_open", True) else 0
        
        # Nearby facilities (if available)
        if nearby_facilities:
            features["pharmacy_distance"] = nearby_facilities.get("pharmacy_distance", float("inf"))
            features["hospital_distance"] = nearby_facilities.get("hospital_distance", float("inf"))
            features["has_pharmacy_nearby"] = 1 if features["pharmacy_distance"] < 1000 else 0
            features["has_hospital_nearby"] = 1 if features["hospital_distance"] < 2000 else 0
        else:
            features["pharmacy_distance"] = float("inf")
            features["hospital_distance"] = float("inf")
            features["has_pharmacy_nearby"] = 0
            features["has_hospital_nearby"] = 0
        
        return features
    
    def extract_collaboration_features(self, business: Dict) -> Dict:
        """Extract features for collaboration authenticity"""
        features = {}
        
        # Business verification features
        features["has_business_registration"] = 1 if business.get("registration_id") else 0
        features["has_phone"] = 1 if business.get("phone") else 0
        features["has_email"] = 1 if business.get("email") else 0
        features["has_website"] = 1 if business.get("website") else 0
        
        # Social media features
        social_media = business.get("social_media", {})
        features["has_instagram"] = 1 if social_media.get("instagram") else 0
        features["has_facebook"] = 1 if social_media.get("facebook") else 0
        features["instagram_followers"] = social_media.get("instagram_followers", 0)
        features["facebook_likes"] = social_media.get("facebook_likes", 0)
        features["social_media_age_days"] = social_media.get("account_age_days", 0)
        
        # Content authenticity
        features["photo_count"] = len(business.get("photos", []))
        features["has_unique_photos"] = 1 if features["photo_count"] > 3 else 0
        
        # Cross-platform presence
        platforms = business.get("platforms_present", [])
        features["platform_count"] = len(platforms)
        features["on_google_maps"] = 1 if "google" in platforms else 0
        features["on_yelp"] = 1 if "yelp" in platforms else 0
        features["on_tripadvisor"] = 1 if "tripadvisor" in platforms else 0
        
        # Review authenticity
        features["total_reviews"] = business.get("total_reviews", 0)
        features["avg_rating"] = business.get("avg_rating", 0)
        features["review_velocity"] = business.get("reviews_last_30_days", 0)
        
        return features

class FeaturePipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self):
        self.place_extractor = PlaceFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
    
    def process_places(self, places_df: pd.DataFrame) -> pd.DataFrame:
        """Process all places and extract features"""
        features_list = []
        
        for _, row in places_df.iterrows():
            place_dict = row.to_dict()
            features = self.place_extractor.extract_place_features(place_dict)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df.columns)} features from {len(features_df)} places")
        
        return features_df
    
    def process_solo_matching(self, users_df: pd.DataFrame, 
                               places_df: pd.DataFrame) -> pd.DataFrame:
        """Process user-place pairs for solo matching"""
        matching_features = []
        
        for _, user_row in users_df.iterrows():
            for _, place_row in places_df.iterrows():
                user_dict = user_row.to_dict()
                place_dict = place_row.to_dict()
                
                features = self.place_extractor.extract_solo_matching_features(
                    user_dict, place_dict
                )
                features["user_id"] = user_dict.get("user_id")
                features["place_name"] = place_dict.get("name")
                
                matching_features.append(features)
        
        return pd.DataFrame(matching_features)
    
    def process_fake_detection(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Process reviews for fake detection"""
        features_list = []
        
        for _, row in reviews_df.iterrows():
            text = row.get("text", "")
            features = self.text_extractor.extract_review_features(text)
            
            # Add metadata features
            features["rating"] = row.get("rating", 0)
            features["is_fake"] = row.get("is_fake", 0)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def process_safety(self, places_df: pd.DataFrame, 
                       facilities_df: pd.DataFrame = None) -> pd.DataFrame:
        """Process places for safety scoring"""
        features_list = []
        
        for _, row in places_df.iterrows():
            place_dict = row.to_dict()
            
            # Get nearby facilities if available
            nearby_facilities = None
            if facilities_df is not None:
                # Simple nearest facility lookup
                place_lat = place_dict.get("latitude", 0)
                place_lng = place_dict.get("longitude", 0)
                
                if len(facilities_df) > 0:
                    # Find nearest pharmacy
                    pharmacies = facilities_df[facilities_df["type"] == "pharmacy"]
                    if len(pharmacies) > 0:
                        distances = np.sqrt(
                            (pharmacies["latitude"] - place_lat)**2 + 
                            (pharmacies["longitude"] - place_lng)**2
                        ) * 111  # Approximate km
                        nearest_pharmacy = distances.min()
                    else:
                        nearest_pharmacy = float("inf")
                    
                    # Find nearest hospital
                    hospitals = facilities_df[facilities_df["type"] == "hospital"]
                    if len(hospitals) > 0:
                        distances = np.sqrt(
                            (hospitals["latitude"] - place_lat)**2 + 
                            (hospitals["longitude"] - place_lng)**2
                        ) * 111
                        nearest_hospital = distances.min()
                    else:
                        nearest_hospital = float("inf")
                    
                    nearby_facilities = {
                        "pharmacy_distance": nearest_pharmacy * 1000,  # Convert to meters
                        "hospital_distance": nearest_hospital * 1000
                    }
            
            features = self.place_extractor.extract_safety_features(
                place_dict, nearby_facilities
            )
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def process_collaborations(self, businesses_df: pd.DataFrame) -> pd.DataFrame:
        """Process business applications for authenticity"""
        features_list = []
        
        for _, row in businesses_df.iterrows():
            business_dict = row.to_dict()
            features = self.place_extractor.extract_collaboration_features(business_dict)
            features_list.append(features)
        
        return pd.DataFrame(features_list)

if __name__ == "__main__":
    # Test feature engineering with mock data
    import json
    
    # Load mock data
    with open(f"{_SQ_ROOT}/data/raw/mock_places.json", "r") as f:
        places = json.load(f)
    
    places_df = pd.DataFrame(places)
    
    # Initialize pipeline
    pipeline = FeaturePipeline()
    
    # Process places
    place_features = pipeline.process_places(places_df)
    print(f"Place features shape: {place_features.shape}")
    print(f"Features: {list(place_features.columns)}")
    
    # Save features
    place_features.to_csv(f"{_SQ_ROOT}/data/processed/place_features.csv", index=False)
    print("Features saved to processed folder")
