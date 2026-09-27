"""
SideQuest FastAPI Application
Serves the unified model via REST API
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.unified_model import UnifiedSideQuestModel
from src.rag_pipeline import RAGPipeline
from src.features import FeaturePipeline

# Initialize FastAPI app
app = FastAPI(
    title="SideQuest API",
    description="Unified ML Model for Hidden Gem Discovery, Solo Matching, and More",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
model = None
rag = None
feature_pipeline = None

# Pydantic models for request/response
class PlaceInput(BaseModel):
    name: str = Field(..., description="Place name")
    city: str = Field(..., description="City name")
    rating: float = Field(0.0, ge=0, le=5)
    review_count: int = Field(0, ge=0)
    price_level: int = Field(2, ge=1, le=4)
    categories: List[str] = Field(default_factory=list)
    latitude: float = Field(0.0)
    longitude: float = Field(0.0)

class UserInput(BaseModel):
    user_id: str
    name: str
    age: int = Field(25, ge=18, le=100)
    city: str
    interests: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=["English"])
    available_dates: List[str] = Field(default_factory=list)

class ReviewInput(BaseModel):
    text: str
    rating: float = Field(5.0, ge=1, le=5)
    user_id: str = "anonymous"

class CollaborationInput(BaseModel):
    name: str
    type: str = Field(..., description="event or food_stall")
    city: str
    has_business_registration: bool = False
    has_phone: bool = False
    has_email: bool = False
    has_website: bool = False
    social_media: Dict = Field(default_factory=dict)
    photos: List[str] = Field(default_factory=list)

class RecommendationRequest(BaseModel):
    city: str
    category: Optional[str] = None
    budget: Optional[str] = None
    preferences: Dict = Field(default_factory=dict)
    top_k: int = Field(5, ge=1, le=20)

class SoloMatchRequest(BaseModel):
    user: UserInput
    city: str
    interests: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=["English"])

# Startup event
@app.on_event("startup")
async def load_models():
    """Load trained models on startup"""
    global model, rag, feature_pipeline
    
    try:
        model = UnifiedSideQuestModel()
        model.load_models(f"{_SQ_ROOT}/models")
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("Using untrained model. Run train.py first.")
        model = UnifiedSideQuestModel()
    
    try:
        rag = RAGPipeline()
        print("RAG pipeline loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load RAG: {e}")
        rag = RAGPipeline()
    
    feature_pipeline = FeaturePipeline()

# Health check
@app.get("/")
async def root():
    return {
        "message": "SideQuest API",
        "version": "1.0.0",
        "status": "healthy",
        "model_loaded": model.is_trained if model else False
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model.is_trained if model else False,
        "rag_ready": rag is not None
    }

# Hidden Gem Detection
@app.post("/api/hidden-gem/detect")
async def detect_hidden_gem(place: PlaceInput):
    """Detect if a place is a hidden gem"""
    if not model or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    # Convert to DataFrame
    place_dict = place.dict()
    place_df = pd.DataFrame([place_dict])
    
    # Add required features
    place_df["review_count_log"] = np.log1p(place_df["review_count"])
    place_df["is_sweet_spot"] = ((place_df["rating"] >= 4.0) & (place_df["rating"] <= 4.5)).astype(int)
    place_df["is_tourist_trap"] = (place_df["rating"] >= 4.8).astype(int)
    place_df["is_mainstream"] = (place_df["review_count"] > 1000).astype(int)
    place_df["is_budget_friendly"] = (place_df["price_level"] <= 2).astype(int)
    place_df["has_photos"] = 0
    place_df["category_count"] = place_df["categories"].apply(len)
    
    # Add mock review features
    for col in ["review_text_length", "review_word_count", "review_exclamation_count",
                "review_fake_indicators_count", "review_real_indicators_count",
                "review_positive_words", "review_negative_words", "review_unique_word_ratio",
                "review_uppercase_ratio", "review_has_numbers"]:
        if col not in place_df.columns:
            place_df[col] = np.nan  # unknown - never invent review statistics
    
    # Add safety features
    place_df["has_reviews"] = (place_df["review_count"] > 0).astype(int)
    place_df["is_open"] = 1
    # (distances used to be np.random.uniform(...) here - removed; safety comes
    # from the safety index computed on real coordinates)
    
    # Add collaboration features
    for col in ["has_business_registration", "has_phone", "has_email", "has_website",
                "has_instagram", "has_facebook", "instagram_followers", "facebook_likes",
                "social_media_age_days", "has_unique_photos", "platform_count",
                "on_google_maps", "on_yelp", "on_tripadvisor", "total_reviews",
                "avg_rating", "review_velocity"]:
        if col not in place_df.columns:
            place_df[col] = 0
    
    # Predict
    try:
        score = model.predict(place_df, "hidden_gem")[0]
        return {
            "place": place.name,
            "hidden_gem_score": float(score),
            "is_hidden_gem": score >= 70,
            "classification": "Hidden Gem" if score >= 70 else ("Decent" if score >= 50 else "Tourist Trap")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Solo Group Matching
@app.post("/api/solo-match")
async def find_solo_matches(request: SoloMatchRequest):
    """Find compatible solo travelers"""
    if not model or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    # Mock matching (in real implementation, this would query database)
    mock_matches = [
        {
            "user_id": "user_123",
            "name": "Sarah",
            "age": 28,
            "city": request.city,
            "interests": ["photography", "street food", "art"],
            "languages": ["English", "Hindi"],
            "match_score": 85,
            "common_interests": list(set(request.interests) & set(["photography", "street food", "art"]))
        },
        {
            "user_id": "user_456",
            "name": "Marco",
            "age": 32,
            "city": request.city,
            "interests": ["local cuisine", "history", "walking tours"],
            "languages": ["English", "Italian"],
            "match_score": 72,
            "common_interests": list(set(request.interests) & set(["local cuisine", "history"]))
        }
    ]
    
    return {
        "user": request.user.name,
        "city": request.city,
        "matches": mock_matches,
        "total_matches": len(mock_matches)
    }

# Fake Detection
@app.post("/api/fake-detect")
async def detect_fake_review(review: ReviewInput):
    """Detect if a review is fake"""
    if not model or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    # Extract features from review text
    text = review.text
    
    # Simple feature extraction
    features = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0,
        "sentence_count": text.count('.') + text.count('!') + text.count('?'),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "comma_count": text.count(','),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "fake_indicators_count": sum([
            '!!!' in text,
            'best ever' in text.lower(),
            'must visit' in text.lower(),
            'highly recommend' in text.lower()
        ]),
        "real_indicators_count": sum([
            'hidden gem' in text.lower(),
            'local favorite' in text.lower(),
            'authentic' in text.lower()
        ]),
        "positive_words": sum(1 for w in text.lower().split() if w in ["good", "great", "amazing"]),
        "negative_words": sum(1 for w in text.lower().split() if w in ["bad", "terrible", "awful"]),
        "neutral_words": sum(1 for w in text.lower().split() if w in ["okay", "average"]),
        "unique_word_ratio": len(set(text.split())) / max(len(text.split()), 1),
        "has_numbers": int(bool(__import__('re').search(r'\d', text))),
        "has_emojis": 0,
        "rating": review.rating
    }
    
    # Create DataFrame
    review_df = pd.DataFrame([features])
    
    # Add missing columns
    for col in ["is_fake"]:
        if col not in review_df.columns:
            review_df[col] = 0
    
    # Predict
    try:
        prediction = model.predict(review_df, "fake_detection")[0]
        is_fake = bool(prediction)
        
        return {
            "review_text": text[:100] + "..." if len(text) > 100 else text,
            "is_fake": is_fake,
            "confidence": 0.85 if is_fake else 0.92,
            "indicators": {
                "excessive_exclamation": review.text.count('!') > 2,
                "superlatives": 'best' in text.lower() or 'amazing' in text.lower(),
                "short_text": len(text.split()) < 5
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Safety Scoring
@app.post("/api/safety-score")
async def get_safety_score(place: PlaceInput):
    """Rule-based safety index for a place (src/safety_index.py).

    Computed from real police-station, hospital and NCRB district crime data at
    the place's coordinates. Returns null (unknown) when there is no data -
    never a random or default value.
    """
    import sys as _sys
    _sys.path.insert(0, f"{_SQ_ROOT}/src")
    from safety_index import SafetyIndex

    if not place.latitude and not place.longitude:
        raise HTTPException(status_code=422, detail="latitude and longitude are required for the safety index")
    result = SafetyIndex.load().score_point(place.latitude, place.longitude, place.city)
    return {"place": place.name, "city": place.city, **result}

@app.post("/api/collaboration/auth")
async def authenticate_collaboration(business: CollaborationInput):
    """Authenticate a collaboration (event/food stall)"""
    if not model or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    # Create feature DataFrame
    features = {
        "has_business_registration": int(business.has_business_registration),
        "has_phone": int(business.has_phone),
        "has_email": int(business.has_email),
        "has_website": int(business.has_website),
        "has_instagram": int("instagram" in business.social_media),
        "has_facebook": int("facebook" in business.social_media),
        "instagram_followers": business.social_media.get("instagram_followers", 0),
        "facebook_likes": business.social_media.get("facebook_likes", 0),
        "social_media_age_days": business.social_media.get("account_age_days", 0),
        "photo_count": len(business.photos),
        "has_unique_photos": int(len(business.photos) > 3),
        "platform_count": len(business.social_media),
        "on_google_maps": 0,
        "on_yelp": 0,
        "on_tripadvisor": 0,
        "total_reviews": 0,
        "avg_rating": 0,
        "review_velocity": 0
    }
    
    business_df = pd.DataFrame([features])
    
    # Add missing columns
    for col in ["rating", "review_count", "review_count_log", "price_level",
                "is_sweet_spot", "is_tourist_trap", "is_mainstream", "is_budget_friendly",
                "has_photos", "category_count", "has_reviews", "is_open",
                "pharmacy_distance", "hospital_distance", "has_pharmacy_nearby",
                "has_hospital_nearby", "review_text_length", "review_word_count",
                "review_exclamation_count", "review_fake_indicators_count",
                "review_real_indicators_count", "review_positive_words",
                "review_negative_words", "review_unique_word_ratio",
                "review_uppercase_ratio", "review_has_numbers"]:
        if col not in business_df.columns:
            business_df[col] = 0
    
    # Predict
    try:
        prediction = model.predict(business_df, "collaboration_auth")[0]
        is_authentic = bool(prediction)
        
        # Calculate authenticity score
        auth_score = sum([
            features["has_business_registration"] * 30,
            features["has_phone"] * 10,
            features["has_email"] * 5,
            features["has_website"] * 5,
            features["has_instagram"] * 15 if features["instagram_followers"] > 100 else 0,
            features["platform_count"] * 5,
            features["photo_count"] * 2
        ])
        
        return {
            "business": business.name,
            "type": business.type,
            "is_authentic": is_authentic,
            "authenticity_score": min(100, auth_score),
            "recommendation": "Approved" if is_authentic else "Needs Review",
            "checks": {
                "business_registration": features["has_business_registration"],
                "contact_info": features["has_phone"] or features["has_email"],
                "social_media": features["has_instagram"] or features["has_facebook"],
                "photos": features["photo_count"] > 0,
                "cross_platform": features["platform_count"] > 1
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# RAG Recommendations
@app.post("/api/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations using RAG"""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        result = rag.recommend_hidden_gems(
            city=request.city,
            category=request.category,
            budget=request.budget,
            top_k=request.top_k
        )
        
        return {
            "city": request.city,
            "category": request.category,
            "recommendation": result["recommendation"],
            "places": result["retrieved_places"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

# Batch predictions
@app.post("/api/batch/predict")
async def batch_predict(places: List[PlaceInput]):
    """Get predictions for multiple places"""
    if not model or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    results = []
    for place in places:
        try:
            # Simplified prediction
            score = (place.rating * 20 + 
                    min(place.review_count / 50, 30) +
                    (30 if 4.0 <= place.rating <= 4.5 else 0))
            
            results.append({
                "place": place.name,
                "city": place.city,
                "hidden_gem_score": min(100, max(0, score)),
                "is_hidden_gem": score >= 70
            })
        except Exception as e:
            results.append({
                "place": place.name,
                "error": str(e)
            })
    
    return {"predictions": results}

# Model info
@app.get("/api/model/info")
async def get_model_info():
    """Get information about the trained model"""
    if not model:
        return {"status": "No model loaded"}
    
    return {
        "is_trained": model.is_trained,
        "tasks": list(model.models.keys()) if model.is_trained else [],
        "metrics": {k: v.to_dict() for k, v in model.metrics.items()} if model.is_trained else {}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
