"""
Test script for SideQuest Inference Module
Tests collaborator verification and hidden gem filtering
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[0])).replace("\\", "/")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from unified_model import UnifiedSideQuestModel
from inference import SideQuestInference, analyze_place, verify_collaborator


def create_mock_place(is_gem: bool = False, is_safe: bool = True, 
                      is_verified: bool = True) -> pd.DataFrame:
    """Create a mock place for testing"""
    
    if is_gem:
        # Place that should qualify as hidden gem
        place = {
            "rating": [4.3],
            "review_count": [45],  # Low reviews = hidden gem
            "review_count_log": [np.log1p(45)],
            "price_level": [2],
            "category_count": [3],
            "is_sweet_spot": [1],  # Rating 4.0-4.5
            "is_budget_friendly": [0],
            "latitude": [40.7580],
            "longitude": [-73.9855],
            "is_mainstream": [0],  # Not mainstream
            "is_tourist_trap": [0],
            "review_text_mean_len": [150],
            "review_text_max_len": [300],
            "review_text_mean_words": [25],
            "review_exclamation_mean": [1.5],
            "review_uppercase_mean": [0.05],
            "review_useful_sum": [10],
            "review_funny_sum": [5],
            "review_cool_sum": [8],
            "review_votes_total": [23],
            "has_reviews": [1],
            "is_open": [1],
            "has_wifi": [1],
            "has_parking": [0],
            "hours_per_week": [60],
            "city_avg_rating": [3.8],
            "city_avg_reviews": [150],
            "city_business_count": [5000],
            "rating_vs_city": [0.5],  # Above average
            "review_ratio_vs_city": [0.3],  # Fewer reviews than avg
            # Safety features
            "pharmacy_distance": [500],
            "hospital_distance": [1500],
            "has_pharmacy_nearby": [1],
            "has_hospital_nearby": [1],
            "violent_crime_rate": [300],
            "property_crime_rate": [1500],
            # Collab auth features
            "has_business_registration": [1],
            "has_phone": [1],
            "has_email": [1],
            "has_website": [1],
            "has_instagram": [1],
            "has_facebook": [1],
            "instagram_followers": [500],
            "facebook_likes": [200],
            "social_media_age_days": [365],
            "has_unique_photos": [1],
            "platform_count": [4],
            "on_google_maps": [1],
            "on_yelp": [1],
            "on_tripadvisor": [1],
            "total_reviews": [45],
            "avg_rating": [4.3],
            "review_velocity": [2],
            "by_appointment": [0],
        }
    else:
        # Regular place
        place = {
            "rating": [3.5],
            "review_count": [500],  # High reviews = mainstream
            "review_count_log": [np.log1p(500)],
            "price_level": [3],
            "category_count": [5],
            "is_sweet_spot": [0],
            "is_budget_friendly": [0],
            "latitude": [40.7580],
            "longitude": [-73.9855],
            "is_mainstream": [1],  # Mainstream
            "is_tourist_trap": [1],
            "review_text_mean_len": [100],
            "review_text_max_len": [200],
            "review_text_mean_words": [15],
            "review_exclamation_mean": [0.5],
            "review_uppercase_mean": [0.1],
            "review_useful_sum": [50],
            "review_funny_sum": [20],
            "review_cool_sum": [30],
            "review_votes_total": [100],
            "has_reviews": [1],
            "is_open": [1],
            "has_wifi": [1],
            "has_parking": [1],
            "hours_per_week": [80],
            "city_avg_rating": [3.8],
            "city_avg_reviews": [150],
            "city_business_count": [5000],
            "rating_vs_city": [-0.3],  # Below average
            "review_ratio_vs_city": [3.0],  # More reviews than avg
            # Safety features
            "pharmacy_distance": [300],
            "hospital_distance": [800],
            "has_pharmacy_nearby": [1],
            "has_hospital_nearby": [1],
            "violent_crime_rate": [400],
            "property_crime_rate": [2000],
            # Collab auth features
            "has_business_registration": [1],
            "has_phone": [1],
            "has_email": [1],
            "has_website": [1],
            "has_instagram": [1],
            "has_facebook": [1],
            "instagram_followers": [1000],
            "facebook_likes": [500],
            "social_media_age_days": [730],
            "has_unique_photos": [1],
            "platform_count": [5],
            "on_google_maps": [1],
            "on_yelp": [1],
            "on_tripadvisor": [1],
            "total_reviews": [500],
            "avg_rating": [3.5],
            "review_velocity": [10],
            "by_appointment": [0],
        }
    
    return pd.DataFrame(place)


def test_model_methods():
    """Test the new methods in unified_model.py"""
    print("="*60)
    print("TESTING UNIFIED MODEL METHODS")
    print("="*60)
    
    # Load model
    model = UnifiedSideQuestModel()
    model.load_models(f"{_SQ_ROOT}/models")
    
    # Test 1: should_list_as_hidden_gem
    print("\n--- Test 1: should_list_as_hidden_gem ---")
    gem_place = create_mock_place(is_gem=True)
    regular_place = create_mock_place(is_gem=False)
    
    is_gem_gem = model.should_list_as_hidden_gem(gem_place, threshold=60.0)
    is_gem_regular = model.should_list_as_hidden_gem(regular_place, threshold=60.0)
    
    print(f"Hidden gem place qualifies: {is_gem_gem} (expected: True)")
    print(f"Regular place qualifies: {is_gem_regular} (expected: False)")
    
    # Test 2: get_place_scores
    print("\n--- Test 2: get_place_scores ---")
    scores_gem = model.get_place_scores(gem_place)
    scores_regular = model.get_place_scores(regular_place)
    
    print(f"Hidden gem place scores: {scores_gem}")
    print(f"Regular place scores: {scores_regular}")
    
    # Test 3: verify_collaborator_place
    print("\n--- Test 3: verify_collaborator_place ---")
    verification_gem = model.verify_collaborator_place(gem_place)
    verification_regular = model.verify_collaborator_place(regular_place)
    
    print(f"Hidden gem verification: can_list={verification_gem['can_list']}, "
          f"type={verification_gem['listing_type']}")
    print(f"Regular verification: can_list={verification_regular['can_list']}, "
          f"type={verification_regular['listing_type']}")
    
    # Test 4: filter_places_by_gem_score
    print("\n--- Test 4: filter_places_by_gem_score ---")
    places_df = pd.concat([gem_place, regular_place], ignore_index=True)
    filtered = model.filter_places_by_gem_score(places_df, threshold=60.0)
    
    print(f"Original places: {len(places_df)}")
    print(f"Filtered hidden gems: {len(filtered)}")
    
    return True


def test_inference_module():
    """Test the inference module"""
    print("\n" + "="*60)
    print("TESTING INFERENCE MODULE")
    print("="*60)
    
    # Initialize inference engine
    inference = SideQuestInference(f"{_SQ_ROOT}/models")
    
    # Test 1: analyze_place
    print("\n--- Test 1: analyze_place ---")
    gem_place = create_mock_place(is_gem=True)
    result_gem = inference.analyze_place(gem_place)
    
    print(f"Place: {result_gem.name}")
    print(f"Hidden Gem Score: {result_gem.hidden_gem_score:.1f}")
    print(f"Safety Score: {result_gem.safety_score:.1f}")
    print(f"Collab Auth Score: {result_gem.collab_auth_score:.2f}")
    print(f"Is Hidden Gem: {result_gem.is_hidden_gem}")
    print(f"Is Verified: {result_gem.is_verified}")
    print(f"Recommendation: {result_gem.recommendation}")
    print(f"Confidence: {result_gem.confidence:.2f}")
    
    # Test 2: verify_collaborator
    print("\n--- Test 2: verify_collaborator ---")
    verification = inference.verify_collaborator(gem_place)
    
    print(f"Can List: {verification['can_list']}")
    print(f"Listing Type: {verification['listing_type']}")
    print(f"Message: {verification['message']}")
    
    # Test 3: filter_hidden_gems
    print("\n--- Test 3: filter_hidden_gems ---")
    regular_place = create_mock_place(is_gem=False)
    places_df = pd.concat([gem_place, regular_place], ignore_index=True)
    
    hidden_gems = inference.filter_hidden_gems(places_df, min_score=60.0)
    print(f"Total places: {len(places_df)}")
    print(f"Hidden gems found: {len(hidden_gems)}")
    
    # Test 4: get_recommendations_for_user
    print("\n--- Test 4: get_recommendations_for_user ---")
    user_prefs = {
        "categories": ["food", "restaurant"],
        "max_price": 3,
        "min_rating": 4.0
    }
    
    recommendations = inference.get_recommendations_for_user(
        user_prefs, places_df, max_results=5
    )
    
    print(f"Recommendations found: {len(recommendations)}")
    for rec in recommendations:
        print(f"  - {rec['name']}: Gem={rec['hidden_gem_score']:.1f}, "
              f"Safety={rec['safety_score']:.1f}")
    
    return True


def test_threshold_adjustment():
    """Test threshold adjustment"""
    print("\n" + "="*60)
    print("TESTING THRESHOLD ADJUSTMENT")
    print("="*60)
    
    inference = SideQuestInference(f"{_SQ_ROOT}/models")
    
    # Get initial thresholds
    print(f"Initial thresholds:")
    print(f"  Hidden Gem: {inference.hidden_gem_threshold}")
    print(f"  Safety: {inference.safety_threshold}")
    print(f"  Collab Auth: {inference.collab_auth_threshold}")
    
    # Update thresholds
    inference.update_thresholds(hidden_gem=70.0, safety=60.0)
    
    print(f"\nUpdated thresholds:")
    print(f"  Hidden Gem: {inference.hidden_gem_threshold}")
    print(f"  Safety: {inference.safety_threshold}")
    print(f"  Collab Auth: {inference.collab_auth_threshold}")
    
    # Test with new thresholds
    gem_place = create_mock_place(is_gem=True)
    result = inference.analyze_place(gem_place)
    
    print(f"\nWith new thresholds:")
    print(f"  Is Hidden Gem (threshold=70): {result.is_hidden_gem}")
    print(f"  Hidden Gem Score: {result.hidden_gem_score:.1f}")
    
    return True


def show_accuracy_metrics():
    """Show model accuracy metrics"""
    print("\n" + "="*60)
    print("MODEL ACCURACY METRICS")
    print("="*60)
    
    inference = SideQuestInference(f"{_SQ_ROOT}/models")
    metrics = inference.get_model_accuracy()
    
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    for task_name, task_metrics in metrics.items():
        print(f"\n{task_name.upper()}:")
        
        if isinstance(task_metrics, dict):
            if "type" in task_metrics and task_metrics["type"] == "regression":
                print(f"  R² Score: {task_metrics.get('r2', 'N/A')}")
                print(f"  RMSE: {task_metrics.get('rmse', 'N/A')}")
            else:
                print(f"  Accuracy: {task_metrics.get('accuracy', 'N/A')}")
                print(f"  F1 Score: {task_metrics.get('f1', 'N/A')}")
                print(f"  AUC-ROC: {task_metrics.get('auc_roc', 'N/A')}")


if __name__ == "__main__":
    print("SideQuest Inference Module - Test Suite")
    print("="*60)
    
    try:
        # Run tests
        test_model_methods()
        test_inference_module()
        test_threshold_adjustment()
        show_accuracy_metrics()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
