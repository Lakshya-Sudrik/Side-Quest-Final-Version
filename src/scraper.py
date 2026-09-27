"""
SideQuest Data Scraper
Scrapes data from Yelp, Google Places, Reddit, and Instagram
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import random

import requests
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlaceData:
    """Standardized place data structure"""
    name: str
    city: str
    address: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    rating: float = 0.0
    review_count: int = 0
    categories: List[str] = None
    price_level: int = 0
    photos: List[str] = None
    reviews: List[Dict] = None
    source: str = ""
    is_open: bool = True
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.photos is None:
            self.photos = []
        if self.reviews is None:
            self.reviews = []

class YelpScraper:
    """Scrape data from Yelp API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.yelp.com/v3"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def search_places(self, city: str, category: str = "restaurants", 
                      limit: int = 50) -> List[PlaceData]:
        """Search for places in a city"""
        places = []
        
        try:
            url = f"{self.base_url}/businesses/search"
            params = {
                "location": city,
                "categories": category,
                "limit": limit,
                "sort_by": "review_count"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            for business in data.get("businesses", []):
                place = PlaceData(
                    name=business.get("name", ""),
                    city=city,
                    address=business.get("location", {}).get("address1", ""),
                    latitude=business.get("coordinates", {}).get("latitude", 0),
                    longitude=business.get("coordinates", {}).get("longitude", 0),
                    rating=business.get("rating", 0),
                    review_count=business.get("review_count", 0),
                    categories=[c.get("title", "") for c in business.get("categories", [])],
                    price_level=len(business.get("price", "")),
                    photos=[business.get("image_url", "")],
                    source="yelp"
                )
                places.append(place)
            
            logger.info(f"Scraped {len(places)} places from Yelp for {city}")
            
        except Exception as e:
            logger.error(f"Error scraping Yelp for {city}: {e}")
        
        return places
    
    def get_reviews(self, business_id: str, limit: int = 20) -> List[Dict]:
        """Get reviews for a specific business"""
        reviews = []
        
        try:
            url = f"{self.base_url}/businesses/{business_id}/reviews"
            params = {"limit": limit}
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            for review in data.get("reviews", []):
                reviews.append({
                    "text": review.get("text", ""),
                    "rating": review.get("rating", 0),
                    "time_created": review.get("time_created", ""),
                    "user": review.get("user", {}).get("name", "")
                })
                
        except Exception as e:
            logger.error(f"Error getting reviews: {e}")
        
        return reviews

class GooglePlacesScraper:
    """Scrape data from Google Places API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"
    
    def search_places(self, city: str, keyword: str = "restaurant",
                      radius: int = 5000, limit: int = 20) -> List[PlaceData]:
        """Search for places using text search"""
        places = []
        
        try:
            url = f"{self.base_url}/textsearch/json"
            params = {
                "query": f"{keyword} in {city}",
                "key": self.api_key,
                "radius": radius
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for result in data.get("results", [])[:limit]:
                place = PlaceData(
                    name=result.get("name", ""),
                    city=city,
                    address=result.get("formatted_address", ""),
                    latitude=result.get("geometry", {}).get("location", {}).get("lat", 0),
                    longitude=result.get("geometry", {}).get("location", {}).get("lng", 0),
                    rating=result.get("rating", 0),
                    review_count=result.get("user_ratings_total", 0),
                    photos=[f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={p.get('photo_reference', '')}&key={self.api_key}" 
                           for p in result.get("photos", [])[:3]],
                    source="google"
                )
                places.append(place)
            
            logger.info(f"Scraped {len(places)} places from Google for {city}")
            
        except Exception as e:
            logger.error(f"Error scraping Google Places for {city}: {e}")
        
        return places

class RedditScraper:
    """Scrape recommendations from Reddit"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.base_url = "https://www.reddit.com"
    
    def search_subreddit(self, subreddit: str, query: str, 
                         limit: int = 100) -> List[Dict]:
        """Search a subreddit for place recommendations"""
        posts = []
        
        try:
            url = f"{self.base_url}/r/{subreddit}/search.json"
            params = {
                "q": query,
                "limit": limit,
                "sort": "relevance"
            }
            headers = {"User-Agent": self.user_agent}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                posts.append({
                    "title": post.get("title", ""),
                    "text": post.get("selftext", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "subreddit": subreddit
                })
            
            logger.info(f"Scraped {len(posts)} posts from r/{subreddit}")
            
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
        
        return posts
    
    def extract_places_from_posts(self, posts: List[Dict]) -> List[Dict]:
        """Extract place names and recommendations from posts"""
        places = []
        
        # Simple NLP to extract place mentions
        for post in posts:
            text = f"{post['title']} {post['text']}"
            
            # Look for common patterns
            import re
            
            # Pattern: "Place Name" or 'Place Name'
            quoted = re.findall(r'["\']([^"\']+)["\']', text)
            for place in quoted:
                if len(place) > 3 and len(place) < 100:
                    places.append({
                        "name": place,
                        "source": "reddit",
                        "context": text[:200],
                        "score": post.get("score", 0)
                    })
        
        return places

class DataScraper:
    """Main scraper that combines all sources"""
    
    def __init__(self, yelp_key: str = None, google_key: str = None,
                 reddit_client_id: str = None, reddit_client_secret: str = None):
        self.yelp = YelpScraper(yelp_key) if yelp_key else None
        self.google = GooglePlacesScraper(google_key) if google_key else None
        self.reddit = RedditScraper(reddit_client_id, reddit_client_secret, 
                                    "SideQuest/1.0") if reddit_client_id else None
    
    def scrape_city(self, city: str, categories: List[str] = None) -> List[PlaceData]:
        """Scrape all sources for a city"""
        all_places = []
        
        if categories is None:
            categories = ["restaurants", "cafes", "bars", "nightlife", 
                         "active", "arts", "shopping"]
        
        # Yelp
        if self.yelp:
            for category in categories:
                places = self.yelp.search_places(city, category, limit=20)
                all_places.extend(places)
                time.sleep(1)  # Rate limiting
        
        # Google Places
        if self.google:
            for category in categories:
                places = self.google.search_places(city, category, limit=20)
                all_places.extend(places)
                time.sleep(1)
        
        # Reddit
        if self.reddit:
            subreddits = [f"{city.lower()}food", f"{city.lower()}travel", 
                         f"{city.lower()}locals"]
            for sub in subreddits:
                posts = self.reddit.search_subreddit(sub, f"hidden gem {city}", limit=50)
                reddit_places = self.reddit.extract_places_from_posts(posts)
                all_places.extend(reddit_places)
                time.sleep(1)
        
        logger.info(f"Total scraped {len(all_places)} places for {city}")
        return all_places
    
    def scrape_all_cities(self, cities: List[str]) -> pd.DataFrame:
        """Scrape all target cities"""
        all_data = []
        
        for city in cities:
            logger.info(f"Scraping {city}...")
            places = self.scrape_city(city)
            
            for place in places:
                if hasattr(place, '__dict__'):
                    all_data.append(asdict(place))
                else:
                    all_data.append(place)
            
            time.sleep(2)  # Pause between cities
        
        df = pd.DataFrame(all_data)
        
        # Save to file
        output_path = Path(f"{_SQ_ROOT}/data/scraped/all_places.json")
        df.to_json(output_path, orient="records", indent=2)
        
        logger.info(f"Total scraped {len(df)} places across {len(cities)} cities")
        return df

# Placeholder implementations for when API keys aren't available
class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    @staticmethod
    def generate_mock_places(n: int = 1000, cities: List[str] = None) -> List[Dict]:
        """Generate mock place data"""
        if cities is None:
            cities = ["Jaipur", "Delhi", "Mumbai", "Bangalore", "Goa"]
        
        categories = ["Cafe", "Restaurant", "Bar", "Museum", "Park", 
                      "Market", "Temple", "Gallery", "Bookshop", "Bakery"]
        
        places = []
        for i in range(n):
            city = random.choice(cities)
            rating = round(random.uniform(2.5, 5.0), 1)
            review_count = random.randint(5, 5000)
            category = random.choice(categories)
            
            places.append({
                "name": f"{category} {chr(65 + i % 26)}{i}",
                "city": city,
                "address": f"{random.randint(1, 100)} Main St, {city}",
                "latitude": random.uniform(26.0, 28.0),
                "longitude": random.uniform(75.0, 78.0),
                "rating": rating,
                "review_count": review_count,
                "categories": [category],
                "price_level": random.randint(1, 4),
                "photos": [],
                "reviews": [],
                "source": random.choice(["yelp", "google", "reddit", "mock"]),
                "is_open": random.choice([True, True, True, False])
            })
        
        return places
    
    @staticmethod
    def generate_mock_solo_users(n: int = 500) -> List[Dict]:
        """Generate mock solo traveler data"""
        interests_pool = [
            "photography", "street food", "history", "art", "nature",
            "adventure", "culture", "nightlife", "shopping", "hiking",
            "local cuisine", "architecture", "music", "books", "yoga"
        ]
        languages = ["English", "Hindi", "Spanish", "French", "German", 
                     "Japanese", "Chinese", "Italian", "Korean", "Arabic"]
        
        users = []
        for i in range(n):
            city = random.choice(["Jaipur", "Delhi", "Mumbai", "Bangalore"])
            age = random.randint(18, 60)
            interests = random.sample(interests_pool, random.randint(2, 5))
            
            users.append({
                "user_id": f"user_{i}",
                "name": f"Traveler_{i}",
                "age": age,
                "city": city,
                "interests": interests,
                "languages": random.sample(languages, random.randint(1, 3)),
                "available_dates": [f"2026-09-{random.randint(1, 30):02d}" 
                                   for _ in range(random.randint(1, 5))],
                "looking_for": random.choice(["cafe", "adventure", "cultural", 
                                             "food", "nightlife", "quiet"]),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "verified": random.choice([True, True, True, False]),
                "profile_photo": f"user_{i}.jpg"
            })
        
        return users
    
    @staticmethod
    def generate_mock_reviews(n: int = 5000) -> List[Dict]:
        """Generate mock reviews for fake detection training"""
        reviews = []
        
        real_templates = [
            "Great place! The {adjective} atmosphere made it perfect.",
            "I loved the {noun} here. Definitely coming back!",
            "Hidden gem - not many tourists know about this.",
            "Perfect for {activity}. Highly recommend!",
            "The {noun} was amazing. Must visit if you're in the area.",
            "Local favorite for a reason. {adjective} experience.",
            "Worth the detour from main attractions.",
            "Authentic {cuisine} food. Best I've had in the city."
        ]
        
        fake_templates = [
            "Best place ever!!! Must visit!!!",
            "Amazing!!! Best in city!!!",
            "Highly recommend this place!!!",
            "Perfect place for everyone!!!",
            "Love this place so much!!!",
            "Best experience of my life!!!",
            "Everyone should visit this place!!!",
            "The best place in the entire world!!!"
        ]
        
        adjectives = ["amazing", "wonderful", "cozy", "charming", "delightful"]
        nouns = ["coffee", "food", "ambiance", "service", "decor"]
        activities = ["solo travel", "couples", "family outing", "friends meetup"]
        cuisines = ["Indian", "Italian", "Chinese", "Thai", "Mexican"]
        
        for i in range(n):
            is_fake = random.random() < 0.3  # 30% fake
            
            if is_fake:
                template = random.choice(fake_templates)
                text = template
                rating = random.choice([1.0, 2.0, 5.0])  # Extreme ratings
            else:
                template = random.choice(real_templates)
                text = template.format(
                    adjective=random.choice(adjectives),
                    noun=random.choice(nouns),
                    activity=random.choice(activities),
                    cuisine=random.choice(cuisines)
                )
                rating = round(random.uniform(3.0, 5.0), 1)
            
            reviews.append({
                "review_id": f"review_{i}",
                "text": text,
                "rating": rating,
                "user_id": f"user_{random.randint(0, 499)}",
                "is_fake": is_fake,
                "review_length": len(text),
                "word_count": len(text.split()),
                "exclamation_count": text.count("!"),
                "has_uppercase_words": any(w.isupper() for w in text.split()),
                "created_at": f"2026-09-{random.randint(1, 30):02d}"
            })
        
        return reviews

if __name__ == "__main__":
    # Generate mock data for testing
    generator = MockDataGenerator()
    
    # Generate and save mock places
    places = generator.generate_mock_places(2000)
    df_places = pd.DataFrame(places)
    df_places.to_json(f"{_SQ_ROOT}/data/raw/mock_places.json", 
                      orient="records", indent=2)
    print(f"Generated {len(df_places)} mock places")
    
    # Generate and save mock solo users
    users = generator.generate_mock_solo_users(1000)
    df_users = pd.DataFrame(users)
    df_users.to_json(f"{_SQ_ROOT}/data/raw/mock_solo_users.json",
                     orient="records", indent=2)
    print(f"Generated {len(df_users)} mock solo users")
    
    # Generate and save mock reviews
    reviews = generator.generate_mock_reviews(10000)
    df_reviews = pd.DataFrame(reviews)
    df_reviews.to_json(f"{_SQ_ROOT}/data/raw/mock_reviews.json",
                       orient="records", indent=2)
    print(f"Generated {len(df_reviews)} mock reviews")
