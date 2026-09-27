"""
SideQuest RAG Pipeline
Retrieval Augmented Generation for personalized recommendations
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Using fallback embeddings.")

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    logger.warning("chromadb not installed. Using in-memory storage.")

@dataclass
class RAGConfig:
    """RAG Pipeline configuration"""
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "sidequest_places"
    top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 2000
    chroma_persist_dir: str = f"{_SQ_ROOT}/offline/chroma_db"

class EmbeddingGenerator:
    """Generate embeddings for places and queries"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load sentence-transformers: {e}")
                self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.model is not None:
            return self.model.encode(text)
        else:
            # Fallback: simple TF-IDF-like embedding
            return self._simple_embedding(text)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if self.model is not None:
            return self.model.encode(texts, show_progress_bar=True)
        else:
            return np.array([self._simple_embedding(text) for text in texts])
    
    def _simple_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        """Simple hash-based embedding as fallback"""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(dim)

class VectorStore:
    """Vector storage for places"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        if HAS_CHROMA and persist_directory:
            try:
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.client.get_or_create_collection(
                    name="sidequest_places",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"ChromaDB initialized at {persist_directory}")
            except Exception as e:
                logger.warning(f"Could not initialize ChromaDB: {e}")
        
        # Fallback: in-memory storage
        self.memory_store = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
    
    def add_place(self, place_id: str, document: str, 
                  embedding: np.ndarray, metadata: Dict = None):
        """Add a place to the vector store"""
        if self.collection is not None:
            self.collection.add(
                ids=[place_id],
                documents=[document],
                embeddings=[embedding.tolist()],
                metadatas=[metadata or {}]
            )
        else:
            self.memory_store["ids"].append(place_id)
            self.memory_store["documents"].append(document)
            self.memory_store["embeddings"].append(embedding)
            self.memory_store["metadatas"].append(metadata or {})
    
    def add_places(self, places: List[Dict]):
        """Add multiple places to the vector store"""
        if self.collection is not None:
            ids = [p["id"] for p in places]
            documents = [p["document"] for p in places]
            embeddings = [p["embedding"].tolist() for p in places]
            metadatas = [p.get("metadata", {}) for p in places]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for place in places:
                self.memory_store["ids"].append(place["id"])
                self.memory_store["documents"].append(place["document"])
                self.memory_store["embeddings"].append(place["embedding"])
                self.memory_store["metadatas"].append(place.get("metadata", {}))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar places"""
        if self.collection is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            return [
                {
                    "id": id_,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for id_, doc, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        else:
            # In-memory search
            if not self.memory_store["embeddings"]:
                return []
            
            embeddings = np.array(self.memory_store["embeddings"])
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [
                {
                    "id": self.memory_store["ids"][i],
                    "document": self.memory_store["documents"][i],
                    "metadata": self.memory_store["metadatas"][i],
                    "similarity": similarities[i]
                }
                for i in top_indices
            ]
    
    def save(self):
        """Save the vector store"""
        if self.persist_directory and self.memory_store["ids"]:
            # Save memory store to disk
            save_path = Path(self.persist_directory) / "memory_store.json"
            data = {
                "ids": self.memory_store["ids"],
                "documents": self.memory_store["documents"],
                "embeddings": [e.tolist() for e in self.memory_store["embeddings"]],
                "metadatas": self.memory_store["metadatas"]
            }
            with open(save_path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Memory store saved to {save_path}")
    
    def load(self):
        """Load the vector store"""
        if self.persist_directory:
            save_path = Path(self.persist_directory) / "memory_store.json"
            if save_path.exists():
                with open(save_path, 'r') as f:
                    data = json.load(f)
                self.memory_store["ids"] = data["ids"]
                self.memory_store["documents"] = data["documents"]
                self.memory_store["embeddings"] = [np.array(e) for e in data["embeddings"]]
                self.memory_store["metadatas"] = data["metadatas"]
                logger.info(f"Memory store loaded from {save_path}")

class RAGPipeline:
    """Complete RAG pipeline for SideQuest recommendations"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        self.embedding_generator = EmbeddingGenerator(self.config.embedding_model)
        self.vector_store = VectorStore(self.config.chroma_persist_dir)
        
        # Load existing data
        self.vector_store.load()
    
    def _place_to_document(self, place: Dict) -> str:
        """Convert place data to searchable document"""
        parts = [
            f"Name: {place.get('name', 'Unknown')}",
            f"City: {place.get('city', 'Unknown')}",
            f"Category: {', '.join(place.get('categories', [])) if isinstance(place.get('categories'), list) else place.get('categories', '')}",
            f"Rating: {place.get('rating', 0)} stars",
            f"Reviews: {place.get('review_count', 0)}",
            f"Price: {'₹' * int(place.get('price_level', 2) if pd.notna(place.get('price_level')) else 2)}",
        ]
        
        if place.get("description"):
            parts.append(f"Description: {place['description']}")
        
        if place.get("vibe"):
            parts.append(f"Vibe: {place['vibe']}")
        
        return " | ".join(parts)
    
    def index_places(self, places_df: pd.DataFrame):
        """Index all places into the vector store"""
        logger.info(f"Indexing {len(places_df)} places...")
        
        places_to_add = []
        
        for idx, row in places_df.iterrows():
            place = row.to_dict()
            document = self._place_to_document(place)
            embedding = self.embedding_generator.generate_embedding(document)
            
            place_id = f"place_{idx}"
            metadata = {
                "name": place.get("name", ""),
                "city": place.get("city", ""),
                "rating": place.get("rating", 0),
                "review_count": place.get("review_count", 0),
                "categories": place.get("categories", []),
                "price_level": place.get("price_level", 0),
                "latitude": place.get("latitude", 0),
                "longitude": place.get("longitude", 0)
            }
            
            places_to_add.append({
                "id": place_id,
                "document": document,
                "embedding": embedding,
                "metadata": metadata
            })
        
        self.vector_store.add_places(places_to_add)
        self.vector_store.save()
        
        logger.info(f"Indexed {len(places_to_add)} places successfully")
    
    def _build_context(self, query: str, retrieved_places: List[Dict], 
                       user_preferences: Dict = None) -> str:
        """Build context for LLM generation"""
        context_parts = [
            f"User Query: {query}",
            "\nRetrieved Places:"
        ]
        
        for i, place in enumerate(retrieved_places, 1):
            context_parts.append(f"\n{i}. {place['document']}")
            if place.get('metadata', {}).get('distance'):
                context_parts.append(f"   Similarity: {1 - place['metadata']['distance']:.2f}")
        
        if user_preferences:
            context_parts.append("\nUser Preferences:")
            for key, value in user_preferences.items():
                context_parts.append(f"  {key}: {value}")
        
        return "\n".join(context_parts)
    
    def _generate_recommendation(self, context: str, retrieved_places: List[Dict] = None) -> str:
        """Generate recommendation using LLM (or rule-based fallback)"""
        # In a real implementation, this would call OpenAI/LLM API
        # For now, we'll use a rule-based approach
        
        # Build recommendation from retrieved places
        if not retrieved_places:
            return "I couldn't find specific places matching your criteria. Please try different preferences."
        
        recommendation_parts = [
            "Based on your preferences, I recommend these hidden gems:\n"
        ]
        
        for i, place in enumerate(retrieved_places[:5], 1):
            metadata = place.get("metadata", {})
            name = metadata.get("name", "Unknown")
            city = metadata.get("city", "")
            rating = metadata.get("rating", 0)
            categories = metadata.get("categories", [])
            category_str = ", ".join(categories) if categories else "Local Gem"
            score = 1 - place.get("distance", 0) if "distance" in place else place.get("similarity", 0)
            
            recommendation_parts.append(
                f"{i}. {name} ({city})\n"
                f"   Category: {category_str}\n"
                f"   Rating: {rating:.1f} stars\n"
                f"   Match Score: {score:.2f}\n"
                f"   Why: This is a local favorite with authentic experience.\n"
            )
        
        recommendation_parts.append(
            "\nThese places are hidden gems - authentic local favorites "
            "that most tourists don't know about. Enjoy your exploration!"
        )
        
        return "\n".join(recommendation_parts)
    
    def recommend(self, query: str, user_preferences: Dict = None,
                  top_k: int = None) -> Dict:
        """
        Generate recommendations using RAG.
        
        Args:
            query: User query/description
            user_preferences: Additional user preferences
            top_k: Number of results to retrieve
        
        Returns:
            Dictionary with recommendations and metadata
        """
        if top_k is None:
            top_k = self.config.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Retrieve similar places
        retrieved_places = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Build context
        context = self._build_context(query, retrieved_places, user_preferences)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(context, retrieved_places)
        
        return {
            "query": query,
            "recommendation": recommendation,
            "retrieved_places": [
                {
                    "name": p.get("metadata", {}).get("name", ""),
                    "city": p.get("metadata", {}).get("city", ""),
                    "rating": p.get("metadata", {}).get("rating", 0),
                    "score": 1 - p.get("distance", 0) if "distance" in p else p.get("similarity", 0)
                }
                for p in retrieved_places
            ],
            "context": context
        }
    
    def recommend_hidden_gems(self, city: str, category: str = None,
                               budget: str = None, top_k: int = 5) -> Dict:
        """Specialized recommendation for hidden gems"""
        query_parts = [f"hidden gem in {city}"]
        if category:
            query_parts.append(category)
        if budget:
            query_parts.append(f"budget {budget}")
        
        query = " ".join(query_parts)
        
        user_prefs = {
            "city": city,
            "category": category,
            "budget": budget,
            "type": "hidden_gem"
        }
        
        return self.recommend(query, user_prefs, top_k=top_k)
    
    def recommend_for_solo_traveler(self, city: str, interests: List[str],
                                     languages: List[str] = None) -> Dict:
        """Specialized recommendation for solo travelers"""
        query_parts = [f"solo travel in {city}"]
        if interests:
            query_parts.append(f"interests: {', '.join(interests)}")
        if languages:
            query_parts.append(f"languages: {', '.join(languages)}")
        
        query = " ".join(query_parts)
        
        user_prefs = {
            "city": city,
            "interests": interests,
            "languages": languages,
            "type": "solo_traveler"
        }
        
        return self.recommend(query, user_prefs, top_k=5)

def create_sample_data():
    """Create sample data for testing"""
    places = [
        {
            "name": "Café Padasanai",
            "city": "Jaipur",
            "categories": ["Cafe", "Restaurant"],
            "rating": 4.2,
            "review_count": 94,
            "price_level": 2,
            "latitude": 26.9124,
            "longitude": 75.7873,
            "vibe": "cozy, authentic, local",
            "description": "Hidden rooftop café in old city"
        },
        {
            "name": "Chai Wala Corner",
            "city": "Delhi",
            "categories": ["Cafe", "Street Food"],
            "rating": 4.3,
            "review_count": 67,
            "price_level": 1,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "vibe": "authentic, street, local",
            "description": "Best chai in Chandni Chowk"
        },
        {
            "name": "Art Studio Café",
            "city": "Mumbai",
            "categories": ["Cafe", "Art Gallery"],
            "rating": 4.1,
            "review_count": 45,
            "price_level": 2,
            "latitude": 19.0760,
            "longitude": 72.8777,
            "vibe": "artistic, quiet, creative",
            "description": "Hidden art café with local artists"
        },
        {
            "name": "Bookworm's Paradise",
            "city": "Bangalore",
            "categories": ["Bookshop", "Cafe"],
            "rating": 4.4,
            "review_count": 89,
            "price_level": 2,
            "latitude": 12.9716,
            "longitude": 77.5946,
            "vibe": "quiet, intellectual, cozy",
            "description": "Hidden bookstore café with rare collections"
        },
        {
            "name": "Spice Garden",
            "city": "Jaipur",
            "categories": ["Restaurant", "Indian"],
            "rating": 4.2,
            "review_count": 156,
            "price_level": 2,
            "latitude": 26.9228,
            "longitude": 75.7873,
            "vibe": "authentic, spicy, local",
            "description": "Hidden gem for authentic Rajasthani food"
        }
    ]
    
    return pd.DataFrame(places)

if __name__ == "__main__":
    # Test RAG pipeline
    print("Testing RAG Pipeline...")
    
    # Create sample data
    places_df = create_sample_data()
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Index places
    rag.index_places(places_df)
    
    # Test recommendation
    result = rag.recommend_hidden_gems("Jaipur", "cafe")
    
    print("\nQuery:", result["query"])
    print("\nRecommendation:")
    print(result["recommendation"])
    print("\nRetrieved Places:")
    for place in result["retrieved_places"]:
        print(f"  - {place['name']} ({place['city']}) - Score: {place['score']:.2f}")
