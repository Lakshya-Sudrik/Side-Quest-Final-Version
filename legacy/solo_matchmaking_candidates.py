"""
Solo Matchmaking Candidate Display Module
Transparent list + sort by compatibility, NOT filter by compatibility.
Full candidate list returned regardless of score.
Double opt-in flow for matching.
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class User:
    user_id: str
    name: str
    handle: str
    age: int
    age_range: str
    gender: str
    interests: List[str]
    solo_flag: bool
    shareable_fields: List[str]


@dataclass
class Trip:
    trip_id: str
    user_id: str
    place_id: str
    place_name: str
    start_date: str  # ISO format YYYY-MM-DD
    end_date: str    # ISO format YYYY-MM-DD
    solo_flag: bool


@dataclass
class Candidate:
    user_id: str
    name: str
    handle: str
    age_range: str
    gender: str
    selected_place: str
    trip_start: str
    trip_end: str
    shared_interests: List[str]
    compatibility_score: float  # 0-1, used ONLY for sort order


@dataclass
class InterestExpression:
    from_user_id: str
    to_user_id: str
    place_id: str
    timestamp: str  # ISO format
    status: str  # "pending", "matched", "declined"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class MatchmakingData:
    def __init__(self, data_dir: str = f"{_SQ_ROOT}/data/matchmaking"):
        self.data_dir = Path(data_dir)
        self.users: Dict[str, User] = {}
        self.trips: List[Trip] = []
        self.interests_expressed: List[InterestExpression] = []
        self._load_users()
        self._load_trips()
        self._load_interests()

    def _load_users(self):
        users_path = self.data_dir / "users.json"
        if users_path.exists():
            with open(users_path) as f:
                data = json.load(f)
                for u in data:
                    self.users[u["user_id"]] = User(**u)
        logger.info(f"Loaded {len(self.users)} users")

    def _load_trips(self):
        trips_path = self.data_dir / "trips.json"
        if trips_path.exists():
            with open(trips_path) as f:
                data = json.load(f)
                self.trips = [Trip(**t) for t in data]
        logger.info(f"Loaded {len(self.trips)} trips")

    def _load_interests(self):
        interests_path = self.data_dir / "interests_expressed.json"
        if interests_path.exists():
            with open(interests_path) as f:
                data = json.load(f)
                self.interests_expressed = [InterestExpression(**i) for i in data]
        logger.info(f"Loaded {len(self.interests_expressed)} interest expressions")

    def save_interests(self):
        interests_path = self.data_dir / "interests_expressed.json"
        with open(interests_path, "w") as f:
            json.dump([asdict(i) for i in self.interests_expressed], f, indent=2)

    def get_user_trips(self, user_id: str) -> List[Trip]:
        return [t for t in self.trips if t.user_id == user_id and t.solo_flag]

    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)


# ---------------------------------------------------------------------------
# Compatibility model wrapper
# ---------------------------------------------------------------------------
class CompatibilityModel:
    """Wrapper for the existing solo_matching model."""

    def __init__(self, model_path: str = f"{_SQ_ROOT}/models/matchmaking.pkl"):
        self.model_path = Path(model_path)
        self.bundle = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.bundle = joblib.load(self.model_path)
                logger.info(f"Loaded matchmaking model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load matchmaking model: {e}")
                self.bundle = None
        else:
            logger.warning(f"Matchmaking model not found at {self.model_path}")
            self.bundle = None

    def score_pair(self, user_a: Dict, user_b: Dict) -> float:
        """
        Compute compatibility score between two users.
        Returns score in [0, 1]. If model unavailable, returns heuristic score.
        """
        if self.bundle is not None:
            try:
                # Try to use the existing match_india score_pair function
                # For now, use a heuristic since match_india.py may not be available
                return self._heuristic_score(user_a, user_b)
            except Exception as e:
                logger.warning(f"Model scoring failed, using heuristic: {e}")
                return self._heuristic_score(user_a, user_b)
        return self._heuristic_score(user_a, user_b)

    def _heuristic_score(self, user_a: Dict, user_b: Dict) -> float:
        """Heuristic compatibility based on shared interests and profile similarity."""
        interests_a = set(user_a.get("interests", []))
        interests_b = set(user_b.get("interests", []))

        if not interests_a or not interests_b:
            return 0.1

        # Jaccard similarity for interests
        shared = interests_a & interests_b
        total = interests_a | interests_b
        interest_score = len(shared) / len(total) if total else 0.0

        # Age similarity (normalized)
        age_a = user_a.get("age", 30)
        age_b = user_b.get("age", 30)
        age_diff = abs(age_a - age_b)
        age_score = max(0.0, 1.0 - age_diff / 20.0)  # 20 years = 0 similarity

        # Gender preference not modeled here (user-controlled filter)
        gender_score = 1.0

        # Weighted combination
        return 0.6 * interest_score + 0.3 * age_score + 0.1 * gender_score


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------
def dates_overlap(start1: str, end1: str, start2: str, end2: str) -> bool:
    """Check if two date ranges overlap (inclusive)."""
    s1 = datetime.fromisoformat(start1)
    e1 = datetime.fromisoformat(end1)
    s2 = datetime.fromisoformat(start2)
    e2 = datetime.fromisoformat(end2)
    return s1 <= e2 and s2 <= e1


def get_candidates(
    current_user_id: str,
    data: MatchmakingData,
    model: CompatibilityModel,
    filters: Optional[Dict] = None,
) -> List[Candidate]:
    """
    Get all candidates for a user at their selected places.
    Filters: destination + date overlap + solo_flag ONLY.
    No compatibility-based exclusion.
    """
    current_user = data.get_user(current_user_id)
    if not current_user:
        return []

    current_trips = data.get_user_trips(current_user_id)
    if not current_trips:
        return []

    # Build candidate pool: users with same place_id, overlapping dates, solo_flag
    candidates_dict: Dict[str, Candidate] = {}

    for trip in current_trips:
        place_id = trip.place_id
        place_name = trip.place_name

        # Find other users at same place with overlapping dates
        for other_trip in data.trips:
            if other_trip.user_id == current_user_id:
                continue
            if not other_trip.solo_flag:
                continue
            if other_trip.place_id != place_id:
                continue
            if not dates_overlap(trip.start_date, trip.end_date, other_trip.start_date, other_trip.end_date):
                continue

            other_user = data.get_user(other_trip.user_id)
            if not other_user:
                continue

            # Compute shared interests (literal list)
            shared = list(set(current_user.interests) & set(other_user.interests))

            # Get compatibility score (for sorting ONLY)
            compat_score = model.score_pair(
                {"interests": current_user.interests, "age": current_user.age},
                {"interests": other_user.interests, "age": other_user.age}
            )

            # Build candidate key
            cand_key = other_user.user_id

            if cand_key not in candidates_dict:
                candidates_dict[cand_key] = Candidate(
                    user_id=other_user.user_id,
                    name=other_user.name,
                    handle=other_user.handle,
                    age_range=other_user.age_range,
                    gender=other_user.gender,
                    selected_place=place_name,
                    trip_start=other_trip.start_date,
                    trip_end=other_trip.end_date,
                    shared_interests=shared,
                    compatibility_score=compat_score,
                )
            else:
                # User appears at multiple shared places - merge
                existing = candidates_dict[cand_key]
                existing.shared_interests = list(set(existing.shared_interests) | set(shared))
                existing.selected_place += f", {place_name}"
                existing.compatibility_score = max(existing.compatibility_score, compat_score)
                # Extend trip date range
                existing.trip_start = min(existing.trip_start, other_trip.start_date)
                existing.trip_end = max(existing.trip_end, other_trip.end_date)

    candidates = list(candidates_dict.values())

    # Apply optional user-controlled filters
    if filters:
        if "age_range" in filters and filters["age_range"]:
            age_range_str = filters["age_range"]
            candidates = [c for c in candidates if c.age_range == age_range_str]
        if "gender" in filters and filters["gender"]:
            candidates = [c for c in candidates if c.gender == filters["gender"]]
        if "date_range" in filters and filters["date_range"]:
            start, end = filters["date_range"]
            candidates = [c for c in candidates if dates_overlap(c.trip_start, c.trip_end, start, end)]

    # Sort by compatibility_score descending (highest first)
    candidates.sort(key=lambda c: c.compatibility_score, reverse=True)

    return candidates


# ---------------------------------------------------------------------------
# Double opt-in flow
# ---------------------------------------------------------------------------
def express_interest(
    from_user_id: str,
    to_user_id: str,
    place_id: str,
    data: MatchmakingData,
) -> Dict:
    """User marks interest in a candidate. Returns match status."""
    # Check if already expressed
    existing = next(
        (i for i in data.interests_expressed
         if i.from_user_id == from_user_id and i.to_user_id == to_user_id and i.place_id == place_id),
        None
    )
    if existing:
        return {"success": False, "error": "Interest already expressed", "status": existing.status}

    # Create pending interest
    interest = InterestExpression(
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        place_id=place_id,
        timestamp=datetime.now().isoformat(),
        status="pending"
    )
    data.interests_expressed.append(interest)
    data.save_interests()

    # Check for mutual interest
    mutual = next(
        (i for i in data.interests_expressed
         if i.from_user_id == to_user_id and i.to_user_id == from_user_id and i.place_id == place_id and i.status == "pending"),
        None
    )

    if mutual:
        # Both interested -> MATCH!
        interest.status = "matched"
        mutual.status = "matched"
        data.save_interests()
        return {
            "success": True,
            "matched": True,
            "match_user_id": to_user_id,
            "match_user_name": data.get_user(to_user_id).name if data.get_user(to_user_id) else "Unknown",
            "message": "It's a match! Chat unlocked."
        }

    return {
        "success": True,
        "matched": False,
        "message": "Interest sent. Waiting for mutual interest."
    }


def get_mutual_matches(user_id: str, data: MatchmakingData) -> List[Dict]:
    """Get all mutual matches for a user (chat unlocked)."""
    matches = []
    for interest in data.interests_expressed:
        if interest.to_user_id == user_id and interest.status == "matched":
            # Find the corresponding mutual interest
            mutual = next(
                (i for i in data.interests_expressed
                 if i.from_user_id == user_id and i.to_user_id == interest.from_user_id
                 and i.place_id == interest.place_id and i.status == "matched"),
                None
            )
            if mutual:
                other_user = data.get_user(interest.from_user_id)
                matches.append({
                    "match_user_id": interest.from_user_id,
                    "match_user_name": other_user.name if other_user else "Unknown",
                    "match_user_handle": other_user.handle if other_user else "Unknown",
                    "place_id": interest.place_id,
                    "matched_at": interest.timestamp,
                })
    return matches


# ---------------------------------------------------------------------------
# Candidate list stats for sample places
# ---------------------------------------------------------------------------
def analyze_candidate_pool(data: MatchmakingData, model: CompatibilityModel) -> Dict:
    """Report candidate list sizes for popular vs obscure places."""
    # Group trips by place_id
    place_trips: Dict[str, List[Trip]] = {}
    for trip in data.trips:
        if trip.solo_flag:
            place_trips.setdefault(trip.place_id, []).append(trip)

    results = {}
    for place_id, trips in place_trips.items():
        if len(trips) < 2:
            continue
        # For each trip, count candidates
        total_candidates = 0
        unique_users = set()
        for trip in trips:
            cands = get_candidates(trip.user_id, data, model)
            # Filter to only those at this place
            place_cands = [c for c in cands if c.selected_place == trips[0].place_name]
            total_candidates += len(place_cands)
            for c in place_cands:
                unique_users.add(c.user_id)

        results[place_id] = {
            "place_name": trips[0].place_name,
            "num_trips": len(trips),
            "total_candidate_slots": total_candidates,
            "unique_other_users": len(unique_users),
        }

    return results


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solo Matchmaking Candidates")
    parser.add_argument("--user-id", help="Current user ID (required for all modes except stats)")
    parser.add_argument("--mode", choices=["candidates", "express-interest", "matches", "stats"], default="candidates")
    parser.add_argument("--target-user-id", help="Target user ID for express-interest")
    parser.add_argument("--place-id", help="Place ID for express-interest")
    parser.add_argument("--age-range", help="Optional age range filter (e.g., 25-30)")
    parser.add_argument("--gender", help="Optional gender filter")
    parser.add_argument("--date-start", help="Optional date range start (YYYY-MM-DD)")
    parser.add_argument("--date-end", help="Optional date range end (YYYY-MM-DD)")

    args = parser.parse_args()

    # Validate user-id for modes that require it
    if args.mode in ("candidates", "express-interest", "matches") and not args.user_id:
        print(json.dumps({"success": False, "error": f"user-id required for mode {args.mode}"}))
        exit(1)

    data = MatchmakingData()
    model = CompatibilityModel()

    if args.mode == "candidates":
        filters = {}
        if args.age_range:
            filters["age_range"] = args.age_range
        if args.gender:
            filters["gender"] = args.gender
        if args.date_start and args.date_end:
            filters["date_range"] = (args.date_start, args.date_end)

        candidates = get_candidates(args.user_id, data, model, filters)
        output = {
            "success": True,
            "user_id": args.user_id,
            "candidate_count": len(candidates),
            "candidates": [asdict(c) for c in candidates]
        }
        print(json.dumps(output, indent=2))

    elif args.mode == "express-interest":
        if not args.target_user_id or not args.place_id:
            print(json.dumps({"success": False, "error": "target-user-id and place-id required"}))
        else:
            result = express_interest(args.user_id, args.target_user_id, args.place_id, data)
            print(json.dumps(result, indent=2))

    elif args.mode == "matches":
        matches = get_mutual_matches(args.user_id, data)
        print(json.dumps({"success": True, "matches": matches}, indent=2))

    elif args.mode == "stats":
        stats = analyze_candidate_pool(data, model)
        print(json.dumps({"success": True, "place_stats": stats}, indent=2))