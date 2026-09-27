"""
SideQuest Trip Suggestion Module
Retrieval + ranking layer on top of existing hidden_gem scoring pipeline.
No new model training - uses pre-computed hidden_gem_score from india_places.csv.
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[1])).replace("\\", "/")

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TripConfig:
    MIN_STOP_TIME: int = 20  # minutes
    ROUTE_BUFFER_KM: float = 10.0  # km
    MAX_RESULTS_PER_LIST: int = 5
    OSRM_BASE_URL: str = "http://router.project-osrm.org"  # free public OSRM
    # OSRM_BASE_URL: str = "http://localhost:5000"  # for local OSRM instance


DEFAULT_CATEGORY_VISIT_MIN = {
    # Default visit durations per primary_category (minutes)
    # These are ESTIMATES - dataset has NO visit-duration-per-category column
    "North Indian": 45,
    "Chinese": 40,
    "South Indian": 40,
    "Biryani": 45,
    "Indian": 45,
    "Desserts": 30,
    "Cafe": 45,
    "Fast Food": 30,
    "Bakery": 25,
    "Beverages": 20,
    "Pizzas": 40,
    "Ice Cream": 20,
    "Continental": 50,
    "American": 45,
    "Italian": 45,
    "Snacks": 25,
    "Arabian": 40,
    "Andhra": 45,
    "Street Food": 25,
    "Sweets": 20,
    "Seafood": 50,
    "Mughlai": 50,
    "Tibetan": 40,
    "Thai": 45,
    "Mexican": 45,
    "Japanese": 45,
    "Korean": 45,
    "Vietnamese": 45,
    "Mediterranean": 50,
    "European": 50,
    "African": 45,
    "Bar": 60,
    "Pub": 60,
    "Brewery": 60,
    "Club": 90,
    "Lounge": 60,
}

# Categories with NO visit duration estimate in DEFAULT_CATEGORY_VISIT_MIN
NO_ESTIMATE_CATEGORIES = []


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in km."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def point_to_polyline_distance_km(lat: float, lon: float, polyline: List[Tuple[float, float]]) -> float:
    """Minimum distance from point to polyline (list of [lat, lon])."""
    min_dist = float("inf")
    for i in range(len(polyline) - 1):
        lat1, lon1 = polyline[i]
        lat2, lon2 = polyline[i + 1]
        # Project point onto segment
        dx = lon2 - lon1
        dy = lat2 - lat1
        if dx == 0 and dy == 0:
            d = haversine_km(lat, lon, lat1, lon1)
        else:
            t = ((lon - lon1) * dx + (lat - lat1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_lon = lon1 + t * dx
            proj_lat = lat1 + t * dy
            d = haversine_km(lat, lon, proj_lat, proj_lon)
        if d < min_dist:
            min_dist = d
    return min_dist


def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Decode Google/OSRM polyline string to list of (lat, lon)."""
    index = 0
    lat = 0
    lon = 0
    coordinates = []
    while index < len(polyline_str):
        for coord in [0, 1]:
            shift = 0
            result = 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if coord == 0:
                lat += delta
            else:
                lon += delta
        coordinates.append((lat / 1e5, lon / 1e5))
    return coordinates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def get_route(
    start_lat: float, start_lon: float, end_lat: float, end_lon: float,
    osrm_url: str = "http://router.project-osrm.org"
) -> Optional[Dict]:
    """
    Call OSRM for driving route. Returns dict with:
    - duration_min: float
    - distance_km: float
    - geometry: list of [lat, lon] (decoded polyline)
    Returns None if no route found.
    """
    url = f"{osrm_url}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {
        "overview": "full",
        "geometries": "polyline",
        "steps": "false",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            logger.warning(f"OSRM no route: {data}")
            return None
        route = data["routes"][0]
        duration_min = route["duration"] / 60.0
        distance_km = route["distance"] / 1000.0
        geometry = decode_polyline(route["geometry"])
        return {
            "duration_min": duration_min,
            "distance_km": distance_km,
            "geometry": geometry,
        }
    except Exception as e:
        logger.error(f"OSRM request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Place filtering & ranking
# ---------------------------------------------------------------------------
def load_places(places_path: str) -> pd.DataFrame:
    """Load india_places.csv with required columns."""
    usecols = [
        "place_id", "name", "city", "state", "area", "address",
        "latitude", "longitude", "stars", "review_count", "review_count_log",
        "primary_category", "categories", "cost_inr", "price_level",
        "hidden_gem_score", "hospital_distance", "police_distance",
    ]
    df = pd.read_csv(places_path, usecols=usecols, low_memory=False)
    return df


def estimate_visit_duration_min(row: pd.Series) -> Tuple[float, str]:
    """
    Estimate visit duration for a place.
    Returns (duration_min, source) where source is 'lookup' or 'default'.
    """
    cat = row.get("primary_category", "")
    if cat in DEFAULT_CATEGORY_VISIT_MIN:
        return float(DEFAULT_CATEGORY_VISIT_MIN[cat]), "lookup"
    return 45.0, "default"  # fallback


def compute_popularity_score(row: pd.Series) -> float:
    """
    Popularity metric: stars * log1p(review_count)
    Explicitly NOT hidden_gem_score (which penalizes popularity).
    """
    stars = float(row.get("stars", 0))
    reviews = float(row.get("review_count", 0))
    return stars * math.log1p(reviews)


def filter_places_along_route(
    df: pd.DataFrame,
    route_geometry: List[Tuple[float, float]],
    buffer_km: float,
    budget_max_inr: float,
    free_time_min: float,
) -> pd.DataFrame:
    """
    Filter places to those within buffer_km of route, within budget,
    and whose visit+detour time fits in free_time_min.
    """
    # Distance to route
    distances = df.apply(
        lambda r: point_to_polyline_distance_km(
            r["latitude"], r["longitude"], route_geometry
        ),
        axis=1
    )
    df = df.copy()
    df["_dist_to_route_km"] = distances

    # Budget filter (cost_inr <= budget)
    if budget_max_inr > 0:
        df = df[df["cost_inr"] <= budget_max_inr]

    # Route buffer filter
    df = df[df["_dist_to_route_km"] <= buffer_km]

    # Time fit: estimate visit duration + detour time
    # Detour time: assume 2 min per km off-route (round trip = 4 min/km)
    visit_durations = []
    detour_times = []
    total_times = []
    for _, row in df.iterrows():
        visit_dur, _ = estimate_visit_duration_min(row)
        detour = row["_dist_to_route_km"] * 4.0  # 2 min/km each way
        visit_durations.append(visit_dur)
        detour_times.append(detour)
        total_times.append(visit_dur + detour)
    df["_est_visit_min"] = visit_durations
    df["_detour_min"] = detour_times
    df["_total_time_min"] = total_times

    # Time fit filter
    df = df[df["_total_time_min"] <= free_time_min]

    return df


def build_ranked_lists(
    df: pd.DataFrame,
    max_per_list: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build two ranked lists:
    a) hidden_gems: top by hidden_gem_score
    b) hidden_and_popular: top by popularity_score (stars * log1p(reviews))
    Deduplicated: a place appears in at most one list.
    """
    if df.empty:
        return [], []

    df = df.copy()
    df["_popularity"] = df.apply(compute_popularity_score, axis=1)

    # Sort for hidden gems
    gems_df = df.sort_values("_gem" if "_gem" in df.columns else "hidden_gem_score", ascending=False)
    # Sort for popularity
    pop_df = df.sort_values("_popularity", ascending=False)

    hidden_gems = []
    hidden_and_popular = []
    used_place_ids = set()

    # Fill hidden_gems first
    for _, row in gems_df.iterrows():
        if len(hidden_gems) >= max_per_list:
            break
        pid = row["place_id"]
        if pid not in used_place_ids:
            used_place_ids.add(pid)
            hidden_gems.append(format_result(row, "hidden_gem"))

    # Fill hidden_and_popular
    for _, row in pop_df.iterrows():
        if len(hidden_and_popular) >= max_per_list:
            break
        pid = row["place_id"]
        if pid not in used_place_ids:
            used_place_ids.add(pid)
            hidden_and_popular.append(format_result(row, "popular"))

    return hidden_gems, hidden_and_popular


def format_result(row: pd.Series, list_type: str) -> Dict:
    """Format a place result for the API response."""
    visit_dur, source = estimate_visit_duration_min(row)
    return {
        "place_id": str(row["place_id"]),
        "name": str(row["name"]),
        "city": str(row["city"]),
        "primary_category": str(row.get("primary_category", "")),
        "cuisines": str(row.get("categories", ""))[:80],
        "rating": float(row.get("stars", 0)),
        "review_count": int(row.get("review_count", 0)),
        "hidden_gem_score": round(float(row.get("hidden_gem_score", 0)), 1) if list_type == "hidden_gem" else None,
        "popularity_score": round(float(row.get("_popularity", 0)), 2) if list_type == "popular" else None,
        "cost_inr": float(row.get("cost_inr", 0)),
        "price_level": float(row.get("price_level", 0)),
        "hospital_distance": round(float(row.get("hospital_distance", 0)), 2),
        "police_distance": round(float(row.get("police_distance", 0)), 2),
        "estimated_visit_duration_min": round(visit_dur, 1),
        "estimated_visit_duration_source": source,
        "detour_distance_km": round(float(row.get("_dist_to_route_km", 0)), 2),
        "detour_time_min": round(float(row.get("_detour_min", 0)), 1),
        "total_time_min": round(float(row.get("_total_time_min", 0)), 1),
        "lat": float(row["latitude"]),
        "lng": float(row["longitude"]),
    }


# ---------------------------------------------------------------------------
# Main trip suggestion function
# ---------------------------------------------------------------------------
def suggest_trip_stops(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    trip_hours: float,
    budget_inr: float,
    config: Optional[TripConfig] = None,
) -> Dict:
    """
    Main entry point for trip suggestions.
    Returns dict with:
    - route_info: duration_min, distance_km
    - free_time_min
    - hidden_gems: list of 5
    - hidden_and_popular: list of 5
    - warnings: list of strings (e.g., missing visit duration data)
    """
    config = config or TripConfig()

    # 1. Get route
    route = get_route(start_lat, start_lon, end_lat, end_lon, config.OSRM_BASE_URL)
    if route is None:
        return {
            "success": False,
            "error": "No route found between start and destination",
            "code": "NO_ROUTE",
        }

    route_duration = route["duration_min"]
    free_time = trip_hours * 60 - route_duration

    warnings = []

    # 2. Check minimum stop time
    if free_time < config.MIN_STOP_TIME:
        return {
            "success": True,
            "route": {
                "duration_min": round(route_duration, 1),
                "distance_km": round(route["distance_km"], 1),
            },
            "free_time_min": round(free_time, 1),
            "message": f"No time for a stop (free_time={free_time:.0f} min < MIN_STOP_TIME={config.MIN_STOP_TIME} min)",
            "hidden_gems": [],
            "hidden_and_popular": [],
            "warnings": warnings,
        }

    # 3. Load places
    places_path = Path(f"{_SQ_ROOT}/data/india_places.csv")
    if not places_path.exists():
        places_path = Path(f"{_SQ_ROOT}/data/india_places.csv")
    if not places_path.exists():
        return {"success": False, "error": "india_places.csv not found"}

    df = load_places(str(places_path))

    # Check which categories have visit duration estimates
    categories_in_data = df["primary_category"].unique()
    missing_estimates = [c for c in categories_in_data if c not in DEFAULT_CATEGORY_VISIT_MIN]
    if missing_estimates:
        warnings.append(
            f"Visit duration estimates MISSING for {len(missing_estimates)} categories: "
            f"{', '.join(sorted(missing_estimates)[:10])}{'...' if len(missing_estimates) > 10 else ''}. "
            f"Using 45 min default for these."
        )

    # 4. Filter places
    filtered = filter_places_along_route(
        df,
        route["geometry"],
        config.ROUTE_BUFFER_KM,
        budget_inr,
        free_time,
    )

    # Flag categories with zero survivors
    if not filtered.empty:
        survivors_by_cat = filtered["primary_category"].value_counts()
        all_cats = set(df["primary_category"].unique())
        survived_cats = set(survivors_by_cat.index)
        zero_survivors = all_cats - survived_cats
        if zero_survivors:
            warnings.append(
                f"{len(zero_survivors)} categories have ZERO places surviving budget+time filter: "
                f"{', '.join(sorted(zero_survivors)[:15])}{'...' if len(zero_survivors) > 15 else ''}"
            )

    # 5. Build ranked lists
    hidden_gems, hidden_and_popular = build_ranked_lists(filtered, config.MAX_RESULTS_PER_LIST)

    # Stats: compute places in buffer from original df
    distances = df.apply(
        lambda r: point_to_polyline_distance_km(
            r["latitude"], r["longitude"], route["geometry"]
        ),
        axis=1
    )
    places_in_buffer = int((distances <= config.ROUTE_BUFFER_KM).sum())

    return {
        "success": True,
        "route": {
            "duration_min": round(route_duration, 1),
            "distance_km": round(route["distance_km"], 1),
        },
        "free_time_min": round(free_time, 1),
        "config": {
            "min_stop_time": config.MIN_STOP_TIME,
            "route_buffer_km": config.ROUTE_BUFFER_KM,
            "max_results_per_list": config.MAX_RESULTS_PER_LIST,
        },
        "hidden_gems": hidden_gems,
        "hidden_and_popular": hidden_and_popular,
        "warnings": warnings,
        "stats": {
            "total_places_in_dataset": len(df),
            "places_in_buffer": places_in_buffer,
            "places_after_budget_time_filter": len(filtered),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="SideQuest Trip Suggestion")
    parser.add_argument("--start-lat", type=float, required=True)
    parser.add_argument("--start-lon", type=float, required=True)
    parser.add_argument("--end-lat", type=float, required=True)
    parser.add_argument("--end-lon", type=float, required=True)
    parser.add_argument("--trip-hours", type=float, required=True)
    parser.add_argument("--budget-inr", type=float, required=True)
    parser.add_argument("--min-stop-time", type=int, default=20)
    parser.add_argument("--route-buffer-km", type=float, default=10.0)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--osrm-url", default="http://router.project-osrm.org")

    args = parser.parse_args()

    config = TripConfig(
        MIN_STOP_TIME=args.min_stop_time,
        ROUTE_BUFFER_KM=args.route_buffer_km,
        MAX_RESULTS_PER_LIST=args.max_results,
        OSRM_BASE_URL=args.osrm_url,
    )

    result = suggest_trip_stops(
        args.start_lat, args.start_lon,
        args.end_lat, args.end_lon,
        args.trip_hours, args.budget_inr,
        config
    )

    print(json.dumps(result, indent=2))