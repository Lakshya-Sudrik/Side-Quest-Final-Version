"""Decide whether a collaborator's listed place is a hidden gem.

Evidence, strongest first:
  1. The place is already in our data (same name within 3 km): use its measured quality
     rank and how well known it is - exactly what the route recommender uses.
  2. A NEW restaurant: the all-India food model predicts quality from what the listing
     says (cost, cuisines, name, neighbourhood, how the places around it are rated).
     It never sees a rating, so it can score a place with no reviews. Held-out R2 0.26 -
     the result is labelled "predicted" and is re-scored once real ratings exist.
  3. A NEW sight (viewpoint, temple, trail...): the sights model learns from visitor
     reviews, so a place with none cannot be scored honestly -> "needs_reviews".
Ratings the collaborator types in are never used: they cannot be verified.
A listing is published only when it is a hidden gem AND the collaborator is verified.
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import ROOT

sys.path.insert(0, str(ROOT / "src"))
GEM_RANK_MIN = 70.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def _words(s: str) -> set:
    stop = {"the", "and", "of", "restaurant", "cafe", "hotel", "temple", "point", "view", "sri", "shri"}
    return {w for w in _norm(s).split() if len(w) >= 3 and w not in stop}


@lru_cache(maxsize=1)
def _food_base() -> pd.DataFrame:
    from gem_features import dedupe_listings
    df = pd.read_csv(ROOT / "data" / "india_places.csv", low_memory=False)
    return dedupe_listings(df).reset_index(drop=True)


@lru_cache(maxsize=1)
def _food_quality_distribution() -> np.ndarray:
    import recommender
    c = recommender.load_candidates()
    return np.sort(c.loc[c["kind"] == "eat", "quality_pct"].dropna().to_numpy())


def _match_existing(kind: str, name: str, lat: float, lon: float) -> Optional[pd.Series]:
    import recommender
    c = recommender.load_candidates()
    c = c[(c["kind"] == kind) & c["lat"].between(lat - 0.03, lat + 0.03) & c["lon"].between(lon - 0.03, lon + 0.03)]
    if c.empty:
        return None
    w = _words(name)
    if not w:
        return None
    d = recommender.hav_km(lat, lon, c["lat"].to_numpy(), c["lon"].to_numpy())
    ok = c[(d <= 3.0) & c["name"].map(lambda n: w <= _words(n) or _words(n) <= w and len(_words(n)) >= 2)]
    if ok.empty:
        return None
    return ok.sort_values("n_reviews", ascending=False).iloc[0]


def predict_new_restaurant(listing: Dict) -> Dict:
    from india_quality import score_places
    base = _food_base()
    row = pd.DataFrame([{
        "place_id": "listing_new", "source": "swiggy", "name": listing["name"], "city": listing["city"],
        "state": "", "area": listing.get("area") or listing["city"], "address": "collaborator listing",
        "latitude": listing["lat"], "longitude": listing["lon"], "stars": np.nan, "review_count": 0,
        "categories": str([c.strip() for c in str(listing.get("category") or "").split(",") if c.strip()]),
        "cost_inr": listing.get("cost_inr") or np.nan, "price_level": np.nan,
        "area_business_count": np.nan, "dist_from_center": np.nan, "category_rarity": np.nan,
    }])
    frame = pd.concat([base, row], ignore_index=True)
    res = score_places(frame).iloc[-1]
    model_pct = float(res["model_pct"])
    dist = _food_quality_distribution()
    rank = 100.0 * np.searchsorted(dist, model_pct) / len(dist)
    return {"model_pct": round(model_pct, 1), "quality_rank": round(float(rank), 1)}


def evaluate(listing: Dict, collaborator_verified: bool) -> Dict:
    import recommender
    kind, name, lat, lon = listing["kind"], listing["name"], float(listing["lat"]), float(listing["lon"])
    ev: Dict = {"inputs_used": ["name", "location", "category", "cost"],
                "inputs_ignored": ["self-reported ratings or review counts (cannot be verified)"]}
    hit = _match_existing(kind, name, lat, lon)
    if hit is not None:
        rank, exposure = float(hit["quality_rank"]), float(hit["exposure_pct"])
        enough = hit["n_reviews"] >= recommender.MIN_REVIEWS[kind]
        ev.update(evidence="existing_place", matched_place={"name": hit["name"], "city": hit["city"],
                  "rating": None if pd.isna(hit["rating"]) else float(hit["rating"]),
                  "reviews": None if pd.isna(hit["n_reviews"]) else int(hit["n_reviews"])},
                  quality_rank=round(rank, 1), exposure_pct=round(exposure, 3), score_basis=hit["basis"])
        if not enough:
            decision = "needs_reviews"
            ev["reason"] = f"Only {int(hit['n_reviews'])} reviews - too few to judge"
        elif rank >= GEM_RANK_MIN and exposure <= recommender.HIDDEN_MAX_EXPOSURE:
            decision = "hidden_gem"
            ev["reason"] = (f"Top {100 - rank:.0f}% for quality and less known than "
                            f"{100 - exposure * 100:.0f}% of places in {hit['city']}")
        elif rank >= GEM_RANK_MIN:
            decision = "not_a_gem"
            ev["reason"] = "Good quality but already well known - listed as famous & good, not hidden"
            ev["famous_and_good"] = exposure >= recommender.FAMOUS_MIN_EXPOSURE
        else:
            decision = "not_a_gem"
            ev["reason"] = f"Quality rank {rank:.0f}/100 is below the gem bar ({GEM_RANK_MIN:.0f})"
    elif kind == "eat":
        p = predict_new_restaurant(listing)
        ev.update(evidence="predicted_from_listing", **p, exposure_pct=0.0, score_basis="india_quality_model",
                  model_accuracy="held-out R2 0.26 across India (no ratings used)")
        if p["quality_rank"] >= GEM_RANK_MIN:
            decision = "hidden_gem"
            ev["reason"] = (f"Model predicts top {100 - p['quality_rank']:.0f}% quality from the listing; "
                            "new, so not yet well known. Will be re-scored when real ratings arrive.")
        else:
            decision = "not_a_gem"
            ev["reason"] = f"Predicted quality rank {p['quality_rank']:.0f}/100 is below the gem bar"
    else:
        decision = "needs_reviews"
        ev.update(evidence="none", reason="Sights are judged from visitor reviews; this place has none in our data "
                                          "yet. It can be re-checked once reviews (or verified YouTube videos) exist.")
    # safety block, same as every recommended option
    row = pd.DataFrame([{"id": "listing", "name": name, "kind": kind, "category": listing.get("category"),
                         "city": listing["city"], "lat": lat, "lon": lon, "n_reviews": 0,
                         "tx_unsafe": np.nan, "tx_safe": np.nan}])
    ev["safety"] = recommender.safety_blocks(row)[0]
    ev["decision"] = decision
    ev["published"] = bool(decision == "hidden_gem" and collaborator_verified)
    if decision == "hidden_gem" and not collaborator_verified:
        ev["publish_note"] = "Will be published after your business verification is approved"
    return ev
