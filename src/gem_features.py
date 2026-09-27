"""Place-level features for the hidden-gem models (shared by training and serving).

None of these read review text or the label's inputs. They are derived from the
listing itself (name, address, area, cost, cuisines, restaurant type).
"""
from __future__ import annotations

import ast
from typing import List

import numpy as np
import pandas as pd

# fixed vocabulary so training and serving build identical columns
TOP_CUISINES: List[str] = [
    "North Indian", "Chinese", "South Indian", "Fast Food", "Continental", "Biryani",
    "Cafe", "Desserts", "Beverages", "Italian", "Street Food", "Bakery", "Pizza",
    "Burger", "Seafood", "Andhra", "Ice Cream", "Mughlai", "American", "Asian",
    "Kerala", "Rolls", "Momos", "Finger Food", "Salad",
]


def cuisine_col(c: str) -> str:
    return "cu_" + c.lower().replace(" ", "_")


CUISINE_COLS = [cuisine_col(c) for c in TOP_CUISINES]


def _parse_list(s) -> List[str]:
    if isinstance(s, list):
        return s
    s = str(s or "")
    if s.startswith("["):
        try:
            return [str(x).strip() for x in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            return []
    return [x.strip() for x in s.split(",") if x.strip()]


def dedupe_listings(places: pd.DataFrame) -> pd.DataFrame:
    """Zomato repeats a restaurant once per listing type with IDENTICAL reviews and
    features (41,492 rows -> ~9,400 restaurants). Copies are not extra data: left
    in, the same restaurant sits in train and test and inflates every score.
    Keep one row per (source, name, address) - the one with the most votes."""
    key = (places["source"].astype(str) + "|"
           + places["name"].astype(str).str.lower().str.strip() + "|"
           + places["address"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip())
    keep = (places.assign(_k=key)
            .sort_values("review_count", ascending=False, kind="stable")
            .drop_duplicates("_k").index)
    return places.loc[sorted(keep)].copy()


def add_gem_features(places: pd.DataFrame) -> pd.DataFrame:
    out = places.copy()
    name_key = out["name"].astype(str).str.lower().str.strip()
    # branches of the same brand in the city - chains are rarely hidden gems
    out["chain_size"] = out.groupby([out["city"], name_key])["place_id"].transform("count").astype(float)
    # price positioning inside the neighbourhood
    area_cost = out.groupby(["city", "area"])["cost_inr"].transform("median")
    out["cost_vs_area"] = pd.to_numeric(out["cost_inr"], errors="coerce") / area_cost.replace(0, np.nan)
    # cuisine multi-hot
    cats = out.get("categories", pd.Series("", index=out.index)).apply(_parse_list)
    for c, col in zip(TOP_CUISINES, CUISINE_COLS):
        out[col] = cats.apply(lambda lst, c=c: int(c in lst)).astype(float)
    # neighbourhood as a category (lat/lon are area-level geocodes anyway)
    out["area_cat"] = (out["city"].astype(str) + "|" + out["area"].astype(str)).astype(object)
    out["rest_type"] = out.get("rest_type", pd.Series("", index=out.index)).fillna("unknown").astype(object)
    return out
