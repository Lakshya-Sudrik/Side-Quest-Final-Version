"""Score every restaurant once -> data/processed/food_scored.csv.gz (used by recommender.py).

    python src/build_food_scores.py

quality_pct (0-100, within city):
  * Zomato Bangalore  -> review-text model (models/hidden_gem_model.pkl, held-out R2 0.41)
  * everywhere else   -> all-India ratings blend (models/india_quality.pkl, held-out R2 0.26)
exposure_pct        -> review-count percentile within the city (observed, not predicted)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "processed" / "food_scored.csv.gz"
log = logging.getLogger("food_scores")


def build() -> pd.DataFrame:
    from gem_features import add_gem_features, dedupe_listings
    from india_quality import score_places
    from inference import SideQuestInference, exposure_for

    df = pd.read_csv(ROOT / "data" / "india_places.csv", low_memory=False)
    df = add_gem_features(dedupe_listings(df))
    engine = SideQuestInference()
    in_domain = (df["source"].astype(str) == "zomato") & (df["city"].astype(str).str.lower() == "bangalore")
    q = pd.Series(np.nan, index=df.index)
    basis = pd.Series("ratings_blend", index=df.index)
    try:
        q[in_domain] = np.clip(engine.model.predict(df[in_domain], task="hidden_gem"), 0, 100)
        basis[in_domain] = "reviews_model"
    except Exception as exc:  # model missing -> everyone uses the India-wide score
        log.warning("review-text model unavailable (%s)", exc)
        in_domain[:] = False
    iq = score_places(df)
    q[~in_domain] = iq.loc[~in_domain, "quality_india"]
    out = pd.DataFrame({
        "place_id": df["place_id"].astype(str), "name": df["name"].astype(str),
        "city": df["city"].astype(str), "area": df["area"].astype(str),
        "lat": pd.to_numeric(df["latitude"], errors="coerce"),
        "lon": pd.to_numeric(df["longitude"], errors="coerce"),
        "category": df["primary_category"].astype(str), "cuisines": df["categories"].astype(str).str[:80],
        "rating": pd.to_numeric(df["stars"], errors="coerce"),
        "n_reviews": pd.to_numeric(df["review_count"], errors="coerce"),
        "cost_inr": pd.to_numeric(df["cost_inr"], errors="coerce"),
        "source": df["source"].astype(str), "brand": df["name"].astype(str).str.lower().str.replace(
            r"[^a-z ]", "", regex=True).str.strip(),
        "quality_pct": q.round(1), "own_rating_weight": iq["rating_weight"].round(2),
        "score_basis": basis, "exposure_pct": exposure_for(df).round(3),
    })
    out = out[out["lat"].notna() & out["quality_pct"].notna()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, compression="gzip")
    log.info("scored %d restaurants -> %s", len(out), OUT)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    o = build()
    print(o[["quality_pct", "exposure_pct"]].describe().round(2))
