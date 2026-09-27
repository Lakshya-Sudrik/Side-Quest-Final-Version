"""India-wide quality model: label and feature rules (synthetic data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import india_quality as iq  # noqa: E402


def _places(n=400, seed=0):
    rng = np.random.default_rng(seed)
    city = rng.choice(["Pune", "Jaipur"], n)
    return pd.DataFrame({
        "place_id": [f"p{i}" for i in range(n)], "source": rng.choice(["swiggy", "zomato"], n),
        "name": ["Place " + "".join(chr(97 + int(d)) for d in str(i)) for i in range(n)], "city": city, "state": "S",
        "area": rng.choice(list("ABCD"), n), "address": [f"{i} road" for i in range(n)],
        "latitude": np.where(city == "Pune", 18.52, 26.91) + rng.normal(0, 0.02, n),
        "longitude": np.where(city == "Pune", 73.85, 75.79) + rng.normal(0, 0.02, n),
        "stars": np.clip(rng.normal(4.0, 0.4, n), 1, 5).round(1),
        "review_count": rng.choice([20, 50, 100, 500], n),
        "categories": rng.choice(["['Pizzas']", "['North Indian', 'Chinese']", "['Cafe']"], n),
        "cost_inr": rng.choice([200, 400, 800], n).astype(float), "price_level": 2.0,
        "area_business_count": 100.0, "dist_from_center": rng.uniform(0, 10, n),
        "category_rarity": rng.uniform(0, 0.3, n),
    })


def test_shrinkage_prefers_many_ratings():
    p = _places()
    lab = iq.RatingLabeler(k=60).fit(p)
    pair = p.iloc[:2].copy()
    pair["stars"] = [4.8, 4.5]
    pair["review_count"] = [20, 500]
    pair[["source", "city"]] = [["swiggy", "Pune"], ["swiggy", "Pune"]]
    y = lab.transform(pair)
    assert y.iloc[1] > y.iloc[0]  # 4.5 on 500 ratings beats 4.8 on 20


def test_label_is_percentile_within_source_and_city():
    p = _places()
    lab = iq.RatingLabeler(k=30).fit(p)
    y = lab.transform(p)
    assert y.between(0, 100).all()
    for _, g in y.groupby([p["source"], p["city"]]):
        assert 35 < g.mean() < 65  # each (source, city) is ranked on its own scale


def test_own_rating_never_a_feature():
    p = _places()
    _, feats, _ = iq.build_features(p)
    assert not {"stars", "review_count", "review_count_log", "rating_vs_city"} & set(feats)


def test_neighbours_exclude_self_and_same_brand():
    p = _places(50)
    p["source"] = "swiggy"  # one rating scale, so centring shifts every neighbour equally
    p["brand"] = iq.brand_of(p["name"])
    base = iq.neighbour_features(p)
    # giving place 0 a perfect rating must not change its own neighbour features
    p2 = p.copy()
    p2.loc[0, "stars"] = 5.0
    p2.loc[1:, "stars"] = p.loc[1:, "stars"]
    after = iq.neighbour_features(p2)
    src_mean_shift = p2.groupby("source")["stars"].transform("mean") - p.groupby("source")["stars"].transform("mean")
    diff = (after["nbr_rating_k"] - base["nbr_rating_k"]).iloc[0]
    assert abs(diff + src_mean_shift.iloc[0]) < 1e-9  # only the source-centring moves, not its own rating
    # a second outlet of the same brand next door is ignored
    twin = p.iloc[[0]].copy()
    twin["place_id"] = "twin"
    twin["stars"] = 1.0
    p3 = pd.concat([p, twin], ignore_index=True)
    p3["brand"] = iq.brand_of(p3["name"])
    n3 = iq.neighbour_features(p3)
    assert n3["nbr_n_2km"].iloc[0] == iq.neighbour_features(p)["nbr_n_2km"].iloc[0]
