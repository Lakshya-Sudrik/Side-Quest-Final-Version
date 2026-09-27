"""Tests for the honest pipeline fixes (run: python -m pytest tests -q).

Uses small synthetic data, so these check that the CODE behaves correctly
(no leakage, no invented scores, reports merge, safety index logic). They do
not measure real-world accuracy - that needs the real data files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import labels_honest as lh  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic world: restaurants whose true "quality" depends partly on features
# ---------------------------------------------------------------------------
def make_world(tmp: Path, n: int = 2400, seed: int = 0):
    rng = np.random.default_rng(seed)
    city = rng.choice(["Bangalore", "Chennai"], size=n, p=[0.7, 0.3])
    stars = np.clip(rng.normal(3.9, 0.35, n), 2.5, 4.9)
    cost = rng.choice([200, 400, 600, 900, 1500], size=n)
    chain = rng.choice([1, 1, 1, 2, 5, 12], size=n).astype(float)
    dish = rng.poisson(3, n).astype(float)
    rest_type = rng.choice(["Casual Dining", "Cafe", "Quick Bites", "Bar"], size=n)
    # latent quality: some signal in stars / chain / dishes, plus noise
    latent = 0.9 * (stars - 3.9) - 0.08 * np.log(chain) + 0.05 * dish + rng.normal(0, 0.25, n)
    p_detailed_pos = 1 / (1 + np.exp(-(latent * 3 - 1)))
    n_rev = rng.integers(5, 80, n)
    review_count = (n_rev * rng.uniform(2, 20, n)).astype(int)  # votes: exposure

    places = pd.DataFrame({
        "place_id": [f"p{i}" for i in range(n)],
        "source": "zomato", "name": [f"R{i}" for i in range(n)],
        "address": [f"{i} Main Road" for i in range(n)],
        "categories": rng.choice(["North Indian, Chinese", "Cafe", "South Indian", "Biryani"], size=n),
        "city": city, "state": np.where(city == "Bangalore", "Karnataka", "Tamil Nadu"),
        "area": rng.choice(list("ABCDEFGH"), size=n),
        "latitude": np.where(city == "Bangalore", 12.97, 13.08) + rng.normal(0, 0.03, n),
        "longitude": np.where(city == "Bangalore", 77.59, 80.27) + rng.normal(0, 0.03, n),
        "stars": stars, "review_count": review_count, "review_count_log": np.log1p(review_count),
        "price_level": np.digitize(cost, [300, 700, 1500]) + 1.0, "category_count": rng.integers(1, 5, n),
        "is_budget_friendly": (cost <= 400).astype(int), "cost_inr": cost,
        "area_business_count": rng.integers(50, 2000, n).astype(float),
        "competitor_density": 0.0, "dist_from_center": rng.uniform(0, 12, n),
        "category_rarity": rng.uniform(0, 0.3, n), "has_online_order": rng.integers(0, 2, n),
        "has_book_table": rng.integers(0, 2, n), "rest_type": rest_type,
        "chain_size": chain, "dish_liked_count": dish, "cost_vs_area": rng.uniform(0.5, 2, n),
        "violent_crime_rate": 10.0, "property_crime_rate": 5.0,
        # a LEAKY legacy column, exactly like india_places.csv carries
        "hidden_gem_score": rng.uniform(0, 100, n),
    })
    rows = []
    for pid, k, p in zip(places["place_id"], n_rev, p_detailed_pos):
        good = rng.random(k) < p
        for g in good:
            rows.append({"place_id": pid, "stars": 5 if g else rng.choice([2, 3, 4]),
                         "review_text": ("word " * (60 if g else 10)).strip()})
    reviews = pd.DataFrame(rows)
    places.to_csv(tmp / "india_places.csv", index=False)
    reviews.to_csv(tmp / "india_reviews.csv", index=False)
    return places, reviews


@pytest.fixture()
def world(tmp_path, monkeypatch):
    places, reviews = make_world(tmp_path)
    monkeypatch.setattr(lh, "PLACES_CSV", tmp_path / "india_places.csv")
    monkeypatch.setattr(lh, "REVIEWS_CSV", tmp_path / "india_reviews.csv")
    monkeypatch.setattr(lh, "SCORED_REVIEWS_CSV", tmp_path / "missing.csv")
    return tmp_path, places, reviews


# ---------------------------------------------------------------------------
def test_gem_label_is_quality_only_and_shrunk(world):
    tmp, places, reviews = world
    end = lh.GemLabeler.endorsement_components(reviews)
    lab = lh.GemLabeler().fit(places, end)
    y = lab.transform(places, end)
    j = places.join(end, on="place_id")
    # every place with >= min_reviews (1) reviews is labelled; 0-100 percentile scale
    assert y[j["n_reviews"] >= 1].notna().all()
    assert y.dropna().between(0, 100).all()
    # label is NOT driven by exposure (review volume) any more
    ok = y.notna()
    corr = pd.Series(y[ok]).corr(places.loc[ok, "review_count"], method="spearman")
    assert abs(corr) < 0.3


def test_shrinkage_pulls_small_samples_to_city_rate():
    """4/4 perfect reviews must NOT outrank 180/220 strong reviews."""
    end = pd.DataFrame({
        "n_reviews": [4.0, 220.0] + [10.0] * 60,
        "n_detailed_positive": [4.0, 180.0] + [2.0] * 60,
        "sum_review_stars": [20.0, 990.0] + [35.0] * 60,
        "n_starred": [4.0, 220.0] + [10.0] * 60,
    }, index=[f"p{i}" for i in range(62)])
    end["detailed_positive_rate"] = end["n_detailed_positive"] / end["n_reviews"]
    places = pd.DataFrame({"place_id": end.index, "city": "X", "review_count": 10})
    lab = lh.GemLabeler().fit(places, end)
    y = lab.transform(places, end)
    assert y.iloc[1] > y.iloc[0]


def test_exposure_pct_is_within_city():
    df = pd.DataFrame({"city": ["A"] * 4 + ["B"] * 2, "review_count": [1, 10, 100, 1000, 5, 50]})
    e = lh.exposure_pct(df)
    assert e.iloc[0] == 0.25 and e.iloc[3] == 1.0 and e.iloc[4] == 0.5


def test_training_drops_legacy_label_and_merges_reports(world, tmp_path, monkeypatch):
    import training_honest as th

    tmp, places, _ = world
    models, reports = tmp_path / "models", tmp_path / "reports"
    models.mkdir(); reports.mkdir()
    monkeypatch.setattr(th, "MODELS_DIR", models)
    monkeypatch.setattr(th, "REPORTS_DIR", reports)
    # a previous run left a collaboration_auth report behind
    (reports / "metrics.json").write_text(json.dumps({"generated_at": "old", "tasks": {
        "collaboration_auth": {"type": "classification", "accuracy": 0.9, "macro_f1": 0.8,
                               "balanced_accuracy": 0.8, "minority_class": 0, "minority_recall": 0.5,
                               "minority_f1": 0.5, "baseline": {"majority_accuracy": 0.6},
                               "confusion_matrix_labels": ["0", "1"], "confusion_matrix": [[1, 0], [0, 1]],
                               "gate_results": {}, "gate_pass": True}}}))
    (reports / "gates.json").write_text(json.dumps({"gates": {"collaboration_auth": {"all_pass": True}}}))

    data, _ = th.prepare_training_frame()
    assert "hidden_gem_score" not in data.columns  # legacy leaky target removed
    tr = th.HonestTrainer({"ebm_outer_bags": 1})
    res = tr.train_task("hidden_gem_class", data, shuffle_test=False)
    tr.train_task("hidden_gem", data, shuffle_test=False)
    # class label must come from the review-quality labeler, not the legacy column
    assert res["report"]["type"] == "classification"
    tr.save()
    m = json.loads((reports / "metrics.json").read_text())
    assert {"hidden_gem", "hidden_gem_class", "collaboration_auth"} <= set(m["tasks"])
    assert "safety_score" not in m["tasks"]
    specs = json.loads((models / "feature_specs.json").read_text())
    feats = specs["hidden_gem"]["features"]
    assert "review_count" not in feats and "competitor_density" not in feats
    assert {"chain_size", "cost_vs_area", "rest_type", "area_cat"} <= set(feats)
    assert "rest_type" in specs["hidden_gem"]["categorical_maps"]
    rep = m["tasks"]["hidden_gem"]
    assert "spearman" in rep and "precision_at_top20pct" in rep


def test_safety_training_is_refused():
    import training_honest as th

    with pytest.raises(ValueError):
        th.train_all(tasks=["safety_score"])


# ---------------------------------------------------------------------------
# safety index
# ---------------------------------------------------------------------------
def _safety_dir(tmp: Path) -> Path:
    d = tmp / "safety"
    d.mkdir()
    square = [[77.0, 12.0], [78.0, 12.0], [78.0, 13.0], [77.0, 13.0], [77.0, 12.0]]
    other = [[80.0, 12.0], [81.0, 12.0], [81.0, 13.0], [80.0, 13.0], [80.0, 12.0]]
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"district": "Bengaluru Urban", "st_nm": "Karnataka"},
         "geometry": {"type": "Polygon", "coordinates": [square]}},
        {"type": "Feature", "properties": {"district": "Chennai", "st_nm": "Tamil Nadu"},
         "geometry": {"type": "Polygon", "coordinates": [other]}},
    ]}
    (d / "districts.geojson").write_text(json.dumps(gj))
    pd.DataFrame({"state": ["Karnataka", "Tamil Nadu", "Kerala"],
                  "district": ["Bengaluru Urban", "Chennai", "Kochi"],
                  "violent_rate": [10, 40, 20], "property_rate": [5, 30, 10],
                  "women_rate": [3, 12, 6]}).to_csv(d / "district_crime.csv", index=False)
    pd.DataFrame({"lat": [12.5], "lon": [77.5], "state": ["Karnataka"]}).to_csv(
        d / "police_stations.csv", index=False)
    pd.DataFrame({"lat": [12.51], "lon": [77.5], "state": ["Karnataka"]}).to_csv(
        d / "hospitals.csv", index=False)
    return d


def test_safety_index_scores_and_explains(tmp_path):
    from safety_index import SafetyIndex

    idx = SafetyIndex.load(_safety_dir(tmp_path), refresh=True)
    near = idx.score_point(12.5, 77.5)
    assert near["district"] == "Bengaluru Urban" and near["crime_resolution"] == "district"
    assert near["safety_score"] is not None and near["level"] == "high"
    assert any("Police station" in r for r in near["reasons"])
    # Chennai: worse crime, no infra data within range -> infra missing, not invented
    ch = idx.score_point(12.5, 80.5)
    assert ch["missing"] == ["police", "hospital"]
    assert ch["safety_score"] < near["safety_score"]
    assert ch["data_coverage"] == 0.5


def test_safety_index_unknown_outside_data(tmp_path):
    from safety_index import SafetyIndex

    idx = SafetyIndex.load(_safety_dir(tmp_path), refresh=True)
    r = idx.score_point(30.0, 70.0)
    assert r["safety_score"] is None and r["level"] == "unknown" and r["women_safe"] is None


def test_safety_index_no_crime_means_women_safe_unknown(tmp_path):
    from safety_index import SafetyIndex

    d = _safety_dir(tmp_path)
    (d / "district_crime.csv").unlink()
    r = SafetyIndex.load(d, refresh=True).score_point(12.5, 77.5)
    assert r["safety_score"] is not None and r["women_safe"] is None


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
def test_stale_model_is_refused(tmp_path):
    import pickle
    from sklearn.linear_model import LinearRegression
    from unified_model import UnifiedSideQuestModel

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 0.0]})
    m = LinearRegression().fit(X, [1.0, 2.0, 3.0])
    (tmp_path / "hidden_gem_model.pkl").write_bytes(pickle.dumps(m))
    (tmp_path / "feature_specs.json").write_text(json.dumps(
        {"hidden_gem": {"features": ["a", "c"], "numeric_medians": {}, "categorical_maps": {}}}))
    um = UnifiedSideQuestModel()
    um.load_models(str(tmp_path))
    assert "hidden_gem" not in um.models and "hidden_gem" in um.stale_models
    with pytest.raises(ValueError):
        um.predict(pd.DataFrame({"a": [1.0], "c": [2.0]}), "hidden_gem")


def test_spec_prediction_uses_train_medians_and_categories(tmp_path):
    import pickle
    from sklearn.linear_model import LinearRegression
    from unified_model import UnifiedSideQuestModel

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "kind": [0.0, 1.0, 0.0]})
    m = LinearRegression().fit(X, [10.0, 25.0, 30.0])
    (tmp_path / "hidden_gem_model.pkl").write_bytes(pickle.dumps(m))
    (tmp_path / "feature_specs.json").write_text(json.dumps({"hidden_gem": {
        "features": ["a", "kind"], "numeric_medians": {"a": 2.0},
        "categorical_maps": {"kind": {"Cafe": 0, "Bar": 1}}}}))
    um = UnifiedSideQuestModel()
    um.load_models(str(tmp_path))
    p = um.predict(pd.DataFrame({"a": [np.nan, 2.0], "kind": ["Bar", "Bar"]}), "hidden_gem")
    assert p[0] == pytest.approx(p[1])  # NaN filled with the TRAIN median (2.0)
