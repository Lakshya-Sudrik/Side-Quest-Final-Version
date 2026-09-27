"""India-wide place quality (all cities, Swiggy + Zomato) - no review text needed.

    python src/india_quality.py            # train, evaluate, save
    python src/india_quality.py --quick    # skip the unseen-city check

Label  : the place's rating, shrunk toward its (source, city) average with K
         pseudo-ratings (a 4.8 on 20 ratings is not better than 4.5 on 500),
         then 0-100 percentile within (source, city). Swiggy and Zomato use
         different rating scales, so they are never ranked against each other.
Model  : LightGBM on what a listing tells you WITHOUT its own rating: cost,
         cuisines, chain size, name words, neighbourhood, and how well the
         OTHER places around it are rated (never itself or its own brand).
Serving: quality = (n * observed_pct + K * model_pct) / (n + K)
         -> places with many ratings are scored by their ratings, places with
            few or none lean on the model.
Split  : grouped by brand, so no outlet of a test brand is seen in training.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gem_features import _parse_list, dedupe_listings  # noqa: E402

PLACES_CSV = ROOT / "data" / "india_places.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BUNDLE = "india_quality.pkl"
TASK = "gem_quality_india"
log = logging.getLogger("india_quality")

# Swiggy and Zomato spell cuisines differently - map to one vocabulary
CUISINE_CANON = {
    "pizzas": "pizza", "burgers": "burger", "kebabs": "kebab", "bbq": "barbecue",
    "salads": "salad", "sweets": "mithai", "thalis": "thali", "pastas": "pasta",
    "indian": "north indian", "punjabi": "north indian", "juices": "beverages",
    "tea": "beverages", "pan-asian": "asian", "sandwiches": "sandwich", "snacks": "fast food",
}
NAME_WORDS = ["cafe", "bakery", "dhaba", "biryani", "sweets", "restaurant", "kitchen", "hotel",
              "bar", "pizza", "foods", "juice", "tiffin", "mess", "bhavan", "express", "house",
              "corner", "point", "the"]
N_CUISINES = 40
MIN_GROUP = 50          # (source, city) needs this many training places for its own reference
RATING_CAP = 500        # Swiggy buckets counts at 500+
LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                  verbose=-1, random_state=42)


def _cuisines(s) -> List[str]:
    return sorted({CUISINE_CANON.get(x.strip().lower(), x.strip().lower()) for x in _parse_list(s)})


def _slug(c: str) -> str:
    return "cu_" + re.sub(r"\W+", "_", c)


def brand_of(name: pd.Series) -> pd.Series:
    n = name.astype(str).str.lower().str.replace(r"[^a-z ]", "", regex=True)
    return n.str.replace(r"\s+", " ", regex=True).str.strip()


# ---------------------------------------------------------------------------
# features that do not use the place's own rating
# ---------------------------------------------------------------------------
def build_features(p: pd.DataFrame, cuisine_vocab: Optional[List[str]] = None):
    p = p.copy()
    cz = p["categories"].apply(_cuisines)
    if cuisine_vocab is None:
        cuisine_vocab = [c for c, _ in Counter(x for l in cz for x in l).most_common(N_CUISINES)]
    for c in cuisine_vocab:
        p[_slug(c)] = cz.apply(lambda l, c=c: float(c in l))
    name = p["name"].astype(str).str.lower().str.strip()
    p["brand"] = brand_of(p["name"])
    p["chain_city"] = p.groupby(["city", "brand"])["brand"].transform("size").astype(float)
    p["chain_india"] = p.groupby("brand")["brand"].transform("size").astype(float)
    p["chain_cities"] = p.groupby("brand")["city"].transform("nunique").astype(float)
    for w in NAME_WORDS:
        p["nm_" + w] = name.str.contains(r"\b" + w + r"\b").astype(float)
    p["name_len"] = name.str.split().str.len().astype(float)
    cost = pd.to_numeric(p["cost_inr"], errors="coerce")
    p["cost_vs_area"] = cost / p.groupby(["city", "area"])["cost_inr"].transform("median").replace(0, np.nan)
    p["cost_vs_city"] = cost / p.groupby("city")["cost_inr"].transform("median").replace(0, np.nan)
    p["city_size"] = p.groupby("city")["city"].transform("size").astype(float)
    p["is_swiggy"] = (p["source"] == "swiggy").astype(float)
    p["n_cuisines"] = cz.str.len().astype(float)
    p = p.join(neighbour_features(p))
    feats = (["cost_inr", "price_level", "cost_vs_area", "cost_vs_city", "area_business_count",
              "dist_from_center", "category_rarity", "chain_city", "chain_india", "chain_cities",
              "name_len", "city_size", "is_swiggy", "n_cuisines"]
             + [_slug(c) for c in cuisine_vocab] + ["nm_" + w for w in NAME_WORDS]
             + ["nbr_rating_2km", "nbr_rating_k", "nbr_n_2km", "nbr_dist_km"])
    return p, feats, cuisine_vocab


def neighbour_features(p: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """How well the places AROUND this one are rated. Uses other places' observed
    ratings (known at serving time) - never the place itself, never its own brand
    (which also removes the same restaurant listed on both Swiggy and Zomato).
    Ratings are centred per source so the two rating scales are comparable."""
    from sklearn.neighbors import BallTree

    out = pd.DataFrame(index=p.index, columns=["nbr_rating_2km", "nbr_rating_k", "nbr_n_2km",
                                               "nbr_dist_km"], dtype=float)
    ok = p["latitude"].notna() & p["longitude"].notna()
    q = p.loc[ok]
    if len(q) < 2:
        return out
    stars = pd.to_numeric(q["stars"], errors="coerce")
    centred = (stars - stars.groupby(q["source"]).transform("mean")).to_numpy(float)
    rad = np.radians(q[["latitude", "longitude"]].to_numpy(float))
    kk = min(k + 10, len(q))
    d, ix = BallTree(rad, metric="haversine").query(rad, k=kk)
    br = q["brand"].to_numpy(dtype=object)
    bad = (br[ix] == br[:, None]) | np.isnan(centred[ix])
    val = np.where(bad, np.nan, centred[ix])
    dkm = np.where(bad, np.nan, d * 6371.0)
    near = dkm <= 2.0
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out.loc[ok, "nbr_rating_2km"] = np.nanmean(np.where(near, val, np.nan), axis=1)
            out.loc[ok, "nbr_rating_k"] = np.nanmean(val, axis=1)
            out.loc[ok, "nbr_n_2km"] = near.sum(axis=1).astype(float)
            out.loc[ok, "nbr_dist_km"] = np.nanmedian(dkm, axis=1)
    return out


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
class RatingLabeler:
    """Shrunk rating -> percentile within (source, city). Fitted on training rows only."""

    def __init__(self, k: float = 30.0):
        self.k = float(k)

    @staticmethod
    def _grp(df):
        return df["source"].astype(str) + "|" + df["city"].astype(str)

    def fit(self, tr: pd.DataFrame) -> "RatingLabeler":
        g = self._grp(tr)
        self.prior_group_ = tr.groupby(g)["stars"].mean().to_dict()
        self.prior_source_ = tr.groupby("source")["stars"].mean().to_dict()
        sh = self.shrunk(tr)
        self.ref_group_ = {k: np.sort(v.to_numpy(float)) for k, v in sh.groupby(g) if len(v) >= MIN_GROUP}
        self.ref_source_ = {k: np.sort(v.to_numpy(float)) for k, v in sh.groupby(tr["source"])}
        # raw (unshrunk) rating references, used to score observed ratings at serving
        self.raw_group_ = {k: np.sort(v.to_numpy(float)) for k, v in tr.groupby(g)["stars"] if len(v) >= MIN_GROUP}
        self.raw_source_ = {k: np.sort(v.to_numpy(float)) for k, v in tr.groupby("source")["stars"]}
        return self

    def prior(self, df):
        g = self._grp(df)
        return g.map(self.prior_group_).fillna(df["source"].map(self.prior_source_)).astype(float)

    def shrunk(self, df):
        n = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).clip(0, RATING_CAP)
        pr = self.prior(df) if hasattr(self, "prior_group_") else df["stars"].mean()
        return (pd.to_numeric(df["stars"], errors="coerce") * n + self.k * pr) / (n + self.k)

    @staticmethod
    def _pct(v, ref):
        return 100.0 * (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / 2.0 / len(ref)

    def _rank(self, values: pd.Series, df, ref_g, ref_s):
        out = pd.Series(np.nan, index=df.index)
        g = self._grp(df)
        for key, idx in values.groupby(g).groups.items():
            src = key.split("|")[0]
            ref = ref_g.get(key, ref_s.get(src))
            if ref is None:
                continue
            v = values.loc[idx].to_numpy(float)
            r = self._pct(v, ref)
            r[np.isnan(v)] = np.nan
            out.loc[idx] = r
        return out

    def transform(self, df):
        return self._rank(self.shrunk(df), df, self.ref_group_, self.ref_source_)

    def observed_pct(self, df):
        return self._rank(pd.to_numeric(df["stars"], errors="coerce"), df, self.raw_group_, self.raw_source_)


# ---------------------------------------------------------------------------
# data / split
# ---------------------------------------------------------------------------
def load_places(path: Path = PLACES_CSV) -> pd.DataFrame:
    p = pd.read_csv(path, low_memory=False)
    p = dedupe_listings(p).reset_index(drop=True)
    return p


def brand_split(brand: pd.Series, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = brand.unique()
    pick = set(rng.choice(u, int(round(frac * len(u))), replace=False))
    return brand.isin(pick).to_numpy()


ALGO = "xgboost"   # final model; LightGBM is trained on the same split for comparison
XGB_PARAMS = dict(n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=100, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=10.0, tree_method="hist", random_state=42)


def _fit_lgb(X, y, fit_m, val_m, algo=None):
    """Fit the chosen algorithm with early stopping on the validation rows."""
    algo = algo or ALGO
    if algo == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=100)
        m.fit(X[fit_m], y[fit_m], eval_set=[(X[val_m], y[val_m])], verbose=False)
        return m
    import lightgbm as lgb

    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X[fit_m], y[fit_m], eval_set=[(X[val_m], y[val_m])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def _metrics(y, pred) -> Dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    y = np.asarray(y, float); pred = np.asarray(pred, float)
    top = y >= np.quantile(y, 0.8)
    sel = pred >= np.quantile(pred, 0.8)
    return {"r2": round(float(r2_score(y, pred)), 4),
            "mae": round(float(mean_absolute_error(y, pred)), 3),
            "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 3),
            "spearman": round(float(spearmanr(y, pred)[0]), 4),
            "precision_at_top20pct": round(float(top[sel].mean()), 4),
            "precision_at_top20pct_random": 0.2, "n": int(len(y))}


def choose_k(p, feats, train_m, seed=42, ks=(10, 30, 60)) -> Dict[float, float]:
    """Pick the shrinkage constant by brand-grouped 3-fold CV on the TRAIN rows only."""
    from sklearn.metrics import r2_score

    tr_idx = np.where(train_m)[0]
    rng = np.random.default_rng(seed)
    brands = p["brand"].iloc[tr_idx].unique()
    fold_of = dict(zip(brands, rng.integers(0, 3, len(brands))))
    fold = p["brand"].map(fold_of).to_numpy()
    X = p[feats].astype(float)
    scores = {}
    for k in ks:
        r2s = []
        for f in range(3):
            va = train_m & (fold == f)
            fit = train_m & (fold != f)
            lab = RatingLabeler(k).fit(p[fit])
            y = lab.transform(p)
            m = _fit_lgb(X, y, fit, va)
            r2s.append(r2_score(y[va], m.predict(X[va])))
        scores[k] = round(float(np.mean(r2s)), 4)
        log.info("  CV K=%s  R2=%.4f", k, scores[k])
    return scores


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train(quick: bool = False, seed: int = 42) -> Dict:
    t0 = time.time()
    p = load_places()
    p = p[pd.to_numeric(p["stars"], errors="coerce").notna()
          & (pd.to_numeric(p["review_count"], errors="coerce") > 0)].reset_index(drop=True)
    p, feats, vocab = build_features(p)
    log.info("places with a rating: %d  (%d cities, %s)", len(p), p["city"].nunique(),
             p["source"].value_counts().to_dict())

    test_m = brand_split(p["brand"], 0.25, seed)
    train_m = ~test_m
    cv = choose_k(p, feats, train_m, seed)
    k = max(cv, key=cv.get)
    log.info("chosen K=%s", k)

    lab = RatingLabeler(k).fit(p[train_m])
    y = lab.transform(p)
    X = p[feats].astype(float)
    val_m = train_m & brand_split(p["brand"].where(train_m), 0.15, seed + 1)
    fit_m = train_m & ~val_m
    compare = {}
    for algo in ("xgboost", "lightgbm"):
        m_ = _fit_lgb(X, y, fit_m, val_m, algo)
        compare[algo] = {"val_r2": _metrics(y[val_m], m_.predict(X[val_m]))["r2"],
                         "test_r2": _metrics(y[test_m], m_.predict(X[test_m]))["r2"],
                         "train_r2": _metrics(y[fit_m], m_.predict(X[fit_m]))["r2"]}
        if algo == ALGO:
            model = m_
    log.info("algorithm comparison: %s", compare)

    test = _metrics(y[test_m], model.predict(X[test_m]))
    train_r = _metrics(y[fit_m], model.predict(X[fit_m]))
    # baseline: always predict the training mean
    base = _metrics(y[test_m], np.full(test_m.sum(), y[fit_m].mean()))

    # shuffled labels must collapse to ~0
    ys = pd.Series(np.random.default_rng(seed).permutation(y.to_numpy()), index=y.index)
    shuf = _fit_lgb(X, ys, fit_m, val_m)
    shuffle_r2 = _metrics(ys[test_m], shuf.predict(X[test_m]))["r2"]

    # per-city breakdown (largest cities)
    per_city = {}
    tp = p[test_m]
    pt = pd.Series(model.predict(X[test_m]), index=tp.index)
    for c in tp["city"].value_counts().index[:10]:
        i = tp.index[tp["city"] == c]
        if len(i) >= 100:
            per_city[c] = _metrics(y[i], pt[i])["r2"]

    unseen = None
    if not quick:
        cities = list(p["city"].value_counts().index[1:80])
        hold = set(np.random.default_rng(seed + 2).choice(cities, 25, replace=False))
        inh = p["city"].isin(hold).to_numpy()
        tr2 = train_m & ~inh
        lab2 = RatingLabeler(k).fit(p[tr2])
        y2 = lab2.transform(p)
        v2 = tr2 & brand_split(p["brand"].where(tr2), 0.15, seed + 3)
        m2 = _fit_lgb(X, y2, tr2 & ~v2, v2)
        te2 = test_m & inh
        unseen = _metrics(y2[te2], m2.predict(X[te2]))
        unseen["held_out_cities"] = sorted(hold)

    gap = train_r["r2"] - test["r2"]
    gates = {
        "r2_min": [test["r2"], ">=", 0.15, test["r2"] >= 0.15],
        "r2_gap_max": [round(gap, 4), "<=", 0.05, gap <= 0.05],
        "shuffle_r2_max": [shuffle_r2, "<=", 0.01, shuffle_r2 <= 0.01],
        "beats_baseline": [test["r2"] - base["r2"], ">=", 0.1, test["r2"] - base["r2"] >= 0.1],
    }
    report = {
        "type": "regression", "model_type": ALGO, "algorithm_comparison": compare, "label_source": "rating_shrunk_pct_within_source_city",
        "label_provenance": "independent (own rating is the label, never a feature)",
        "split": "grouped by brand (25% of brands held out)", "shrinkage_k": k, "k_cv": cv,
        **test, "train_r2": train_r["r2"], "r2_gap": round(gap, 4), "baseline_r2": base["r2"],
        "shuffle_r2": shuffle_r2, "n_train": int(fit_m.sum()), "n_val": int(val_m.sum()),
        "n_test": int(test_m.sum()), "n_places": int(len(p)), "n_cities": int(p["city"].nunique()),
        "n_features": len(feats), "per_city_r2": per_city, "unseen_cities": unseen,
        "r2_minus_baseline": round(test["r2"] - base["r2"], 4), "n_samples": test["n"],
        "target_mean": round(float(y[test_m].mean()), 3), "target_std": round(float(y[test_m].std()), 3),
        "gate_results": {k_: {"value": v[0], "required": f"{v[1]} {v[2]}", "pass": bool(v[3])}
                         for k_, v in gates.items()},
        "gates": {k_: {"value": v[0], "op": v[1], "threshold": v[2], "pass": bool(v[3])}
                  for k_, v in gates.items()},
        "gate_pass": all(v[3] for v in gates.values()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(time.time() - t0, 1),
    }
    save(model, lab, feats, vocab, report)
    return report


def save(model, lab, feats, vocab, report) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / BUNDLE, "wb") as fh:
        pickle.dump({"model": model, "labeler": lab, "features": feats, "cuisine_vocab": vocab,
                     "trained_at": report["generated_at"]}, fh)
    # merge into the shared reports so other tasks' results are kept
    mp = REPORTS_DIR / "metrics.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"tasks": {}}
    m.setdefault("tasks", {})[TASK] = report
    mp.write_text(json.dumps(m, indent=2, default=float), encoding="utf-8")
    gp = REPORTS_DIR / "gates.json"
    g = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {"gates": {}}
    g.setdefault("gates", {})[TASK] = {**report["gates"], "all_pass": report["gate_pass"]}
    gp.write_text(json.dumps(g, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
_BUNDLE_CACHE: Dict[str, dict] = {}


def load_bundle(models_dir: Path = MODELS_DIR) -> Optional[dict]:
    key = str(models_dir)
    if key not in _BUNDLE_CACHE:
        path = Path(models_dir) / BUNDLE
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            _BUNDLE_CACHE[key] = pickle.load(fh)
    return _BUNDLE_CACHE[key]


def score_places(places: pd.DataFrame, models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """quality_india (0-100) for every place in any city.

    Neighbour features are computed over the WHOLE frame passed in, so pass the
    full city (or country) listing, not a single place."""
    b = load_bundle(models_dir)
    if b is None:
        raise FileNotFoundError(f"{BUNDLE} missing - run python src/india_quality.py")
    lab: RatingLabeler = b["labeler"]
    df, _, _ = build_features(places, b["cuisine_vocab"])
    model_pct = pd.Series(np.clip(b["model"].predict(df[b["features"]].astype(float)), 0, 100),
                          index=df.index)
    n = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).clip(0, RATING_CAP)
    n = n.where(pd.to_numeric(df["stars"], errors="coerce").notna(), 0.0)
    obs = lab.observed_pct(df)
    w = n / (n + lab.k)
    quality = (w * obs.fillna(0) + (1 - w) * model_pct).where(obs.notna() | (n == 0), model_pct)
    return pd.DataFrame({"quality_india": quality.round(2), "model_pct": model_pct.round(2),
                         "observed_pct": obs.round(2), "rating_weight": w.round(3)}, index=places.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the unseen-city check")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = train(quick=a.quick)
    show = {k: r[k] for k in ["r2", "train_r2", "r2_gap", "spearman", "precision_at_top20pct",
                              "shuffle_r2", "shrinkage_k", "n_places", "n_cities", "n_test", "gate_pass"]}
    print(json.dumps(show, indent=2))
    if r["unseen_cities"]:
        print("unseen cities:", {k: r["unseen_cities"][k] for k in ["r2", "spearman", "n"]})
    print("per city R2:", r["per_city_r2"])


if __name__ == "__main__":
    # run through the importable module so the pickle references
    # india_quality.RatingLabeler, not __main__.RatingLabeler
    import india_quality
    india_quality.main()
