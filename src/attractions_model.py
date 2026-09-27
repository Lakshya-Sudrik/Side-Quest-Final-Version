"""Sightseeing hidden-gem model (15k Indian attractions, 1.48M reviews).

    python src/attractions_model.py          # build, train, evaluate, score all places

What it predicts (and why that is honest)
-----------------------------------------
Every place's reviews are split at random into two halves:
    half A -> everything the model is allowed to see (ratings, text, counts)
    half B -> the TARGET: the average rating that *other* visitors gave.
So the model answers a real question - "how will the next visitors rate this
place?" - and is scored against reviews it never saw. Places are split into
train / test by name (a place is never in both), and results are compared with
the obvious baseline "just use the half-A average rating".

Hidden gem = predicted quality in the top 30% AND the place is among the less
reviewed half of its town (exposure <= 0.5). Famous & good = top-30% quality
AND among the most reviewed 20% of its town.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

REVIEWS = ROOT / "data" / "attractions" / "Review_db.csv"
OUT_DIR = ROOT / "data" / "attractions"
SCORED = OUT_DIR / "attractions_scored.csv.gz"
MODEL = ROOT / "models" / "attractions_model.pkl"
REPORTS = ROOT / "reports"
TASK = "attractions_gem"
SEED = 42
PRIOR_K = 5.0          # pseudo-reviews toward the global mean for the half-A rating feature
MIN_REVIEWS = 4        # need >= 2 reviews on each side of the split to be evaluated
GOOD_QUANTILE = 0.70   # "gem quality" = top 30% of predicted future rating
log = logging.getLogger("attractions")

# review-text signals (share of a place's reviews that mention each)
LEXICON = {
    "tx_scenic": r"beautiful|scenic|breathtaking|stunning|serene|peaceful|calm|picturesque|lush|panoram",
    "tx_worth": r"must visit|must see|worth (?:a )?visit|worth it|highly recommend|don.t miss|do not miss",
    "tx_hidden": r"hidden|less crowded|offbeat|off beat|not many tourists|untouched|secluded|unexplored|few people",
    "tx_crowd": r"crowded|overcrowded|\brush\b|long queue|long line|too many people",
    "tx_dirty": r"dirty|filthy|garbage|litter|smell|stink|unclean|poorly maintained|not maintained",
    "tx_clean": r"\bclean\b|well maintained|well-maintained|neat",
    "tx_overpriced": r"overpriced|over priced|rip ?off|waste of money|too expensive|not worth",
    "tx_disappoint": r"disappoint|nothing special|boring|avoid|waste of time|not recommended",
    "tx_family": r"family|kids|children",
    "tx_unsafe": r"unsafe|danger|theft|pickpocket|harass|eve teas|\bscam|\btouts?\b|cheat|robbed|stolen|slippery|accident|drown",
    "tx_safe": r"safe for|safe place|feel safe|felt safe|very safe|\bsecure\b|police",
}
PLACE_TYPES = {
    "temple": r"temple|mandir|devasthan|kovil|koil|matha?\b|mutt|dham\b",
    "church_mosque": r"church|cathedral|basilica|mosque|masjid|dargah|gurudwara|gurdwara|monastery|gompa|stupa|vihar",
    "fort_palace": r"fort|qila|kila|palace|mahal|haveli|castle",
    "monument": r"tomb|maqbara|memorial|monument|gate|minar|ruins|caves?|stepwell|baori|vav\b",
    "water": r"lake|falls|waterfall|beach|river|dam|backwater|ghat|sagar|talav|tal\b|kund|spring",
    "hill_view": r"hill|peak|view ?point|point\b|valley|trek|top\b|sunset|sunrise",
    "nature_park": r"park|garden|sanctuary|zoo|forest|reserve|national park|botanical|bird",
    "museum": r"museum|gallery|planetarium|science|art",
    "market_mall": r"market|bazaar|mall|street|shopping|haat",
}


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]+", " ", regex=True).str.replace(
        r"\s+", " ", regex=True).str.strip()


# ---------------------------------------------------------------------------
# 1. reviews -> per-review signals
# ---------------------------------------------------------------------------
CACHE = OUT_DIR / "_reviews_cache.pkl"


def load_reviews(path: Path = REVIEWS) -> pd.DataFrame:
    if CACHE.exists() and CACHE.stat().st_mtime > path.stat().st_mtime:
        return pd.read_pickle(CACHE)
    r = _load_reviews(path)
    r.to_pickle(CACHE)
    return r


def _load_reviews(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, usecols=["City", "Place", "Rating", "Raw_Review"], low_memory=False)
    r["Raw_Review"] = r["Raw_Review"].fillna("").astype(str)
    n0 = len(r)
    # the same review text posted twice for a place would sit in BOTH halves -> leak
    r = r.drop_duplicates(["City", "Place", "Raw_Review"]).reset_index(drop=True)
    log.info("reviews: %d (%d exact duplicates removed)", len(r), n0 - len(r))
    r["place_id"] = (r["City"].astype(str) + "|" + r["Place"].astype(str))
    t = r["Raw_Review"].str.lower()
    r["words"] = t.str.count(r"\S+").astype(np.float32)
    for k, pat in LEXICON.items():
        r[k] = t.str.contains(pat, regex=True).astype(np.float32)
    r["Rating"] = pd.to_numeric(r["Rating"], errors="coerce").astype(np.float32)
    r = r[r["Rating"].between(1, 5)]
    rng = np.random.default_rng(SEED)
    r["half"] = rng.integers(0, 2, len(r)).astype(np.int8)
    return r


def aggregate(r: pd.DataFrame, prior: float) -> pd.DataFrame:
    """Per-place features from the given reviews (half A in training, all reviews at serving)."""
    g = r.groupby("place_id", observed=True)
    f = pd.DataFrame({
        "n": g.size().astype(float),
        "rating_mean": g["Rating"].mean(),
        "rating_std": g["Rating"].std().fillna(0),
        "share5": g["Rating"].apply(lambda s: (s == 5).mean()),
        "share_low": g["Rating"].apply(lambda s: (s <= 2).mean()),
        "words_mean": g["words"].mean(),
        "share_detailed": g["words"].apply(lambda s: (s >= 40).mean()),
    })
    f["rating_shrunk"] = (f["rating_mean"] * f["n"] + PRIOR_K * prior) / (f["n"] + PRIOR_K)
    f["log_n"] = np.log1p(f["n"])
    for k in LEXICON:
        f[k] = g[k].mean()
    return f


def place_meta(places: pd.Series) -> pd.DataFrame:
    """City / name derived columns for a place_id index."""
    pid = pd.Series(places.astype(str).to_numpy(), index=places.to_numpy())
    city = pid.str.split("|").str[0]
    name = pid.str.split("|", n=1).str[1]
    m = pd.DataFrame({"city": city.values, "place": name.values}, index=places)
    low = _norm(m["place"])
    for k, pat in PLACE_TYPES.items():
        m["pt_" + k] = low.str.contains(pat, regex=True).astype(float)
    m["name_group"] = low
    return m


# ---------------------------------------------------------------------------
# 2. text model (out-of-fold, so its prediction is an honest feature)
# ---------------------------------------------------------------------------
def _texts(r: pd.DataFrame, ids) -> pd.Series:
    t = r.groupby("place_id", observed=True)["Raw_Review"].apply(lambda s: " ".join(s.astype(str).str[:600]))
    return t.reindex(ids).fillna("")


def text_oof(texts_tr, y_tr, w_tr, groups_tr, texts_te, n_folds=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    def make():
        return TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                               stop_words="english", dtype=np.float32)
    oof = np.zeros(len(texts_tr))
    rng = np.random.default_rng(SEED)
    ug = np.unique(groups_tr)
    fold_of = dict(zip(ug, rng.integers(0, n_folds, len(ug))))
    folds = np.array([fold_of[g] for g in groups_tr])
    ym = float(np.average(y_tr, weights=w_tr))
    for k in range(n_folds):
        tr, va = folds != k, folds == k
        vec = make(); X = vec.fit_transform(texts_tr[tr])
        m = Ridge(alpha=3.0).fit(X, y_tr[tr] - ym, sample_weight=w_tr[tr])
        oof[va] = m.predict(vec.transform(texts_tr[va])) + ym
    vec = make(); X = vec.fit_transform(texts_tr)
    m = Ridge(alpha=3.0).fit(X, y_tr - ym, sample_weight=w_tr)
    return oof, m.predict(vec.transform(texts_te)) + ym, (vec, m, ym)


# ---------------------------------------------------------------------------
# 3. train + evaluate
# ---------------------------------------------------------------------------
ALGO = "xgboost"   # final model; LightGBM is trained on the same split for comparison
XGB = dict(n_estimators=3000, learning_rate=0.02, max_depth=3, min_child_weight=40, subsample=0.8,
           colsample_bytree=0.6, reg_lambda=20.0, reg_alpha=1.0, tree_method="hist", random_state=SEED)
LGB = dict(n_estimators=3000, learning_rate=0.02, num_leaves=7, min_child_samples=120,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0,
           verbose=-1, random_state=SEED)


def _metrics(y, p, w=None) -> Dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    return {"r2": round(float(r2_score(y, p, sample_weight=w)), 4),
            "spearman": round(float(spearmanr(y, p)[0]), 4),
            "mae": round(float(np.average(np.abs(y - p), weights=w)), 4), "n": int(len(y))}


def _gem_classification(y_future, pred, baseline, exposure, good_cut_y):
    """Among LESS-REVIEWED test places: does 'predicted top-30%' find places whose
    FUTURE visitors really rate them in the top 30%?"""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    hid = exposure <= 0.5
    truth = (y_future >= good_cut_y)[hid]
    out = {}
    for name, score in (("model", pred), ("baseline_rating_only", baseline)):
        s = score[hid]
        cut = np.quantile(score, GOOD_QUANTILE)  # same threshold rule as serving
        guess = s >= cut
        out[name] = {
            "accuracy": round(float(accuracy_score(truth, guess)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(truth, guess)), 4),
            "precision": round(float(precision_score(truth, guess, zero_division=0)), 4),
            "recall": round(float(recall_score(truth, guess)), 4),
            "f1": round(float(f1_score(truth, guess)), 4),
            "roc_auc": round(float(roc_auc_score(truth, s)), 4),
        }
    out["n_hidden_test_places"] = int(hid.sum())
    out["share_truly_good"] = round(float(truth.mean()), 4)
    return out


def exposure_pct(n_reviews: pd.Series, city: pd.Series, min_city: int = 8) -> pd.Series:
    """Percentile of review count within the town; small towns use the national ranking."""
    nat = n_reviews.rank(pct=True)
    loc = n_reviews.groupby(city).rank(pct=True)
    size = city.map(city.value_counts())
    return loc.where(size >= min_city, nat)


def train() -> Dict:
    import lightgbm as lgb

    t0 = time.time()
    r = load_reviews()
    prior = float(r["Rating"].mean())
    A, B = r[r["half"] == 0], r[r["half"] == 1]
    fa = aggregate(A, prior)
    gb = B.groupby("place_id", observed=True)["Rating"]
    tgt = pd.DataFrame({"y": gb.mean(), "nB": gb.size().astype(float)})
    d = fa.join(tgt, how="inner")
    d = d[(d["n"] >= 2) & (d["nB"] >= 2)]
    meta = place_meta(pd.Series(d.index))
    meta.index = d.index
    d = d.join(meta)
    # town-level context from half A of all places (no half-B information)
    d["city_places"] = d.groupby("city")["n"].transform("size").astype(float)
    d["city_rating"] = d.groupby("city")["rating_mean"].transform("mean")
    d["rating_vs_city"] = d["rating_mean"] - d["city_rating"]
    n_total = r.groupby("place_id", observed=True).size().reindex(d.index).astype(float)
    d["exposure"] = exposure_pct(n_total, d["city"])
    log.info("places with >=2 reviews in each half: %d", len(d))

    # split places by NAME (same-name places in two towns stay together)
    rng = np.random.default_rng(SEED)
    ug = d["name_group"].unique()
    test_g = set(rng.choice(ug, int(0.2 * len(ug)), replace=False))
    te = d["name_group"].isin(test_g).to_numpy()
    tr = ~te
    w = np.minimum(d["nB"].to_numpy(), 30.0)  # noisy targets (2-3 reviews) count less
    y = d["y"].to_numpy()

    texts = _texts(A, d.index)
    oof, te_text, text_model = text_oof(texts[tr].to_numpy(), y[tr], w[tr],
                                        d["name_group"].to_numpy()[tr], texts[te].to_numpy())
    d["text_score"] = np.nan
    d.loc[tr, "text_score"] = oof
    d.loc[te, "text_score"] = te_text

    feats = (["rating_shrunk", "rating_mean", "rating_std", "share5", "share_low", "log_n",
              "words_mean", "share_detailed", "city_places", "city_rating", "rating_vs_city",
              "text_score"] + list(LEXICON) + ["pt_" + k for k in PLACE_TYPES])
    X = d[feats].astype(float)
    # inner validation (by name) for early stopping
    vg = set(rng.choice(d.loc[tr, "name_group"].unique(), int(0.15 * len(set(d.loc[tr, "name_group"]))), replace=False))
    va = tr & d["name_group"].isin(vg).to_numpy()
    fit = tr & ~va
    def fit_model(algo, yy):
        """Same features, same split, same early-stopping rule for both algorithms."""
        if algo == "xgboost":
            import xgboost as xgb
            m = xgb.XGBRegressor(**XGB, early_stopping_rounds=150)
            m.fit(X[fit], yy[fit], sample_weight=w[fit], eval_set=[(X[va], yy[va])],
                  sample_weight_eval_set=[w[va]], verbose=False)
            return m, int(m.best_iteration + 1)
        m = lgb.LGBMRegressor(**LGB)
        m.fit(X[fit], yy[fit], sample_weight=w[fit], eval_set=[(X[va], yy[va])],
              eval_sample_weight=[w[va]], callbacks=[lgb.early_stopping(150, verbose=False)])
        return m, int(m.best_iteration_ or LGB["n_estimators"])

    base_te = d["rating_shrunk"].to_numpy()[te]
    base = _metrics(y[te], base_te, w[te])
    compare = {}
    for algo in ("xgboost", "lightgbm"):
        m_, it_ = fit_model(algo, y)
        compare[algo] = {"val": _metrics(y[va], m_.predict(X[va]), w[va]),
                         "test": _metrics(y[te], m_.predict(X[te]), w[te]),
                         "train_r2": _metrics(y[fit], m_.predict(X[fit]), w[fit])["r2"], "iterations": it_}
        if algo == ALGO:
            model, best_it = m_, it_
    log.info("algorithm comparison (validation R2): %s",
             {k: v["val"]["r2"] for k, v in compare.items()})
    p_te, p_fit = model.predict(X[te]), model.predict(X[fit])
    test = _metrics(y[te], p_te, w[te])
    train_m = _metrics(y[fit], p_fit, w[fit])
    # shuffled target must give ~0
    ys = rng.permutation(y)
    sh, _ = fit_model(ALGO, ys)
    shuffle_r2 = _metrics(y[te], sh.predict(X[te]), w[te])["r2"]

    good_cut = float(np.quantile(y[tr], GOOD_QUANTILE))
    gem = _gem_classification(y[te], p_te, base_te, d["exposure"].to_numpy()[te], good_cut)

    by_size = {}
    nt = d["n"].to_numpy()[te] + d["nB"].to_numpy()[te]
    for lo, hi, lab in ((4, 10, "4-9 reviews"), (10, 40, "10-39"), (40, 200, "40-199"), (200, 1e9, "200+")):
        m = (nt >= lo) & (nt < hi)
        if m.sum() > 50:
            by_size[lab] = {"model": _metrics(y[te][m], p_te[m], w[te][m]),
                            "baseline": _metrics(y[te][m], base_te[m], w[te][m])}

    gap = train_m["r2"] - test["r2"]
    gates = {"r2_beats_baseline": test["r2"] > base["r2"], "r2_gap_max_0.05": gap <= 0.05,
             "shuffle_r2_max_0.01": shuffle_r2 <= 0.01,
             "gem_f1_beats_baseline": gem["model"]["f1"] >= gem["baseline_rating_only"]["f1"]}
    report = {
        "task": TASK, "model_type": f"{ALGO} + tfidf ridge (out-of-fold)", "algorithm_comparison": compare,
        "target": "average rating of a random held-out half of each place's reviews",
        "split": "20% of place names held out; features only from the other half of reviews",
        "test": test, "train_r2": train_m["r2"], "r2_gap": round(gap, 4),
        "baseline_half_A_rating": base, "shuffle_r2": shuffle_r2,
        "hidden_gem_classification": gem, "good_cut_rating": round(good_cut, 3),
        "by_review_count": by_size, "gates": {k: bool(v) for k, v in gates.items()},
        "gate_pass": bool(all(gates.values())), "n_places_modelled": int(len(d)),
        "n_train": int(fit.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "n_features": len(feats), "best_iteration": best_it,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round(time.time() - t0, 1),
    }
    MODEL.parent.mkdir(exist_ok=True)
    with open(MODEL, "wb") as fh:
        pickle.dump({"model": model, "features": feats, "text_model": text_model, "prior": prior,
                     "good_quantile": GOOD_QUANTILE, "trained_at": report["generated_at"]}, fh)
    _write_report(report)
    score_all(r, prior, model, feats, text_model)
    return report


def _write_report(rep: Dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    mp = REPORTS / "metrics.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"tasks": {}}
    m.setdefault("tasks", {})[TASK] = {
        **rep, "type": "regression", "r2": rep["test"]["r2"], "spearman": rep["test"]["spearman"],
        "rmse": None, "mae": rep["test"]["mae"], "r2_minus_baseline": round(rep["test"]["r2"] - rep["baseline_half_A_rating"]["r2"], 4),
        "target_mean": None, "target_std": None, "n_samples": rep["test"]["n"],
        "label_provenance": "independent (held-out half of reviews)",
        "gate_results": {k: {"value": v, "required": "true", "pass": v} for k, v in rep["gates"].items()},
    }
    mp.write_text(json.dumps(m, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. score every place with ALL its reviews (serving table)
# ---------------------------------------------------------------------------
def score_all(r: pd.DataFrame, prior: float, model, feats, text_model) -> pd.DataFrame:
    import geo_india as gi

    f = aggregate(r, prior)
    meta = place_meta(pd.Series(f.index)); meta.index = f.index
    f = f.join(meta)
    f["city_places"] = f.groupby("city")["n"].transform("size").astype(float)
    f["city_rating"] = f.groupby("city")["rating_mean"].transform("mean")
    f["rating_vs_city"] = f["rating_mean"] - f["city_rating"]
    vec, ridge, ym = text_model
    f["text_score"] = ridge.predict(vec.transform(_texts(r, f.index).to_numpy())) + ym
    f["quality_pred"] = np.clip(model.predict(f[feats].astype(float)), 1, 5)
    # places with a single review are too thin to call a gem
    f.loc[f["n"] < 2, "quality_pred"] = np.nan
    f["quality_pct"] = (f["quality_pred"].rank(pct=True) * 100).round(1)
    f["exposure_pct"] = exposure_pct(f["n"], f["city"]).round(3)
    # locations: exact site when GeoNames has it, else the town centre
    towns = {c: gi.town_coords(c) for c in f["city"].unique()}
    lat, lon, prec = [], [], []
    for c, p in zip(f["city"], f["place"]):
        t = towns.get(c)
        if t is None:
            lat.append(np.nan); lon.append(np.nan); prec.append("unknown"); continue
        s = gi.site_coords(p, t)
        if s:
            lat.append(s[0]); lon.append(s[1]); prec.append("exact_site")
        else:
            lat.append(t[0]); lon.append(t[1]); prec.append("town_centre")
    f["lat"], f["lon"], f["location_precision"] = lat, lon, prec
    types = ["pt_" + k for k in PLACE_TYPES]
    f["place_type"] = f[types].idxmax(axis=1).str[3:].where(f[types].max(axis=1) > 0, "attraction")
    keep = ["city", "place", "lat", "lon", "location_precision", "place_type", "n", "rating_mean",
            "quality_pred", "quality_pct", "exposure_pct", "tx_unsafe", "tx_safe", "tx_hidden",
            "tx_scenic", "tx_crowd"]
    out = f[keep].reset_index().rename(columns={"index": "place_id", "n": "n_reviews"})
    out.to_csv(SCORED, index=False, compression="gzip")
    log.info("scored %d places -> %s (%.0f%% located, %.0f%% exact site)", len(out), SCORED.name,
             100 * out["lat"].notna().mean(), 100 * (out["location_precision"] == "exact_site").mean())
    return out


def relocate() -> pd.DataFrame:
    """Recompute sight locations in the scored table (after gazetteer fixes) - no retraining."""
    import geo_india as gi
    gi._towns.cache_clear(); gi._anchors.cache_clear()
    out = pd.read_csv(SCORED)
    towns = {c: gi.town_coords(c) for c in out["city"].unique()}
    lat, lon, prec = [], [], []
    for c, p in zip(out["city"], out["place"]):
        t = towns.get(c)
        if t is None:
            lat.append(np.nan); lon.append(np.nan); prec.append("unknown"); continue
        s_ = gi.site_coords(p, t)
        if s_:
            lat.append(s_[0]); lon.append(s_[1]); prec.append("exact_site")
        else:
            lat.append(t[0]); lon.append(t[1]); prec.append("town_centre")
    moved = (np.hypot(np.array(lat) - out["lat"].to_numpy(), np.array(lon) - out["lon"].to_numpy()) > 0.5).sum()
    out["lat"], out["lon"], out["location_precision"] = lat, lon, prec
    out.to_csv(SCORED, index=False, compression="gzip")
    log.info("relocated %d places (%d moved > ~50 km)", len(out), int(moved))
    return out


def main():
    import sys as _sys
    if "--relocate" in _sys.argv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        relocate()
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rep = train()
    show = {"test_r2": rep["test"]["r2"], "baseline_r2 (half-A rating only)": rep["baseline_half_A_rating"]["r2"],
            "train_r2": rep["train_r2"], "gap": rep["r2_gap"], "spearman": rep["test"]["spearman"],
            "shuffle_r2": rep["shuffle_r2"], "gate_pass": rep["gate_pass"]}
    print(json.dumps(show, indent=2))
    print(json.dumps(rep["hidden_gem_classification"], indent=2))
    print(json.dumps(rep["by_review_count"], indent=1))


if __name__ == "__main__":
    import attractions_model
    attractions_model.main()
