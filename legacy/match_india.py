"""SOLO MATCHMAKING - Phase 4.

Hybrid collaborative matching trained on Indianized OkCupid-style profiles
(59,946 real profiles) with ground truth from essay embedding similarity.

Features (all available at inference for any two users):
- age/height/income/education compatibility
- shared interest count + Jaccard (interest taxonomy from essays)
- language match (Indianized: Hindi/Tamil/... + English)
- shared values: diet, drinks, smokes, religion, education
- same-city proximity
- essay embedding cosine similarity (MiniLM)

Output: models/matchmaking.pkl, reports/match_metrics.json
POST /api/ml/match consumes scoring_model().
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # skip hub checks (30-60s saved per run)

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("match_india")

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Indianization: US locations/languages -> Indian equivalents
# ---------------------------------------------------------------------------
INDIAN_CITIES = [
    "Mumbai, Maharashtra", "Delhi, Delhi", "Bangalore, Karnataka",
    "Hyderabad, Telangana", "Chennai, Tamil Nadu", "Pune, Maharashtra",
    "Kolkata, West Bengal", "Ahmedabad, Gujarat", "Jaipur, Rajasthan",
    "Lucknow, Uttar Pradesh", "Surat, Gujarat", "Kanpur, Uttar Pradesh",
    "Nagpur, Maharashtra", "Indore, Madhya Pradesh", "Bhopal, Madhya Pradesh",
    "Patna, Bihar", "Vadodara, Gujarat", "Ludhiana, Punjab", "Agra, Uttar Pradesh",
    "Nashik, Maharashtra", "Faridabad, Haryana", "Meerut, Uttar Pradesh",
    "Rajkot, Gujarat", "Varanasi, Uttar Pradesh", "Srinagar, Jammu & Kashmir",
    "Amritsar, Punjab", "Ranchi, Jharkhand", "Coimbatore, Tamil Nadu",
    "Kochi, Kerala", "Chandigarh, Chandigarh", "Mysore, Karnataka",
    "Thiruvananthapuram, Kerala", "Gurgaon, Haryana", "Noida, Uttar Pradesh",
    "Visakhapatnam, Andhra Pradesh", "Madurai, Tamil Nadu", "Dehradun, Uttarakhand",
]
CITY_WEIGHTS = np.array(
    [12.5, 11.0, 8.5, 7.5, 7.0, 5.5, 4.5, 4.0, 3.5, 3.2, 3.0, 2.8,
     2.5, 2.2, 2.0, 1.9, 1.7, 1.6, 1.5, 1.4, 1.3, 1.3, 1.2, 1.2, 0.8,
     1.1, 0.9, 1.3, 1.2, 1.0, 0.9, 0.8, 2.5, 2.4, 1.6, 1.2, 0.9],
    dtype=float,
)
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

INDIAN_LANGS = [
    "hindi", "bengali", "telugu", "marathi", "tamil", "gujarati", "kannada",
    "malayalam", "punjabi", "urdu", "odia", "assamese",
]

# interest taxonomy (keywords found in essays -> category)
INTEREST_TAXONOMY = {
    "music": ["music", "song", "guitar", "band", "concert", "spotify", "sing", "harmonica", "drum"],
    "travel": ["travel", "trip", "backpack", "wanderlust", "explore", "trek", "hike", "vacation", "passport"],
    "food": ["cook", "foodie", "cuisine", "baking", "restaurant", "eat", "recipe", "coffee", "chai", "pizza"],
    "fitness": ["gym", "fitness", "run", "marathon", "yoga", "sport", "swim", "cycle", "workout"],
    "movies": ["movie", "film", "cinema", "netflix", "series", "bollywood", "hollywood", "director"],
    "reading": ["book", "read", "novel", "poetry", "author", "literature", "library"],
    "tech": ["tech", "code", "programming", "startup", "gadget", "computer", "ai", "geek", "nerd"],
    "art": ["art", "paint", "draw", "photograph", "design", "sketch", "museum", "creative"],
    "outdoors": ["nature", "beach", "mountain", "camp", "outdoor", "forest", "bird", "garden"],
    "gaming": ["game", "gaming", "chess", "video game", "board game", "playstation"],
    "volunteering": ["volunteer", "charity", "ngo", "social cause", "donate", "teach"],
    "spirituality": ["spiritual", "meditat", "temple", "church", "mosque", "god", "astrology", "karma"],
    "pets": ["dog", "cat", "pet", "puppy", "animal"],
    "nightlife": ["party", "bar", "club", "dance", "cocktail", "pub"],
    "fashion": ["fashion", "style", "clothes", "shopping", "outfit"],
}

# Indian city coordinates for distance feature
CITY_COORDS = {
    "mumbai": (19.076, 72.877), "delhi": (28.613, 77.209),
    "bangalore": (12.971, 77.594), "hyderabad": (17.385, 78.486),
    "chennai": (13.082, 80.270), "pune": (18.520, 73.856),
    "kolkata": (22.572, 88.363), "ahmedabad": (23.022, 72.571),
    "jaipur": (26.912, 75.787), "lucknow": (26.846, 80.946),
}


def _essay_text(row) -> str:
    parts = [str(row.get(f"essay{i}", "") or "") for i in range(10)]
    return " ".join(p for p in parts if p and p != "nan")


def indianize(profiles: pd.DataFrame) -> pd.DataFrame:
    """Map US locations to Indian cities, inject Indian languages."""
    rng = np.random.default_rng(SEED)
    n = len(profiles)
    profiles = profiles.copy()
    profiles["city"] = rng.choice(INDIAN_CITIES, size=n, p=CITY_WEIGHTS)

    # languages: keep original + add Hindi + 0-2 regional
    def mk_langs(s: str) -> str:
        base = ["english"]
        if isinstance(s, str) and s.strip():
            # keep only natural languages (drop c++/python etc. from 'speaks')
            langs = re.findall(r"[a-z]{4,}", s.lower())
            keep = [l for l in langs if l in INDIAN_LANGS or l == "english"][:2]
            base += keep
        if "hindi" not in base:
            base.append("hindi")
        if rng.random() < 0.35:
            extra = rng.choice([l for l in INDIAN_LANGS if l != "hindi"])
            if extra not in base:
                base.append(extra)
        return ", ".join(base)

    profiles["speaks"] = profiles["speaks"].apply(mk_langs)
    return profiles


def extract_interests(text: str) -> set[str]:
    low = (text or "").lower()
    found = set()
    for cat, kws in INTEREST_TAXONOMY.items():
        if any(k in low for k in kws):
            found.add(cat)
    return found


# ---------------------------------------------------------------------------
# build pairs + features
# ---------------------------------------------------------------------------
def build_dataset(profiles: pd.DataFrame, n_pairs: int = 400_000) -> tuple[pd.DataFrame, np.ndarray, object]:
    from sentence_transformers import SentenceTransformer

    # sample: this CPU embeds ~36 docs/s -> 25K = ~12 min one-time, cached after
    MAX_PROFILES = 25_000
    if len(profiles) > MAX_PROFILES:
        profiles = profiles.sample(n=MAX_PROFILES, random_state=SEED).reset_index(drop=True)
        logger.info(f"  sampled {MAX_PROFILES} profiles for embedding speed")

    logger.info("  loading MiniLM for essay embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    essays = profiles["essay_text"].tolist()
    emb_path = ROOT / "data" / "cache" / f"essay_emb_{MAX_PROFILES}.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        if emb.shape[0] == len(essays):
            logger.info(f"  loaded cached embeddings {emb.shape} from {emb_path.name}")
        else:
            emb = None
    else:
        emb = None
    if emb is None:
        logger.info(f"  embedding {len(essays)} essays... (~10-12 min, cached next time)")
        t0 = time.time()
        emb = model.encode(
            essays, batch_size=256, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        logger.info(f"    embedded in {time.time()-t0:.0f}s shape={emb.shape}")
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb)

    interests = [extract_interests(e) for e in essays]
    profiles = profiles.reset_index(drop=True)
    profiles["_interests"] = interests
    profiles["_idx"] = np.arange(len(profiles))

    rng = np.random.default_rng(SEED)
    logger.info(f"  sampling {n_pairs} pairs...")

    # pair sampling: half random, half curated (same interest / nearby age)
    n_half = n_pairs // 2
    i0 = rng.integers(0, len(profiles), size=n_pairs)
    j0 = rng.integers(0, len(profiles), size=n_pairs)
    # curated: same top interest
    by_int: dict[str, list[int]] = {}
    for idx, ints in enumerate(interests):
        for it in ints:
            by_int.setdefault(it, []).append(idx)
    cat_pool = list(by_int.keys())
    for k in range(n_half):
        c = cat_pool[rng.integers(0, len(cat_pool))]
        members = by_int[c]
        i0[n_half + k] = members[rng.integers(0, len(members))]
        j0[n_half + k] = members[rng.integers(0, len(members))]

    diff = i0 != j0
    i0, j0 = i0[diff], j0[diff]
    logger.info(f"  valid pairs: {len(i0)}")

    A = profiles.iloc[i0].reset_index(drop=True)
    B = profiles.iloc[j0].reset_index(drop=True)

    # ---- features
    feats = pd.DataFrame()
    feats["age_diff"] = (A["age"].values - B["age"].values).astype(float)
    feats["age_sim"] = 1.0 - np.clip(feats["age_diff"].abs() / 30.0, 0, 1)
    feats["age_both_adult"] = ((A["age"].values >= 21) & (B["age"].values >= 21)).astype(float)

    h_diff = (A["height"].fillna(65).values - B["height"].fillna(65).values).astype(float)
    feats["height_diff"] = h_diff
    feats["income_diff"] = (A["income"].fillna(0).values - B["income"].fillna(0).values).astype(float)

    # orientation compatibility
    def orient_ok(a_sex, a_or, b_sex, b_or):
        # simplistic: straight/straight, gay/gay, bi anything
        if a_or == "gay" and b_or == "gay":
            return a_sex != b_sex or a_sex == b"b"
        if "straight" in (a_or, b_or) and a_sex != b_sex:
            return 1
        if "bi" in (a_or, b_or):
            return 1
        return int(a_or == b_or)

    sexes = A["sex"].astype(str).values, B["sex"].astype(str).values
    ors = A["orientation"].astype(str).values, B["orientation"].astype(str).values
    feats["orientation_ok"] = np.array([
        orient_ok(sexes[0][i], ors[0][i], sexes[1][i], ors[1][i])
        for i in range(len(A))
    ], dtype=float)

    # interests
    ia, ib = A["_interests"], B["_interests"]
    shared = np.array([len(a & b) for a, b in zip(ia, ib)], dtype=float)
    union = np.array([len(a | b) or 1 for a, b in zip(ia, ib)], dtype=float)
    feats["shared_interests"] = shared
    feats["interest_jaccard"] = shared / union
    feats["both_no_interests"] = ((shared == 0) & (union == 1)).astype(float)

    # language match
    def langs(s):
        return {x.strip() for x in str(s).split(",") if x.strip()}
    la = A["speaks"].apply(langs)
    lb = B["speaks"].apply(langs)
    feats["lang_overlap"] = np.array([len(a & b) for a, b in zip(la, lb)], dtype=float)
    feats["has_hindi"] = np.array([int("hindi" in a) for a in la], dtype=float)

    # shared values (categorical equality, neutral if missing)
    for col in ["diet", "drinks", "smokes", "religion", "education", "ethnicity"]:
        a = A[col].fillna("").astype(str).str.lower().values
        b = B[col].fillna("").astype(str).str.lower().values
        both_known = (a != "") & (b != "")
        feats[f"same_{col}"] = (a == b).astype(float) * both_known
        feats[f"{col}_known"] = both_known.astype(float)

    # same city
    feats["same_city"] = (A["city"].values == B["city"].values).astype(float)
    dist = []
    for ca, cb in zip(A["city"], B["city"]):
        ka, kb = ca.split(",")[0].lower().strip(), cb.split(",")[0].lower().strip()
        pa, pb = CITY_COORDS.get(ka), CITY_COORDS.get(kb)
        if pa and pb:
            d = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) * 111.0
        else:
            d = 0.0 if ka == kb else 800.0
        dist.append(d)
    feats["city_distance_km"] = dist

    # embedding cosine
    cos = np.sum(emb[i0] * emb[j0], axis=1)
    feats["essay_cosine"] = cos

    # ---- ground truth: similarity of personalities (embedding + interests + values)
    interest_j = feats["interest_jaccard"].values
    val_score = (
        1.0 * feats["same_diet"] + 1.0 * feats["same_drinks"] + 1.0 * feats["same_smokes"]
        + 0.5 * feats["same_education"] + 0.5 * feats["same_religion"]
    )
    age_pen = np.clip((feats["age_diff"].abs().values - 8) / 15.0, 0, 1)
    label_f = (
        0.55 * np.clip(cos, 0, 1)          # personality alignment
        + 0.20 * interest_j                 # shared interests
        + 0.15 * val_score                  # lifestyle values
        + 0.10 * (feats["same_city"].values) # proximity
        - 0.15 * age_pen                     # age-gap penalty
    )
    # normalize to 0..1 then binary label at median-biased threshold
    label_f = (label_f - label_f.min()) / (label_f.max() - label_f.min() + 1e-9)
    y = (label_f >= 0.62).astype(int)
    feats["match_score_gt"] = label_f  # ground-truth score kept for R² eval
    logger.info(f"  label balance: match={y.mean()*100:.1f}% ({y.sum()}/{len(y)})")

    return feats, y, model


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train_matchmaking() -> dict:
    import joblib
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, r2_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier, XGBRegressor

    logger.info("=" * 60)
    logger.info("PHASE 4: MATCHMAKING MODEL")
    logger.info("=" * 60)

    p = pd.read_parquet(ROOT / "data" / "raw" / "india" / "dating_profiles.parquet")
    logger.info(f"  loaded {len(p)} profiles")
    # parquet Arrow dtypes: coerce numerics explicitly
    for col in ["age", "height", "income"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    profiles = indianize(p)
    profiles["essay_text"] = profiles.apply(_essay_text, axis=1)
    # drop empty essays (no personality signal)
    keep = profiles["essay_text"].str.len() > 40
    profiles = profiles[keep].reset_index(drop=True)
    logger.info(f"  {len(profiles)} profiles with essays")

    feats, y, model = build_dataset(profiles, n_pairs=400_000)
    X = feats.drop(columns=["match_score_gt"])

    # 60/20/20 train / val(for early stopping) / test(final metrics only)
    X_trv, X_te, y_trv, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=0.25, random_state=SEED, stratify=y_trv)
    gt_tr = feats.loc[X_tr.index, "match_score_gt"]
    gt_va = feats.loc[X_va.index, "match_score_gt"]
    gt_te = feats.loc[X_te.index, "match_score_gt"]

    logger.info(f"  splits: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # Robustness masking (train only): real demo users rarely fill diet/religion/
    # education/etc, so all-zero flag rows must be in-distribution. Model learns
    # to fall back to essay cosine + interests + city when lifestyle fields missing.
    X_tr_full = X_tr.copy()
    MASK_COLS = ["diet", "drinks", "smokes", "religion", "education", "ethnicity"]
    rng = np.random.default_rng(SEED)
    mode = rng.random(len(X_tr))
    drop = np.zeros((len(X_tr), len(MASK_COLS)), dtype=bool)
    drop[mode < 0.15] = True  # 15%: ALL lifestyle flags hidden
    partial = (mode >= 0.15) & (mode < 0.40)
    drop[partial] = rng.random((int(partial.sum()), len(MASK_COLS))) < 0.45  # 25%: subset hidden
    arr = X_tr.to_numpy(copy=True)
    for k, c in enumerate(MASK_COLS):
        arr[drop[:, k], X_tr.columns.get_loc(f"same_{c}")] = 0.0
        arr[drop[:, k], X_tr.columns.get_loc(f"{c}_known")] = 0.0
    X_tr = pd.DataFrame(arr, columns=X_tr.columns, index=X_tr.index)
    logger.info(f"  masked lifestyle flags on {int(drop.any(axis=1).sum())}/{len(X_tr)} train rows")

    logger.info("  training XGBClassifier (early stopping on val)...")
    clf = XGBClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.5,
        min_child_weight=20,         objective="binary:logistic", eval_metric="logloss",
        random_state=SEED, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=50,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        verbose=False,
    )
    best_iter = getattr(clf, "best_iteration", None)
    logger.info(f"    best_iteration={best_iter} (of 2000)")

    def _proba(model, X):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            return model.predict(X)

    proba = _proba(clf, X_te)
    pred = (proba >= 0.5).astype(int)
    # regressor head for continuous match score (R² gate)
    logger.info("  training XGBRegressor (early stopping on val)...")
    reg = XGBRegressor(
        n_estimators=2000, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
        random_state=SEED, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=50,
    )
    reg.fit(
        X_tr, gt_tr,
        eval_set=[(X_tr, gt_tr), (X_va, gt_va)],
        verbose=False,
    )
    gt_pred = reg.predict(X_te)

    # train-vs-test gap (anti-overfit check; full-info train copy for fair gap)
    proba_tr = _proba(clf, X_tr_full)
    train_acc = float(accuracy_score(y_tr, (proba_tr >= 0.5).astype(int)))
    test_acc = float(accuracy_score(y_te, pred))
    acc_gap = train_acc - test_acc
    gt_pred_tr = reg.predict(X_tr_full)
    train_r2 = float(r2_score(gt_tr, gt_pred_tr))
    test_r2 = float(r2_score(gt_te, gt_pred))
    r2_gap = train_r2 - test_r2

    metrics = {
        "accuracy": test_acc,
        "train_accuracy": train_acc,
        "accuracy_gap": float(acc_gap),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, proba)),
        "score_r2": test_r2,
        "train_score_r2": train_r2,
        "score_r2_gap": float(r2_gap),
        "n_pairs": int(len(y)), "n_features": int(X.shape[1]),
        "n_estimators_fitted": int(best_iter + 1) if best_iter else None,
        "model": "XGBClassifier + XGBRegressor (early-stopped, depth<=6)",
    }
    logger.info("  TEST METRICS: " + ", ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
    # anti-overfit gates: performance + small generalization gap
    gate = (
        metrics["accuracy"] >= 0.80
        and metrics["score_r2"] >= 0.60
        and acc_gap <= 0.05
        and r2_gap <= 0.08
    )
    metrics["gate_pass"] = bool(gate)
    metrics["overfit_check"] = {
        "acc_gap<=0.05": bool(acc_gap <= 0.05),
        "r2_gap<=0.08": bool(r2_gap <= 0.08),
    }
    logger.info(f"  GATE (acc>=0.80, R2>=0.60, gaps<=0.05/0.08): {'PASS' if gate else 'FAIL'}")
    logger.info(f"  overfit: train_acc={train_acc:.4f} test_acc={test_acc:.4f} gap={acc_gap:.4f} | "
                f"train_r2={train_r2:.4f} test_r2={test_r2:.4f} gap={r2_gap:.4f}")

    # feature importance
    imp = dict(zip(X.columns, [float(v) for v in clf.feature_importances_]))
    imp_sorted = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    logger.info("  top features: " + ", ".join(
        f"{k}={v:.3f}" for k, v in list(imp_sorted.items())[:8]))
    metrics["top_features"] = dict(list(imp_sorted.items())[:8])

    bundle = {
        "clf": clf, "reg": reg, "feature_cols": list(X.columns),
        "metrics": metrics, "interest_taxonomy": INTEREST_TAXONOMY,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "matchmaking.pkl")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "match_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("  saved models/matchmaking.pkl + reports/match_metrics.json")

    # quick sanity: score a sample pair
    idx = np.where(y_te == 1)[0][:1] if y.sum() else [0]
    return bundle


# ---------------------------------------------------------------------------
# INFERENCE: score one candidate pair (used by server /api/ml/match)
# ---------------------------------------------------------------------------
def score_pair(bundle: dict, user_a: dict, user_b: dict) -> dict:
    """Compute match score for two user dicts (API contract)."""
    import numpy as np

    feats = {}
    feats["age_diff"] = float(user_a["age"]) - float(user_b["age"])
    feats["age_sim"] = 1.0 - min(abs(feats["age_diff"]) / 30.0, 1.0)
    feats["age_both_adult"] = float(user_a["age"] >= 21 and user_b["age"] >= 21)
    feats["height_diff"] = float(user_a.get("height") or 65) - float(user_b.get("height") or 65)
    feats["income_diff"] = float(user_a.get("income") or 0) - float(user_b.get("income") or 0)
    feats["orientation_ok"] = 1.0
    ia = set(user_a.get("interests") or [])
    ib = set(user_b.get("interests") or [])
    feats["shared_interests"] = float(len(ia & ib))
    feats["interest_jaccard"] = float(len(ia & ib)) / float(len(ia | ib) or 1)
    feats["both_no_interests"] = float(len(ia | ib) == 0)
    la = set(user_a.get("languages") or ["english"])
    lb = set(user_b.get("languages") or ["english"])
    feats["lang_overlap"] = float(len(la & lb))
    feats["has_hindi"] = float("hindi" in la or "hindi" in lb)
    for col in ["diet", "drinks", "smokes", "religion", "education", "ethnicity"]:
        a = str(user_a.get(col, "") or "").lower()
        b = str(user_b.get(col, "") or "").lower()
        known = bool(a) and bool(b)
        feats[f"same_{col}"] = float(a == b) * float(known)
        feats[f"{col}_known"] = float(known)
    feats["same_city"] = float(user_a.get("city") == user_b.get("city"))
    feats["city_distance_km"] = 0.0 if feats["same_city"] else 600.0
    feats["essay_cosine"] = float(user_a.get("essay_cosine") or 0.0)

    X = pd.DataFrame([feats])[bundle["clf"].feature_names_in_]
    p = float(bundle["clf"].predict_proba(X)[0, 1])
    s = float(bundle["reg"].predict(X)[0])
    s = max(0.0, min(1.0, s))
    return {
        "compatibility_probability": round(p, 4),
        "match_score": round(s * 100, 1),
        "tier": ("excellent" if s >= 0.75 else "good" if s >= 0.6 else "possible" if s >= 0.45 else "low"),
    }


if __name__ == "__main__":
    train_matchmaking()
