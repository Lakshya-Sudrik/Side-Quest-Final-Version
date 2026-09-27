"""Honest label construction for the SideQuest India pipeline.

Every label in this module is built from signals that are NOT model features.
The set of columns a label is allowed to read is declared in config/criteria.json
under ``label_input_columns``; the set it is forbidden to share with X is
``features_excluded_from_X``. ``assert_no_leakage`` enforces the contract and is
called by the trainer before any fitting happens.

Design rules
------------
1. No cross-row statistic (median, quantile, percentile) may be computed on the
   full dataset. Labelers expose ``fit(train_rows)`` + ``transform(rows)`` so the
   reference distribution comes from the training fold only.
2. Labels whose ingredients are also features are rejected at training time.
3. Provenance is recorded: independent | weak_distilled | synthetic. Only
   ``independent`` may be advertised as genuine accuracy.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
logger = logging.getLogger(__name__)

CRITERIA_PATH = ROOT / "config" / "criteria.json"

REVIEWS_CSV = ROOT / "data" / "india_reviews.csv"
PLACES_CSV = ROOT / "data" / "india_places.csv"
SCORED_REVIEWS_CSV = ROOT / "data" / "zomato_reviews_scored.csv"


# ---------------------------------------------------------------------------
# criteria helpers
# ---------------------------------------------------------------------------
def load_criteria() -> dict:
    with open(CRITERIA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def task_criteria(task: str) -> dict:
    crit = load_criteria()
    if task not in crit["tasks"]:
        raise KeyError(f"task {task!r} is not defined in {CRITERIA_PATH}")
    return crit["tasks"][task]


def assert_no_leakage(task: str, features: Iterable[str]) -> None:
    """Fail loudly if a label's inputs are also model features."""
    tc = task_criteria(task)
    forbidden = set(tc.get("features_excluded_from_X", [])) | set(
        tc.get("label_input_columns", [])
    )
    overlap = forbidden & set(features)
    if overlap:
        raise AssertionError(
            f"LEAKAGE: task {task!r} reads label inputs as model features: "
            f"{sorted(overlap)}. Remove them from X or change the label."
        )


def prohibited_features(task: str) -> List[str]:
    tc = task_criteria(task)
    return sorted(
        set(tc.get("features_excluded_from_X", [])) | set(tc.get("label_input_columns", []))
    )


# ---------------------------------------------------------------------------
# hidden_gem: review endorsement x exposure
# ---------------------------------------------------------------------------
class GemLabeler:
    """gem_quality = 100 * pct_rank_within_city( w * z(detail_rate) + (1-w) * z(review_star_mean) )

    Both parts are measured on the place's scraped reviews (text + per-review
    stars), shrunk towards the training-fold average by ``shrinkage_k``
    pseudo-reviews, because Zomato gives at most 5 reviews per place:

        detail_rate      = (n_detailed_positive + k * prior) / (n_reviews + k)
                           detailed positive = >= 40 words AND >= 4 stars
        review_star_mean = (sum_review_stars   + k * prior) / (n_reviews + k)

    Why this label (measured on the real data, restaurant-grouped test split):
      * the old 0.6*quality + 0.4*(1 - popularity) label put 40% of the target
        on review volume, which is banned from X - held-out R2 0.08 once
        duplicate listings stopped leaking across the split;
      * the binary detail rate alone is very noisy with <= 5 reviews; blending
        in the per-review star mean lowers that noise.
    "Hidden" (low exposure) is applied at serving time from the observed review
    count (see ``exposure_pct`` / criteria gem_rule), not predicted.
    Priors, z-score stats and percentile references are fitted on TRAIN only.
    """

    def __init__(self, criteria: Optional[dict] = None):
        self.crit = (criteria or task_criteria("hidden_gem"))["label_constants"]
        self._ref: Dict[str, np.ndarray] = {}
        self._global_ref: Optional[np.ndarray] = None
        self.fitted = False

    # -- per-place raw counts (no cross-row statistics) -------------------------
    @staticmethod
    def endorsement_components(reviews: pd.DataFrame) -> pd.DataFrame:
        """Per-place review aggregates: n_reviews, n_detailed_positive, sum_review_stars."""
        kc = task_criteria("hidden_gem")["label_constants"]
        min_words = int(kc.get("detailed_review_min_words", 40))
        pos_min = float(kc.get("positive_rating_min", 4.0))
        df = reviews.dropna(subset=["place_id"]).copy()
        text = df.get("review_text", pd.Series("", index=df.index)).fillna("")
        stars = pd.to_numeric(df.get("stars", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["_words"] = text.str.split().str.len()
        df["_stars"] = stars
        df["_detailed_positive"] = ((df["_words"] >= min_words) & (stars >= pos_min)).astype(int)
        g = df.groupby("place_id").agg(
            n_reviews=("_detailed_positive", "size"),
            n_detailed_positive=("_detailed_positive", "sum"),
            sum_review_stars=("_stars", "sum"),
            n_starred=("_stars", "count"),
        )
        g["detailed_positive_rate"] = g["n_detailed_positive"] / g["n_reviews"]
        return g

    def _raw(self, j: pd.DataFrame) -> np.ndarray:
        k = float(self.crit.get("shrinkage_k", 20.0))
        w = float(self.crit.get("w_detail", 0.3))
        n = j["n_reviews"].fillna(0).to_numpy(float)
        ns = j["n_starred"].fillna(0).to_numpy(float)
        a = (j["n_detailed_positive"].fillna(0).to_numpy(float) + k * self.prior_detail_) / (n + k)
        b = (j["sum_review_stars"].fillna(0).to_numpy(float) + k * self.prior_stars_) / (ns + k)
        za = (a - self.mu_a_) / self.sd_a_
        zb = (b - self.mu_b_) / self.sd_b_
        return w * za + (1.0 - w) * zb

    # -- fit / transform -------------------------------------------------------
    def fit(self, train_places: pd.DataFrame, endorsement: pd.DataFrame) -> "GemLabeler":
        j = train_places.join(endorsement, on="place_id", how="left")
        j = j[j["n_reviews"].fillna(0) >= int(self.crit.get("min_reviews", 1))]
        if j.empty:
            raise RuntimeError("no training places with reviews to fit the gem label")
        k = float(self.crit.get("shrinkage_k", 20.0))
        self.prior_detail_ = float(j["n_detailed_positive"].sum() / j["n_reviews"].sum())
        self.prior_stars_ = float(j["sum_review_stars"].sum() / max(j["n_starred"].sum(), 1))
        a = (j["n_detailed_positive"] + k * self.prior_detail_) / (j["n_reviews"] + k)
        b = (j["sum_review_stars"] + k * self.prior_stars_) / (j["n_starred"] + k)
        self.mu_a_, self.sd_a_ = float(a.mean()), float(a.std() or 1.0)
        self.mu_b_, self.sd_b_ = float(b.mean()), float(b.std() or 1.0)
        raw = self._raw(j)
        self._global_ref = np.sort(raw)
        for city, idx in j.groupby("city").indices.items():
            self._ref[city] = np.sort(raw[idx])
        self.fitted = True
        return self

    @staticmethod
    def _pct_rank(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if ref is None or len(ref) == 0:
            return np.full(values.shape, 0.5)
        idx = np.searchsorted(ref, values, side="right")
        return idx / max(len(ref), 1)

    def transform(self, places: pd.DataFrame, endorsement: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("GemLabeler.fit() must run on the training fold first")
        j = places.join(endorsement, on="place_id", how="left")
        eligible = j["n_reviews"].fillna(0) >= int(self.crit.get("min_reviews", 1))
        raw = self._raw(j)
        q = self._pct_rank(raw, self._global_ref)
        for city, idx in j.groupby("city").indices.items():
            if city in self._ref and len(self._ref[city]) >= 50:
                q[idx] = self._pct_rank(raw[idx], self._ref[city])
        out = pd.Series(np.clip(100.0 * q, 0, 100), index=places.index, name="hidden_gem_score")
        out[~eligible.to_numpy()] = np.nan
        return out


def exposure_pct(places: pd.DataFrame, reference: Optional[pd.DataFrame] = None) -> pd.Series:
    """Percentile of a place's review count WITHIN ITS CITY (0 = least known).

    This is the observable "hidden" half of a hidden gem. It is never a model
    input or target - serving combines it with predicted quality:
    gem = quality >= threshold AND exposure_pct <= criteria max_exposure_pct.
    ``reference`` defaults to ``places`` itself (all known places in the city).
    """
    ref = places if reference is None else reference
    rc = pd.to_numeric(places["review_count"], errors="coerce")
    out = pd.Series(np.nan, index=places.index, name="exposure_pct")
    ref_sorted = {c: np.sort(pd.to_numeric(g["review_count"], errors="coerce").dropna().to_numpy(float))
                  for c, g in ref.groupby("city")}
    for city, idx in places.groupby("city").indices.items():
        r = ref_sorted.get(city)
        if r is None or len(r) == 0:
            continue
        v = rc.iloc[idx].to_numpy(float)
        out.iloc[idx] = np.where(np.isfinite(v), np.searchsorted(r, v, side="right") / len(r), np.nan)
    return out


# ---------------------------------------------------------------------------
# safety_score: NCRB crime only
# ---------------------------------------------------------------------------
def _norm_name(s: str) -> str:
    s = str(s).strip().lower()
    for a, b in (
        ("bengaluru", "bangalore"),
        ("chennai", "madras"),
        ("kolkata", "calcutta"),
        ("thiruvananthapuram", "trivandrum"),
        ("varanasi", "benares"),
        ("gautam buddh nagar", "gautam buddh"),
        ("mumbai suburban", "mumbai"),
        ("mumbai city", "mumbai"),
    ):
        s = s.replace(a, b)
    return " ".join(s.split())


class SafetyLabeler:
    """safety = 100 * (w_v * exp(-v/v_scale) + w_p * exp(-p/p_scale)).

    The decay constants are FIT ON THE TRAINING FOLD (p75 of each rate) because
    Indian NCRB rates are an order of magnitude smaller than US FBI rates - a
    fixed 800/4000 collapses the target to a near-constant and any R^2 on it is
    meaningless. Crime rates are label inputs and are stripped from X by
    ``prohibited_features('safety_score')``.
    """

    def __init__(self, criteria: Optional[dict] = None):
        # legacy: safety is now a rule-based index (src/safety_index.py); this
        # labeler is kept only so old notebooks keep running
        self.crit = (criteria or task_criteria("safety_score")).get("label_constants", {})
        self.v_scale_: float = float(self.crit.get("fallback_v_scale", 30.0))
        self.p_scale_: float = float(self.crit.get("fallback_p_scale", 10.0))
        self.fitted = False

    @staticmethod
    def district_group(places: pd.DataFrame) -> pd.Series:
        """Best-effort district for group splitting; falls back to state."""
        try:
            from india_data import load_crime

            district_rates, _ = load_crime()
        except Exception as exc:  # pragma: no cover - data optional
            logger.warning("NCRB district rates unavailable (%s); grouping by state", exc)
            return places["state"].astype(str)

        dnorm = {_norm_name(d): d for d in district_rates["district_name"].astype(str)}
        city_norm = places["city"].map(_norm_name)
        matched = city_norm.map(lambda c: dnorm.get(c, None))
        # substring fallback (city "Bengaluru Urban" vs district "Bangalore")
        unmatched = matched.isna()
        if unmatched.any():
            keys = list(dnorm.keys())
            for c in sorted(city_norm[unmatched].unique()):
                if not keys:
                    break
                hit = [k for k in keys if c and (c in k or k in c)]
                if hit:
                    matched[city_norm == c] = dnorm[hit[0]]
        return matched.fillna(places["state"].astype(str))

    @staticmethod
    def enrich_crime_with_district(places: pd.DataFrame) -> pd.DataFrame:
        """Replace state-level rates with district rates where a city matches."""
        try:
            from india_data import load_crime

            district_rates, _ = load_crime()
        except Exception as exc:  # pragma: no cover
            logger.warning("district crime enrichment skipped: %s", exc)
            out = places.copy()
            out["ncrb_group"] = out["state"].astype(str)
            return out

        d = district_rates.drop_duplicates("district_name").copy()
        d["_key"] = d["district_name"].astype(str).map(_norm_name)
        lookup = d.set_index("_key")[["violent_rate", "property_rate", "district_name"]]
        lookup = lookup[~lookup.index.duplicated(keep="first")]
        city_key = places["city"].map(_norm_name)

        v = places["violent_crime_rate"].astype(float).values.copy()
        p = places["property_crime_rate"].astype(float).values.copy()
        grp = places["state"].astype(str).values.copy()
        hit = 0
        for i, key in enumerate(city_key.values):
            if key in lookup.index:
                row = lookup.loc[key]
                v[i] = float(row["violent_rate"])
                p[i] = float(row["property_rate"])
                grp[i] = str(row["district_name"])
                hit += 1
        logger.info("  safety label: %d/%d places matched to an NCRB district", hit, len(places))
        out = places.copy()
        out["violent_crime_rate"] = v
        out["property_crime_rate"] = p
        out["ncrb_group"] = grp
        return out

    def fit(self, train_places: pd.DataFrame) -> "SafetyLabeler":
        q = float(self.crit.get("scale_quantile", 0.75))
        eps = float(self.crit.get("min_scale", 1.0))
        v = pd.to_numeric(train_places["violent_crime_rate"], errors="coerce").dropna()
        p = pd.to_numeric(train_places["property_crime_rate"], errors="coerce").dropna()
        self.v_scale_ = max(float(v.quantile(q)), eps) if len(v) else self.v_scale_
        self.p_scale_ = max(float(p.quantile(q)), eps) if len(p) else self.p_scale_
        # missing-value fill must also come from the train fold, never from the
        # frame being transformed (which may contain held-out rows)
        self.v_median_ = float(v.median()) if len(v) else 0.0
        self.p_median_ = float(p.median()) if len(p) else 0.0
        self.fitted = True
        logger.info("  safety label scales fitted on train: v_scale=%.2f p_scale=%.2f",
                    self.v_scale_, self.p_scale_)
        return self

    def transform(self, places: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("SafetyLabeler.fit() must run on the training fold first")
        kc = self.crit
        v = pd.to_numeric(places["violent_crime_rate"], errors="coerce")
        p = pd.to_numeric(places["property_crime_rate"], errors="coerce")
        v = v.fillna(getattr(self, "v_median_", 0.0))
        p = p.fillna(getattr(self, "p_median_", 0.0))
        wv = float(kc.get("violent_weight", 0.6))
        wp = float(kc.get("property_weight", 0.4))
        score = 100.0 * (
            wv * np.exp(-v / self.v_scale_) + wp * np.exp(-p / self.p_scale_)
        ) / max(wv + wp, 1e-9)
        return pd.Series(np.clip(score, 0, 100), index=places.index, name="safety_score")


# ---------------------------------------------------------------------------
# collaboration_auth: distillation label (provenance = weak_distilled)
# ---------------------------------------------------------------------------
def build_collab_labels(places: pd.DataFrame, scored_reviews: Optional[pd.DataFrame] = None) -> pd.Series:
    """is_authentic = 1 when the place's fake-review fraction < threshold.

    The text verdict (fraud_rate) is a label input only and is excluded from X by
    ``prohibited_features('collaboration_auth')``.
    """
    thr = float(task_criteria("collaboration_auth")["label_constants"]["fake_fraction_threshold"])
    if scored_reviews is None:
        scored_reviews = pd.read_csv(SCORED_REVIEWS_CSV)

    stats = scored_reviews.groupby("place_id").agg(
        fraud_rate=("is_fake", "mean"), rev_n=("is_fake", "size")
    )
    joined = places[["place_id"]].join(stats, on="place_id")
    labels = pd.Series(np.nan, index=places.index, name="is_authentic")
    known = joined["fraud_rate"].notna()
    labels[known] = (joined.loc[known, "fraud_rate"] < thr).astype(float)
    return labels


# ---------------------------------------------------------------------------
# public entry point used by the trainer
# ---------------------------------------------------------------------------
def build_all_labels(places: pd.DataFrame, gem_labeler: Optional[GemLabeler] = None,
                     fit_on: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, dict]:
    """Attach honest labels to a places frame.

    ``fit_on`` is the training fold; every cross-row constant (gem percentile
    references, safety decay scales) is estimated from it only. Returns
    (frame, info) where info carries provenance and the fitted labelers so the
    same transforms can be replayed at serving time.
    """
    crit = load_criteria()["tasks"]
    out = places.copy()
    fit_index = (fit_on if fit_on is not None else places).index

    reviews = pd.read_csv(REVIEWS_CSV) if REVIEWS_CSV.exists() else None

    # crime enrichment must happen before any fit so train rows carry the same
    # district-level rates they will carry at transform time
    out = SafetyLabeler.enrich_crime_with_district(out)
    fit_rows = out.loc[fit_index]

    # hidden_gem
    lab = gem_labeler
    if reviews is not None:
        end = GemLabeler.endorsement_components(reviews)
        if lab is None:
            lab = GemLabeler()
        if not lab.fitted:
            lab.fit(fit_rows, end)
        out["hidden_gem_score"] = lab.transform(out, end)
    else:
        logger.warning("india_reviews.csv missing -> hidden_gem labels unavailable")
        out["hidden_gem_score"] = np.nan

    # safety_score is not a label any more - see src/safety_index.py
    out["ncrb_group"] = out.get("ncrb_group", out["state"]).fillna(out["state"])

    # collaboration_auth
    scored = pd.read_csv(SCORED_REVIEWS_CSV) if SCORED_REVIEWS_CSV.exists() else None
    out["is_authentic"] = build_collab_labels(out, scored)

    provenance = {t: c["label_provenance"] for t, c in crit.items()}
    info = {
        "provenance": provenance,
        "labelers": {"hidden_gem": lab},
    }
    return out, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    places = pd.read_csv(PLACES_CSV)
    labelled, info = build_all_labels(places)
    for col in ("hidden_gem_score", "is_authentic"):
        s = labelled[col]
        print(f"{col}: n={s.notna().sum()} coverage={s.notna().mean():.1%} "
              f"mean={s.mean():.2f} std={s.std():.2f}")
    print("provenance:", info["provenance"])
