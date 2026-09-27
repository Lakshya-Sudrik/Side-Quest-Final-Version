"""India data pipeline: builds model-ready datasets from Indian sources.

Outputs:
  data/india_places.csv         - all places + features + hidden_gem/safety targets
  data/india_reviews.csv        - sampled review texts (for fake detection + RAG)

collab_auth labels (is_authentic) are added later by train_india.py after the
fake-review classifier is trained (text-model labels -> metadata prediction,
so no target/feature leakage).
"""
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from india_data import (
    BLR_CENTER, build_infra_index, geocode_all, haversine_km, load_crime,
    load_json_cache, load_swiggy, load_zomato, nearest_km, GEOCODE_CACHE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
RAW = Path(__file__).parent / "data" / "raw" / "india"


# ---------------------------------------------------------------------------
# STEP 1: unified places + geocodes
# ---------------------------------------------------------------------------
def build_places() -> pd.DataFrame:
    z = load_zomato()
    s = load_swiggy()

    # attach geocodes (prefer raw/locality key, fall back to city key)
    cache = load_json_cache(GEOCODE_CACHE)

    def geo_for(row, kind):
        if kind == "zomato":
            keys = [f"{row['area']}, Bangalore, India"]
        else:
            keys = [f"{row['raw_city']}, India", f"{row['city']}, India"]
        lat = lng = None
        state = ""
        for q in keys:
            g = cache.get(q) or {}
            if g.get("lat") is not None and lat is None:
                lat, lng = g.get("lat"), g.get("lng")
            if g.get("state") and not state:
                state = g.get("state", "")
            if lat is not None and state:
                break
        return lat, lng, state

    for kind, df in (("zomato", z), ("swiggy", s)):
        res = df.apply(lambda r: geo_for(r, kind), axis=1, result_type="expand")
        res.columns = ["latitude", "longitude", "state_geo"]
        df["latitude"] = pd.to_numeric(res["latitude"], errors="coerce").astype(float)
        df["longitude"] = pd.to_numeric(res["longitude"], errors="coerce").astype(float)
        if kind == "swiggy":
            df["state"] = np.where(res["state_geo"].fillna("").ne(""), res["state_geo"], df["state"])
        else:
            df["state"] = "Karnataka"

    places = pd.concat([z, s], ignore_index=True)
    has_geo = places[["latitude", "longitude"]].notna().all(axis=1)
    logger.info(f"Places: {len(places)} total, {has_geo.sum()} with coords "
                f"(dropping {len(places)-has_geo.sum()} without)")
    places = places[has_geo].reset_index(drop=True)
    places["state"] = places["state"].fillna("")
    return places


# ---------------------------------------------------------------------------
# STEP 2: spatial + relative context features
# ---------------------------------------------------------------------------
def add_context_features(places: pd.DataFrame) -> pd.DataFrame:
    # city mean coords -> city centre (per city)
    city_center = places.groupby("city")[["latitude", "longitude"]].transform("mean")
    places["dist_from_center"] = haversine_km(
        places["latitude"].values, places["longitude"].values,
        city_center["latitude"].values, city_center["longitude"].values,
    )

    # competitor density: businesses sharing (city, area)
    grp = places.groupby(["city", "area"])["place_id"].transform("count")
    places["competitor_density"] = grp.astype(float)

    # category frequency within city (lower = rarer = more gem-like)
    primary = places["categories"].apply(lambda c: c[0] if isinstance(c, list) and c else "unknown")
    places["primary_category"] = primary
    cat_city = places.groupby(["city", "primary_category"])["place_id"].transform("count")
    city_n = places.groupby("city")["place_id"].transform("count")
    places["category_rarity"] = cat_city / city_n.clip(lower=1)

    # --- restaurant-level features (none of them feed a label) ---
    # branches of the same brand in the city: chains are rarely hidden gems
    name_key = places["name"].astype(str).str.lower().str.strip()
    places["chain_size"] = places.groupby([places["city"], name_key])["place_id"].transform("count").astype(float)
    # how many dishes diners call out by name (zomato only)
    dl = places.get("dish_liked", pd.Series("", index=places.index)).fillna("").astype(str)
    places["dish_liked_count"] = dl.apply(lambda s: len([d for d in s.split(",") if d.strip()])).astype(float)
    # price positioning inside the neighbourhood
    area_cost = places.groupby(["city", "area"])["cost_inr"].transform("median")
    places["cost_vs_area"] = (places["cost_inr"] / area_cost.replace(0, np.nan)).astype(float)

    # relative rating/review context
    cg = places.groupby("city")
    places["city_avg_rating"] = cg["stars"].transform("mean")
    places["city_avg_reviews"] = cg["review_count"].transform("mean")
    places["city_business_count"] = cg["place_id"].transform("count")
    places["rating_vs_city"] = places["stars"] - places["city_avg_rating"]
    places["review_ratio_vs_city"] = (
        places["review_count"] / places["city_avg_reviews"].clip(lower=1)
    )

    # review text stats (zomato only; swiggy -> 0)
    places["has_reviews"] = places["has_reviews_text"]
    places["review_count_log"] = np.log1p(places["review_count"])
    places["category_count"] = places["categories"].apply(len)

    places["is_sweet_spot"] = ((places["stars"] >= 4.0) & (places["stars"] <= 4.5)).astype(int)
    p90 = places["review_count"].quantile(0.90)
    places["is_mainstream"] = (places["review_count"] > p90).astype(int)
    places["is_budget_friendly"] = (places["price_level"].fillna(2) <= 2).astype(int)
    places["is_tourist_trap"] = (
        (places["stars"] >= 4.8) & (places["review_count"] > 500)
    ).astype(int)

    # area business count (geographic cluster size)
    places["area_business_count"] = places["competitor_density"]
    places["price_inr_log"] = np.log1p(places["cost_inr"].fillna(0))
    return places


# ---------------------------------------------------------------------------
# STEP 3: OSM infrastructure -> hospital/police distance
# ---------------------------------------------------------------------------
def add_infrastructure(places: pd.DataFrame) -> pd.DataFrame:
    # query infra for cities covering the most places + Bangalore always
    top_cities = places["city"].value_counts().head(45).index.tolist()
    if "Bangalore" not in top_cities:
        top_cities.append("Bangalore")
    city_coords = (
        places[places["city"].isin(top_cities)]
        .groupby("city")[["latitude", "longitude"]].mean()
        .to_dict("index")
    )
    infra = build_infra_index(city_coords)  # city -> {points: [...]}

    hosp_d, police_d = [], []
    for _, r in places.iterrows():
        pts = (infra.get(r["city"]) or {}).get("points", [])
        if not pts:
            hosp_d.append(np.nan)
            police_d.append(np.nan)
            continue
        h = [p for p in pts if p["type"] in ("hospital", "clinic", "doctors")]
        pol = [p for p in pts if p["type"] == "police"]
        hosp_d.append(nearest_km(r["latitude"], r["longitude"], h))
        police_d.append(nearest_km(r["latitude"], r["longitude"], pol))

    places["hospital_distance"] = hosp_d
    places["police_distance"] = police_d
    places["has_hospital_nearby"] = (places["hospital_distance"] <= 2.0).astype(int)
    places["has_police_nearby"] = (places["police_distance"] <= 2.0).astype(int)
    covered = places["hospital_distance"].notna().sum()
    logger.info(f"  infra: {covered}/{len(places)} places have OSM hospital/police data")
    return places


# ---------------------------------------------------------------------------
# STEP 4: crime (district for Karnataka places, state otherwise)
# ---------------------------------------------------------------------------
def add_crime(places: pd.DataFrame, district_rates: pd.DataFrame, state_rates: dict) -> pd.DataFrame:
    # Bangalore -> Bengaluru Urban / Bangalore district
    dist = district_rates.drop_duplicates(subset=["district_name"])
    blr = dist[dist["district_name"].str.contains("Bengaluru|Bangalore", case=False, na=False)]
    blr_row = blr.iloc[0] if len(blr) else None

    v, p = [], []
    for _, r in places.iterrows():
        if r["city"] == "Bangalore" and blr_row is not None:
            v.append(blr_row["violent_rate"])
            p.append(blr_row["property_rate"])
        else:
            sr = state_rates.get(str(r["state"]).strip().lower(), {})
            v.append(sr.get("violent_rate", np.nan))
            p.append(sr.get("property_rate", np.nan))
    places["violent_crime_rate"] = v
    places["property_crime_rate"] = p
    if blr_row is not None:
        logger.info(f"  Bengaluru Urban district: violent={blr_row['violent_rate']:.1f}/100k, "
                    f"property={blr_row['property_rate']:.1f}/100k")
    have = places["violent_crime_rate"].notna().sum()
    logger.info(f"  crime: {have}/{len(places)} places have crime rates")
    return places


# ---------------------------------------------------------------------------
# STEP 4b: CLEANING - part 1: drop low-quality rows BEFORE features
# ---------------------------------------------------------------------------
def drop_low_quality(places: pd.DataFrame) -> pd.DataFrame:
    n0 = len(places)
    places = places[places["stars"].notna()].copy()                       # no rating
    places = places[places["stars"].between(1.0, 5.0)].copy()             # sane rating
    places = places[places["categories"].apply(lambda c: isinstance(c, list) and len(c) > 0)].copy()
    places = places[places["review_count"].notna()].copy()
    places = places[places["latitude"].notna() & places["longitude"].notna()].copy()
    places = places[places["city"].ne("") & places["city"].ne("unknown")].copy()
    logger.info(f"  DROP: {n0} -> {len(places)} rows ({n0 - len(places)} low-quality removed)")
    return places.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 4c: CLEANING - part 2: impute remaining nulls -> zero-null assertion
# ---------------------------------------------------------------------------
def impute_nulls(places: pd.DataFrame) -> pd.DataFrame:
    # flag infra coverage BEFORE imputation (model knows real vs filled)
    places["has_infra_data"] = places["hospital_distance"].notna().astype(int)

    # price/cost: city median -> global median
    for col in ["price_level", "cost_inr"]:
        if col in places.columns:
            gm = places[col].median()
            places[col] = places.groupby("city")[col].transform(lambda s: s.fillna(s.median()))
            places[col] = places[col].fillna(gm)

    # infra distances: city median -> neutral 3km
    for col in ["hospital_distance", "police_distance"]:
        gm = places[col].median()
        if pd.isna(gm):
            gm = 3.0
        places[col] = (
            places.groupby("city")[col].transform(lambda s: s.fillna(s.median() if s.notna().any() else np.nan))
            .fillna(gm)
        )

    # crime rates: state median -> global median
    for col in ["violent_crime_rate", "property_crime_rate"]:
        gm = places[col].median()
        if pd.isna(gm):
            gm = 10.0
        places[col] = (
            places.groupby("state")[col].transform(lambda s: s.fillna(s.median() if s.notna().any() else np.nan))
            .fillna(gm)
        )

    places["stars"] = places["stars"].astype(float)
    places["price_level"] = places["price_level"].astype(float)

    # VERIFY: no nulls in any model-numeric column
    numeric_cols = places.select_dtypes(include=[np.number]).columns.tolist()
    nulls = places[numeric_cols].isna().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        raise ValueError(f"NULLS REMAIN in numeric columns:\n{bad}")
    logger.info(f"  IMPUTE: 0 nulls in {len(numeric_cols)} numeric cols ({len(places)} rows)")
    return places


# ---------------------------------------------------------------------------
# STEP 5: TARGETS
# ---------------------------------------------------------------------------
def target_hidden_gem(places: pd.DataFrame) -> pd.Series:
    """External-signal target (no leakage: none of these are model features
    directly except relative context which is engineered separately)."""
    score = np.zeros(len(places))

    med_density = places["competitor_density"].median()
    score += np.where(places["competitor_density"] < med_density * 0.3, 30,
             np.where(places["competitor_density"] < med_density, 15, -5))

    med_dist = places["dist_from_center"].median()
    score += np.where(places["dist_from_center"] > med_dist * 1.5, 20,
             np.where(places["dist_from_center"] > med_dist, 10, -5))

    med_rarity = places["category_rarity"].median()
    score += np.where(places["category_rarity"] < med_rarity * 0.5, 20,
             np.where(places["category_rarity"] < med_rarity, 10, 0))

    score += np.where((places["rating_vs_city"] > 0) & (places["rating_vs_city"] < 0.5), 15,
             np.where(places["rating_vs_city"] > 0.5, 5, -10))

    score += np.where((places["review_ratio_vs_city"] < 0.5) & (places["rating_vs_city"] > 0), 15,
             np.where(places["review_ratio_vs_city"] > 2.0, -10, 0))

    score += np.where(places["review_count"] > 200, -10, 0)  # mainstream penalty
    return np.clip(score, 0, 100)


def target_safety(places: pd.DataFrame) -> pd.Series:
    """Exponential decay on real NCRB crime rates + local safety infrastructure."""
    v = places["violent_crime_rate"].fillna(places["violent_crime_rate"].median())
    p = places["property_crime_rate"].fillna(places["property_crime_rate"].median())
    base = np.exp(-v / 800.0) * 50 + np.exp(-p / 4000.0) * 30

    # infra access (hospital + police within 2km) -> up to 20 pts
    hosp = np.where(
        places["hospital_distance"].isna(), 0.5,
        np.clip(1.0 - places["hospital_distance"].fillna(5.0) / 5.0, 0, 1),
    )
    police = np.where(
        places["police_distance"].isna(), 0.5,
        np.clip(1.0 - places["police_distance"].fillna(5.0) / 5.0, 0, 1),
    )
    infra = (hosp + police) / 2.0 * 20.0
    return np.clip(base + infra, 0, 100)


# ---------------------------------------------------------------------------
# STEP 6: review extraction (fake detection + RAG corpus)
# ---------------------------------------------------------------------------
def extract_reviews(places: pd.DataFrame, max_per_place: int = 5) -> pd.DataFrame:
    z = places[places["source"] == "zomato"].copy()
    rows = []
    for _, r in z.iterrows():
        raw = r["reviews_raw"]
        if not isinstance(raw, str) or raw in ("[]", "nan", ""):
            continue
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            continue
        if not isinstance(parsed, list):
            continue
        picked = parsed[:max_per_place]
        for rating_s, text in picked:
            if not isinstance(text, str):
                continue
            text = text.replace("RATED", "").strip()
            if len(text) < 25:
                continue
            stars = None
            try:
                stars = float(str(rating_s).replace("Rated", "").strip())
            except Exception:
                pass
            rows.append({
                "place_id": r["place_id"],
                "city": r["city"],
                "area": r["area"],
                "stars": stars,
                "review_text": text[:1200],
            })
    df = pd.DataFrame(rows)
    logger.info(f"  extracted {len(df)} reviews from {df.place_id.nunique()} places")
    return df


# ---------------------------------------------------------------------------
def run() -> None:
    logger.info("=" * 60)
    logger.info("INDIA DATA PIPELINE")
    logger.info("=" * 60)

    places = build_places()
    places = drop_low_quality(places)
    places = add_context_features(places)

    district_rates, state_rates = load_crime()
    places = add_crime(places, district_rates, state_rates)
    places = add_infrastructure(places)
    places = impute_nulls(places)

    places["hidden_gem_score"] = target_hidden_gem(places)
    places["safety_score"] = target_safety(places)

    # deterministic sort for reproducibility
    places = places.sort_values("place_id").reset_index(drop=True)

    out_cols = [
        "place_id", "source", "name", "city", "state", "area", "address",
        "latitude", "longitude", "stars", "review_count", "review_count_log",
        "categories", "primary_category", "category_count", "rest_type",
        "cost_inr", "price_level", "has_phone", "has_website",
        "has_online_order", "has_book_table", "has_license",
        "competitor_density", "area_business_count", "dist_from_center",
        "category_rarity", "city_avg_rating", "city_avg_reviews",
        "city_business_count", "rating_vs_city", "review_ratio_vs_city",
        "is_sweet_spot", "is_mainstream", "is_budget_friendly",
        "is_tourist_trap", "has_reviews", "is_open",
        "hospital_distance", "police_distance", "has_hospital_nearby",
        "has_police_nearby", "has_infra_data", "violent_crime_rate",
        "property_crime_rate", "hidden_gem_score", "safety_score",
        "chain_size", "dish_liked_count", "cost_vs_area",
        "dish_liked", "reviews_raw",
    ]
    places["is_open"] = 1
    missing = [c for c in out_cols if c not in places.columns]
    if missing:
        logger.warning(f"missing cols: {missing}")
        for c in missing:
            places[c] = 0
    out = places[out_cols].copy()

    out_path = Path(__file__).parent / "data" / "india_places.csv"
    # reviews_raw is huge -> keep only in sidecar for zomato
    rev_cols = ["place_id", "reviews_raw"]
    reviews_raw_df = out[rev_cols].copy()
    out = out.drop(columns=["reviews_raw", "dish_liked"])
    out.to_csv(out_path, index=False)
    reviews_raw_df.to_csv(Path(__file__).parent / "data" / "india_reviews_raw.csv", index=False)
    logger.info(f"wrote {out_path} ({len(out)} places)")

    revs = extract_reviews(places, max_per_place=5)
    revs.to_csv(Path(__file__).parent / "data" / "india_reviews.csv", index=False)
    logger.info(f"wrote data/india_reviews.csv ({len(revs)} reviews)")

    logger.info("\nTARGET DISTRIBUTIONS")
    logger.info(f"  hidden_gem: mean={out.hidden_gem_score.mean():.1f} "
                f"p90={out.hidden_gem_score.quantile(.9):.1f} "
                f"qualifying(>=60)={100*(out.hidden_gem_score>=60).mean():.1f}%")
    logger.info(f"  safety: mean={out.safety_score.mean():.1f} "
                f"p10={out.safety_score.quantile(.1):.1f} p90={out.safety_score.quantile(.9):.1f}")
    logger.info(f"  cities={out.city.nunique()} states={out.state.nunique()} "
                f"sources={out.source.value_counts().to_dict()}")


if __name__ == "__main__":
    run()
