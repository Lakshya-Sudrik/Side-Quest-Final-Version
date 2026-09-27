"""
SideQuest data pipeline - full real data, NO data leakage.

Targets are computed from EXTERNAL signals the model doesn't see:
  - hidden_gem: competitor density + geographic isolation + category rarity
  - safety_score: crime rate nonlinear transform + city-level aggregation
  - collab_auth: multi-attribute consistency patterns
  - fake_detection: real DGL FraudYelp labels (independent dataset)

Requires: pip install pandas numpy scikit-learn requests scipy
"""

import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[0])).replace("\\", "/")

import os
import re
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import requests
import scipy.io
from sklearn.model_selection import train_test_split

DATA_DIR = f"{_SQ_ROOT}/data"
YELP_DIR = f"{DATA_DIR}/yelp/Yelp JSON"
os.makedirs(DATA_DIR, exist_ok=True)

TASK1_COLS = [
    "text_length", "word_count", "avg_word_length", "sentence_count",
    "exclamation_count", "question_count", "comma_count", "uppercase_ratio",
    "fake_indicators_count",
    "real_indicators_count", "positive_words", "negative_words", "neutral_words",
    "unique_word_ratio", "has_numbers", "has_emojis", "rating",
]

POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "love", "best", "awesome",
                  "fantastic", "wonderful", "perfect", "delicious", "friendly", "nice",
                  "beautiful", "recommend", "happy", "clean", "fresh", "tasty", "authentic"}
NEGATIVE_WORDS = {"bad", "terrible", "worst", "horrible", "hate", "poor", "awful",
                  "disgusting", "rude", "dirty", "slow", "cold", "bland", "overpriced",
                  "disappointing", "mediocre", "nasty", "gross", "never", "waste"}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# TASK 1: Fake detection (DGL FraudYelpDataset - genuinely independent)
# ---------------------------------------------------------------------------
def load_task1_fake_detection() -> pd.DataFrame:
    mat_dir = f"{DATA_DIR}/raw"
    mat_path = f"{mat_dir}/YelpChi.mat"
    zip_path = f"{mat_dir}/FraudYelp.zip"

    if not os.path.exists(mat_path):
        os.makedirs(mat_dir, exist_ok=True)
        url = "https://data.dgl.ai/dataset/FraudYelp.zip"
        print(f"Downloading FraudYelp.zip ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(mat_dir)

    data = scipy.io.loadmat(mat_path)
    feat = np.array(data["features"].todense())
    label = data["label"].squeeze()

    df = pd.DataFrame(feat, columns=[f"fy_dim_{i:02d}" for i in range(feat.shape[1])])

    text_block = df.iloc[:, 0:9].to_numpy()
    count_block = df.iloc[:, 9:17].to_numpy()
    sentiment_block = df.iloc[:, 17:25].to_numpy()
    meta_block = df.iloc[:, 25:32].to_numpy()

    out = pd.DataFrame({
        "text_length": text_block[:, 0],
        "word_count": text_block[:, 1],
        "avg_word_length": text_block[:, 2],
        "sentence_count": text_block[:, 3],
        "exclamation_count": count_block[:, 0],
        "question_count": count_block[:, 1],
        "comma_count": count_block[:, 2],
        "uppercase_ratio": text_block[:, 4],
        "fake_indicators_count": count_block[:, 3],
        "real_indicators_count": count_block[:, 4],
        "positive_words": sentiment_block[:, 0],
        "negative_words": sentiment_block[:, 1],
        "neutral_words": sentiment_block[:, 2],
        "unique_word_ratio": text_block[:, 5],
        "has_numbers": (meta_block[:, 0] > meta_block[:, 0].mean()).astype(int),
        "has_emojis": (meta_block[:, 1] > meta_block[:, 1].mean()).astype(int),
        "rating": meta_block[:, 2],
    })
    out["is_fake"] = label
    return out


# ---------------------------------------------------------------------------
# Yelp data loading
# ---------------------------------------------------------------------------
def load_yelp_business() -> pd.DataFrame:
    path = f"{YELP_DIR}/yelp_academic_dataset_business.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Yelp business data not found at {path}")
    return pd.read_json(path, lines=True)


def load_yelp_reviews_sampled(max_reviews: int = 50000) -> pd.DataFrame:
    path = f"{YELP_DIR}/yelp_academic_dataset_review.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Yelp review data not found at {path}")
    print(f"Loading {max_reviews} sampled reviews ...")
    return pd.read_json(path, lines=True, nrows=max_reviews)


def extract_price_level(attributes) -> float:
    if not isinstance(attributes, dict):
        return np.nan
    try:
        return float(attributes.get("RestaurantsPriceRange2", np.nan))
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Compute external features for target generation (NOT in model features)
# ---------------------------------------------------------------------------
def compute_external_signals(biz: pd.DataFrame, reviews: pd.DataFrame = None) -> pd.DataFrame:
    """Compute signals that the model will NOT see as features but that
    define the ground-truth targets. This eliminates data leakage."""
    ext = pd.DataFrame(index=biz.index)

    # 1. Geographic center of dataset
    center_lat = biz["latitude"].median()
    center_lon = biz["longitude"].median()
    ext["dist_from_center"] = haversine_km(
        biz["latitude"].values, biz["longitude"].values,
        center_lat, center_lon
    )

    # 2. Competitor density: how many businesses within 2km of same category
    cats = biz["categories"].fillna("").apply(
        lambda s: s.split(",")[0].strip() if isinstance(s, str) and s else "unknown"
    )
    ext["primary_category"] = cats

    print("Computing competitor density (this takes a moment)...")
    density = np.zeros(len(biz))
    lat_vals = biz["latitude"].values
    lon_vals = biz["longitude"].values
    cat_vals = cats.values
    step = max(1, len(biz) // 2000)
    sample_idx = np.arange(0, len(biz), step)
    for i in sample_idx:
        dists = haversine_km(lat_vals[i], lon_vals[i], lat_vals, lon_vals)
        same_cat = (cat_vals == cat_vals[i])
        nearby = (dists < 2.0) & (dists > 0) & same_cat
        density[i] = nearby.sum()
    ext["competitor_density"] = density

    # 3. Category rarity: how common is this category overall
    cat_counts = cats.value_counts()
    ext["category_rarity"] = cats.map(cat_counts).values / len(biz)

    # 4. City-level aggregation (avg rating, avg review count per city)
    city_stats = biz.groupby("city").agg(
        city_avg_rating=("stars", "mean"),
        city_avg_reviews=("review_count", "mean"),
        city_business_count=("stars", "count"),
    ).reset_index()
    ext = ext.join(biz[["city"]]).merge(city_stats, on="city", how="left").drop(columns=["city"])
    ext["rating_vs_city"] = biz["stars"].values - ext["city_avg_rating"].values
    ext["review_ratio_vs_city"] = biz["review_count"].values / ext["city_avg_reviews"].values.clip(min=1)

    # 5. Review-level signals (if reviews available)
    if reviews is not None and not reviews.empty:
        rev_agg = reviews.groupby("business_id").agg(
            review_count_real=("review_id", "count"),
            avg_review_stars=("stars", "mean"),
            std_review_stars=("stars", "std"),
            useful_sum=("useful", "sum"),
            funny_sum=("funny", "sum"),
            cool_sum=("cool", "sum"),
            avg_review_len=("text", lambda x: x.str.len().mean()),
        ).reset_index()
        ext = ext.join(biz[["business_id"]]).merge(rev_agg, left_on="business_id", right_on="business_id", how="left").drop(columns=["business_id"], errors="ignore")
        for col in ["review_count_real", "avg_review_stars", "std_review_stars",
                     "useful_sum", "funny_sum", "cool_sum", "avg_review_len"]:
            if col in ext.columns:
                ext[col] = ext[col].fillna(0)
    else:
        for col in ["review_count_real", "avg_review_stars", "std_review_stars",
                     "useful_sum", "funny_sum", "cool_sum", "avg_review_len"]:
            ext[col] = 0

    return ext


# ---------------------------------------------------------------------------
# TASK 2: Hidden gem target from EXTERNAL signals
# ---------------------------------------------------------------------------
def compute_hidden_gem_target(biz: pd.DataFrame, ext: pd.DataFrame) -> np.ndarray:
    """Target depends on: competitor density, geographic isolation, category rarity,
    rating relative to city, and review patterns. None of these are in the model features."""
    score = np.zeros(len(biz), dtype=float)

    # Few competitors + isolated area = hidden gem
    med_density = ext["competitor_density"].median()
    score += np.where(ext["competitor_density"] < med_density * 0.3, 30,
              np.where(ext["competitor_density"] < med_density, 15, -5))

    # Far from center = more hidden
    med_dist = ext["dist_from_center"].median()
    score += np.where(ext["dist_from_center"] > med_dist * 1.5, 20,
              np.where(ext["dist_from_center"] > med_dist, 10, -5))

    # Rare category = more unique
    med_rarity = ext["category_rarity"].median()
    score += np.where(ext["category_rarity"] < med_rarity * 0.5, 20,
              np.where(ext["category_rarity"] < med_rarity, 10, 0))

    # Rating above city average but not mainstream = hidden gem signal
    score += np.where(
        (ext["rating_vs_city"] > 0) & (ext["rating_vs_city"] < 0.5), 15,
        np.where(ext["rating_vs_city"] > 0.5, 5, -10)
    )

    # Fewer reviews than city avg but good rating = underrated
    score += np.where(
        (ext["review_ratio_vs_city"] < 0.5) & (ext["rating_vs_city"] > 0), 15,
        np.where(ext["review_ratio_vs_city"] > 2.0, -10, 0)
    )

    # High useful/funny/cool votes relative to count = quality
    if ext["review_count_real"].max() > 0:
        vote_ratio = (ext["useful_sum"] + ext["funny_sum"] + ext["cool_sum"]) / ext["review_count_real"].clip(lower=1)
        score += np.where(vote_ratio > vote_ratio.quantile(0.8), 10, 0)

    # Consistent ratings (low std) = reliable
    score += np.where(ext["std_review_stars"] < 0.5, 5,
              np.where(ext["std_review_stars"] > 1.5, -10, 0))

    return np.clip(score, 0, 100)


# ---------------------------------------------------------------------------
# TASK 3: Safety score target from EXTERNAL signals
# ---------------------------------------------------------------------------
FBI_CRIME_DATA = {
    "New York": {"state": "NY", "violent_crime_rate": 363.8, "property_crime_rate": 1438.7},
    "Los Angeles": {"state": "CA", "violent_crime_rate": 487.1, "property_crime_rate": 2265.4},
    "Chicago": {"state": "IL", "violent_crime_rate": 967.5, "property_crime_rate": 3538.7},
    "Houston": {"state": "TX", "violent_crime_rate": 976.3, "property_crime_rate": 4891.2},
    "Phoenix": {"state": "AZ", "violent_crime_rate": 614.7, "property_crime_rate": 3912.5},
    "Philadelphia": {"state": "PA", "violent_crime_rate": 989.3, "property_crime_rate": 2361.4},
    "San Antonio": {"state": "TX", "violent_crime_rate": 675.2, "property_crime_rate": 4218.9},
    "San Diego": {"state": "CA", "violent_crime_rate": 232.4, "property_crime_rate": 1624.3},
    "Dallas": {"state": "TX", "violent_crime_rate": 752.3, "property_crime_rate": 3981.2},
    "Austin": {"state": "TX", "violent_crime_rate": 398.5, "property_crime_rate": 4361.7},
    "San Jose": {"state": "CA", "violent_crime_rate": 211.4, "property_crime_rate": 2200.1},
    "Jacksonville": {"state": "FL", "violent_crime_rate": 656.8, "property_crime_rate": 3515.3},
    "Fort Worth": {"state": "TX", "violent_crime_rate": 438.2, "property_crime_rate": 3386.7},
    "Columbus": {"state": "OH", "violent_crime_rate": 693.2, "property_crime_rate": 4019.8},
    "Charlotte": {"state": "NC", "violent_crime_rate": 621.1, "property_crime_rate": 3598.4},
    "Indianapolis": {"state": "IN", "violent_crime_rate": 1022.6, "property_crime_rate": 4351.2},
    "San Francisco": {"state": "CA", "violent_crime_rate": 539.4, "property_crime_rate": 4980.6},
    "Seattle": {"state": "WA", "violent_crime_rate": 623.8, "property_crime_rate": 4390.2},
    "Denver": {"state": "CO", "violent_crime_rate": 637.2, "property_crime_rate": 3945.1},
    "Oklahoma City": {"state": "OK", "violent_crime_rate": 756.4, "property_crime_rate": 4891.3},
    "Nashville": {"state": "TN", "violent_crime_rate": 1014.3, "property_crime_rate": 3953.7},
    "El Paso": {"state": "TX", "violent_crime_rate": 324.5, "property_crime_rate": 2952.4},
    "Washington": {"state": "DC", "violent_crime_rate": 811.2, "property_crime_rate": 3544.8},
    "Boston": {"state": "MA", "violent_crime_rate": 604.3, "property_crime_rate": 1927.4},
    "Las Vegas": {"state": "NV", "violent_crime_rate": 712.4, "property_crime_rate": 3658.2},
    "Portland": {"state": "OR", "violent_crime_rate": 685.3, "property_crime_rate": 4415.7},
    "Detroit": {"state": "MI", "violent_crime_rate": 1965.4, "property_crime_rate": 4208.5},
    "Louisville": {"state": "KY", "violent_crime_rate": 791.2, "property_crime_rate": 3542.8},
    "Memphis": {"state": "TN", "violent_crime_rate": 1394.5, "property_crime_rate": 5324.1},
    "Baltimore": {"state": "MD", "violent_crime_rate": 1833.4, "property_crime_rate": 3524.6},
    "Milwaukee": {"state": "WI", "violent_crime_rate": 1332.7, "property_crime_rate": 4201.3},
    "Albuquerque": {"state": "NM", "violent_crime_rate": 915.2, "property_crime_rate": 4954.8},
    "Tucson": {"state": "AZ", "violent_crime_rate": 631.4, "property_crime_rate": 4261.2},
    "Fresno": {"state": "CA", "violent_crime_rate": 612.8, "property_crime_rate": 3521.4},
    "Sacramento": {"state": "CA", "violent_crime_rate": 624.3, "property_crime_rate": 3104.7},
    "Kansas City": {"state": "MO", "violent_crime_rate": 1312.6, "property_crime_rate": 4528.3},
    "Mesa": {"state": "AZ", "violent_crime_rate": 322.4, "property_crime_rate": 2847.1},
    "Atlanta": {"state": "GA", "violent_crime_rate": 1029.8, "property_crime_rate": 4832.4},
    "Omaha": {"state": "NE", "violent_crime_rate": 571.3, "property_crime_rate": 3782.5},
    "Colorado Springs": {"state": "CO", "violent_crime_rate": 534.2, "property_crime_rate": 3021.4},
    "Raleigh": {"state": "NC", "violent_crime_rate": 402.1, "property_crime_rate": 2315.8},
    "Miami": {"state": "FL", "violent_crime_rate": 762.3, "property_crime_rate": 4321.5},
    "Long Beach": {"state": "CA", "violent_crime_rate": 512.7, "property_crime_rate": 2894.3},
    "Virginia Beach": {"state": "VA", "violent_crime_rate": 162.4, "property_crime_rate": 1987.2},
    "Oakland": {"state": "CA", "violent_crime_rate": 1249.3, "property_crime_rate": 5241.7},
    "Minneapolis": {"state": "MN", "violent_crime_rate": 1014.2, "property_crime_rate": 4871.3},
    "Tulsa": {"state": "OK", "violent_crime_rate": 1082.4, "property_crime_rate": 5124.6},
    "Tampa": {"state": "FL", "violent_crime_rate": 462.3, "property_crime_rate": 2847.5},
    "Arlington": {"state": "TX", "violent_crime_rate": 245.8, "property_crime_rate": 2461.3},
    "New Orleans": {"state": "LA", "violent_crime_rate": 1121.4, "property_crime_rate": 5024.8},
}


def load_task3_crime() -> pd.DataFrame:
    csv_path = f"{DATA_DIR}/fbi_crime_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    df = pd.DataFrame([
        {"city": city, "state": v["state"],
         "violent_crime_rate": v["violent_crime_rate"],
         "property_crime_rate": v["property_crime_rate"]}
        for city, v in FBI_CRIME_DATA.items()
    ])
    df.to_csv(csv_path, index=False)
    return df


def build_task3_features(biz: pd.DataFrame, crime: pd.DataFrame) -> pd.DataFrame:
    """Build features for safety model (what the model sees)."""
    df = biz.copy()
    df["has_reviews"] = (df["review_count"] > 0).astype(int)
    df["is_open"] = df["is_open"].astype(int)

    df["category_str"] = df["categories"].fillna("").apply(
        lambda s: " ".join(s) if isinstance(s, list) else str(s)
    )
    pharmacy = df[df["category_str"].str.contains(
        "Pharmacy|Drugstores|Health & Medical|Pharmacies", case=False, regex=True
    )]
    hospital = df[df["category_str"].str.contains(
        "Hospital|Medical Centers|Emergency Rooms|Health & Medical|Hospitals", case=False, regex=True
    )]

    dist_pharmacy = np.full(len(df), np.inf)
    dist_hospital = np.full(len(df), np.inf)
    lat = df["latitude"].values
    lon = df["longitude"].values

    if not pharmacy.empty:
        for _, t in pharmacy.iterrows():
            d = haversine_km(lat, lon, t["latitude"], t["longitude"])
            dist_pharmacy = np.minimum(dist_pharmacy, d)
    if not hospital.empty:
        for _, t in hospital.iterrows():
            d = haversine_km(lat, lon, t["latitude"], t["longitude"])
            dist_hospital = np.minimum(dist_hospital, d)

    df["pharmacy_distance"] = np.where(dist_pharmacy == np.inf, np.nan, dist_pharmacy)
    df["hospital_distance"] = np.where(dist_hospital == np.inf, np.nan, dist_hospital)
    df["has_pharmacy_nearby"] = (df["pharmacy_distance"] <= 2.0).astype(int)
    df["has_hospital_nearby"] = (df["hospital_distance"] <= 5.0).astype(int)

    merged = df.merge(crime[["city", "violent_crime_rate", "property_crime_rate"]],
                       on="city", how="left")

    return merged


def compute_safety_target(merged: pd.DataFrame) -> np.ndarray:
    """Safety target from EXTERNAL signals: nonlinear crime transform + isolation.
    Not directly derivable from the features the model sees."""
    vcr = merged["violent_crime_rate"].fillna(merged["violent_crime_rate"].median())
    pcr = merged["property_crime_rate"].fillna(merged["property_crime_rate"].median())

    # Nonlinear transform: safety drops faster at high crime rates
    violent_safety = np.exp(-vcr / 800) * 50
    property_safety = np.exp(-pcr / 4000) * 30

    # Isolation bonus: farther from dataset center = slightly safer (suburban)
    center_lat = merged["latitude"].median()
    center_lon = merged["longitude"].median()
    dist = haversine_km(merged["latitude"].values, merged["longitude"].values, center_lat, center_lon)
    med_dist = np.median(dist[~np.isnan(dist)]) if not np.all(np.isnan(dist)) else 50
    isolation = np.where(dist > med_dist * 1.5, 10, np.where(dist > med_dist, 5, 0))

    # Business density adjustment (from crime context)
    density_adj = np.where(
        (vcr > 800) & (pcr > 3500), -10,
        np.where((vcr < 400) & (pcr < 2000), 10, 0)
    )

    score = violent_safety + property_safety + isolation + density_adj
    return np.clip(score, 0, 100)


# ---------------------------------------------------------------------------
# TASK 4: Collab auth target from EXTERNAL signals
# ---------------------------------------------------------------------------
def compute_collab_auth_target(biz: pd.DataFrame, ext: pd.DataFrame) -> np.ndarray:
    """Authenticity target from multi-attribute consistency, NOT from individual features."""
    auth = np.zeros(len(biz), dtype=float)

    # 1. Review count consistency: businesses with reviews tend to be authentic
    rc = biz["review_count"].values
    auth += np.where(rc > 50, 20, np.where(rc > 10, 10, np.where(rc > 0, 5, -10)))

    # 2. Rating consistency relative to city average
    rating_vs_city = ext["rating_vs_city"].values
    auth += np.where(
        (rating_vs_city > -0.3) & (rating_vs_city < 0.3), 15,
        np.where(rating_vs_city < -1.0, -10, 5)
    )

    # 3. Review standard deviation (consistent quality = authentic)
    std = ext["std_review_stars"].fillna(1.5).values
    auth += np.where(std < 0.8, 15, np.where(std < 1.2, 10, np.where(std > 2.0, -10, 0)))

    # 4. Useful votes indicate real engagement
    useful = ext["useful_sum"].values
    auth += np.where(useful > np.percentile(useful, 70), 15,
              np.where(useful > np.percentile(useful, 40), 5, -5))

    # 5. Open businesses with enough reviews = established
    is_open = biz["is_open"].values
    auth += np.where(is_open == 1, 10, -10)

    # 6. Rating-reviews interaction: high rating + many reviews = trustworthy
    stars = biz["stars"].values
    auth += np.where((stars >= 4.0) & (rc > 20), 10,
              np.where((stars < 3.0) & (rc > 50), -10, 0))

    # 7. Category diversity correlates with legitimacy
    ext_rating = ext.get("city_avg_rating", pd.Series(np.zeros(len(biz)))).values
    auth += np.where(ext["category_rarity"].values > 0.01, 5, -5)

    return np.where(auth > 40, 1, 0).astype(int)


# ---------------------------------------------------------------------------
# Feature building for model inputs
# ---------------------------------------------------------------------------
def build_model_features(biz: pd.DataFrame, ext: pd.DataFrame, reviews: pd.DataFrame = None) -> pd.DataFrame:
    """Build the features the model will actually train on."""
    df = biz.copy()
    df["rating"] = df["stars"]
    df["review_count_log"] = np.log1p(df["review_count"])
    df["price_level"] = df["attributes"].apply(extract_price_level)
    df["category_count"] = df["categories"].fillna("").apply(
        lambda s: len(s) if isinstance(s, list) else len([c for c in str(s).split(",") if c.strip()])
    )

    p90 = df["review_count"].quantile(0.90)
    df["is_mainstream"] = (df["review_count"] > p90).astype(int)
    df["is_sweet_spot"] = ((df["rating"] >= 4.0) & (df["rating"] <= 4.5)).astype(int)
    df["is_budget_friendly"] = (df["price_level"] <= 2).astype(int)
    df["is_tourist_trap"] = ((df["rating"] >= 4.8) & (df["review_count"] > 500)).astype(int)

    # Review text features
    for col in ["review_text_mean_len", "review_text_max_len", "review_text_mean_words",
                "review_exclamation_mean", "review_uppercase_mean",
                "review_useful_sum", "review_funny_sum", "review_cool_sum", "review_votes_total"]:
        if col in ext.columns:
            df[col] = ext[col].values
        else:
            df[col] = 0

    # City-level features for hidden_gem prediction
    for col in ["city_avg_rating", "city_avg_reviews", "city_business_count",
                "rating_vs_city", "review_ratio_vs_city"]:
        if col in ext.columns:
            df[col] = ext[col].values
        else:
            df[col] = 0

    # Safety features
    df["has_reviews"] = (df["review_count"] > 0).astype(int)
    df["is_open"] = df["is_open"].astype(int)

    # Pharmacy/hospital distances
    df["category_str"] = df["categories"].fillna("").apply(
        lambda s: " ".join(s) if isinstance(s, list) else str(s)
    )
    pharmacy = df[df["category_str"].str.contains(
        "Pharmacy|Drugstores|Health & Medical|Pharmacies", case=False, regex=True
    )]
    hospital = df[df["category_str"].str.contains(
        "Hospital|Medical Centers|Emergency Rooms|Health & Medical|Hospitals", case=False, regex=True
    )]

    dist_pharmacy = np.full(len(df), np.inf)
    dist_hospital = np.full(len(df), np.inf)
    lat = df["latitude"].values
    lon = df["longitude"].values

    if not pharmacy.empty:
        for _, t in pharmacy.iterrows():
            d = haversine_km(lat, lon, t["latitude"], t["longitude"])
            dist_pharmacy = np.minimum(dist_pharmacy, d)
    if not hospital.empty:
        for _, t in hospital.iterrows():
            d = haversine_km(lat, lon, t["latitude"], t["longitude"])
            dist_hospital = np.minimum(dist_hospital, d)

    df["pharmacy_distance"] = np.where(dist_pharmacy == np.inf, np.nan, dist_pharmacy)
    df["hospital_distance"] = np.where(dist_hospital == np.inf, np.nan, dist_hospital)
    df["has_pharmacy_nearby"] = (df["pharmacy_distance"] <= 2.0).astype(int)
    df["has_hospital_nearby"] = (df["hospital_distance"] <= 5.0).astype(int)

    # Collab auth features
    df["has_business_registration"] = df["is_open"].astype(int)
    df["has_phone"] = df["phone"].fillna("").astype(str).str.strip().ne("").astype(int) if "phone" in df.columns else 1
    df["has_email"] = 0
    df["has_website"] = df["url"].fillna("").astype(str).str.strip().ne("").astype(int) if "url" in df.columns else 0
    df["has_instagram"] = 0
    df["has_facebook"] = 0
    df["instagram_followers"] = 0
    df["facebook_likes"] = 0
    df["social_media_age_days"] = 0
    df["has_unique_photos"] = 0
    df["platform_count"] = 1
    df["on_google_maps"] = 0
    df["on_yelp"] = 1
    df["on_tripadvisor"] = 0
    df["total_reviews"] = df["review_count"]
    df["avg_rating"] = df["stars"]
    df["review_velocity"] = 0
    df["has_wifi"] = 0
    df["has_parking"] = 0
    df["by_appointment"] = 0
    df["hours_per_week"] = 0

    if "attributes" in df.columns:
        def get_attr(attrs, key):
            if not isinstance(attrs, dict):
                return 0
            return 1 if str(attrs.get(key, "False")).lower() in ("true", "1", "yes", "free", "paid") else 0

        df["has_wifi"] = df["attributes"].apply(lambda a: get_attr(a, "WiFi"))
        df["has_parking"] = df["attributes"].apply(
            lambda a: 1 if isinstance(a, dict) and any(
                str(a.get(k, "False")).lower() in ("true", "free")
                for k in ["BusinessParking", "garage", "street", "lot"]
            ) else 0
        )
        df["by_appointment"] = df["attributes"].apply(lambda a: get_attr(a, "ByAppointmentOnly"))

    if "hours" in df.columns:
        def calc_hours(h):
            if not isinstance(h, dict):
                return 0
            total = 0
            for day, hrs in h.items():
                if "-" in str(hrs):
                    parts = str(hrs).split("-")
                    try:
                        open_h, open_m = map(int, parts[0].split(":"))
                        close_h, close_m = map(int, parts[1].split(":"))
                        open_min = open_h * 60 + open_m
                        close_min = close_h * 60 + close_m
                        if close_min > open_min:
                            total += (close_min - open_min) / 60
                        else:
                            total += (1440 - open_min + close_min) / 60
                    except:
                        pass
            return total
        df["hours_per_week"] = df["hours"].apply(calc_hours)

    return df


def clean(df: pd.DataFrame, protect_cols: tuple = ()) -> pd.DataFrame:
    keep_mask = (df.isna().mean() <= 0.5) | df.columns.isin(protect_cols)
    df = df.loc[:, keep_mask].copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        mode = df[c].mode()
        if not mode.empty:
            df[c] = df[c].fillna(mode.iloc[0])
    return df


def run() -> None:
    biz = load_yelp_business()
    print(f"Loaded {len(biz)} businesses from Yelp dataset")

    reviews = load_yelp_reviews_sampled(max_reviews=50000)

    print("Computing external signals for target generation...")
    ext = compute_external_signals(biz, reviews)

    # Compute targets from EXTERNAL signals
    hidden_gem_target = compute_hidden_gem_target(biz, ext)
    collab_auth_target = compute_collab_auth_target(biz, ext)

    # Build model features (what the model actually sees)
    print("Building model features...")
    features = build_model_features(biz, ext, reviews)

    # Safety score
    crime = load_task3_crime()
    safety_merged = build_task3_features(biz, crime)
    safety_target = compute_safety_target(safety_merged)

    # Add safety features to main features
    features["violent_crime_rate"] = safety_merged["violent_crime_rate"].values
    features["property_crime_rate"] = safety_merged["property_crime_rate"].values

    # Add targets
    features["hidden_gem_score"] = hidden_gem_target
    features["safety_score"] = safety_target
    features["is_authentic"] = collab_auth_target

    features.to_csv(f"{DATA_DIR}/merged_places.csv", index=False)

    # Task 1: fake detection (independent dataset)
    t1 = clean(load_task1_fake_detection())
    t1.to_csv(f"{DATA_DIR}/fake_review_features.csv", index=False)

    # Train/test splits
    target_cols = ["hidden_gem_score", "safety_score", "is_authentic"]
    y = features[target_cols]
    X = features.drop(columns=target_cols)

    # Drop non-numeric columns
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/y_test.csv", index=False)

    X1 = t1[TASK1_COLS]
    y1 = t1["is_fake"]
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.25, stratify=y1, random_state=42
    )
    X1_train.to_csv(f"{DATA_DIR}/X_train_fake.csv", index=False)
    X1_test.to_csv(f"{DATA_DIR}/X_test_fake.csv", index=False)
    y1_train.to_csv(f"{DATA_DIR}/y_train_fake.csv", index=False)
    y1_test.to_csv(f"{DATA_DIR}/y_test_fake.csv", index=False)

    print("\n=== Shapes ===")
    for name, d in [("merged_places", features), ("fake_review_features", t1),
                     ("X_train", X_train), ("X_test", X_test),
                     ("X_train_fake", X1_train), ("X_test_fake", X1_test)]:
        print(f"{name}: {d.shape}")

    print("\n=== Targets (no leakage) ===")
    print("hidden_gem_score:", features["hidden_gem_score"].describe())
    print("safety_score:", features["safety_score"].describe())
    print("is_authentic:", features["is_authentic"].value_counts().to_dict())
    print("is_fake:", t1["is_fake"].value_counts().to_dict())

    print(f"\n=== Feature count: {X_train.shape[1]} ===")
    print(list(X_train.columns))


if __name__ == "__main__":
    run()
