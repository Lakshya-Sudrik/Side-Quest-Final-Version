"""India data loaders + geocoding + OSM infrastructure for SideQuest.

Replaces the US Yelp pipeline. Sources:
  - zomato.csv            (Bangalore, 51,717 rows, reviews + phone)
  - swiggy.csv            (India-wide, 148,541 rows, license numbers)
  - NCRB district crime   (2017-2022, 788 districts, 25 categories)
  - census2011            (district population -> per-capita crime rates)
  - Google Places API     (exact geocoding of areas/cities, website, phone)
  - Nominatim (OSM)       (geocoding fallback, no key)
  - Overpass API          (hospitals/police infrastructure per city)
"""
import ast
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "india"
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

GEOCODE_CACHE = CACHE / "geocode_cache.json"
INFRA_CACHE = CACHE / "infra_cache.json"

# Bangalore city centre (for dist_from_center)
BLR_CENTER = (12.9716, 77.5946)

# Indian metro weights for Indianizing user profiles
INDIAN_CITIES = [
    ("Mumbai", 0.13), ("Delhi", 0.12), ("Bangalore", 0.11), ("Hyderabad", 0.08),
    ("Chennai", 0.07), ("Pune", 0.06), ("Kolkata", 0.06), ("Ahmedabad", 0.05),
    ("Jaipur", 0.04), ("Lucknow", 0.04), ("Surat", 0.03), ("Indore", 0.03),
    ("Chandigarh", 0.03), ("Kochi", 0.03), ("Goa", 0.03), ("Nagpur", 0.03),
    ("Bhopal", 0.02), ("Guwahati", 0.02), ("Dehradun", 0.02), ("Shimla", 0.02),
]


def _get_env(name: str, default: str = "") -> str:
    """Read key from .env without python-dotenv."""
    val = os.environ.get(name, "")
    if val:
        return val
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return v
    return default


def mask_secret(s: str) -> str:
    """Mask a secret for display: ****last4."""
    if not s:
        return "(none)"
    return f"****{s[-4:]}" if len(s) > 4 else "****"


def load_json_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_json_cache(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------
def load_zomato() -> pd.DataFrame:
    """Bangalore restaurants with reviews, phone, 93 areas."""
    path = RAW / "zomato.csv"
    logger.info("Loading Zomato Bangalore...")
    df = pd.read_csv(path, on_bad_lines="skip")
    out = pd.DataFrame(index=range(len(df)))
    out["source"] = "zomato"
    out["place_id"] = ["zl_" + str(i) for i in range(len(df))]
    out["name"] = df["name"].astype(str)
    out["address"] = df["address"].fillna("")
    out["area"] = df["location"].fillna("unknown")
    out["city"] = "Bangalore"
    out["state"] = "Karnataka"

    # rating "4.1/5" -> 4.1
    rate = df["rate"].astype(str).str.extract(r"([\d.]+)")[0]
    out["stars"] = pd.to_numeric(rate, errors="coerce")
    out["review_count"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)

    # cuisines -> list
    out["categories"] = df["cuisines"].fillna("").apply(
        lambda s: [c.strip() for c in str(s).split(",") if c.strip()]
    )
    # rest_type
    out["rest_type"] = df["rest_type"].fillna("")

    # price bands for India (INR for two people)
    cost = pd.to_numeric(
        df["approx_cost(for two people)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    out["cost_inr"] = cost
    out["price_level"] = pd.cut(
        cost, bins=[0, 300, 700, 1500, 100000], labels=[1, 2, 3, 4]
    ).astype(float)

    out["has_phone"] = df["phone"].fillna("").astype(str).str.strip().ne("").astype(int)
    out["has_website"] = df["url"].fillna("").astype(str).str.strip().ne("").astype(int)
    out["has_online_order"] = (df["online_order"] == "Yes").astype(int)
    out["has_book_table"] = (df["book_table"] == "Yes").astype(int)
    out["has_license"] = 0  # zomato has no license column
    out["dish_liked"] = df["dish_liked"].fillna("")

    # reviews: parse the python-literal list of (rating, text) tuples
    out["reviews_raw"] = df["reviews_list"].fillna("[]")
    out["has_reviews_text"] = out["reviews_raw"].ne("[]").astype(int)
    out["address_for_geocode"] = (
        out["address"].astype(str) + ", " + out["area"].astype(str) + ", Bangalore, India"
    )
    # Zomato lists the SAME restaurant once per listing type (Delivery, Dine-out,
    # Cafes, ...), with identical reviews. Left in, copies of one restaurant land
    # on both sides of a random train/test split and inflate every metric.
    # Keep one row per (name, address): the copy with the most votes.
    key = (out["name"].str.lower().str.strip() + "|"
           + out["address"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip())
    n_before = len(out)
    keep = out.assign(_k=key).sort_values("review_count", ascending=False).drop_duplicates("_k").index
    out = out.loc[sorted(keep)].copy()
    logger.info(f"  Zomato dedupe: {n_before} rows -> {len(out)} unique restaurants")

    logger.info(f"  Zomato: {len(out)} places, {out.has_phone.sum()} with phone, "
                f"{out.stars.notna().sum()} with rating")
    return out


def load_swiggy() -> pd.DataFrame:
    """India-wide Swiggy restaurants with license numbers."""
    path = RAW / "swiggy.csv"
    logger.info("Loading Swiggy...")
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=range(len(df)))
    out["source"] = "swiggy"
    out["place_id"] = ["sw_" + str(x) for x in df["id"].values]
    out["name"] = df["name"].astype(str)
    out["address"] = df["address"].fillna("")

    # city may be "BTM,Bangalore" (locality,city) or plain city
    raw_city = df["city"].fillna("unknown").astype(str)
    out["raw_city"] = raw_city
    has_comma = raw_city.str.contains(",")
    out["area"] = np.where(has_comma, raw_city.str.split(",").str[0], raw_city)
    out["city"] = np.where(has_comma, raw_city.str.split(",").str[-1], raw_city)
    out["state"] = ""  # filled after geocoding

    out["stars"] = pd.to_numeric(df["rating"], errors="coerce")

    # "50+ ratings" -> 50, "Too Few Ratings" -> 0
    rc = df["rating_count"].astype(str)
    rc_num = pd.to_numeric(rc.str.extract(r"(\d+)")[0], errors="coerce")
    out["review_count"] = rc_num.fillna(0).astype(int)

    out["categories"] = df["cuisine"].fillna("").apply(
        lambda s: [c.strip() for c in str(s).split(",") if c.strip()]
    )
    out["rest_type"] = ""

    cost = pd.to_numeric(
        df["cost"].astype(str).str.replace("₹", "", regex=False).str.strip().str.replace(",", "", regex=False),
        errors="coerce",
    )
    out["cost_inr"] = cost
    out["price_level"] = pd.cut(
        cost, bins=[0, 300, 700, 1500, 100000], labels=[1, 2, 3, 4]
    ).astype(float)

    out["has_phone"] = 0  # swiggy csv has no phone column
    out["has_website"] = df["link"].fillna("").astype(str).str.strip().ne("").astype(int)
    out["has_online_order"] = 1  # swiggy is delivery by definition
    out["has_book_table"] = 0
    out["has_license"] = df["lic_no"].notna().astype(int)
    out["dish_liked"] = ""
    out["reviews_raw"] = "[]"
    out["has_reviews_text"] = 0
    out["address_for_geocode"] = (
        out["address"].astype(str) + ", " + out["city"].astype(str) + ", India"
    )
    logger.info(f"  Swiggy: {len(out)} places, {out.has_license.sum()} with license, "
                f"{out.stars.notna().sum()} with rating, {out.city.nunique()} cities")
    return out


def load_crime() -> tuple[pd.DataFrame, dict]:
    """NCRB district crime + census population -> per-capita rates.

    Returns (district_rates DataFrame, state_rates dict by state name).
    """
    ncrb = pd.read_csv(RAW / "NCRB_Crime_Analysis_2017_2022_with_25Crime_categories_Districtwise.csv")
    census = pd.read_csv(RAW / "india-districts-census-2011.csv")

    # population by district (census 2011 names differ from NCRB: case + aliases)
    def _norm(s: str) -> str:
        s = str(s).strip().lower()
        for a, b in (
            ("bengaluru", "bangalore"), ("mumbai", "mumbai"),
            ("chennai", "madras"), ("kolkata", "calcutta"),
            ("thiruvananthapuram", "trivandrum"), ("varanasi", "benares"),
            ("gautam buddh nagar", "gautam buddh"),
            ("north  & middle andaman", "north and middle andaman"),
        ):
            s = s.replace(a, b)
        return s

    pop_lookup = {}
    for _, r in census.iterrows():
        pop_lookup[_norm(r["District name"])] = r["Population"]
    # census state names are UPPERCASE -> normalize keys
    state_pop = {_norm(k): v for k, v in census.groupby("State name")["Population"].sum().to_dict().items()}

    # violent / property crime groups
    ncrb["violent"] = (
        ncrb["murder_homicide"].fillna(0) + ncrb["rape_sexual_violence"].fillna(0)
        + ncrb["assault_modesty"].fillna(0) + ncrb["kidnapping_abduction"].fillna(0)
    )
    ncrb["property"] = (
        ncrb["theft_property_crimes"].fillna(0) + ncrb["forgery_counterfeiting"].fillna(0)
        + ncrb["cheating_fraud"].fillna(0)
    )

    # match population by district name (normalized), else state total share
    def get_pop(row):
        d = _norm(row["district_name"])
        if d in pop_lookup:
            return pop_lookup[d]
        # fuzzy contains both directions
        for k, v in pop_lookup.items():
            if d in k or k in d:
                return v
        sp = state_pop.get(_norm(row["state_name"]), np.nan)
        return sp / 50 if not np.isnan(sp) else np.nan  # rough: districts ~50/state

    ncrb["population"] = ncrb.apply(get_pop, axis=1)
    ncrb["violent_rate"] = ncrb["violent"] / ncrb["population"] * 100000
    ncrb["property_rate"] = ncrb["property"] / ncrb["population"] * 100000
    # crimes against women (the "women solo traveller" signal for the safety index)
    ncrb["women"] = (
        ncrb["rape_sexual_violence"].fillna(0) + ncrb["assault_modesty"].fillna(0)
    )
    ncrb["women_rate"] = ncrb["women"] / ncrb["population"] * 100000

    # latest year per district (2022 preferred)
    latest = ncrb[ncrb["year"] == ncrb["year"].max()].copy()
    district_rates = latest[
        ["state_name", "district_name", "population", "violent_rate", "property_rate",
         "women_rate"]
    ].reset_index(drop=True)

    # state aggregate: mean of district rates weighted by population
    st = ncrb.copy()
    st_agg = st.groupby(["year", "state_name"]).apply(
        lambda g: pd.Series({
            "violent_rate": np.average(g["violent_rate"].fillna(0), weights=g["population"].fillna(1)),
            "property_rate": np.average(g["property_rate"].fillna(0), weights=g["population"].fillna(1)),
        }),
        include_groups=False,
    ).reset_index()
    state_rates = (
        st_agg[st_agg["year"] == st_agg["year"].max()]
        .set_index("state_name")[["violent_rate", "property_rate"]]
        .to_dict("index")
    )
    # casefold keys so Nominatim "karnataka" matches NCRB "Karnataka"
    state_rates = {str(k).strip().lower(): v for k, v in state_rates.items()}
    logger.info(f"  Crime: {len(district_rates)} districts, {len(state_rates)} states, "
                f"year {int(ncrb['year'].max())}")
    return district_rates, state_rates


# ---------------------------------------------------------------------------
# GEOCODING (Google Places first, Nominatim fallback, cached)
# ---------------------------------------------------------------------------
def geocode_place(query: str, cache: dict, google_key: str = "",
                  session: requests.Session = None) -> dict:
    """Return {'lat':..,'lng':..,'state':..,'source':..} or {}."""
    if query in cache:
        return cache[query]
    sess = session or requests
    # 1. Google Places Text Search (fast, exact)
    if google_key:
        try:
            r = sess.post(
                "https://places.googleapis.com/v1/places:searchText",
                headers={
                    "X-Goog-Api-Key": google_key,
                    "Content-Type": "application/json",
                    "X-Goog-FieldMask": "places.location,places.formattedAddress",
                },
                json={"textQuery": query, "languageCode": "en", "regionCode": "IN"},
                timeout=20,
            )
            if r.status_code == 200:
                places = r.json().get("places", [])
                if places:
                    loc = places[0].get("location", {})
                    addr = places[0].get("formattedAddress", "")
                    state = ""
                    for part in str(addr).split(","):
                        p = part.strip()
                        if p in (
                            "Karnataka", "Maharashtra", "Delhi", "Tamil Nadu", "Telangana",
                            "West Bengal", "Gujarat", "Rajasthan", "Uttar Pradesh",
                            "Madhya Pradesh", "Kerala", "Punjab", "Haryana", "Bihar",
                            "Odisha", "Assam", "Goa", "Chhattisgarh", "Jharkhand",
                            "Uttarakhand", "Himachal Pradesh", "Andhra Pradesh",
                        ):
                            state = p
                    res = {
                        "lat": loc.get("latitude"), "lng": loc.get("longitude"),
                        "state": state, "src": "google",
                    }
                    cache[query] = res
                    return res
            time.sleep(0.15)
        except Exception as e:
            logger.debug(f"google geocode failed for {query}: {e}")

    # 2. Nominatim fallback (1 req/sec policy)
    try:
        r = sess.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "countrycodes": "in"},
            headers={"User-Agent": "SideQuestResearch/1.0"},
            timeout=20,
        )
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            res = {"lat": float(d["lat"]), "lng": float(d["lon"]), "state": "", "src": "nominatim"}
            cache[query] = res
            time.sleep(1.05)  # nominatim rate limit
            return res
    except Exception as e:
        logger.debug(f"nominatim failed for {query}: {e}")

    cache[query] = {}
    return {}


def geocode_all(queries: list[str], google_key: str = "", label: str = "areas") -> dict:
    """Geocode unique query strings with persistent cache."""
    cache = load_json_cache(GEOCODE_CACHE)
    todo = [q for q in dict.fromkeys(queries) if q and q not in cache]
    logger.info(f"Geocoding {len(todo)} new {label} ({len(queries)} total refs)...")
    sess = requests.Session()
    for i, q in enumerate(todo, 1):
        geocode_place(q, cache, google_key, sess)
        if i % 25 == 0:
            save_json_cache(GEOCODE_CACHE, cache)
            logger.info(f"  {i}/{len(todo)} geocoded")
    save_json_cache(GEOCODE_CACHE, cache)
    hits = sum(1 for q in cache if cache[q])
    logger.info(f"  geocode cache: {hits}/{len(cache)} resolved")
    return cache


# ---------------------------------------------------------------------------
# OSM INFRASTRUCTURE (hospitals / police near each city)
# ---------------------------------------------------------------------------
def query_city_infra(south, west, north, east) -> dict:
    """Overpass: hospitals+clinics+police within bbox. Returns counts + coords."""
    q = (
        f"[out:json][timeout:40];"
        f"((node[amenity~'hospital|clinic|doctors|police']({south},{west},{north},{east});"
        f"way[amenity~'hospital|clinic|doctors|police']({south},{west},{north},{east}););"
        f");out center;"
    )
    try:
        r = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": q},
            headers={"User-Agent": "SideQuestResearch/1.0"},
            timeout=60,
        )
        if r.status_code != 200:
            return {}
        els = r.json().get("elements", [])
        pts = []
        for e in els:
            lat = e.get("lat") or (e.get("center", {}) or {}).get("lat")
            lng = e.get("lon") or (e.get("center", {}) or {}).get("lon")
            if lat and lng:
                pts.append({
                    "lat": lat, "lng": lng,
                    "type": e.get("tags", {}).get("amenity", ""),
                })
        return {"points": pts}
    except Exception as e:
        logger.debug(f"overpass failed: {e}")
        return {}


def build_infra_index(city_coords: dict) -> dict:
    """city -> {hospitals: [...], police: [...]} with persistent cache."""
    cache = load_json_cache(INFRA_CACHE)
    todo = [c for c in city_coords if c not in cache]
    logger.info(f"Fetching OSM infra for {len(todo)} new cities...")
    for i, city in enumerate(todo, 1):
        cc = city_coords[city]
        lat = float(cc["latitude"])
        lng = float(cc["longitude"])
        pad = 0.12  # ~13km
        res = query_city_infra(lat - pad, lng - pad, lat + pad, lng + pad)
        cache[city] = res or {"points": []}
        if i % 10 == 0:
            save_json_cache(INFRA_CACHE, cache)
            logger.info(f"  {i}/{len(todo)} cities OSM-queried")
        time.sleep(1.2)  # be polite to Overpass
    save_json_cache(INFRA_CACHE, cache)
    return cache


# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_km(lat, lng, points) -> float:
    """Distance (km) to nearest point in [{'lat','lng','type'}, ...]."""
    if not points:
        return np.nan
    lats = np.array([p["lat"] for p in points])
    lngs = np.array([p["lng"] for p in points])
    return float(haversine_km(lat, lng, lats, lngs).min())
