"""India place/town coordinates from the GeoNames gazetteer (data/geo/IN.txt).

    python src/geo_india.py      # builds data/geo/towns_india.csv.gz (compact, committed)

- towns : populated places (feature class P) -> used for "From / To" typed by the
          user and for placing an attraction in its town.
- sites : temples, forts, ruins, hills, lakes, falls ... (classes S/T/H/L) -> an
          attraction whose name matches a site within 60 km of its town gets the
          site's EXACT coordinates; otherwise it gets the town centre.
"""
from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEO_DIR = ROOT / "data" / "geo"
RAW = GEO_DIR / "IN.txt"
TOWNS = GEO_DIR / "towns_india.csv.gz"
ANCHORS = GEO_DIR / "city_anchors.csv"      # verified city coordinates (Vynex + restaurant data)
# Tourist towns whose name also exists elsewhere (Manali is also a Chennai suburb): which
# state the traveller means. Coordinates still come from GeoNames within that state.
TOURIST_STATE = {
    "manali": "himachal pradesh", "kasol": "himachal pradesh", "kullu": "himachal pradesh",
    "dharamshala": "himachal pradesh", "mcleod ganj": "himachal pradesh", "bir": "himachal pradesh",
    "mandu": "madhya pradesh", "orchha": "madhya pradesh", "pachmarhi": "madhya pradesh",
    "dwarka": "gujarat", "somnath": "gujarat", "katra": "jammu and kashmir", "pahalgam": "jammu and kashmir",
    "gulmarg": "jammu and kashmir", "digha": "west bengal", "bishnupur": "west bengal",
    "mayapur": "west bengal", "murshidabad": "west bengal", "halebid": "karnataka", "gokarna": "karnataka",
    "coorg": "karnataka", "madikeri": "karnataka", "hampi": "karnataka", "badami": "karnataka",
    "munnar": "kerala", "varkala": "kerala", "alleppey": "kerala", "alappuzha": "kerala", "wayanad": "kerala",
    "ooty": "tamil nadu", "kodaikanal": "tamil nadu", "yercaud": "tamil nadu", "rameswaram": "tamil nadu",
    "pushkar": "rajasthan", "mount abu": "rajasthan", "jaisalmer": "rajasthan", "bundi": "rajasthan",
    "rishikesh": "uttarakhand", "mussoorie": "uttarakhand", "nainital": "uttarakhand", "auli": "uttarakhand",
    "darjeeling": "west bengal", "gangtok": "sikkim", "shillong": "meghalaya", "cherrapunji": "meghalaya",
    "lonavala": "maharashtra", "mahabaleshwar": "maharashtra", "matheran": "maharashtra",
    "alibag": "maharashtra", "tawang": "arunachal pradesh", "ziro": "arunachal pradesh",
}
# same name twice in the SAME state: pick the GeoNames candidate nearest this point
TOURIST_NEAR = {"mahabaleshwar": (17.92, 73.66)}
SITES = GEO_DIR / "sites_india.csv.gz"
COLS = ["gid", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2",
        "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
# site feature codes that are real places to visit (not hotels, stations, farms ...)
SITE_CODES = {"TMPL", "SHRN", "CH", "MSQE", "MSTY", "RLG", "FT", "RUIN", "CSTL", "PAL", "MUS",
              "MNMT", "ANS", "HSTS", "TOWR", "GDN", "PRK", "ZOO", "OBPT", "BDG", "DAM", "LTHSE",
              "CAVE", "MT", "HLL", "PK", "PASS", "VAL", "GRGE", "CLF", "WTRFL", "LK", "LKS", "RSV",
              "BCH", "BCHS", "ISL", "SPNG", "SPNT", "PRKS", "RESN", "RESF", "RESW", "PPLH",
              "CMTY", "GRVE", "SQR", "MKT", "MALL", "AMTH", "STDM", "UNIV", "LIBR", "PIER"}
# spellings used by review sites that GeoNames stores differently
CITY_ALIASES = {
    "new delhi": "delhi", "trivandrum": "thiruvananthapuram", "panjim": "panaji",
    "bengaluru": "bangalore", "kochi (cochin)": "kochi", "cochin": "kochi", "calicut": "kozhikode",
    "pondicherry": "puducherry", "south andaman island": "port blair", "jammu city": "jammu",
    "sohra": "cherrapunji", "hubli-dharwad": "hubli", "mahabalipuram": "mamallapuram",
    "gurugram": "gurgaon", "vizag": "visakhapatnam", "prayagraj": "allahabad",
    "chittaurgarh": "chittorgarh", "bodh gaya": "bodhgaya", "alibaug": "alibag",
    "mysuru": "mysore", "vadodara": "baroda", "kanyakumari": "kanniyakumari",
    "mcleod ganj": "dharamsala", "thekkady": "kumily", "kutch": "bhuj", "havelock island": "port blair",
    "neil island": "port blair", "mollem national park": "mollem", "chikkaballapur": "chik ballapur",
    "thiruvannamalai": "tiruvannamalai", "raigad": "mahad", "santiniketan": "bolpur",
    "ranthambore national park": "sawai madhopur", "athirappilly": "chalakudi", "naggar": "kulu",
    "kumbhalgarh": "kelwara", "mawlynnong": "pynursla", "murdeshwar": "bhatkal", "badrinath": "joshimath",
    "belakavadi": "malavalli", "thuckalay": "thuckalay", "kaushambi": "manjhanpur",
    # common tourist names people type
    "coorg": "madikeri", "kodagu": "madikeri", "pondy": "puducherry", "chikmagalur": "chikmagalur",
    "chickmagalur": "chikmagalur", "ooty": "ooty", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "benares": "varanasi", "banaras": "varanasi", "kashi": "varanasi",
    "gurugram": "gurgaon", "trichy": "tiruchirappalli", "vizag": "visakhapatnam", "cochin": "kochi",
}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build(raw: Path = RAW) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_csv(raw, sep="\t", header=None, names=COLS, low_memory=False, quoting=3,
                    usecols=["gid", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "a1", "pop"])
    g["pop"] = pd.to_numeric(g["pop"], errors="coerce").fillna(0)
    towns = g[g["fclass"] == "P"]
    rows = []
    for r in towns.itertuples(index=False):
        names = {norm(r.name), norm(r.ascii)}
        # alternate names only for towns with people - keeps the file small
        if r.pop >= 5000 and isinstance(r.alt, str):
            names |= {norm(a) for a in r.alt.split(",")[:30] if a and a.isascii()}
        for n in names:
            if n:
                rows.append((n, r.lat, r.lon, r.pop, r.fcode, r.a1, r.name))
    t = pd.DataFrame(rows, columns=["key", "lat", "lon", "pop", "fcode", "admin1", "display"])
    # several places share a name (Manali in Himachal AND a Chennai suburb): keep up to 8
    # candidates per name; town_coords() picks between them
    t["exact"] = (t["key"] == t["display"].map(norm)).astype(int)
    t = t.sort_values(["exact", "pop"], ascending=False).groupby("key").head(8)
    sites = g[(g["fclass"].isin(["S", "T", "H", "L"])) & g["fcode"].isin(SITE_CODES)]
    s = pd.DataFrame({"key": sites["name"].map(norm), "lat": sites["lat"], "lon": sites["lon"],
                      "fcode": sites["fcode"], "display": sites["name"]})
    s = s[s["key"].str.len() >= 4]
    return t, s


def ensure_built() -> None:
    if TOWNS.exists() and SITES.exists():
        return
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} missing - unzip GeoNames IN.zip into data/geo/")
    t, s = build()
    t.to_csv(TOWNS, index=False, compression="gzip")
    s.to_csv(SITES, index=False, compression="gzip")


@lru_cache(maxsize=1)
def _towns() -> Dict[str, list]:
    """name -> list of (lat, lon, pop), best first (exact spelling, then population)."""
    ensure_built()
    t = pd.read_csv(TOWNS)
    if "exact" not in t.columns:
        t["exact"] = 1
    t = t.sort_values(["exact", "pop"], ascending=False)
    out: Dict[str, list] = {}
    for k, la, lo, p in zip(t["key"], t["lat"], t["lon"], t["pop"]):
        out.setdefault(k, []).append((float(la), float(lo), float(p)))
    return out


@lru_cache(maxsize=1)
def _sites() -> pd.DataFrame:
    ensure_built()
    return pd.read_csv(SITES)


@lru_cache(maxsize=1)
def _anchors() -> Dict[str, Tuple[float, float]]:
    if not ANCHORS.exists():
        return {}
    a = pd.read_csv(ANCHORS)
    return {k: (float(la), float(lo)) for k, la, lo in zip(a["key"], a["lat"], a["lon"])}


@lru_cache(maxsize=1)
def _district_polys():
    try:
        from safety_index import load_districts
        return load_districts(ROOT / "data" / "safety" / "districts.geojson")
    except Exception:
        return []


def _state_of(lat: float, lon: float) -> str:
    from safety_index import locate_districts
    st, _ = locate_districts(np.array([lat]), np.array([lon]), _district_polys())
    return str(st[0]).lower()


def candidates(name: str) -> list:
    t = _towns()
    n = norm(name)
    for cand in (norm(CITY_ALIASES.get(n, n)), n, n.replace(" city", ""), n.split(" ")[0]):
        if cand in t:
            return t[cand]
    return []


def town_coords(name: str) -> Optional[Tuple[float, float]]:
    """Coordinates of a town/city name as people type it ('New Delhi', 'Panjim', 'Hampi',
    'Manali', 'Manali, Tamil Nadu'). Order: typed state > known tourist town's state >
    verified city list > most populous exact-spelling match > named site."""
    raw = str(name)
    state = None
    if "," in raw:
        raw, state = raw.split(",", 1)
        state = norm(state)
    n = norm(raw)
    key = norm(CITY_ALIASES.get(n, n))
    cands = candidates(raw)
    want = state or TOURIST_STATE.get(n) or TOURIST_STATE.get(key)
    if want and cands:
        in_state = [c for c in cands if want in norm(_state_of(c[0], c[1]))]
        if in_state:
            near = TOURIST_NEAR.get(n) or TOURIST_NEAR.get(key)
            if near:
                in_state.sort(key=lambda c: haversine_km(near[0], near[1], c[0], c[1]))
            return in_state[0][0], in_state[0][1]
    anchors = _anchors()
    for k in (n, key):
        if k in anchors:
            return anchors[k]
    if cands:
        return cands[0][0], cands[0][1]
    # islands, parks, hills named as the "city" (e.g. 'Havelock Island')
    s = _sites()
    hit = s[s["key"] == n]
    if len(hit):
        return float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"])
    return None


def build_anchors() -> pd.DataFrame:
    """Verified coordinates for cities: the Vynex list of 500+ large cities, plus cities with
    >= 20 restaurants in our data IF their restaurants sit within 30 km of a GeoNames place of
    that name (catches mislabelled data such as 'Mysore' restaurants that are in Bangalore)."""
    import json as _json
    rows = {}
    v = _json.loads((ROOT / "data" / "attractions" / "vynex_cities.json").read_text(encoding="utf-8"))
    for c in v:
        rows[norm(c["city"])] = (float(c["latitude"]), float(c["longitude"]), "vynex")
    p = pd.read_csv(ROOT / "data" / "india_places.csv", usecols=["city", "latitude", "longitude"], low_memory=False)
    g = p.groupby("city").agg(n=("city", "size"), lat=("latitude", "median"), lon=("longitude", "median"))
    for city, r in g[g["n"] >= 20].iterrows():
        k = norm(city)
        if k in rows:
            continue
        cands = candidates(city)
        if not cands:
            continue
        d = [haversine_km(r["lat"], r["lon"], la, lo) for la, lo, _ in cands]
        i = int(np.argmin(d))
        if d[i] <= 30:
            rows[k] = (cands[i][0], cands[i][1], "restaurants+geonames")
    df = pd.DataFrame([{"key": k, "lat": la, "lon": lo, "source": src} for k, (la, lo, src) in rows.items()])
    df.to_csv(ANCHORS, index=False)
    _anchors.cache_clear()
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


_STOP = {"the", "of", "and", "sri", "shri", "shree", "temple", "fort", "lake", "falls", "hill",
         "beach", "park", "garden", "museum", "palace", "mandir", "church", "point", "view"}


def site_coords(place: str, near: Tuple[float, float], max_km: float = 60.0
                ) -> Optional[Tuple[float, float, str]]:
    """Exact coordinates of a named site near a town, or None.
    Match = same normalised name, or every distinctive word of the place name
    appears in the site name (e.g. 'Dhauligiri Shanti Stupa' ~ 'Dhauli Shanti Stupa' fails;
    'Golconda Fort' ~ 'Golconda Fort' matches)."""
    s = _sites()
    n = norm(place)
    words = {w for w in n.split() if w not in _STOP and len(w) >= 4}
    if not words:
        return None
    lat0, lon0 = near
    box = s[(s["lat"].between(lat0 - 0.6, lat0 + 0.6)) & (s["lon"].between(lon0 - 0.6, lon0 + 0.6))]
    if box.empty:
        return None
    exact = box[box["key"] == n]
    cand = exact if len(exact) else box[box["key"].apply(lambda k: words <= set(k.split()))]
    if cand.empty:
        return None
    d = haversine_km(lat0, lon0, cand["lat"].to_numpy(), cand["lon"].to_numpy())
    i = int(np.argmin(d))
    if d[i] > max_km:
        return None
    r = cand.iloc[i]
    return float(r["lat"]), float(r["lon"]), str(r["fcode"])


if __name__ == "__main__":
    import sys as _sys
    if "--anchors" in _sys.argv:
        a = build_anchors()
        print(f"verified city anchors: {len(a)}")
        for q in ["Manali", "Kasol", "Mahabaleshwar", "Manali, Tamil Nadu", "Chandigarh", "Mysore", "Vadodara",
                  "Rajkot", "Patna", "Ujjain", "Mandu", "Coorg", "Ooty", "Munnar", "Hampi", "Gokarna"]:
            print(q, town_coords(q))
        raise SystemExit
    t, s = build()
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(TOWNS, index=False, compression="gzip")
    s.to_csv(SITES, index=False, compression="gzip")
    print(f"towns: {len(t)} names   sites: {len(s)}")
    for q in ["New Delhi", "Panjim", "Hampi", "Munnar", "Trivandrum", "Kochi (Cochin)", "Sohra"]:
        print(q, town_coords(q))
