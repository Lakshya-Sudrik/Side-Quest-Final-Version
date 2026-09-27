"""Build the data files used by src/safety_index.py.

    python scripts/fetch_safety_data.py                 # everything
    python scripts/fetch_safety_data.py --only police   # one layer
    python scripts/fetch_safety_data.py --nhfr path/to/hospital_directory.csv

Writes to data/safety/:
    districts.geojson    India district boundaries (GitHub: udit-001/india-maps-data)
    district_crime.csv   NCRB district crime per 100k (needs data/raw NCRB + census CSVs)
    police_stations.csv  OpenStreetMap amenity=police, downloaded STATE BY STATE
    hospitals.csv        OpenStreetMap amenity=hospital|clinic (+ optional NHFR/data.gov.in CSV)

Every state is fetched separately and the script prints a coverage table at the
end. A state that returns zero points is reported as FAILED - it is never
silently treated as "no police stations here" (the old per-city fetch lost
Mumbai, Delhi, Pune, Lucknow and ~20 other cities that way).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "safety"
DISTRICTS_URL = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
STATES = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
]
AMENITY = {"police": '"amenity"="police"', "hospitals": '"amenity"~"^(hospital|clinic)$"'}

log = logging.getLogger("fetch_safety")


# ---------------------------------------------------------------------------
def fetch_districts() -> None:
    dst = OUT / "districts.geojson"
    if dst.exists():
        log.info("districts.geojson already present")
        return
    log.info("downloading district boundaries ...")
    urllib.request.urlretrieve(DISTRICTS_URL, dst)
    n = len(json.loads(dst.read_text(encoding="utf-8"))["features"])
    log.info("  %d districts", n)


def build_crime() -> None:
    from india_data import load_crime

    rates, _ = load_crime()
    df = rates.rename(columns={"state_name": "state", "district_name": "district"})
    cols = ["state", "district", "population", "violent_rate", "property_rate", "women_rate"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT / "district_crime.csv", index=False)
    log.info("district_crime.csv: %d districts, %d states", len(df), df["state"].nunique())


# ---------------------------------------------------------------------------
def _overpass(query: str, retries: int = 3) -> Optional[dict]:
    data = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                req = urllib.request.Request(url, data=data,
                                             headers={"User-Agent": "SideQuest-safety/1.0"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as exc:  # try next mirror
                log.warning("    %s failed (%s)", url.split("/")[2], exc)
        time.sleep(10 * (attempt + 1))
    return None


def fetch_osm_layer(layer: str, states: List[str]) -> pd.DataFrame:
    rows, status = [], {}
    for st in states:
        q = f"""[out:json][timeout:220];
area["name"="{st}"]["boundary"="administrative"]["admin_level"="4"]->.s;
(node[{AMENITY[layer]}](area.s); way[{AMENITY[layer]}](area.s); relation[{AMENITY[layer]}](area.s););
out center tags;"""
        log.info("  %s / %s ...", layer, st)
        res = _overpass(q)
        if res is None:
            status[st] = "FAILED"
            continue
        n0 = len(rows)
        for el in res.get("elements", []):
            lat = el.get("lat", (el.get("center") or {}).get("lat"))
            lon = el.get("lon", (el.get("center") or {}).get("lon"))
            if lat is None or lon is None:
                continue
            tags = el.get("tags", {})
            rows.append({"lat": lat, "lon": lon, "name": tags.get("name", ""),
                         "kind": tags.get("amenity", ""), "state": st, "source": "osm"})
        status[st] = len(rows) - n0 if len(rows) > n0 else "ZERO (check state name)"
        time.sleep(2)
    for st, v in status.items():
        log.info("    %-45s %s", st, v)
    return pd.DataFrame(rows)


def seed_from_infra_cache() -> Dict[str, pd.DataFrame]:
    """Reuse the OSM points already cached by data_pipeline_india.py (~20 cities)."""
    path = ROOT / "data" / "cache" / "infra_cache.json"
    if not path.exists():
        return {}
    cache = json.loads(path.read_text(encoding="utf-8"))
    pol, hos = [], []
    for city, v in cache.items():
        for p in (v or {}).get("points", []):
            row = {"lat": p["lat"], "lon": p["lng"], "name": "", "kind": p["type"],
                   "state": "", "source": f"osm_cache:{city}"}
            (pol if p["type"] == "police" else hos).append(row)
    out = {"police": pd.DataFrame(pol), "hospitals": pd.DataFrame(hos)}
    # attach state from coordinates
    from safety_index import load_districts, locate_districts

    gj = OUT / "districts.geojson"
    if gj.exists():
        d = load_districts(gj)
        for k, df in out.items():
            if len(df):
                st, _ = locate_districts(df["lat"].to_numpy(float), df["lon"].to_numpy(float), d)
                df["state"] = st
    return out


def import_nhfr(path: Path) -> pd.DataFrame:
    """National Health Facility Registry / data.gov.in hospital directory CSV."""
    df = pd.read_csv(path, low_memory=False)
    low = {c.lower().strip(): c for c in df.columns}
    lat = next((low[c] for c in low if c in ("latitude", "lat", "location_coordinates_lat")), None)
    lon = next((low[c] for c in low if c in ("longitude", "lon", "long", "lng")), None)
    if lat is None or lon is None:
        raise SystemExit(f"{path.name}: no latitude/longitude columns found ({list(df.columns)[:15]})")
    name = next((low[c] for c in low if "name" in c), None)
    state = next((low[c] for c in low if c.startswith("state")), None)
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat], errors="coerce"),
        "lon": pd.to_numeric(df[lon], errors="coerce"),
        "name": df[name] if name else "",
        "kind": "hospital",
        "state": df[state] if state else "",
        "source": "nhfr",
    })
    out = out[out["lat"].between(6, 38) & out["lon"].between(68, 98)]
    log.info("NHFR: %d hospitals with valid India coordinates", len(out))
    return out


def merge_write(layer: str, frames: List[pd.DataFrame]) -> None:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        log.warning("%s: no data collected", layer)
        return
    df = pd.concat(frames, ignore_index=True)
    # same facility from two sources -> keep one (≈10 m grid)
    df["_k"] = (df["lat"].astype(float).round(4).astype(str) + ","
                + df["lon"].astype(float).round(4).astype(str))
    df = df.drop_duplicates("_k").drop(columns="_k")
    name = "police_stations.csv" if layer == "police" else "hospitals.csv"
    df.to_csv(OUT / name, index=False)
    log.info("%s: %d points across %d states", name, len(df), df["state"].replace("", np.nan).nunique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["districts", "crime", "police", "hospitals", "seed"])
    ap.add_argument("--states", nargs="*", default=STATES)
    ap.add_argument("--nhfr", type=Path, help="hospital directory CSV from data.gov.in / NHFR")
    ap.add_argument("--no-osm", action="store_true", help="skip Overpass downloads")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    if a.only in (None, "districts"):
        fetch_districts()
    if a.only in (None, "crime"):
        try:
            build_crime()
        except FileNotFoundError as exc:
            log.error("crime: raw NCRB/census files missing (%s)", exc)
    seed = seed_from_infra_cache() if a.only in (None, "police", "hospitals", "seed") else {}
    for layer in ("police", "hospitals"):
        if a.only not in (None, layer, "seed"):
            continue
        frames = [seed.get(layer)]
        if not a.no_osm and a.only != "seed":
            frames.append(fetch_osm_layer(layer, a.states))
        if layer == "hospitals" and a.nhfr:
            frames.append(import_nhfr(a.nhfr))
        existing = OUT / ("police_stations.csv" if layer == "police" else "hospitals.csv")
        if existing.exists():
            frames.append(pd.read_csv(existing))
        merge_write(layer, frames)

    from safety_index import SafetyIndex

    print(json.dumps(SafetyIndex.load(OUT, refresh=True).coverage(), indent=2))


if __name__ == "__main__":
    main()
