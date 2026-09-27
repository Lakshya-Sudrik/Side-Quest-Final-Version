"""Rule-based safety index for SideQuest (replaces the safety regression model).

Why this is not a trained model
-------------------------------
The old ``safety_score`` target was a formula we wrote over district crime
rates. Training a regressor to reproduce our own formula can only do one of two
things: memorise the formula when its inputs are features (the fake R^2=0.9999)
or fail when they are removed (R^2 ~ 0, gate failure). Neither adds
information. The honest version is to compute the index directly and show the
user WHY a place scored what it did.

Inputs (all under ``data/safety/``; build them with ``scripts/fetch_safety_data.py``)
    districts.geojson      India district boundaries (point -> district lookup)
    district_crime.csv     state, district, violent_rate, property_rate, women_rate
                           (per 100k, NCRB district data / census population)
    police_stations.csv    lat, lon[, name, state, source]
    hospitals.csv          lat, lon[, name, state, source]

Score (weights live in config/criteria.json -> safety_index)
    crime      = 100 * (1 - national percentile of the district's crime level)
    police     = 100 * exp(-km_to_nearest_police / police_decay_km)
    hospital   = 100 * exp(-km_to_nearest_hospital / hospital_decay_km)
    safety     = weighted mean of the components that are AVAILABLE

A component with no data is dropped and reported in ``missing``; if nothing is
available the score is NaN ("unknown"). No default value is ever invented.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

SAFETY_DIR = ROOT / "data" / "safety"
CRITERIA_PATH = ROOT / "config" / "criteria.json"
EARTH_KM = 6371.0088

DEFAULT_CONFIG: Dict[str, Any] = {
    "weights": {"crime": 0.5, "police": 0.25, "hospital": 0.25},
    "crime_mix": {"violent": 0.45, "women": 0.35, "property": 0.20},
    "police_decay_km": 2.0,
    "hospital_decay_km": 3.0,
    "nearby_radius_km": 2.0,
    "max_infra_distance_km": 25.0,
    "levels": {"high": 70.0, "moderate": 50.0},
}


def load_config() -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        crit = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        user = crit.get("safety_index") or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except FileNotFoundError:
        pass
    return cfg


def norm_name(s: Any) -> str:
    s = str(s or "").strip().lower()
    for a, b in (
        ("bengaluru", "bangalore"), ("bangalore urban", "bangalore"),
        ("mysuru", "mysore"), ("mumbai suburban", "mumbai"), ("mumbai city", "mumbai"),
        ("gurugram", "gurgaon"), ("prayagraj", "allahabad"),
        ("thiruvananthapuram", "trivandrum"), ("kolkata", "calcutta"),
        ("chennai", "madras"), ("varanasi", "benares"),
    ):
        s = s.replace(a, b)
    return " ".join(s.replace("-", " ").split())


# ---------------------------------------------------------------------------
# geometry helpers (pure numpy - no shapely/geopandas dependency)
# ---------------------------------------------------------------------------
def _points_in_ring(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for many points against one ring."""
    inside = np.zeros(len(lon), dtype=bool)
    x0, y0 = ring[:-1, 0], ring[:-1, 1]
    x1, y1 = ring[1:, 0], ring[1:, 1]
    for a, b, c, d in zip(x0, y0, x1, y1):
        crosses = (b > lat) != (d > lat)
        if not crosses.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = a + (lat - b) * (c - a) / (d - b)
        inside ^= crosses & (lon < xint)
    return inside


@dataclass
class _District:
    state: str
    district: str
    rings: List[np.ndarray]
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]


def load_districts(path: Path) -> List[_District]:
    gj = json.loads(path.read_text(encoding="utf-8"))
    out: List[_District] = []
    for f in gj["features"]:
        props = f.get("properties", {})
        geom = f.get("geometry") or {}
        polys = geom.get("coordinates", [])
        if geom.get("type") == "Polygon":
            polys = [polys]
        rings = [np.asarray(r, dtype=float) for poly in polys for r in poly if len(r) >= 4]
        if not rings:
            continue
        allpts = np.vstack(rings)
        out.append(_District(
            state=str(props.get("st_nm", props.get("state", ""))),
            district=str(props.get("district", props.get("DISTRICT", ""))),
            rings=rings,
            bbox=(allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max()),
            centroid=(float(allpts[:, 0].mean()), float(allpts[:, 1].mean())),
        ))
    return out


def locate_districts(lat: np.ndarray, lon: np.ndarray,
                     districts: List[_District]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (state, district) arrays for each point ('' when outside India)."""
    n = len(lat)
    st = np.full(n, "", dtype=object)
    di = np.full(n, "", dtype=object)
    ok = np.isfinite(lat) & np.isfinite(lon)
    for d in districts:
        x0, y0, x1, y1 = d.bbox
        cand = ok & (di == "") & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        if not cand.any():
            continue
        idx = np.where(cand)[0]
        inside = np.zeros(len(idx), dtype=bool)
        for ring in d.rings:
            inside ^= _points_in_ring(lon[idx], lat[idx], ring)
        hit = idx[inside]
        st[hit] = d.state
        di[hit] = d.district
    # coastal / border points just outside the simplified outlines (e.g. south Mumbai):
    # take the nearest district centre if it is within 25 km
    miss = np.where(ok & (di == ""))[0]
    if len(miss) and districts:
        cx = np.array([d.centroid[0] for d in districts]); cy = np.array([d.centroid[1] for d in districts])
        for i in miss:
            dk = np.hypot((cx - lon[i]) * np.cos(np.radians(lat[i])), cy - lat[i]) * 111.0
            j = int(np.argmin(dk))
            if dk[j] <= 25.0:
                st[i], di[i] = districts[j].state, districts[j].district
    return st, di


class _PointIndex:
    """Nearest-neighbour + radius counts on a sphere (sklearn BallTree)."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        from sklearn.neighbors import BallTree

        self.n = len(lat)
        self.tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine") if self.n else None

    def nearest_km(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        if not self.n:
            return np.full(len(lat), np.nan)
        d, _ = self.tree.query(np.radians(np.c_[lat, lon]), k=1)
        return d[:, 0] * EARTH_KM

    def count_within(self, lat: np.ndarray, lon: np.ndarray, km: float) -> np.ndarray:
        if not self.n:
            return np.zeros(len(lat), dtype=int)
        return self.tree.query_radius(np.radians(np.c_[lat, lon]), r=km / EARTH_KM, count_only=True)


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------
@dataclass
class SafetyIndex:
    config: Dict[str, Any]
    districts: List[_District] = field(default_factory=list)
    crime: Optional[pd.DataFrame] = None           # indexed by (state_key, district_key)
    crime_state: Optional[pd.DataFrame] = None     # indexed by state_key
    police: Optional[_PointIndex] = None
    hospitals: Optional[_PointIndex] = None
    police_states: set = field(default_factory=set)
    hospital_states: set = field(default_factory=set)

    _CACHE: ClassVar[Optional["SafetyIndex"]] = None

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, data_dir: Path = SAFETY_DIR, refresh: bool = False) -> "SafetyIndex":
        if cls._CACHE is not None and not refresh and data_dir == SAFETY_DIR:
            return cls._CACHE
        idx = cls(config=load_config())
        gj = data_dir / "districts.geojson"
        if gj.exists():
            idx.districts = load_districts(gj)
        crime_csv = data_dir / "district_crime.csv"
        if crime_csv.exists():
            idx._set_crime(pd.read_csv(crime_csv))
        for name, attr, states_attr in (("police_stations.csv", "police", "police_states"),
                                        ("hospitals.csv", "hospitals", "hospital_states")):
            p = data_dir / name
            if p.exists():
                df = pd.read_csv(p)
                df = df[pd.to_numeric(df["lat"], errors="coerce").notna()
                        & pd.to_numeric(df["lon"], errors="coerce").notna()]
                setattr(idx, attr, _PointIndex(df["lat"].astype(float).values,
                                               df["lon"].astype(float).values))
                if "state" in df.columns:
                    setattr(idx, states_attr, {norm_name(s) for s in df["state"].dropna()})
        if data_dir == SAFETY_DIR:
            cls._CACHE = idx
        return idx

    def _set_crime(self, df: pd.DataFrame) -> None:
        mix = self.config["crime_mix"]
        df = df.copy()
        for c in ("violent_rate", "property_rate", "women_rate"):
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # national percentile of each rate across ALL districts (no train/test:
        # this is a published reference table, not a fitted model)
        parts, wsum = [], 0.0
        for key, col in (("violent", "violent_rate"), ("women", "women_rate"),
                         ("property", "property_rate")):
            w = float(mix.get(key, 0.0))
            if w > 0 and df[col].notna().any():
                parts.append(w * df[col].rank(pct=True))
                wsum += w
        df["crime_level_pct"] = (sum(parts) / wsum) if parts else np.nan
        df["state_key"] = df["state"].map(norm_name)
        df["district_key"] = df["district"].map(norm_name)
        self.crime = df.drop_duplicates(["state_key", "district_key"]).set_index(
            ["state_key", "district_key"])
        self.crime_state = df.groupby("state_key")[["crime_level_pct"]].mean()

    # -- scoring -------------------------------------------------------------
    def _crime_lookup(self, state: str, district: str, city: str) -> Tuple[float, str, str]:
        """(crime_level_pct, resolution, matched district)."""
        if self.crime is None:
            return np.nan, "none", ""
        sk, dk, ck = norm_name(state), norm_name(district), norm_name(city)
        for key in (dk, ck):
            if key and (sk, key) in self.crime.index:
                return float(self.crime.loc[(sk, key), "crime_level_pct"]), "district", key
        # district names drift between census and NCRB -> substring match in state
        if sk and (dk or ck):
            sub = self.crime.loc[self.crime.index.get_level_values(0) == sk]
            for key in (dk, ck):
                if not key:
                    continue
                hits = [k for k in sub.index.get_level_values(1) if key in k or k in key]
                if hits:
                    return float(sub.loc[(sk, hits[0]), "crime_level_pct"]), "district", hits[0]
        if sk and sk in self.crime_state.index:
            return float(self.crime_state.loc[sk, "crime_level_pct"]), "state", ""
        return np.nan, "none", ""

    def score_frame(self, places: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        n = len(places)
        lat = pd.to_numeric(places.get("latitude", pd.Series(np.nan, index=places.index)),
                            errors="coerce").to_numpy(float)
        lon = pd.to_numeric(places.get("longitude", pd.Series(np.nan, index=places.index)),
                            errors="coerce").to_numpy(float)
        city = places.get("city", pd.Series("", index=places.index)).fillna("").astype(str).to_numpy()
        state_in = places.get("state", pd.Series("", index=places.index)).fillna("").astype(str).to_numpy()

        # district from coordinates (authoritative), else from supplied state/city
        if self.districts:
            geo_state, geo_district = locate_districts(lat, lon, self.districts)
        else:
            geo_state = np.full(n, "", dtype=object)
            geo_district = np.full(n, "", dtype=object)
        state = np.where(geo_state != "", geo_state, state_in)

        crime_pct = np.full(n, np.nan)
        resolution = np.full(n, "none", dtype=object)
        for i in range(n):
            crime_pct[i], resolution[i], _ = self._crime_lookup(state[i], geo_district[i], city[i])

        maxd = float(cfg["max_infra_distance_km"])
        coords_ok = np.isfinite(lat) & np.isfinite(lon)
        pol_km = self.police.nearest_km(np.where(coords_ok, lat, 0), np.where(coords_ok, lon, 0)) \
            if self.police else np.full(n, np.nan)
        hos_km = self.hospitals.nearest_km(np.where(coords_ok, lat, 0), np.where(coords_ok, lon, 0)) \
            if self.hospitals else np.full(n, np.nan)
        pol_km = np.where(coords_ok, pol_km, np.nan)
        hos_km = np.where(coords_ok, hos_km, np.nan)
        # a "nearest station 300 km away" means we have no data for that area,
        # not that the place is 300 km from help
        pol_km = np.where(pol_km <= maxd, pol_km, np.nan)
        hos_km = np.where(hos_km <= maxd, hos_km, np.nan)
        r = float(cfg["nearby_radius_km"])
        pol_n = self.police.count_within(np.where(coords_ok, lat, 0), np.where(coords_ok, lon, 0), r) \
            if self.police else np.zeros(n, dtype=int)
        hos_n = self.hospitals.count_within(np.where(coords_ok, lat, 0), np.where(coords_ok, lon, 0), r) \
            if self.hospitals else np.zeros(n, dtype=int)

        comp = {
            "crime": 100.0 * (1.0 - crime_pct),
            "police": 100.0 * np.exp(-pol_km / float(cfg["police_decay_km"])),
            "hospital": 100.0 * np.exp(-hos_km / float(cfg["hospital_decay_km"])),
        }
        w = cfg["weights"]
        num = np.zeros(n)
        den = np.zeros(n)
        for k, v in comp.items():
            ok = np.isfinite(v)
            num += np.where(ok, v * float(w[k]), 0.0)
            den += np.where(ok, float(w[k]), 0.0)
        score = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

        out = pd.DataFrame(index=places.index)
        out["safety_score"] = np.round(score, 1)
        out["safety_level"] = [self.level(s) for s in score]
        out["crime_component"] = np.round(comp["crime"], 1)
        out["police_component"] = np.round(comp["police"], 1)
        out["hospital_component"] = np.round(comp["hospital"], 1)
        out["police_km"] = np.round(pol_km, 2)
        out["hospital_km"] = np.round(hos_km, 2)
        out["police_within_2km"] = pol_n
        out["hospitals_within_2km"] = hos_n
        out["district"] = geo_district
        out["state"] = state
        out["crime_resolution"] = resolution
        out["data_coverage"] = np.round(den / sum(float(x) for x in w.values()), 2)
        out["missing"] = [
            ",".join(k for k, v in comp.items() if not np.isfinite(v[i])) for i in range(n)
        ]
        return out

    def level(self, s: float) -> str:
        if not np.isfinite(s):
            return "unknown"
        lv = self.config["levels"]
        return "high" if s >= lv["high"] else "moderate" if s >= lv["moderate"] else "low"

    def explain(self, row: pd.Series) -> List[str]:
        """Human-readable reasons for one scored row."""
        reasons = []
        if np.isfinite(row["police_km"]):
            reasons.append(f"Police station {row['police_km']:.1f} km away"
                           f" ({int(row['police_within_2km'])} within 2 km)")
        else:
            reasons.append("No police-station data for this area")
        if np.isfinite(row["hospital_km"]):
            reasons.append(f"Hospital {row['hospital_km']:.1f} km away"
                           f" ({int(row['hospitals_within_2km'])} within 2 km)")
        else:
            reasons.append("No hospital data for this area")
        if np.isfinite(row["crime_component"]):
            pct = 100 - row["crime_component"]
            band = "low" if pct < 33 else "medium" if pct < 67 else "high"
            where = row["district"] or row["state"]
            scope = "district" if row["crime_resolution"] == "district" else "state average"
            reasons.append(f"Crime in {where}: {band} vs rest of India ({scope}, NCRB)")
        else:
            reasons.append("No crime data for this area")
        return reasons

    def score_point(self, lat: float, lon: float, city: str = "", state: str = "") -> Dict[str, Any]:
        df = pd.DataFrame([{"latitude": lat, "longitude": lon, "city": city, "state": state}])
        row = self.score_frame(df).iloc[0]
        val = row["safety_score"]
        return {
            "safety_score": None if not np.isfinite(val) else float(val),
            "level": row["safety_level"],
            # needs the crime layer (crimes against women) - infra alone can't say this
            "women_safe": (None if not np.isfinite(row["crime_component"])
                           else bool(np.isfinite(val) and val >= self.config["levels"]["high"])),
            "data_coverage": float(row["data_coverage"]),
            "reasons": self.explain(row),
            "district": row["district"] or None,
            "state": row["state"] or None,
            "crime_resolution": row["crime_resolution"],
            "missing": [m for m in row["missing"].split(",") if m],
            "components": {
                "crime": None if not np.isfinite(row["crime_component"]) else float(row["crime_component"]),
                "police": None if not np.isfinite(row["police_component"]) else float(row["police_component"]),
                "hospital": None if not np.isfinite(row["hospital_component"]) else float(row["hospital_component"]),
            },
        }

    # -- coverage report -----------------------------------------------------
    def coverage(self) -> Dict[str, Any]:
        n_d = len(self.districts)
        n_c = 0 if self.crime is None else len(self.crime)
        return {
            "district_boundaries": n_d,
            "districts_with_crime": n_c,
            "police_points": 0 if self.police is None else self.police.n,
            "hospital_points": 0 if self.hospitals is None else self.hospitals.n,
            "states_with_police_data": sorted(self.police_states),
            "states_with_hospital_data": sorted(self.hospital_states),
        }


if __name__ == "__main__":  # quick check: python src/safety_index.py 12.97 77.59 Bangalore
    logging.basicConfig(level=logging.INFO)
    idx = SafetyIndex.load()
    print(json.dumps(idx.coverage(), indent=2)[:2000])
    if len(sys.argv) >= 3:
        print(json.dumps(idx.score_point(float(sys.argv[1]), float(sys.argv[2]),
                                         sys.argv[3] if len(sys.argv) > 3 else ""), indent=2))
