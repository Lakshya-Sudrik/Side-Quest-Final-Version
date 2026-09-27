"""SideQuest hidden-gem recommender: 5 hidden gems + 5 famous-and-good places.

    python src/recommender.py --from "Bangalore" --to "Mysore"
    python src/recommender.py --from "12.97,77.59" --to "Hampi" --buffer-km 15 --youtube live

Pipeline
    1. Route      : OSRM road route (real km / minutes); if OSRM is unreachable, a
                    straight line with road distance estimated as 1.3x (flagged).
    2. Candidates : restaurants (70k, food_scored.csv.gz) + attractions (15k,
                    attractions_scored.csv.gz) within --buffer-km of the route.
    3. Classify   : quality_rank = percentile of model quality within its type.
                    hidden gem     : quality_rank >= 70 and exposure <= 0.50
                    famous & good  : quality_rank >= 70 and exposure >= 0.80
    4. YouTube    : places with videos that passed the authenticity check get a
                    small, capped boost (+ up to 5 rank points) - never a new label.
    5. Safety     : official district crime (NCRB 2014 / census) + nearest police &
                    hospital (OpenStreetMap) + what reviewers say (theft, scams,
                    harassment, 'felt safe'). Classes: safe / caution / avoid.
    6. Pick       : rank by safety class first, then gem score; 'avoid' is used
                    only if nothing else is left and is clearly flagged. Each list
                    mixes eat + visit (>= 2 of each when the route has them) and
                    never repeats a brand/place.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FOOD = ROOT / "data" / "processed" / "food_scored.csv.gz"
ATTR = ROOT / "data" / "attractions" / "attractions_scored.csv.gz"
OSRM_URL = os.environ.get("OSRM_URL", "https://router.project-osrm.org")
ROAD_FACTOR = 1.3          # straight line -> road km when OSRM is unreachable
AVG_SPEED_KMH = 45.0       # for estimated drive times
GEM_RANK_MIN = 70.0
HIDDEN_MAX_EXPOSURE = 0.50
FAMOUS_MIN_EXPOSURE = 0.80
# minimum evidence before anything can be called a gem (a 5.0 from 3 reviews is not evidence)
MIN_REVIEWS = {"eat": 20, "visit": 8}
# shops, malls and similar are not "sights" even if reviewed on a travel site
NOT_A_SIGHT = (r"\b(?:gems?|jewell?ers?|jewell?ery|showroom|emporium|boutique|silks?|sarees?|textiles?|spa|salon|"
               r"agency|travels|tours?|trips?|rentals?|holidays|adventures|pvt|ltd|p ltd|private limited|tour operator|"
               r"taxi|cabs?|expeditions?|guide|guides|club|academy|school|college|university|hospital|clinic|office|"
               r"bank|atm|stadium|gym|studio|salon|parlou?r|apartments?|residency|hostel|pg)\b")
MAX_CHAIN_OUTLETS = 5      # a brand with more outlets than this in India is not a hidden gem
log = logging.getLogger("recommender")


# ---------------------------------------------------------------------------
# geography
# ---------------------------------------------------------------------------
def parse_point(s: str) -> Tuple[float, float, str]:
    s = str(s).strip()
    if "," in s:
        a, b = s.split(",", 1)
        try:
            return float(a), float(b), s
        except ValueError:
            pass
    import geo_india as gi
    c = gi.town_coords(s)
    if c is None:
        raise ValueError(f"Unknown place '{s}' - try a nearby town name or 'lat,lon'")
    return c[0], c[1], s


def hav_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def get_route(a: Tuple[float, float], b: Tuple[float, float], timeout: float = 8.0) -> Dict:
    """Road route from OSRM; falls back to a straight line (estimated) if unreachable."""
    try:
        import requests
        url = f"{OSRM_URL}/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}"
        r = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        if js.get("code") == "Ok" and js.get("routes"):
            rt = js["routes"][0]
            coords = [(lat, lon) for lon, lat in rt["geometry"]["coordinates"]]
            return {"geometry": coords, "distance_km": rt["distance"] / 1000.0,
                    "duration_min": rt["duration"] / 60.0, "source": "osrm_road"}
    except Exception as exc:  # offline / blocked / timeout
        log.info("OSRM unavailable (%s) - using straight-line estimate", type(exc).__name__)
    straight = float(hav_km(a[0], a[1], b[0], b[1]))
    n = max(2, int(straight // 2) + 2)  # a point every ~2 km
    geom = list(zip(np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)))
    road = straight * ROAD_FACTOR
    return {"geometry": geom, "distance_km": road, "duration_min": road / AVG_SPEED_KMH * 60,
            "source": "estimated_straight_line_x1.3"}


def distance_to_route(lat: np.ndarray, lon: np.ndarray, geom: List[Tuple[float, float]]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """(km from route, km along route from start) for many points, vectorised."""
    g = np.asarray(geom, float)
    if len(g) > 600:
        g = g[np.linspace(0, len(g) - 1, 600).astype(int)]
    lat0 = np.radians(np.mean(g[:, 0]))
    kx, ky = 111.32 * np.cos(lat0), 110.57  # km per degree
    px, py = lon * kx, lat * ky
    ax, ay = g[:-1, 1] * kx, g[:-1, 0] * ky
    bx, by = g[1:, 1] * kx, g[1:, 0] * ky
    seg = np.hypot(bx - ax, by - ay)
    cum = np.concatenate([[0], np.cumsum(seg)])[:-1]
    best = np.full(len(px), np.inf)
    along = np.zeros(len(px))
    for s in range(0, len(px), 4000):
        X, Y = px[s:s + 4000, None], py[s:s + 4000, None]
        dx, dy = bx - ax, by - ay
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / np.where(seg > 0, seg ** 2, 1), 0, 1)
        d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
        i = np.argmin(d, axis=1)
        r = np.arange(len(i))
        best[s:s + 4000] = d[r, i]
        along[s:s + 4000] = cum[i] + t[r, i] * seg[i]
    return best, along


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_candidates() -> pd.DataFrame:
    frames = []
    if FOOD.exists():
        f = pd.read_csv(FOOD)
        frames.append(pd.DataFrame({
            "id": "food:" + f["place_id"].astype(str), "name": f["name"], "kind": "eat",
            "category": f["category"], "city": f["city"], "lat": f["lat"], "lon": f["lon"],
            "location_precision": "exact", "rating": f["rating"], "n_reviews": f["n_reviews"],
            "quality_pct": f["quality_pct"], "exposure_pct": f["exposure_pct"],
            "dedupe_key": f["brand"].fillna(f["name"].str.lower()), "basis": f["score_basis"],
            "cost_inr": f["cost_inr"], "tx_unsafe": np.nan, "tx_safe": np.nan, "tx_hidden": np.nan}))
    if ATTR.exists():
        a = pd.read_csv(ATTR)
        a = a[a["quality_pred"].notna() & a["lat"].notna()]
        frames.append(pd.DataFrame({
            "id": "visit:" + a["place_id"].astype(str), "name": a["place"], "kind": "visit",
            "category": a["place_type"], "city": a["city"], "lat": a["lat"], "lon": a["lon"],
            "location_precision": a["location_precision"], "rating": a["rating_mean"].round(2),
            "n_reviews": a["n_reviews"], "quality_pct": a["quality_pct"], "exposure_pct": a["exposure_pct"],
            "dedupe_key": a["place"].str.lower().str.replace(r"[^a-z ]", "", regex=True),
            "basis": "attractions_model", "cost_inr": np.nan, "tx_unsafe": a["tx_unsafe"],
            "tx_safe": a["tx_safe"], "tx_hidden": a["tx_hidden"]}))
    if not frames:
        raise FileNotFoundError("run src/build_food_scores.py and src/attractions_model.py first")
    c = pd.concat(frames, ignore_index=True)
    c["chain_outlets"] = c.groupby(["kind", "dedupe_key"])["id"].transform("size")
    # comparable 0-100 rank within each kind (restaurant and attraction models use different scales)
    c["quality_rank"] = c.groupby("kind")["quality_pct"].rank(pct=True) * 100
    return c


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _safety_index():
    from safety_index import SafetyIndex
    return SafetyIndex.load()


def safety_blocks(c: pd.DataFrame) -> List[Dict]:
    si = _safety_index()
    s = si.score_frame(pd.DataFrame({"latitude": c["lat"].to_numpy(), "longitude": c["lon"].to_numpy(),
                                     "city": c["city"].to_numpy()}, index=c.index))
    blocks = []
    for i, row in c.iterrows():
        r = s.loc[i]
        reasons = si.explain(r)
        # 1. official crime: 100 = lowest-crime district in India, 0 = highest
        crime = float(r["crime_component"]) if np.isfinite(r["crime_component"]) else np.nan
        crime_pct = 100 - crime if np.isfinite(crime) else np.nan
        # 2. what visitors report (attractions with >= 10 reviews)
        unsafe_share = safe_share = np.nan
        review = np.nan
        if row["kind"] == "visit" and np.isfinite(row.get("tx_unsafe", np.nan)) and row["n_reviews"] >= 10:
            unsafe_share, safe_share = float(row["tx_unsafe"]), float(row["tx_safe"])
            review = float(np.clip(100 - 1000 * unsafe_share + 300 * safe_share, 0, 100))
            n_bad = int(round(unsafe_share * row["n_reviews"]))
            if unsafe_share >= 0.03:
                reasons.append(f"{n_bad} of {int(row['n_reviews'])} reviews mention theft, scams, harassment or hazards")
            elif safe_share >= 0.02 and unsafe_share < 0.01:
                reasons.append(f"Reviewers describe it as safe ({int(round(safe_share * row['n_reviews']))} mentions)")
            else:
                reasons.append(f"No safety complaints stand out in {int(row['n_reviews'])} reviews")
        # 3. help nearby (hospital, police): full marks within 5 km, none beyond 25 km
        def near_score(km):
            return float(np.clip(100 * (25 - km) / 20, 0, 100)) if np.isfinite(km) else np.nan
        access_parts = [v for v in (near_score(r["hospital_km"]), near_score(r["police_km"])) if np.isfinite(v)]
        access = float(np.mean(access_parts)) if access_parts else np.nan
        parts = {"crime": (crime, 0.5), "reviews": (review, 0.3), "help_nearby": (access, 0.2)}
        have = {k: v for k, v in parts.items() if np.isfinite(v[0])}
        score = (sum(v * w for v, w in have.values()) / sum(w for _, w in have.values())) if have else None
        # class from EVIDENCE, not from rank alone (a rank would call a third of India unsafe)
        far_hospital = np.isfinite(r["hospital_km"]) and r["hospital_km"] > 15
        if (np.isfinite(crime_pct) and crime_pct >= 90) or (np.isfinite(unsafe_share) and unsafe_share >= 0.06):
            cls = "avoid"
        elif ((np.isfinite(crime_pct) and crime_pct >= 65) or (np.isfinite(unsafe_share) and unsafe_share >= 0.03)
              or far_hospital):
            cls = "caution"
        elif score is not None and score < 50:
            cls = "caution"   # no single red flag, but the combined picture is below average
        elif np.isfinite(crime_pct) or np.isfinite(unsafe_share):
            cls = "safe"
        else:
            cls = "unknown"
        if np.isfinite(crime_pct):
            reasons.append(f"Official crime rate is higher than in {crime_pct:.0f}% of Indian districts")
        if r["state"] and str(r["state"]).lower() in ("kerala", "delhi") and np.isfinite(crime):
            reasons.append("Note: official figures depend on how many complaints a state registers; "
                           "high-reporting states can look worse than they are")
        blocks.append({
            "score": None if score is None else round(float(score), 1), "class": cls,
            "components": {k: (None if not np.isfinite(v[0]) else round(float(v[0]), 1)) for k, v in parts.items()},
            "district": r["district"] or None, "state": r["state"] or None,
            "police_km": None if not np.isfinite(r["police_km"]) else float(r["police_km"]),
            "hospital_km": None if not np.isfinite(r["hospital_km"]) else float(r["hospital_km"]),
            "data_coverage": round(sum(w for _, w in have.values()), 2),
            "missing": [k for k in parts if k not in have],
            "reasons": reasons,
            "sources": "NCRB Crime in India 2014 + Census 2011 (district rates), OpenStreetMap "
                       "police/hospitals, visitor reviews",
        })
    return blocks


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------
SAFETY_ORDER = {"safe": 0, "caution": 1, "unknown": 2, "avoid": 3}


def _pick(pool: pd.DataFrame, k: int, min_each: int, used: set, min_gap_km: float) -> pd.DataFrame:
    """Top-k by (safety class, gem score), spread along the route (stops at least
    min_gap_km apart where possible), >= min_each 'eat' and 'visit' when available,
    one entry per brand/place name."""
    pool = pool[~pool["dedupe_key"].isin(used)].drop_duplicates("dedupe_key")
    pool = pool.sort_values(["safety_rank", "gem_score"], ascending=[True, False])
    safe_enough = pool[pool["safety_class"] != "avoid"]
    if len(safe_enough) >= k:
        pool = safe_enough
    chosen: List = []

    def spaced(i, gap):
        return all(abs(pool.at[i, "along_km"] - pool.at[j, "along_km"]) >= gap for j in chosen)

    def take(rows, limit, gap):
        for i in rows:
            if len(chosen) >= limit:
                return
            if i not in chosen and spaced(i, gap):
                chosen.append(i)

    for gap in (min_gap_km, min_gap_km / 3, 0.0):  # relax spacing only if the route is too short
        for kind in ("visit", "eat"):
            have = sum(pool.at[i, "kind"] == kind for i in chosen)
            take(pool.index[pool["kind"] == kind], len(chosen) + max(0, min_each - have), gap)
        take(pool.index, k, gap)
        if len(chosen) >= k:
            break
    out = pool.loc[chosen[:k]].sort_values("along_km")
    used.update(out["dedupe_key"])
    return out


def listing_candidates() -> pd.DataFrame:
    """Published collaborator listings the model judged hidden gems (backend DB).
    Re-read on every request so approvals show up immediately."""
    try:
        import json as _json
        sys.path.insert(0, str(ROOT))
        from backend import db as bdb
        with bdb.session() as con:
            rows = con.execute("SELECT * FROM listings WHERE published=1 AND (decision='hidden_gem' OR review_status='admin_approved')").fetchall()
    except Exception:
        return pd.DataFrame()
    out = []
    for r in rows:
        ev = _json.loads(r["evaluation"])
        human_approved = r["review_status"] == "admin_approved"
        out.append({"id": "listing:" + r["id"], "name": r["name"], "kind": r["kind"], "category": r["category"] or "",
                    "city": r["city"], "lat": r["lat"], "lon": r["lon"], "location_precision": "exact",
                    "rating": ev.get("matched_place", {}).get("rating") if ev.get("matched_place") else np.nan,
                    "n_reviews": np.nan, "quality_pct": ev.get("quality_rank", np.nan),
                    "quality_rank": ev.get("quality_rank", np.nan), "exposure_pct": ev.get("exposure_pct", 0.0),
                    "dedupe_key": str(r["name"]).lower(), "basis": "collaborator_listing:" + ev.get("evidence", ""),
                    "cost_inr": r["cost_inr"], "tx_unsafe": np.nan, "tx_safe": np.nan, "tx_hidden": np.nan,
                    "chain_outlets": 1, "verified_collaborator": True, "human_approved": human_approved,
                    "photo_url": f"/api/v2/listings/{r['id']}/photo" if r["photo_path"] else None})
    return pd.DataFrame(out)


def recommend(start: str, end: str, buffer_km: float = 10.0, k: int = 5, min_each: int = 2,
              youtube: str = "cache", kinds: Tuple[str, ...] = ("eat", "visit")) -> Dict:
    a_lat, a_lon, a_name = parse_point(start)
    b_lat, b_lon, b_name = parse_point(end)
    route = get_route((a_lat, a_lon), (b_lat, b_lon))
    estimated = route["source"] != "osrm_road"
    c = load_candidates()
    lc = listing_candidates()
    if len(lc):
        c = pd.concat([c, lc], ignore_index=True)
    c = c[c["kind"].isin(kinds)]
    g = np.asarray(route["geometry"])
    pad = buffer_km / 100.0 + 0.05
    box = c[c["lat"].between(g[:, 0].min() - pad, g[:, 0].max() + pad)
            & c["lon"].between(g[:, 1].min() - pad, g[:, 1].max() + pad)]
    off, along = distance_to_route(box["lat"].to_numpy(), box["lon"].to_numpy(), route["geometry"])
    # town-centre locations are fuzzy -> allow a little extra room, but say so
    slack = np.where(box["location_precision"].to_numpy() == "town_centre", 5.0, 0.0)
    near = box[off <= buffer_km + slack].copy()
    near["off_route_km"] = off[off <= buffer_km + slack]
    near["along_km"] = along[off <= buffer_km + slack]
    straight_total = float(hav_km(a_lat, a_lon, b_lat, b_lon))
    scale = route["distance_km"] / max(straight_total, 1e-6) if not estimated else ROAD_FACTOR

    # YouTube: verified videos give a small capped boost (never a new label on their own)
    from youtube_verify import signals_for
    yt = signals_for(near, mode=youtube)
    near["yt_boost"] = near["id"].map(lambda i: yt.get(i, {}).get("boost", 0.0)).astype(float)
    near["gem_score"] = np.clip(near["quality_rank"] + near["yt_boost"], 0, 100)

    # verified-collaborator listings the model judged gems count even with no reviews yet
    is_listing = near["id"].str.startswith("listing:")
    enough = (near["n_reviews"] >= near["kind"].map(MIN_REVIEWS)) | is_listing
    not_sight = (near["kind"] == "visit") & near["name"].str.lower().str.contains(NOT_A_SIGHT, regex=True)
    near = near[enough & ~not_sight]
    human_approved = near.get("human_approved", pd.Series(False, index=near.index)).fillna(False).astype(bool)
    near["is_hidden"] = (((near["gem_score"] >= GEM_RANK_MIN) & (near["exposure_pct"] <= HIDDEN_MAX_EXPOSURE)
                         & (near["chain_outlets"] <= MAX_CHAIN_OUTLETS)) | human_approved)
    # famous & good: well known locally, and not a national chain you can find anywhere
    near["is_famous_good"] = ((near["gem_score"] >= GEM_RANK_MIN) & (near["exposure_pct"] >= FAMOUS_MIN_EXPOSURE)
                              & (near["chain_outlets"] <= 3 * MAX_CHAIN_OUTLETS))
    cand = near[near["is_hidden"] | near["is_famous_good"]].copy()
    # safety only for the shortlist (fast), then choose
    cand["segment"] = (cand["along_km"] // max(route["distance_km"] / 10.0, 1.0)).astype(int)
    cand = cand.sort_values("gem_score", ascending=False).groupby(["is_hidden", "kind", "segment"]).head(12)
    blocks = safety_blocks(cand) if len(cand) else []
    cand["safety"] = blocks
    cand["safety_class"] = [b["class"] for b in blocks]
    cand["safety_rank"] = cand["safety_class"].map(SAFETY_ORDER)
    used: set = set()
    gap = max(3.0, route["distance_km"] / (2.0 * k))
    hidden = _pick(cand[cand["is_hidden"]], k, min_each, used, gap)
    famous = _pick(cand[cand["is_famous_good"]], k, min_each, used, gap)

    def fmt(r) -> Dict:
        road_from_start = r["along_km"] * scale
        detour_km = r["off_route_km"] * ROAD_FACTOR * 2
        kind = "restaurants" if r["kind"] == "eat" else "sights"
        human_approved_flag = pd.notna(r.get("human_approved")) and bool(r.get("human_approved"))
        why = (["Approved by a SideQuest human reviewer"] if human_approved_flag else
               [f"Top {max(1, round(100 - r['quality_rank']))}% for quality among {kind} in India"])
        if r["exposure_pct"] <= 0.5:
            why.append(f"Little known: fewer reviews than {round(100 - r['exposure_pct'] * 100)}% of places in {r['city']}")
        else:
            why.append(f"Well known: more reviews than {round(r['exposure_pct'] * 100)}% of places in {r['city']}")
        if r["yt_boost"] > 0:
            why.append(f"Backed by {yt[r['id']]['verified_videos']} verified YouTube video(s)")
        return {
            "id": r["id"], "name": r["name"], "type": "place to eat" if r["kind"] == "eat" else "place to visit",
            "category": r["category"], "city": r["city"],
            "rating": None if pd.isna(r["rating"]) else float(r["rating"]),
            "reviews": None if pd.isna(r["n_reviews"]) else int(r["n_reviews"]),
            "gem_score_10": None if pd.isna(r["gem_score"]) else round(float(r["gem_score"]) / 10, 1),
            "quality_rank": None if pd.isna(r["quality_rank"]) else round(float(r["quality_rank"]), 1),
            "human_approved": human_approved_flag,
            "photo_url": None if pd.isna(r.get("photo_url")) else r.get("photo_url"),
            "exposure_pct": round(float(r["exposure_pct"]), 3),
            "why": why, "score_basis": r["basis"],
            "distance": {
                "from_route_km": round(float(r["off_route_km"]), 1),
                "from_start_km": round(float(road_from_start), 1),
                "detour_km_round_trip": round(float(detour_km), 1),
                "detour_min_est": round(float(detour_km / AVG_SPEED_KMH * 60)),
                "location": r["location_precision"],
                "note": ("road distance along OSRM route" if not estimated else
                         "estimated (straight line x1.3) - OSRM unreachable") +
                        ("; attraction placed at its town centre" if r["location_precision"] == "town_centre" else ""),
            },
            "lat": round(float(r["lat"]), 5), "lon": round(float(r["lon"]), 5),
            "safety": r["safety"],
            "youtube": yt.get(r["id"], {"verified_videos": 0}),
        }

    return {
        "success": True,
        "route": {"from": a_name, "to": b_name, "distance_km": round(route["distance_km"], 1),
                  "duration_min": round(route["duration_min"]), "source": route["source"],
                  "buffer_km": buffer_km},
        "hidden_gems": [fmt(r) for _, r in hidden.iterrows()],
        "famous_and_good": [fmt(r) for _, r in famous.iterrows()],
        "stats": {"places_near_route": int(len(near)), "hidden_candidates": int(near["is_hidden"].sum()),
                  "famous_candidates": int(near["is_famous_good"].sum()),
                  "youtube_mode": youtube, "youtube_places_boosted": int((near["yt_boost"] > 0).sum())},
        "unverified_new_places": yt.get("_new_places", []),
        "rules": {"hidden_gem": f"quality_rank >= {GEM_RANK_MIN:.0f} and exposure <= {HIDDEN_MAX_EXPOSURE}",
                  "famous_and_good": f"quality_rank >= {GEM_RANK_MIN:.0f} and exposure >= {FAMOUS_MIN_EXPOSURE}",
                  "order": "safety class (safe > caution > unknown > avoid), then gem score"},
    }


def main():
    ap = argparse.ArgumentParser(description="5 hidden gems + 5 famous-and-good places along a route")
    ap.add_argument("--from", dest="start", required=True, help="town name or 'lat,lon'")
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--buffer-km", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--youtube", choices=["off", "cache", "live"], default="cache")
    ap.add_argument("--only", choices=["eat", "visit"], default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        kinds = (a.only,) if a.only else ("eat", "visit")
        out = recommend(a.start, a.end, a.buffer_km, a.k, youtube=a.youtube, kinds=kinds)
    except Exception as exc:
        out = {"success": False, "error": str(exc)}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
