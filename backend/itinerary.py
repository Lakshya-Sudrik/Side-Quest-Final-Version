"""Dynamic trip planner: stops the user adds/removes, a timeline, and re-planning when a
flight (or anything before departure) is delayed.

Time model (all local times)
    start       = depart_at + delay
    drive       = route duration split in proportion to distance along the route
    each stop   = detour there and back (off-route km x 1.3 road factor, 45 km/h) + visit time
    must finish at the destination by arrive_by
If the stops no longer fit (e.g. after a delay), the planner keeps the set with the highest
total gem score that fits (greedy by score per minute, then fills gaps), lists what it
dropped, and suggests other gems on the same route that fit the time that is left -
the same gems when they still fit, shorter/closer ones when they don't.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ROOT

sys.path.insert(0, str(ROOT / "src"))

AVG_SPEED_KMH = 45.0
ROAD_FACTOR = 1.3
MAX_OFF_ROUTE_KM = 40.0
VISIT_MIN = {"eat": 45, "temple": 45, "church_mosque": 40, "fort_palace": 90, "monument": 60, "water": 60,
             "hill_view": 60, "nature_park": 90, "museum": 75, "market_mall": 60, "attraction": 60}


def visit_minutes(kind: str, category: str) -> int:
    if kind == "eat":
        return VISIT_MIN["eat"]
    return VISIT_MIN.get(str(category), VISIT_MIN["attraction"])


def parse_local(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace(" ", "T")[:16])


def build_route(origin: str, destination: str) -> Dict:
    import recommender as rec
    a = rec.parse_point(origin)
    b = rec.parse_point(destination)
    r = rec.get_route(a[:2], b[:2])
    g = np.asarray(r["geometry"])
    if len(g) > 300:
        g = g[np.linspace(0, len(g) - 1, 300).astype(int)]
    straight = float(rec.hav_km(a[0], a[1], b[0], b[1]))
    along_total = float(rec.distance_to_route(np.array([b[0]]), np.array([b[1]]), g.tolist())[1][0]) or max(straight, 1e-6)
    return {"distance_km": round(r["distance_km"], 1), "duration_min": round(r["duration_min"], 1),
            "source": r["source"], "geometry": [[round(x, 5), round(y, 5)] for x, y in g.tolist()],
            "along_total_km": along_total, "from": [a[0], a[1]], "to": [b[0], b[1]]}


def place_snapshot(place_id: str, route: Dict) -> Dict:
    """Details of a recommendable place + where it sits relative to this route."""
    import pandas as pd
    import recommender as rec
    c = rec.load_candidates()
    row = c[c["id"] == place_id]
    if row.empty:
        lc = rec.listing_candidates()
        row = lc[lc["id"] == place_id] if len(lc) else lc
    if row.empty:
        raise LookupError(f"unknown place {place_id}")
    r = row.iloc[0]
    off, along = rec.distance_to_route(np.array([r["lat"]]), np.array([r["lon"]]), route["geometry"])
    off, along = float(off[0]), float(along[0])
    if off > MAX_OFF_ROUTE_KM:
        raise ValueError(f"{r['name']} is {off:.0f} km off this route (max {MAX_OFF_ROUTE_KM:.0f})")
    detour_min = off * ROAD_FACTOR * 2 / AVG_SPEED_KMH * 60
    return {"place_id": place_id, "name": str(r["name"]), "kind": str(r["kind"]), "category": str(r["category"]),
            "city": str(r["city"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
            "rating": None if pd.isna(r["rating"]) else float(r["rating"]),
            "gem_score": round(float(r["quality_rank"]), 1) if not pd.isna(r["quality_rank"]) else 50.0,
            "off_route_km": round(off, 1), "along_km": round(along, 1),
            "detour_min": int(round(detour_min)), "visit_min": visit_minutes(str(r["kind"]), str(r["category"]))}


def schedule(itin: Dict, route: Dict, stops: List[Dict]) -> Dict:
    """Timeline for the given stops (ordered along the route)."""
    import recommender as rec
    start = parse_local(itin["depart_at"]) + timedelta(minutes=int(itin.get("delay_min") or 0))
    deadline = parse_local(itin["arrive_by"])
    total_along = max(route["along_total_km"], 1e-6)
    per_km = route["duration_min"] / total_along
    fixed_events = []
    planning = itin.get("planning_data") or {}
    for fixed in planning.get("fixed_stops", []):
        off, along = rec.distance_to_route(np.array([float(fixed["lat"])]), np.array([float(fixed["lon"])]), route["geometry"])
        off, along = float(off[0]), float(along[0])
        fixed_events.append({"fixed": True, "name": fixed["place"], "category": "User itinerary",
                             "city": "", "kind": "visit", "along_km": along,
                             "detour_min": int(round(off * ROAD_FACTOR * 2 / AVG_SPEED_KMH * 60)),
                             "visit_min": int(fixed.get("duration_min", 60)), "fixed_at": fixed["time"]})
    t, prev_along, legs = start, 0.0, []
    fixed_on_time = True
    for s in sorted([*stops, *fixed_events], key=lambda x: (x["along_km"], x.get("fixed_at", ""))):
        drive = max(0.0, s["along_km"] - prev_along) * per_km
        t += timedelta(minutes=drive + s["detour_min"] / 2)
        if s.get("fixed"):
            appointment = parse_local(s["fixed_at"])
            if t > appointment:
                fixed_on_time = False
            else:
                t = appointment
        arrive = t
        t += timedelta(minutes=s["visit_min"])
        leave = t
        t += timedelta(minutes=s["detour_min"] / 2)
        prev_along = s["along_km"]
        legs.append({**s, "arrive": arrive.strftime("%Y-%m-%d %H:%M"), "leave": leave.strftime("%Y-%m-%d %H:%M")})
    t += timedelta(minutes=max(0.0, total_along - prev_along) * per_km)
    slack = (deadline - t).total_seconds() / 60
    return {"start": start.strftime("%Y-%m-%d %H:%M"), "arrive_destination": t.strftime("%Y-%m-%d %H:%M"),
            "deadline": deadline.strftime("%Y-%m-%d %H:%M"), "slack_min": int(round(slack)),
            "fits": slack >= 0 and fixed_on_time, "fixed_stops_on_time": fixed_on_time, "stops": legs,
            "free_time_min": int(round((deadline - start).total_seconds() / 60 - route["duration_min"])),
            "distance_note": (f"road-network route; {planning.get('transport', 'car')} ETA estimate" if route["source"] == "osrm_road"
                              else "estimated drive times (straight line x1.3) - road routing unavailable")}


def cost_min(s: Dict) -> float:
    return s["detour_min"] + s["visit_min"]


def best_fit(itin: Dict, route: Dict, stops: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Keep the stops with the most gem score that still fit the time window."""
    if schedule(itin, route, stops)["fits"]:
        return stops, []
    order = sorted(stops, key=lambda s: -(s["gem_score"] / max(cost_min(s), 1)))
    keep: List[Dict] = []
    for s in order:
        if schedule(itin, route, keep + [s])["fits"]:
            keep.append(s)
    dropped = [s for s in stops if s not in keep]
    return keep, dropped


def replacements(itin: Dict, route: Dict, kept: List[Dict], exclude: set, limit: int = 5) -> List[Dict]:
    """Other gems on this route that fit in the time left after the kept stops."""
    import recommender as rec
    sched = schedule(itin, route, kept)
    if sched["slack_min"] <= 20:
        return []
    c = rec.load_candidates()
    g = np.asarray(route["geometry"])
    pad = 0.2
    box = c[c["lat"].between(g[:, 0].min() - pad, g[:, 0].max() + pad)
            & c["lon"].between(g[:, 1].min() - pad, g[:, 1].max() + pad)]
    enough = box["n_reviews"] >= box["kind"].map(rec.MIN_REVIEWS)
    # same rules as the recommender: no shops, spas, agencies or tour operators; no big chains
    not_sight = (box["kind"] == "visit") & box["name"].str.lower().str.contains(rec.NOT_A_SIGHT, regex=True)
    box = box[enough & ~not_sight & (box["quality_rank"] >= rec.GEM_RANK_MIN) & ~box["id"].isin(exclude)
              & (box["chain_outlets"] <= rec.MAX_CHAIN_OUTLETS)]
    if box.empty:
        return []
    off, along = rec.distance_to_route(box["lat"].to_numpy(), box["lon"].to_numpy(), route["geometry"])
    box = box.assign(off=off, along=along)
    box = box[box["off"] <= 10.0]
    # Prefer evidence-based safety alternatives and never suggest a place the
    # project's safety model classifies as avoid.
    safety = rec.safety_blocks(box) if len(box) else []
    box = box.assign(safety=safety, safety_class=[b["class"] for b in safety])
    box = box[box["safety_class"] != "avoid"]
    box = box.assign(visit=[visit_minutes(k, cat) for k, cat in zip(box["kind"], box["category"])],
                     detour=box["off"] * ROAD_FACTOR * 2 / AVG_SPEED_KMH * 60)
    box = box[box["visit"] + box["detour"] <= sched["slack_min"]]
    box = box.sort_values("quality_rank", ascending=False).drop_duplicates("dedupe_key")
    out = []
    for r in box.head(limit * 4).itertuples(index=False):
        s = {"place_id": r.id, "name": r.name, "kind": r.kind, "category": r.category, "city": r.city,
             "gem_score": round(float(r.quality_rank), 1), "off_route_km": round(float(r.off), 1),
             "along_km": round(float(r.along), 1), "detour_min": int(round(r.detour)), "visit_min": int(r.visit),
             "hidden": bool(r.exposure_pct <= rec.HIDDEN_MAX_EXPOSURE),
             "safety": r.safety,
             "rating": None if r.rating != r.rating else float(r.rating)}
        if schedule(itin, route, kept + [s])["fits"]:
            out.append(s)
        if len(out) >= limit:
            break
    return out
