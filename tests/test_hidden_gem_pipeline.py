"""Hidden-gem pipeline: YouTube checks, route maths, picking rules, end-to-end run."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import recommender as rec  # noqa: E402
import youtube_verify as yv  # noqa: E402


def _iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _video(vid, title, views, likes, comments, channel="c1", desc="", paid=False, synthetic=False, loc=None):
    v = {"id": vid, "snippet": {"title": title, "description": desc, "channelId": channel,
                                "channelTitle": "Traveller", "tags": []},
         "statistics": {"viewCount": str(views), "likeCount": str(likes), "commentCount": str(comments)},
         "status": {"containsSyntheticMedia": synthetic},
         "paidProductPlacementDetails": {"hasPaidProductPlacement": paid}}
    if loc:
        v["recordingDetails"] = {"location": {"latitude": loc[0], "longitude": loc[1]}}
    return v


OLD_CHANNEL = {"id": "c1", "snippet": {"publishedAt": _iso(4 * 365)},
               "statistics": {"subscriberCount": "52000", "videoCount": "240"}}
NEW_CHANNEL = {"id": "c2", "snippet": {"publishedAt": _iso(30)},
               "statistics": {"subscriberCount": "300", "videoCount": "2"}}
REAL_COMMENTS = ["We visited last month, the water was so clear", "Went there with family, amazing",
                 "Thanks for the directions!", "Is it open in monsoon?", "Been there in 2019, still lovely"]
BOT_COMMENTS = ["Nice video"] * 12 + ["Great"] * 6


# --- authenticity -----------------------------------------------------------
def test_genuine_creator_is_genuine():
    a = yv.assess_video(_video("v1", "Zalor Beach Goa hidden gem", 40000, 1800, 120), OLD_CHANNEL, REAL_COMMENTS)
    assert a["verdict"] == "genuine" and a["trust"] >= yv.GENUINE_MIN


def test_bought_views_are_suspicious():
    a = yv.assess_video(_video("v2", "hidden gem", 900000, 900, 10, channel="c2"), NEW_CHANNEL, BOT_COMMENTS)
    assert a["verdict"] == "suspicious"
    assert {"low_likes", "view_spike", "duplicate_comments"} <= set(a["flags"])


def test_declared_ai_content_is_rejected():
    a = yv.assess_video(_video("v3", "x", 5000, 300, 20, synthetic=True), OLD_CHANNEL, REAL_COMMENTS)
    assert a["verdict"] == "rejected" and a["trust"] == 0


def test_sponsored_video_is_never_genuine_evidence():
    a = yv.assess_video(_video("v4", "x", 40000, 1800, 120, paid=True), OLD_CHANNEL, REAL_COMMENTS)
    assert a["verdict"] != "genuine" and "sponsored" in a["flags"]


# --- matching ---------------------------------------------------------------
PLACES = pd.DataFrame({"id": ["visit:Margao|Zalor Beach", "visit:Margao|Colva Beach"],
                       "name": ["Zalor Beach", "Colva Beach"], "city": ["Margao", "Margao"],
                       "lat": [15.15, 15.28], "lon": [73.95, 73.92]})


def test_match_needs_place_and_town():
    v = _video("v5", "Zalor beach - Margao's secret", 1, 1, 1)
    assert yv.match_places(v, PLACES) == ["visit:Margao|Zalor Beach"]
    v2 = _video("v6", "Zalor beach vlog", 1, 1, 1)  # no town, no location -> not trusted as a match
    assert yv.match_places(v2, PLACES) == []


def test_match_rejects_video_recorded_far_away():
    v = _video("v7", "Zalor beach Margao", 1, 1, 1, loc=(28.6, 77.2))  # recorded in Delhi
    assert yv.match_places(v, PLACES) == []


class FakeClient(yv.YouTubeClient):
    def __init__(self):
        super().__init__(fetch=self._fetch)

    def _fetch(self, endpoint, params):
        if endpoint == "search":
            return {"items": [{"id": {"videoId": v}} for v in ("g1", "g2", "b1", "n1")]}
        if endpoint == "videos":
            vids = {"g1": _video("g1", "Zalor Beach Margao hidden gem", 40000, 1800, 120),
                    "g2": _video("g2", "Zalor beach, Margao - quiet sunset", 22000, 900, 60),
                    "b1": _video("b1", "Colva Beach Margao", 900000, 800, 5, channel="c2"),
                    "n1": _video("n1", "Hidden waterfall near Margao nobody knows", 30000, 1500, 90)}
            return {"items": [vids[i] for i in params["id"].split(",") if i in vids]}
        if endpoint == "channels":
            return {"items": [c for c in (OLD_CHANNEL, NEW_CHANNEL) if c["id"] in params["id"]]}
        if endpoint == "commentThreads":
            txt = BOT_COMMENTS if params["videoId"] == "b1" else REAL_COMMENTS
            return {"items": [{"snippet": {"topLevelComment": {"snippet": {"textDisplay": t}}}} for t in txt]}
        raise AssertionError(endpoint)


def test_scan_boosts_only_verified_and_lists_new_places():
    sig = yv.scan(PLACES, [("Margao", 15.27, 73.96)], client=FakeClient(), max_towns=1)
    zalor = sig["visit:Margao|Zalor Beach"]
    assert zalor["verified_videos"] == 2 and 0 < zalor["boost"] <= yv.MAX_BOOST
    colva = sig["visit:Margao|Colva Beach"]  # only a bought-views video
    assert colva["verified_videos"] == 0 and colva["boost"] == 0
    assert [p["video_id"] for p in sig["_new_places"]] == ["n1"]
    assert "unverified" in sig["_new_places"][0]["status"]
    assert sig["_units_used"] >= 200  # two searches


def test_boost_is_capped():
    assert yv.boost_for(0) == 0 and yv.boost_for(50) == yv.MAX_BOOST


# --- route maths ------------------------------------------------------------
def test_distance_to_route():
    geom = [(12.0, 77.0), (12.0, 78.0)]  # ~109 km east-west line
    off, along = rec.distance_to_route(np.array([12.0, 12.09]), np.array([77.5, 77.0]), geom)
    assert off[0] < 0.1 and abs(along[0] - 54.4) < 1.5
    assert abs(off[1] - 10.0) < 0.3 and along[1] < 0.1


def test_parse_point():
    assert rec.parse_point("12.9,77.6")[:2] == (12.9, 77.6)
    lat, lon, _ = rec.parse_point("Munnar")
    assert abs(lat - 10.09) < 0.1 and abs(lon - 77.06) < 0.1
    with pytest.raises(ValueError):
        rec.parse_point("Qwertyuiopville")


def test_pick_spreads_mixes_and_dedupes():
    n = 12
    pool = pd.DataFrame({
        "kind": ["eat"] * 8 + ["visit"] * 4, "gem_score": np.linspace(99, 70, n),
        "along_km": [0, 0.5, 1, 1.5, 2, 50, 100, 150, 1, 60, 120, 160],
        "dedupe_key": [f"p{i}" for i in range(n - 1)] + ["p0"],  # last one repeats a brand
        "safety_class": ["safe"] * n, "safety_rank": [0] * n})
    out = rec._pick(pool, 5, 2, set(), min_gap_km=20)
    assert len(out) == 5 and out["dedupe_key"].is_unique
    assert (out["kind"] == "visit").sum() >= 2 and (out["kind"] == "eat").sum() >= 2
    assert np.diff(np.sort(out["along_km"].to_numpy())).min() >= 20


def test_pick_skips_avoid_when_alternatives_exist():
    pool = pd.DataFrame({"kind": ["eat"] * 6, "gem_score": [99, 90, 80, 70, 60, 50],
                         "along_km": [0, 30, 60, 90, 120, 150], "dedupe_key": list("abcdef"),
                         "safety_class": ["avoid", "safe", "safe", "caution", "safe", "safe"]})
    pool["safety_rank"] = pool["safety_class"].map(rec.SAFETY_ORDER)
    out = rec._pick(pool, 5, 0, set(), 10)
    assert "a" not in set(out["dedupe_key"])


# --- end to end on the real data (skipped if the scored tables aren't built) --
@pytest.mark.skipif(not (rec.FOOD.exists() and rec.ATTR.exists()), reason="scored tables not built")
def test_end_to_end_route():
    out = rec.recommend("Mumbai", "Pune", youtube="off")
    assert out["success"]
    assert len(out["hidden_gems"]) == 5 and len(out["famous_and_good"]) == 5
    ids = [g["id"] for g in out["hidden_gems"] + out["famous_and_good"]]
    assert len(set(ids)) == 10
    for g in out["hidden_gems"] + out["famous_and_good"]:
        assert g["safety"]["class"] in ("safe", "caution", "unknown", "avoid")
        assert g["safety"]["reasons"] and "distance" in g and g["distance"]["from_route_km"] <= 15.1
        assert g["type"] in ("place to eat", "place to visit")
    for g in out["hidden_gems"]:
        assert g["exposure_pct"] <= rec.HIDDEN_MAX_EXPOSURE and g["quality_rank"] >= 60
    for g in out["famous_and_good"]:
        assert g["exposure_pct"] >= rec.FAMOUS_MIN_EXPOSURE


# --- attractions model: features never see the target half ------------------
def test_attraction_features_use_only_half_a():
    import attractions_model as am
    r = pd.DataFrame({"place_id": ["x|p"] * 6, "Rating": [5, 5, 5, 1, 1, 1], "words": [50.0] * 6,
                      "half": [0, 0, 0, 1, 1, 1], **{k: [0.0] * 6 for k in am.LEXICON}})
    fa = am.aggregate(r[r["half"] == 0], prior=4.0)
    assert fa.loc["x|p", "rating_mean"] == 5.0  # the 1-star half B is invisible to the features
