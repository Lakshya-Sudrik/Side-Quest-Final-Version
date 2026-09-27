"""YouTube layer: find videos about places, check they look genuine, match them to
known places, and turn verified evidence into a SMALL, capped ranking boost.

    python src/youtube_verify.py --town "Hampi"            # live (needs YOUTUBE_API_KEY + internet)
    python src/youtube_verify.py --video https://youtu.be/XXXX  # check one video

What counts as "genuine" (each check adds or removes trust points, with a reason)
    channel      : age, subscribers, number of uploads
    engagement   : likes/views and comments/views - bought views have almost none
    view spikes  : a young/small channel with a huge view count
    comments     : copy-pasted / bot-like comments vs people saying they visited
    disclosures  : YouTube's own flags - paid promotion (sponsored, not organic) and
                   altered/synthetic media (AI) -> never counted as evidence
    location     : the video's recorded location must be near the place it names
A place gets a boost only from videos judged "genuine" that name it AND its town.
A video about a place we don't know is listed as "unverified new place" - it is
never shown as a hidden gem until real visitor reviews exist.

Quota (free tier 10,000 units/day): search = 100 units, everything else = 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "youtube" / "signals.json"
API = "https://www.googleapis.com/youtube/v3"
MAX_BOOST = 5.0
GENUINE_MIN = 65
SUSPICIOUS_MAX = 40
log = logging.getLogger("youtube")

STOP = {"the", "and", "of", "in", "at", "to", "a", "sri", "shri", "shree", "temple", "fort", "lake",
        "falls", "hill", "beach", "park", "garden", "museum", "palace", "point", "view", "restaurant",
        "cafe", "hotel", "kitchen", "foods", "house", "india", "best", "top", "places", "place"}
VISITED = re.compile(r"\b(visited|went there|been there|was there|we went|i went|last (week|month|year)|"
                     r"this place is|we visited|my visit|going there)\b", re.I)


def api_key() -> Optional[str]:
    k = os.environ.get("YOUTUBE_API_KEY")
    if k:
        return k.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("YOUTUBE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class YouTubeUnavailable(RuntimeError):
    """Raised with a message that never contains the API key."""


def _http_get(endpoint: str, params: Dict) -> Dict:
    import requests
    p = dict(params, key=api_key())
    try:
        r = requests.get(f"{API}/{endpoint}", params=p, timeout=10)
    except requests.RequestException as exc:  # the exception text contains the full URL (and key)
        raise YouTubeUnavailable(f"cannot reach YouTube ({type(exc).__name__})") from None
    if r.status_code == 403 and "quota" in r.text.lower():
        raise YouTubeUnavailable("YouTube quota exceeded for today")
    if r.status_code in (400, 403):
        raise YouTubeUnavailable(f"YouTube refused the request (HTTP {r.status_code}) - check the API key "
                                 "and that YouTube Data API v3 is enabled for it")
    if r.status_code >= 300:
        raise YouTubeUnavailable(f"YouTube error HTTP {r.status_code}")
    return r.json()


class YouTubeClient:
    """Thin API wrapper. `fetch` is injectable so tests run without the network."""

    def __init__(self, fetch: Callable[[str, Dict], Dict] = _http_get):
        self.fetch = fetch
        self.units = 0

    def search(self, q: str, n: int = 25, lat: float = None, lon: float = None, radius_km: int = 50) -> List[str]:
        params = {"part": "id", "q": q, "type": "video", "maxResults": n, "regionCode": "IN",
                  "relevanceLanguage": "en", "order": "relevance"}
        if lat is not None:
            params.update(location=f"{lat},{lon}", locationRadius=f"{radius_km}km")
        self.units += 100
        js = self.fetch("search", params)
        return [it["id"]["videoId"] for it in js.get("items", []) if it.get("id", {}).get("videoId")]

    def videos(self, ids: List[str]) -> List[Dict]:
        out = []
        for i in range(0, len(ids), 50):
            self.units += 1
            js = self.fetch("videos", {"part": "snippet,statistics,contentDetails,status,recordingDetails,"
                                               "paidProductPlacementDetails", "id": ",".join(ids[i:i + 50])})
            out += js.get("items", [])
        return out

    def channels(self, ids: List[str]) -> Dict[str, Dict]:
        out = {}
        for i in range(0, len(ids), 50):
            self.units += 1
            js = self.fetch("channels", {"part": "snippet,statistics", "id": ",".join(ids[i:i + 50])})
            out.update({c["id"]: c for c in js.get("items", [])})
        return out

    def comments(self, video_id: str, n: int = 50) -> List[str]:
        self.units += 1
        try:
            js = self.fetch("commentThreads", {"part": "snippet", "videoId": video_id, "maxResults": n,
                                               "textFormat": "plainText", "order": "relevance"})
        except Exception:  # comments disabled
            return []
        return [it["snippet"]["topLevelComment"]["snippet"].get("textDisplay", "")
                for it in js.get("items", [])]


# ---------------------------------------------------------------------------
# authenticity
# ---------------------------------------------------------------------------
def _age_days(iso: str) -> float:
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t).days
    except Exception:
        return float("nan")


def _i(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def assess_video(video: Dict, channel: Optional[Dict], comments: List[str]) -> Dict:
    """Trust score 0-100 + verdict + the reasons behind it."""
    s, reasons, flags = 50.0, [], []
    st, sn = video.get("statistics", {}), video.get("snippet", {})
    views, likes, ncom = _i(st.get("viewCount")), _i(st.get("likeCount")), _i(st.get("commentCount"))
    status = video.get("status", {})
    paid = video.get("paidProductPlacementDetails", {}).get("hasPaidProductPlacement")
    if status.get("containsSyntheticMedia") is True:
        return {"trust": 0, "verdict": "rejected", "flags": ["synthetic_media"],
                "reasons": ["Creator declared altered or synthetic (AI) content"]}
    if paid:
        flags.append("sponsored")
        s -= 20
        reasons.append("Declared paid promotion - sponsored, not an independent visit")
    # channel
    if channel:
        cs = channel.get("statistics", {})
        subs, uploads = _i(cs.get("subscriberCount")), _i(cs.get("videoCount"))
        age = _age_days(channel.get("snippet", {}).get("publishedAt", ""))
        if age >= 365:
            s += 10; reasons.append(f"Channel is {age / 365:.1f} years old")
        elif age < 90:
            s -= 15; reasons.append("Channel is less than 3 months old")
        if uploads >= 20:
            s += 5; reasons.append(f"{int(uploads)} uploads on the channel")
        elif uploads < 3:
            s -= 10; reasons.append("Channel has almost no other videos")
        if not cs.get("hiddenSubscriberCount") and subs > 0 and views > 50 * subs and views > 100000 and age < 365:
            s -= 20; flags.append("view_spike")
            reasons.append("Views are far above what this young channel normally gets")
    else:
        reasons.append("Channel details unavailable")
    # engagement
    if views >= 2000:
        lr = likes / views if views and not math.isnan(likes) else float("nan")
        if not math.isnan(lr):
            if lr < 0.004:
                s -= 20; flags.append("low_likes")
                reasons.append(f"Only {lr:.2%} of viewers liked it - typical of bought views")
            elif lr > 0.25:
                s -= 10; flags.append("like_farming")
                reasons.append(f"{lr:.0%} like rate is unusually high")
            else:
                s += 10; reasons.append(f"Normal like rate ({lr:.1%})")
        cr = ncom / views if views and not math.isnan(ncom) else float("nan")
        if not math.isnan(cr) and cr < 0.0003 and views > 20000:
            s -= 10; reasons.append("Very few comments for its view count")
    # comments
    if comments:
        txt = [re.sub(r"\W+", " ", c.lower()).strip() for c in comments if c.strip()]
        dup = 1 - len(set(txt)) / max(len(txt), 1)
        generic = np.mean([len(t.split()) <= 3 for t in txt]) if txt else 0
        visited = np.mean([bool(VISITED.search(c)) for c in comments]) if comments else 0
        if dup > 0.3:
            s -= 20; flags.append("duplicate_comments")
            reasons.append(f"{dup:.0%} of comments are copies of each other (bot-like)")
        if generic > 0.7 and len(txt) >= 10:
            s -= 10; reasons.append("Most comments are generic one-liners")
        if visited >= 0.05:
            s += 15; reasons.append(f"{visited:.0%} of commenters say they have been there")
    s = float(np.clip(s, 0, 100))
    verdict = "genuine" if s >= GENUINE_MIN and "sponsored" not in flags else \
        "suspicious" if s <= SUSPICIOUS_MAX else "uncertain"
    return {"trust": round(s), "verdict": verdict, "flags": flags, "reasons": reasons}


# ---------------------------------------------------------------------------
# matching videos to places
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def _words(name: str) -> set:
    return {w for w in _norm(name).split() if len(w) >= 4 and w not in STOP}


def match_places(video: Dict, places: pd.DataFrame, max_km: float = 50.0) -> List[str]:
    """Place ids whose distinctive name words AND town appear in the title/description
    (or the video was recorded within max_km of the place)."""
    sn = video.get("snippet", {})
    text = _norm(sn.get("title", "") + " " + sn.get("description", "")[:1500] + " " + " ".join(sn.get("tags", [])))
    words = set(text.split())
    loc = video.get("recordingDetails", {}).get("location", {})
    vlat, vlon = loc.get("latitude"), loc.get("longitude")
    hits = []
    for pid, name, city, lat, lon in places[["id", "name", "city", "lat", "lon"]].itertuples(index=False):
        w = _words(name)
        if not w or not w <= words:
            continue
        near = False
        if vlat is not None and not pd.isna(lat):
            d = 6371 * 2 * math.asin(math.sqrt(math.sin(math.radians(lat - vlat) / 2) ** 2 + math.cos(
                math.radians(lat)) * math.cos(math.radians(vlat)) * math.sin(math.radians(lon - vlon) / 2) ** 2))
            if d > max_km:
                continue  # recorded somewhere else -> not about this place
            near = True
        if near or _norm(city).split(" ")[0] in words:
            hits.append(pid)
    return hits


def boost_for(n_genuine: int) -> float:
    return 0.0 if n_genuine <= 0 else min(MAX_BOOST, 2.0 + 1.5 * (n_genuine - 1))


# ---------------------------------------------------------------------------
# run for a set of candidate places
# ---------------------------------------------------------------------------
def scan(places: pd.DataFrame, towns: Iterable[tuple], client: Optional[YouTubeClient] = None,
         per_town: int = 25, max_towns: int = 3) -> Dict:
    """Search each town, assess videos, match to places. Returns {place_id: signal, '_new_places': [...]}."""
    client = client or YouTubeClient()
    out: Dict = {"_new_places": []}
    towns = list(towns)[:max_towns]
    for town, lat, lon in towns:
        ids = []
        for q in (f"hidden gem {town}", f"{town} places to visit offbeat"):
            ids += client.search(q, n=per_town, lat=lat, lon=lon)
        ids = list(dict.fromkeys(ids))
        vids = client.videos(ids)
        chans = client.channels(list({v["snippet"]["channelId"] for v in vids if v.get("snippet")}))
        local = places[(places["city"].str.lower() == str(town).lower())
                       | (np.hypot(places["lat"] - lat, places["lon"] - lon) < 0.5)]
        for v in vids:
            matched = match_places(v, local)
            if not matched and "hidden" not in v.get("snippet", {}).get("title", "").lower():
                continue
            a = assess_video(v, chans.get(v["snippet"]["channelId"]), client.comments(v["id"], 50))
            rec = {"video_id": v["id"], "url": f"https://youtu.be/{v['id']}",
                   "title": v["snippet"].get("title", ""), "channel": v["snippet"].get("channelTitle", ""),
                   "views": _i(v.get("statistics", {}).get("viewCount")), **a}
            if matched:
                for pid in matched:
                    sig = out.setdefault(pid, {"videos": [], "verified_videos": 0, "boost": 0.0})
                    sig["videos"].append(rec)
                    if a["verdict"] == "genuine":
                        sig["verified_videos"] += 1
                    sig["boost"] = boost_for(sig["verified_videos"])
            elif a["verdict"] == "genuine":
                out["_new_places"].append({**rec, "town": town,
                                           "status": "unverified new place - needs visitor reviews "
                                                     "before it can be shown as a hidden gem"})
    out["_units_used"] = client.units
    return out


def _load_cache() -> Dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def _save_cache(sig: Dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cur = _load_cache()
    for k, v in sig.items():
        if k == "_new_places":
            seen = {x["video_id"] for x in cur.get(k, [])}
            cur[k] = cur.get(k, []) + [x for x in v if x["video_id"] not in seen]
        elif not k.startswith("_"):
            cur[k] = v
    cur["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CACHE.write_text(json.dumps(cur, indent=1, default=float), encoding="utf-8")


def signals_for(near: pd.DataFrame, mode: str = "cache") -> Dict:
    """Used by recommender.py. mode: off | cache (saved results) | live (call the API now)."""
    if mode == "off" or near is None or len(near) == 0:
        return {}
    if mode == "live":
        if not api_key():
            log.warning("YOUTUBE_API_KEY not set - using cached YouTube results")
        else:
            towns = (near.groupby("city").agg(lat=("lat", "median"), lon=("lon", "median"), n=("id", "size"))
                     .sort_values("n", ascending=False).head(3))
            try:
                sig = scan(near, [(c, r.lat, r.lon) for c, r in towns.iterrows()])
                _save_cache(sig)
            except YouTubeUnavailable as exc:  # offline / quota / blocked - message has no key
                log.warning("YouTube live scan failed (%s) - using cached results", exc)
            except Exception as exc:
                log.warning("YouTube live scan failed (%s) - using cached results", type(exc).__name__)
    cache = _load_cache()
    ids = set(near["id"])
    res = {k: v for k, v in cache.items() if k in ids}
    towns = set(near["city"].str.lower())
    res["_new_places"] = [x for x in cache.get("_new_places", []) if str(x.get("town", "")).lower() in towns]
    return res


def check_video(url_or_id: str, client: Optional[YouTubeClient] = None) -> Dict:
    """Assess a single video link a user submits."""
    client = client or YouTubeClient()
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url_or_id)
    vid = m.group(1) if m else url_or_id.strip()
    v = client.videos([vid])
    if not v:
        return {"video_id": vid, "verdict": "not_found"}
    ch = client.channels([v[0]["snippet"]["channelId"]])
    a = assess_video(v[0], ch.get(v[0]["snippet"]["channelId"]), client.comments(vid))
    return {"video_id": vid, "title": v[0]["snippet"].get("title"), **a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town")
    ap.add_argument("--video")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.video:
        try:
            print(json.dumps(check_video(a.video), indent=2))
        except YouTubeUnavailable as exc:
            print(json.dumps({"success": False, "error": str(exc)}))
        return
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import geo_india as gi
    from recommender import load_candidates
    c = gi.town_coords(a.town)
    if c is None:
        raise SystemExit(f"unknown town {a.town}")
    places = load_candidates()
    near = places[np.hypot(places["lat"] - c[0], places["lon"] - c[1]) < 0.5]
    sig = scan(near, [(a.town, c[0], c[1])])
    _save_cache(sig)
    print(json.dumps({k: v for k, v in sig.items() if k != "_new_places"}, indent=1, default=float)[:4000])
    print(f"new places: {len(sig['_new_places'])}   quota units used: {sig['_units_used']}")


if __name__ == "__main__":
    main()
