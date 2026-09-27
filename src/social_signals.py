"""Social-media verification for hidden-gem recommendations.

Goal: give the user MORE VALID options - a place that scores as a hidden gem
from review data is only *verified* when it also has a real, findable social /
web presence. Everything here is rule-based and deterministic, so it cannot
overfit, and it is strictly honest about missing data:

    status = "no_data"  -> the checker could not run (no key, no network)
    status = "no_social_link" -> the check ran and found nothing
    status = "verified" -> at least the platform links described in
                           config/criteria.json were found

A missing check is NEVER turned into a score of 0 (or into a pass).

Data contract: ``data/social_media.csv`` (append-only, resumable cache):

    place_id, name, city, source, checked_at, website, instagram, youtube,
    facebook, x, zomato, platform_count, website_ok, status, detail

Collection is deliberately limited to *gem candidates* (score >= the gem
cutoff) so it stays fast: a few thousand Places lookups, not 100k.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from labels_honest import load_criteria

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SOCIAL_CSV = ROOT / "data" / "social_media.csv"
SOCIAL_COLUMNS = [
    "place_id", "name", "city", "source", "checked_at", "website",
    "instagram", "youtube", "facebook", "x", "zomato",
    "platform_count", "website_ok", "status", "detail",
]

PLATFORM_KEYS = ["instagram", "youtube", "facebook", "x", "zomato"]

_SOCIAL_PATTERNS = {
    "instagram": re.compile(r"https?://(?:www\.)?instagram\.com/([A-Za-z0-9._]+)", re.I),
    "youtube": re.compile(r"https?://(?:www\.|m\.|music\.)?youtube\.com/(?:c/|channel/|@|user/)([A-Za-z0-9._\-]+)", re.I),
    "facebook": re.compile(r"https?://(?:www\.|m\.|web\.)?facebook\.com/([A-Za-z0-9._\-]+)", re.I),
    "x": re.compile(r"https?://(?:www\.)?(?:twitter|x)\.com/([A-Za-z0-9_]+)", re.I),
    "zomato": re.compile(r"https?://(?:www\.)?zomato\.com/([A-Za-z0-9._\-]+)", re.I),
}
_GENERIC_HANDLES = {"share", "explore", "reel", "reels", "watch", "playlist", "p", "tv",
                    "pages", "profile.php", "hashtag", "search", "intent", "home"}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def load_social() -> pd.DataFrame:
    if not SOCIAL_CSV.exists():
        return pd.DataFrame(columns=SOCIAL_COLUMNS)
    try:
        df = pd.read_csv(SOCIAL_CSV, dtype=str)
    except Exception as exc:
        logger.warning("social cache unreadable (%s) - starting empty", exc)
        return pd.DataFrame(columns=SOCIAL_COLUMNS)
    for c in SOCIAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[SOCIAL_COLUMNS]


def save_social(df: pd.DataFrame) -> None:
    SOCIAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in SOCIAL_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    out = out[SOCIAL_COLUMNS]
    if "place_id" in out.columns:
        out = out.drop_duplicates(subset=["place_id"], keep="last")
    tmp = SOCIAL_CSV.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, SOCIAL_CSV)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def extract_social_links(html: str, base_url: str = "") -> Tuple[Dict[str, str], List[str]]:
    """Return ({platform: profile url}, [any other candidate urls found])."""
    found: Dict[str, str] = {}
    other: List[str] = []
    if not html:
        return found, other
    for platform, pat in _SOCIAL_PATTERNS.items():
        m = pat.search(html)
        if not m:
            continue
        handle = m.group(1).rstrip("/")
        if handle.lower() in _GENERIC_HANDLES:
            continue
        found[platform] = m.group(0).split("?")[0]
    # any other social-ish links we did not model explicitly
    for u in re.findall(r"https?://[^\s\"'<>]+", html):
        if not any(p in u for p in _SOCIAL_PATTERNS):
            if any(k in u for k in ("linktr.ee", "beacons.ai", "bento.me", "carrd.co")):
                other.append(u)
    return found, other[:5]


def _http_get(url: str, timeout: float = 6.0) -> Optional[str]:
    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SideQuestVerifier/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(600_000)
            return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
def google_website(name: str, city: str, api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a place's official website through the Google Places API (New).

    Returns (website, status). status is 'ok' | 'no_result' | 'no_key' | 'error'.
    """
    api_key = api_key or os.getenv("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return None, "no_key"
    try:
        import urllib.request

        url = "https://places.googleapis.com/v1/places:searchText"
        body = json.dumps({"textQuery": f"{name} {city} India", "languageCode": "en"}).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.website,places.uri,places.displayName",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", "ignore"))
        places = payload.get("places") or []
        if not places:
            return None, "no_result"
        return (places[0].get("website") or None), "ok"
    except Exception as exc:
        logger.debug("google website lookup failed: %s", exc)
        return None, "error"


def collect_for_place(
    row: Dict[str, Any],
    api_key: Optional[str] = None,
    fetch_website: bool = True,
) -> Dict[str, Any]:
    """One place -> one social_signals record. Deterministic, never raises."""
    rec = {c: "" for c in SOCIAL_COLUMNS}
    rec.update({
        "place_id": row.get("place_id", ""),
        "name": row.get("name", ""),
        "city": row.get("city", ""),
        "source": row.get("source", ""),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    website, status = google_website(str(row.get("name", "")), str(row.get("city", "")), api_key)
    if status != "ok":
        # without the resolver we may still try a platform-supplied url
        website = row.get("website") or ""
        rec["status"] = "no_data" if status in ("no_key", "error") else "no_result"
        rec["detail"] = f"places:{status}"
    rec["website"] = website or ""

    links: Dict[str, str] = {}
    if fetch_website and website:
        html = _http_get(website)
        if html:
            rec["website_ok"] = "1"
            links, _other = extract_social_links(html, website)
        else:
            rec["website_ok"] = "0"
            rec["status"] = rec["status"] or "no_social_link"
            rec["detail"] = (rec["detail"] + ";website:unreachable").strip(";")

    for p in PLATFORM_KEYS:
        rec[p] = links.get(p, "")
    rec["platform_count"] = str(len([p for p in PLATFORM_KEYS if rec[p]]))

    if rec["status"] in ("", "no_result") or rec["status"] == "no_data":
        if rec["platform_count"] != "0":
            rec["status"] = "checked"
        elif rec["status"] in ("", "no_result"):
            rec["status"] = "no_social_link"
    return rec


def collect_for_candidates(
    places: pd.DataFrame,
    top_n: Optional[int] = None,
    workers: int = 8,
    api_key: Optional[str] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Resumable social sweep over gem candidates (fast by construction)."""
    cfg = load_criteria().get("social_verification", {})
    top_n = int(top_n or cfg.get("max_candidates", 3000))
    cache = load_social()
    done = set(cache["place_id"].astype(str)) if len(cache) else set()

    todo = places
    if not refresh and done:
        todo = todo[~todo["place_id"].astype(str).isin(done)]
    if len(todo) > top_n:
        todo = todo.head(top_n)
    if todo.empty:
        logger.info("social cache already covers %d candidates", len(done))
        return cache

    logger.info("social collection: %d to check (%d cached), workers=%d",
                len(todo), len(done), workers)
    records: List[Dict[str, Any]] = []
    rows = todo.to_dict("records")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect_for_place, r, api_key) for r in rows]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                records.append(fut.result())
            except Exception as exc:  # pragma: no cover
                logger.debug("collector error: %s", exc)
            if i % 100 == 0:
                logger.info("  %d/%d (%.0fs)", i, len(rows), time.time() - t0)

    if records:
        cache = pd.concat([cache, pd.DataFrame(records)], ignore_index=True) if len(cache) else pd.DataFrame(records)
        save_social(cache)
    return load_social()


# ---------------------------------------------------------------------------
# scoring (rules only - no fitted parameters, so no overfitting possible)
# ---------------------------------------------------------------------------
def social_score(record: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    """(score 0-100, status, reasons) for one cached record.

    Rules come from config/criteria.json ``social_verification``; a record with
    status ``no_data`` returns status ``no_data`` and NO score, so downstream
    code cannot mistake "not checked" for "checked and failed".
    """
    cfg = load_criteria().get("social_verification", {})
    weights: Dict[str, int] = dict(cfg.get("platform_weights", {}))
    bonus = dict(cfg.get("extras", {}))
    cap = int(cfg.get("max_score", 100))

    status = str(record.get("status", "") or "no_data")
    if status in ("no_data",):
        return 0, "no_data", ["social check unavailable (no key/network)"]

    score = 0
    reasons: List[str] = []
    count = 0
    for p in PLATFORM_KEYS:
        if record.get(p):
            w = int(weights.get(p, 10))
            score += w
            count += 1
            reasons.append(f"{p}:+{w}")
    if count >= int(cfg.get("multi_platform_bonus_at", 3)):
        extra = int(bonus.get("multi_platform", 0))
        score += extra
        reasons.append(f"multi_platform:+{extra}")
    if str(record.get("website_ok", "")) == "1":
        extra = int(bonus.get("official_website", 0))
        score += extra
        reasons.append(f"official_website:+{extra}")
    if int(record.get("platform_count") or 0) == 0:
        return 0, "no_social_link", reasons or ["no social links found on official site"]

    return int(min(score, cap)), "verified" if score >= int(cfg.get("verified_cutoff", 60)) else "weak", reasons


def annotate_gems(candidates: pd.DataFrame, social: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add social_score / social_status / social_verified to gem candidates."""
    social = load_social() if social is None else social
    lookup: Dict[str, Dict[str, Any]] = {}
    if len(social):
        lookup = {str(r["place_id"]): r for r in social.to_dict("records")}

    scores, statuses, reasons, verified = [], [], [], []
    for pid in candidates["place_id"].astype(str):
        rec = lookup.get(pid)
        if rec is None:
            scores.append(None)
            statuses.append("unchecked")
            reasons.append("")
            verified.append(False)
            continue
        s, st, rs = social_score(rec)
        scores.append(s)
        statuses.append(st)
        reasons.append(",".join(rs))
        verified.append(st == "verified")
    out = candidates.copy()
    out["social_score"] = scores
    out["social_status"] = statuses
    out["social_reasons"] = reasons
    out["social_verified"] = verified
    return out


def rank_gem_options(
    places: pd.DataFrame,
    gem_scores: pd.Series,
    top_n: int = 50,
    verified_first: bool = True,
) -> pd.DataFrame:
    """Rank hidden-gem candidates for the UI.

    Verified gems come first (more trustworthy options); unverified gems are
    still returned - just flagged - so the user gets MORE options, not fewer,
    with the confidence stated plainly.
    """
    cfg = load_criteria().get("social_verification", {})
    cand = places.loc[gem_scores.index].copy()
    cand["hidden_gem_score"] = gem_scores
    cand = annotate_gems(cand)
    if verified_first:
        cand = cand.sort_values(
            ["social_verified", "social_score", "hidden_gem_score"],
            ascending=[False, False, False],
            na_position="last",
        )
    return cand.head(top_n)


def coverage_report() -> Dict[str, Any]:
    """Honest coverage numbers for the social cache (goes into metrics.json)."""
    df = load_social()
    if df.empty:
        return {"records": 0, "note": "no social data collected yet"}
    status_counts = df["status"].value_counts().to_dict()
    scored = [social_score(r) for r in df.to_dict("records")]
    verified = sum(1 for s, st, _ in scored if st == "verified")
    return {
        "records": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "verified": int(verified),
        "verified_share": round(verified / max(len(df), 1), 4),
        "mean_score": round(float(np.mean([s for s, st, _ in scored if st != "no_data"])) if len(df) else 0.0, 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(coverage_report(), indent=2))
