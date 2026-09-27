"""Solo travel matchmaking.

Flow
  1. A user adds a trip (destination + dates) and switches on "solo match".
  2. Candidates = other solo-match trips to the SAME destination (within 25 km) whose dates
     overlap by >= 1 day, where BOTH people fit each other's companion settings (gender,
     age range) and neither has blocked the other.
  3. Every candidate is shown (nobody is hidden by a low score) as a profile card:
     first name, age band, travel preferences, overlapping dates, compatibility + reasons.
     No surname, phone, email or exact trip dates are shared at this stage.
  4. A sends a request; only when B accepts does a match exist, and only a match can chat.
  5. Either side can block (closes the match and hides both ways) or report.
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import date
from typing import Dict, List, Optional

from .config import MIN_OVERLAP_DAYS, SAME_DESTINATION_KM
from .config import UPLOAD_DIR
from .preferences import compatibility


def km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def overlap(a_start: str, a_end: str, b_start: str, b_end: str) -> Optional[tuple]:
    s, e = max(date.fromisoformat(a_start), date.fromisoformat(b_start)), \
        min(date.fromisoformat(a_end), date.fromisoformat(b_end))
    days = (e - s).days + 1
    return (s.isoformat(), e.isoformat(), days) if days >= MIN_OVERLAP_DAYS else None


def age_band(age: Optional[int]) -> str:
    if not age:
        return "not shared"
    lo = (age // 5) * 5
    return f"{lo}-{lo + 4}"


def avatar_url(uid: str) -> Optional[str]:
    if next(iter(UPLOAD_DIR.glob(f"{uid}_avatar.*")), None):
        return f"/api/v2/avatars/{uid}"
    return None


def _prefs(con, uid) -> Optional[Dict]:
    r = con.execute("SELECT * FROM preferences WHERE user_id=?", (uid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    try:
        d["travel_style"] = json.loads(d.get("travel_styles") or "[]")
    except (TypeError, ValueError):
        d["travel_style"] = []
    if not d["travel_style"]:
        legacy = dict(r).get("travel_style")
        d["travel_style"] = [legacy] if legacy else []
    d.pop("travel_styles", None)
    d["languages"] = json.loads(d["languages"])
    return d


def _fits(me: Dict, my_p: Dict, them: Dict) -> bool:
    """Does `them` satisfy `me`'s companion settings?"""
    if my_p["companion_gender"] == "same" and (not me["gender"] or them["gender"] != me["gender"]):
        return False
    if them["age"] is None:
        return False
    return my_p["companion_age_min"] <= them["age"] <= my_p["companion_age_max"]


def blocked_between(con, a: str, b: str) -> bool:
    return con.execute("SELECT 1 FROM blocks WHERE (blocker=? AND blocked=?) OR (blocker=? AND blocked=?)",
                       (a, b, b, a)).fetchone() is not None


def candidates(con, user_id: str, trip_id: str) -> Dict:
    trip = con.execute("SELECT * FROM trips WHERE id=? AND user_id=?", (trip_id, user_id)).fetchone()
    if not trip:
        raise LookupError("trip not found")
    if not trip["solo_match"]:
        return {"trip": dict(trip), "candidates": [], "note": "Switch on solo matchmaking for this trip first"}
    me = dict(con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())
    my_p = _prefs(con, user_id)
    if not my_p:
        raise ValueError("complete your 5 preferences first")
    rows = con.execute(
        """SELECT t.*, u.name, u.age, u.gender FROM trips t JOIN users u ON u.id=t.user_id
           WHERE t.solo_match=1 AND t.user_id<>? AND u.is_active=1 AND u.role='user'
             AND t.end_date>=? AND t.start_date<=?""",
        (user_id, trip["start_date"], trip["end_date"])).fetchall()
    out = []
    for r in rows:
        d = km(trip["dest_lat"], trip["dest_lon"], r["dest_lat"], r["dest_lon"])
        ov = overlap(trip["start_date"], trip["end_date"], r["start_date"], r["end_date"])
        if d > SAME_DESTINATION_KM or not ov or blocked_between(con, user_id, r["user_id"]):
            continue
        them = {"gender": r["gender"], "age": r["age"]}
        their_p = _prefs(con, r["user_id"])
        if not their_p or not _fits(me, my_p, them) or not _fits(them, their_p, me):
            continue
        score, reasons = compatibility(my_p, their_p)
        req = con.execute(
            """SELECT id, from_user, status FROM match_requests WHERE
               ((from_user=? AND to_user=?) OR (from_user=? AND to_user=?)) AND
               ((from_trip=? AND to_trip=?) OR (from_trip=? AND to_trip=?)) ORDER BY created_at DESC""",
            (user_id, r["user_id"], r["user_id"], user_id, trip_id, r["id"], r["id"], trip_id)).fetchone()
        state = "none"
        if req:
            state = req["status"] if req["status"] != "pending" else (
                "request_sent" if req["from_user"] == user_id else "request_received")
        out.append({
            "user_id": r["user_id"], "trip_id": r["id"], "first_name": r["name"].split()[0],
            "avatar_url": avatar_url(r["user_id"]),
            "age_band": age_band(r["age"]), "gender": r["gender"],
            "preferences": {k: their_p[k] for k in ("travel_style", "budget", "pace", "food", "languages")},
            "destination": r["destination"], "distance_between_destinations_km": round(d, 1),
            "overlap": {"from": ov[0], "to": ov[1], "days": ov[2]},
            "compatibility": score, "reasons": reasons,
            "request_state": state, "request_id": req["id"] if req else None,
        })
    out.sort(key=lambda c: (-c["compatibility"], -c["overlap"]["days"]))
    return {"trip": dict(trip), "candidates": out,
            "note": "All matching travellers are listed; the score only orders them. "
                    "Compatibility is a transparent rule over the 5 preferences, not a trained model."}


def send_request(con, user_id: str, my_trip: str, to_user: str, their_trip: str) -> Dict:
    ids = [c["trip_id"] for c in candidates(con, user_id, my_trip)["candidates"] if c["user_id"] == to_user]
    if their_trip not in ids:
        raise PermissionError("that traveller is not a candidate for this trip")
    # if they already asked us, sending back = accepting
    back = con.execute("SELECT id FROM match_requests WHERE from_user=? AND to_user=? AND from_trip=? "
                       "AND to_trip=? AND status='pending'", (to_user, user_id, their_trip, my_trip)).fetchone()
    if back:
        return respond(con, user_id, back["id"], True)
    prev = con.execute("SELECT status FROM match_requests WHERE from_user=? AND to_user=? AND from_trip=? AND to_trip=?",
                       (user_id, to_user, my_trip, their_trip)).fetchone()
    if prev and prev["status"] in ("declined", "cancelled"):
        raise PermissionError("This traveller declined your request for this trip")
    rid = str(uuid.uuid4())
    con.execute("INSERT INTO match_requests (id, from_user, to_user, from_trip, to_trip) VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                (rid, user_id, to_user, my_trip, their_trip))
    r = con.execute("SELECT * FROM match_requests WHERE from_user=? AND to_user=? AND from_trip=? AND to_trip=?",
                    (user_id, to_user, my_trip, their_trip)).fetchone()
    return {"request_id": r["id"], "status": r["status"]}


def respond(con, user_id: str, request_id: str, accept: bool) -> Dict:
    r = con.execute("SELECT * FROM match_requests WHERE id=?", (request_id,)).fetchone()
    if not r or r["to_user"] != user_id:
        raise PermissionError("not your request")
    if r["status"] != "pending":
        return {"request_id": request_id, "status": r["status"]}
    if blocked_between(con, r["from_user"], r["to_user"]):
        raise PermissionError("blocked")
    status = "accepted" if accept else "declined"
    con.execute("UPDATE match_requests SET status=?, responded_at=CURRENT_TIMESTAMP WHERE id=?", (status, request_id))
    out = {"request_id": request_id, "status": status}
    if accept:
        existing = con.execute("SELECT id FROM matches WHERE status='active' AND ((user_a=? AND user_b=?) OR "
                               "(user_a=? AND user_b=?)) AND ((trip_a=? AND trip_b=?) OR (trip_a=? AND trip_b=?))",
                               (r["from_user"], r["to_user"], r["to_user"], r["from_user"],
                                r["from_trip"], r["to_trip"], r["to_trip"], r["from_trip"])).fetchone()
        mid = existing["id"] if existing else str(uuid.uuid4())
        if not existing:
            con.execute("INSERT INTO matches (id, user_a, user_b, trip_a, trip_b) VALUES (?,?,?,?,?)",
                        (mid, r["from_user"], r["to_user"], r["from_trip"], r["to_trip"]))
        out["match_id"] = mid
    return out


def list_requests(con, user_id: str) -> Dict:
    def rows(sql):
        return [dict(x) for x in con.execute(sql, (user_id,)).fetchall()]
    inc = rows("""SELECT r.id, r.status, r.created_at, r.from_trip, r.to_trip, u.name AS from_name, t.destination
                  FROM match_requests r JOIN users u ON u.id=r.from_user JOIN trips t ON t.id=r.to_trip
                  WHERE r.to_user=? ORDER BY r.created_at DESC""")
    out = rows("""SELECT r.id, r.status, r.created_at, r.from_trip, r.to_trip, u.name AS to_name, t.destination
                  FROM match_requests r JOIN users u ON u.id=r.to_user JOIN trips t ON t.id=r.from_trip
                  WHERE r.from_user=? ORDER BY r.created_at DESC""")
    for x in inc:
        x["from_name"] = x["from_name"].split()[0]
    for x in out:
        x["to_name"] = x["to_name"].split()[0]
    return {"incoming": inc, "outgoing": out}


def my_matches(con, user_id: str) -> List[Dict]:
    rs = con.execute("""SELECT m.*, ua.name AS name_a, ub.name AS name_b, ta.destination AS dest,
                               ta.start_date AS a_s, ta.end_date AS a_e, tb.start_date AS b_s, tb.end_date AS b_e
                        FROM matches m JOIN users ua ON ua.id=m.user_a JOIN users ub ON ub.id=m.user_b
                        JOIN trips ta ON ta.id=m.trip_a JOIN trips tb ON tb.id=m.trip_b
                        WHERE (m.user_a=? OR m.user_b=?) AND m.status='active' ORDER BY m.created_at DESC""",
                     (user_id, user_id)).fetchall()
    out = []
    for m in rs:
        other_is_b = m["user_a"] == user_id
        ov = overlap(m["a_s"], m["a_e"], m["b_s"], m["b_e"])
        last = con.execute("SELECT body, created_at, sender_id FROM messages WHERE match_id=? ORDER BY id DESC LIMIT 1",
                           (m["id"],)).fetchone()
        out.append({"match_id": m["id"], "with_user": m["user_b"] if other_is_b else m["user_a"],
                    "with_name": (m["name_b"] if other_is_b else m["name_a"]).split()[0],
                    "avatar_url": avatar_url(m["user_b"] if other_is_b else m["user_a"]),
                    "destination": m["dest"], "overlap": {"from": ov[0], "to": ov[1], "days": ov[2]} if ov else None,
                    "last_message": dict(last) if last else None, "since": m["created_at"]})
    return out


def is_participant(con, match_id: str, user_id: str) -> bool:
    m = con.execute("SELECT user_a, user_b, status FROM matches WHERE id=?", (match_id,)).fetchone()
    return bool(m) and m["status"] == "active" and user_id in (m["user_a"], m["user_b"])


def block(con, user_id: str, other: str) -> None:
    con.execute("INSERT INTO blocks (blocker, blocked) VALUES (?,?) ON CONFLICT DO NOTHING", (user_id, other))
    con.execute("UPDATE matches SET status='closed' WHERE (user_a=? AND user_b=?) OR (user_a=? AND user_b=?)",
                (user_id, other, other, user_id))
    con.execute("UPDATE match_requests SET status='cancelled' WHERE status='pending' AND "
                "((from_user=? AND to_user=?) OR (from_user=? AND to_user=?))", (user_id, other, other, user_id))
