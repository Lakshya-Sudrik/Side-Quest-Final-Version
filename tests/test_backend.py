"""Backend API: accounts, 5 preferences, collaborators, solo matchmaking, chat."""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp()
os.environ["SIDEQUEST_DB"] = os.path.join(_tmp, "test.db")
os.environ["SIDEQUEST_UPLOADS"] = os.path.join(_tmp, "uploads")
os.environ["SIDEQUEST_ADMIN_EMAILS"] = "admin@example.com"
os.environ["SIDEQUEST_JWT_SECRET"] = "test-secret"

from fastapi.testclient import TestClient  # noqa: E402

from backend import db, gstin, preferences  # noqa: E402
from backend.app import app  # noqa: E402

db.meta.drop_all(db.engine())    # fresh tables on every run (SQLite file or PostgreSQL)
db.init()
client = TestClient(app)
D = lambda n: (date.today() + timedelta(days=n)).isoformat()  # noqa: E731

PREFS = {"travel_style": "nature_wildlife", "budget": "mid_range", "pace": "balanced", "food": "vegetarian",
         "languages": ["english", "kannada"]}


def register(email, role="user", **kw):
    body = {"role": role, "email": email, "password": "travel123", "name": kw.pop("name", "Asha Rao")}
    if role == "user":
        body.update(age=kw.pop("age", 26), gender=kw.pop("gender", "female"), preferences={**PREFS, **kw.pop("prefs", {})})
    body.update(kw)
    r = client.post("/api/v2/auth/register", json=body)
    return r


def auth(tok):
    return {"Authorization": f"Bearer {tok}"}


# --- accounts ---------------------------------------------------------------
def test_register_login_and_role_tab():
    r = register("asha@example.com")
    assert r.status_code == 200, r.text
    assert r.json()["user"]["preferences"]["travel_style"] == "nature_wildlife"
    assert client.post("/api/v2/auth/login", json={"email": "asha@example.com", "password": "travel123"}).status_code == 200
    assert client.post("/api/v2/auth/login", json={"email": "asha@example.com", "password": "wrong999"}).status_code == 401
    # logging in on the collaborator tab with a traveller account is refused
    r = client.post("/api/v2/auth/login", json={"email": "asha@example.com", "password": "travel123", "role": "collaborator"})
    assert r.status_code == 403


def test_register_rules():
    assert register("young@example.com", age=16).status_code == 422                       # 18+
    assert register("weak@example.com", password="short").status_code == 422
    assert register("badpref@example.com", prefs={"budget": "luxury"}).status_code == 422   # unknown option
    assert register("asha@example.com").status_code in (409, 200)                         # duplicate email
    assert register("asha@example.com").status_code == 409


def test_password_is_hashed():
    with db.session() as con:
        h = con.execute("SELECT password_hash FROM users WHERE email='asha@example.com'").fetchone()["password_hash"]
    assert h.startswith("scrypt$") and "travel123" not in h


def test_protected_routes_need_login():
    assert client.get("/api/v2/me").status_code == 401
    assert client.get("/api/v2/me", headers=auth("garbage")).status_code == 401


# --- GSTIN ------------------------------------------------------------------
def test_gstin_checksum():
    body = "29AAGCB7383J1Z"            # Karnataka + PAN + entity + Z
    good = body + gstin.check_digit(body)
    assert gstin.validate(good)["status"] == "valid_format"
    assert gstin.validate(good)["state"] == "Karnataka"
    bad_digit = good[:-1] + ("A" if good[-1] != "A" else "B")
    assert gstin.validate(bad_digit)["status"] == "invalid"
    assert gstin.validate("99AAGCB7383J1Z5")["status"] == "invalid"          # unknown state
    assert gstin.validate("hello")["status"] == "invalid"
    assert gstin.validate("")["status"] == "not_provided"


# --- collaborators ------------------------------------------------------------
def test_collaborator_flow_and_admin_approval():
    body = "32ABCDE1234F1Z"
    r = register("host@example.com", role="collaborator", name="Hill Homestay",
                 collaborator={"business_name": "Misty Hills Homestay", "business_type": "homestay",
                               "gstin": body + gstin.check_digit(body)})
    assert r.status_code == 200, r.text
    assert r.json()["gstin_check"]["status"] == "valid_format"
    host = r.json()["token"]
    # travellers-only endpoints are closed to collaborators
    assert client.get("/api/v2/trips", headers=auth(host)).status_code == 403
    # document upload: type is checked against file content
    fake = client.post("/api/v2/collaborator/documents", headers=auth(host), data={"doc_type": "gst_certificate"},
                       files={"file": ("cert.pdf", b"not really a pdf", "application/pdf")})
    assert fake.status_code == 415
    ok = client.post("/api/v2/collaborator/documents", headers=auth(host), data={"doc_type": "gst_certificate"},
                     files={"file": ("cert.pdf", b"%PDF-1.4 test", "application/pdf")})
    assert ok.status_code == 200 and ok.json()["status"] == "pending"
    # only an admin can verify
    uid = client.get("/api/v2/me", headers=auth(host)).json()["id"]
    assert client.post(f"/api/v2/admin/collaborators/{uid}/verify", json={"approve": True},
                       headers=auth(host)).status_code == 403
    admin = register("admin@example.com", role="collaborator", name="Admin",
                     collaborator={"business_name": "SideQuest", "business_type": "admin"}).json()["token"]
    r = client.post(f"/api/v2/admin/collaborators/{uid}/verify", json={"approve": True, "notes": "cert ok"},
                    headers=auth(admin))
    assert r.json()["verification_status"] == "verified"
    assert client.get("/api/v2/collaborator/status", headers=auth(host)).json()["collaborator"][
        "verification_status"] == "verified"


# --- solo matchmaking -------------------------------------------------------------
def _trip(tok, dest="Munnar", s=10, e=14, solo=True):
    r = client.post("/api/v2/trips", headers=auth(tok),
                    json={"destination": dest, "start_date": D(s), "end_date": D(e), "solo_match": solo})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_solo_match_request_accept_chat_block():
    a = register("meera@example.com", name="Meera Nair").json()["token"]
    b = register("divya@example.com", name="Divya S", age=29, prefs={"budget": "budget"}).json()["token"]
    c = register("ravi@example.com", name="Ravi K", gender="male",
                 prefs={}).json()["token"]
    far = register("zoya@example.com", name="Zoya", age=27).json()["token"]
    late = register("nina@example.com", name="Nina", age=31).json()["token"]
    ta = _trip(a, "Munnar", 10, 14)
    tb = _trip(b, "Munnar", 12, 18)                 # overlaps 3 days
    _trip(c, "Munnar", 10, 14)
    _trip(far, "Jaipur", 10, 14)                    # different destination
    _trip(late, "Munnar", 30, 33)                   # no overlap
    # a only wants same-gender companions -> Ravi is excluded
    client.put("/api/v2/me/preferences", headers=auth(a), json={**PREFS, "companion_gender": "same"})
    cands = client.get(f"/api/v2/solo/candidates?trip_id={ta}", headers=auth(a)).json()["candidates"]
    names = [x["first_name"] for x in cands]
    assert names == ["Divya"], names
    card = cands[0]
    assert card["overlap"]["days"] == 3 and card["age_band"] == "25-29"
    assert "email" not in card and "phone" not in card and 0 <= card["compatibility"] <= 100
    # chat is closed before a match
    req = client.post("/api/v2/solo/requests", headers=auth(a),
                      json={"trip_id": ta, "to_user": card["user_id"], "their_trip_id": tb}).json()
    assert req["status"] == "pending"
    assert client.get("/api/v2/matches", headers=auth(a)).json() == []
    # b sees the request and accepts -> match -> chat opens for both, nobody else
    inc = client.get("/api/v2/solo/requests", headers=auth(b)).json()["incoming"]
    assert inc[0]["from_name"] == "Meera"
    res = client.post(f"/api/v2/solo/requests/{inc[0]['id']}/respond", headers=auth(b), json={"accept": True}).json()
    mid = res["match_id"]
    assert client.post(f"/api/v2/matches/{mid}/messages", headers=auth(a), json={"body": "Hi! Sunrise trek on day 2?"}).status_code == 200
    msgs = client.get(f"/api/v2/matches/{mid}/messages", headers=auth(b)).json()
    assert msgs[0]["body"].startswith("Hi!") and msgs[0]["mine"] is False
    assert client.get(f"/api/v2/matches/{mid}/messages", headers=auth(c)).status_code == 403
    # a stranger can't respond to someone else's request
    assert client.post(f"/api/v2/solo/requests/{inc[0]['id']}/respond", headers=auth(c),
                       json={"accept": True}).status_code == 403
    # block closes the chat and hides each other
    uid_b = client.get("/api/v2/me", headers=auth(b)).json()["id"]
    client.post("/api/v2/block", headers=auth(a), json={"user_id": uid_b})
    assert client.post(f"/api/v2/matches/{mid}/messages", headers=auth(a), json={"body": "hello?"}).status_code == 403
    assert client.get(f"/api/v2/solo/candidates?trip_id={ta}", headers=auth(a)).json()["candidates"] == []


def test_solo_toggle_off_hides_trip():
    a = register("k1@example.com", name="Kavya").json()["token"]
    b = register("k2@example.com", name="Leela").json()["token"]
    ta = _trip(a, "Hampi", 5, 7)
    tb = _trip(b, "Hampi", 6, 8)
    assert len(client.get(f"/api/v2/solo/candidates?trip_id={ta}", headers=auth(a)).json()["candidates"]) == 1
    client.patch(f"/api/v2/trips/{tb}", headers=auth(b), json={"solo_match": False})
    assert client.get(f"/api/v2/solo/candidates?trip_id={ta}", headers=auth(a)).json()["candidates"] == []


def test_trip_validation():
    a = register("t1@example.com", name="Tara").json()["token"]
    bad = client.post("/api/v2/trips", headers=auth(a), json={"destination": "Munnar", "start_date": D(5),
                                                              "end_date": D(2), "solo_match": True})
    assert bad.status_code == 422
    unknown = client.post("/api/v2/trips", headers=auth(a), json={"destination": "Qwertyville", "start_date": D(5),
                                                                  "end_date": D(6)})
    assert unknown.status_code == 422


def test_compatibility_is_symmetric_and_explained():
    a = {**PREFS}
    b = {**PREFS, "budget": "premium", "food": "non_vegetarian", "languages": ["hindi"]}
    s1, r1 = preferences.compatibility(a, b)
    s2, _ = preferences.compatibility(b, a)
    assert s1 == s2 and s1 < preferences.compatibility(a, a)[0] == 100
    assert any("language" in x.lower() for x in r1)


# --- model-backed endpoints (skipped when the scored data / index are not built) ------
_have_data = (ROOT / "data" / "processed" / "food_scored.csv.gz").exists()
_have_index = (ROOT / "data" / "chroma_db").exists()


@pytest.mark.skipif(not _have_data, reason="scored tables not built")
def test_listing_is_scored_and_published_only_when_verified():
    body = "29AAGCB7383J1Z"
    tok = register("cafe@example.com", role="collaborator", name="Cafe Owner",
                   collaborator={"business_name": "Wild Thyme", "business_type": "restaurant",
                                 "gstin": body + gstin.check_digit(body)}).json()["token"]
    listing = {"name": "Wild Thyme Restaurant", "kind": "eat", "lat": 12.93484, "lon": 77.61898,
               "town": "Bangalore", "category": "Chinese"}
    r = client.post("/api/v2/listings", headers=auth(tok), json=listing).json()
    assert r["decision"] == "hidden_gem" and r["evidence"] == "existing_place"
    assert r["published"] is False and "verification" in r["publish_note"]      # not verified yet
    assert r["safety"]["class"] in ("safe", "caution", "unknown", "avoid")
    sight = client.post("/api/v2/listings", headers=auth(tok),
                        json={"name": "Secret Cardamom Falls", "kind": "visit", "town": "Munnar"}).json()
    assert sight["decision"] == "needs_reviews"                                   # no reviews -> no claim
    # admin approval publishes the gem listing and it reaches the route recommender
    admin = client.post("/api/v2/auth/login", json={"email": "admin@example.com", "password": "travel123"}).json()["token"]
    uid = client.get("/api/v2/me", headers=auth(tok)).json()["id"]
    client.post(f"/api/v2/admin/collaborators/{uid}/verify", json={"approve": True}, headers=auth(admin))
    listings = client.get("/api/v2/collaborator/status", headers=auth(tok)).json()["listings"]
    assert any(l["published"] for l in listings if l["decision"] == "hidden_gem")
    import recommender
    assert "Wild Thyme Restaurant" in set(recommender.listing_candidates()["name"])


@pytest.mark.skipif(not _have_index, reason="RAG index not built")
def test_rag_search_endpoint():
    r = client.post("/api/v2/rag/search", json={"query": "quiet waterfall near Munnar", "top_k": 5}).json()
    assert r["city_filter"] == "munnar" and len(r["results"]) == 5
    assert all(x["city_key"] == "munnar" for x in r["results"])
    assert all("safety" in x for x in r["results"]) and r["answer"].startswith("Top matches")
    names = [x["name"] for x in r["results"]]
    assert len(set(n.lower() for n in names)) == len(names)                        # no duplicate branches


@pytest.mark.skipif(not _have_data, reason="scored tables not built")
def test_trip_planner_add_remove_and_flight_delay():
    tok = register("planner@example.com", name="Ira Planner").json()["token"]
    day = D(20)
    it = client.post("/api/v2/itineraries", headers=auth(tok),
                     json={"origin": "Mumbai", "destination": "Pune", "depart_at": f"{day}T08:00",
                           "arrive_by": f"{day}T16:00"}).json()
    iid = it["id"]
    assert it["schedule"]["fits"] and it["schedule"]["free_time_min"] > 120
    sug = client.get(f"/api/v2/itineraries/{iid}/suggestions", headers=auth(tok)).json()["suggestions"]
    assert sug, "expected gems that fit the free time"
    for s in sug[:3]:
        r = client.post(f"/api/v2/itineraries/{iid}/stops", headers=auth(tok), json={"place_id": s["place_id"]})
        assert r.status_code == 200, r.text
    plan = client.get(f"/api/v2/itineraries/{iid}", headers=auth(tok)).json()
    stops = plan["schedule"]["stops"]
    assert len(stops) == 3
    assert [x["along_km"] for x in stops] == sorted(x["along_km"] for x in stops)     # in route order
    assert all(x["arrive"] < x["leave"] for x in stops)
    # remove one
    first = stops[0]["place_id"]
    r = client.delete(f"/api/v2/itineraries/{iid}/stops/{first}", headers=auth(tok))
    assert r.status_code == 200 and len(r.json()["schedule"]["stops"]) == 2
    # flight delayed by 3 hours: free time shrinks, plan must still end on time
    d = client.post(f"/api/v2/itineraries/{iid}/delay", headers=auth(tok), json={"minutes": 180}).json()
    assert d["delay_min"] == 180 and d["schedule"]["start"].endswith("11:00")
    assert d["schedule"]["fits"]
    assert len(d["kept"]) + len(d["dropped"]) == 2
    for s in d["suggested_replacements"]:
        assert s["detour_min"] + s["visit_min"] <= d["schedule"]["slack_min"]
    # someone else's plan is invisible
    other = register("nosy@example.com", name="Nosy Parker").json()["token"]
    assert client.get(f"/api/v2/itineraries/{iid}", headers=auth(other)).status_code == 404


def test_planner_rejects_impossible_windows():
    tok = register("planner2@example.com", name="Om Planner").json()["token"]
    day = D(21)
    r = client.post("/api/v2/itineraries", headers=auth(tok), json={"origin": "Mumbai", "destination": "Pune",
                    "depart_at": f"{day}T08:00", "arrive_by": f"{day}T07:00"})
    assert r.status_code == 422
    r = client.post("/api/v2/itineraries", headers=auth(tok), json={"origin": "Mumbai", "destination": "Pune",
                    "depart_at": f"{day}T08:00", "arrive_by": f"{day}T09:00"})
    assert r.status_code == 422 and "drive alone" in r.json()["detail"]


def test_declined_request_cannot_be_resent():
    a = register("dec1@example.com", name="Anu").json()["token"]
    b = register("dec2@example.com", name="Bela").json()["token"]
    ta, tb = _trip(a, "Coorg", 40, 43), _trip(b, "Coorg", 41, 44)
    ub = client.get("/api/v2/me", headers=auth(b)).json()["id"]
    client.post("/api/v2/solo/requests", headers=auth(a), json={"trip_id": ta, "to_user": ub, "their_trip_id": tb})
    rid = client.get("/api/v2/solo/requests", headers=auth(b)).json()["incoming"][0]["id"]
    client.post(f"/api/v2/solo/requests/{rid}/respond", headers=auth(b), json={"accept": False})
    again = client.post("/api/v2/solo/requests", headers=auth(a), json={"trip_id": ta, "to_user": ub, "their_trip_id": tb})
    assert again.status_code == 403


@pytest.mark.skipif(not _have_data, reason="scored tables not built")
def test_city_gems_follow_recommender_rules():
    import recommender as rec
    r = client.get("/api/v2/gems/city?city=Munnar&limit=6").json()
    assert r["success"] and 0 < r["count"] <= 6
    for g in r["gems"]:
        assert g["city"].lower().startswith("munnar")
        assert g["hidden_gem_score"] >= rec.GEM_RANK_MIN and g["exposure_pct"] <= rec.HIDDEN_MAX_EXPOSURE
        assert g["safety_class"] in ("safe", "caution", "unknown", "avoid")
    india = client.get("/api/v2/gems/city?limit=10").json()["gems"]
    assert len({g["city"] for g in india}) == len(india)          # one per town nationally
