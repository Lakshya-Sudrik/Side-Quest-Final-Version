"""SideQuest backend (FastAPI). Run: uvicorn backend.app:app --port 8000

The Node/Express server serves the web app and forwards /api/v2/* here. Models and
data tables are loaded ONCE at start-up, so a route request takes ~1 s instead of
starting Python for every call.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field, field_validator

from . import db, gstin, matchmaking, preferences
from .config import (ADMIN_EMAILS, ALLOWED_DOC_TYPES, MAX_UPLOAD_MB, MESSAGE_MAX_CHARS, MESSAGES_PER_MINUTE,
                     ROOT, UPLOAD_DIR)
from .security import hash_password, make_token, password_problem, read_token, verify_password

sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger("backend")
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    db.init()
    try:  # warm the recommender tables so the first user doesn't wait
        import recommender
        recommender.load_candidates()
    except Exception as exc:  # data not built yet - endpoints report it
        log.warning("recommender not warmed: %s", exc)
    yield


app = FastAPI(title="SideQuest backend", version="2.0", lifespan=lifespan)


def _avatar_path(uid: str):
    return next(iter(UPLOAD_DIR.glob(f"{uid}_avatar.*")), None)


def _avatar_url(uid: str):
    path = _avatar_path(uid)
    return f"/api/v2/avatars/{uid}?v={path.stat().st_mtime_ns}" if path else None


# ---------------------------------------------------------------------------
# auth helpers
# ---------------------------------------------------------------------------
def current_user(authorization: str = Header(default="")) -> Dict:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Log in first")
    try:
        claims = read_token(authorization.split(" ", 1)[1])
    except Exception:
        raise HTTPException(401, "Session expired - log in again")
    with db.session() as con:
        u = con.execute("SELECT id, email, role, name, phone, age, gender, is_active FROM users WHERE id=?",
                        (claims["sub"],)).fetchone()
    if not u or not u["is_active"]:
        raise HTTPException(401, "Account not found")
    return dict(u)


def require_role(*roles):
    def dep(user: Dict = Depends(current_user)) -> Dict:
        if user["role"] not in roles:
            raise HTTPException(403, f"Only for {' / '.join(roles)} accounts")
        return user
    return dep


# ---------------------------------------------------------------------------
# accounts
# ---------------------------------------------------------------------------
class Prefs(BaseModel):
    travel_style: List[str] | str
    budget: str
    pace: str
    food: str
    languages: List[str]
    companion_gender: str = "any"
    companion_age_min: int = 18
    companion_age_max: int = 99


class ProfileBasics(BaseModel):
    age: int = Field(ge=18, le=99)
    gender: str = Field(pattern="^(female|male|non_binary|prefer_not)$")


class CollaboratorInfo(BaseModel):
    business_name: str = Field(min_length=2, max_length=120)
    business_type: str = Field(min_length=2, max_length=60)   # homestay, restaurant, guide, cafe ...
    gstin: Optional[str] = None


class Register(BaseModel):
    role: str = Field(pattern="^(user|collaborator)$")
    email: EmailStr
    password: str
    name: str = Field(min_length=2, max_length=80)
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    preferences: Optional[Prefs] = None
    collaborator: Optional[CollaboratorInfo] = None

    @field_validator("phone")
    @classmethod
    def _phone(cls, v):
        if v and not re.fullmatch(r"\+?[0-9 -]{10,15}", v):
            raise ValueError("phone must be 10-15 digits")
        return v


def _profile(con, uid: str) -> Dict:
    u = dict(con.execute("SELECT id, email, role, name, phone, age, gender, created_at FROM users WHERE id=?",
                         (uid,)).fetchone())
    p = con.execute("SELECT * FROM preferences WHERE user_id=?", (uid,)).fetchone()
    if p:
        p = dict(p)
        try:
            styles = json.loads(p.get("travel_styles") or "[]")
        except (TypeError, ValueError):
            styles = []
        if not styles and p.get("travel_style"):
            legacy = p["travel_style"]
            try:
                decoded = json.loads(legacy)
                styles = decoded if isinstance(decoded, list) else [legacy]
            except (TypeError, ValueError):
                styles = [legacy]
        p["travel_style"] = styles
        p.pop("travel_styles", None)
        p["languages"] = json.loads(p["languages"])
        p.pop("user_id", None)
    u["preferences"] = p
    u["avatar_url"] = _avatar_url(uid)
    c = con.execute("SELECT * FROM collaborators WHERE user_id=?", (uid,)).fetchone()
    if c:
        c = dict(c)
        c.pop("document_path", None)
        u["collaborator"] = c
    return u


@app.get("/api/v2/preferences/options")
def pref_options():
    return {"questions": preferences.OPTIONS, "companion": preferences.COMPANION}


@app.post("/api/v2/auth/register")
def register(body: Register):
    if (msg := password_problem(body.password)):
        raise HTTPException(422, msg)
    email = body.email.lower()
    role = "admin" if email in ADMIN_EMAILS else body.role
    errors: List[str] = []
    prefs = None
    if role == "user":
        if body.age is not None and body.age < 18:
            errors.append("age: solo travel matching is for 18+ only")
        if body.gender is not None and body.gender not in ("female", "male", "non_binary", "prefer_not"):
            errors.append("gender: female | male | non_binary | prefer_not")
        if body.preferences:
            prefs, perr = preferences.validate(body.preferences.model_dump())
            errors += perr
    if role == "collaborator" and not body.collaborator:
        errors.append("collaborator: business name and type are required")
    if errors:
        raise HTTPException(422, errors)
    uid = str(uuid.uuid4())
    with db.session() as con:
        if con.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise HTTPException(409, "An account with this email already exists")
        con.execute("INSERT INTO users (id, email, password_hash, role, name, phone, age, gender) VALUES (?,?,?,?,?,?,?,?)",
                    (uid, email, hash_password(body.password), role, body.name.strip(), body.phone, body.age, body.gender))
        if prefs:
            con.execute("""INSERT INTO preferences (user_id, travel_style, travel_styles, budget, pace, food, languages,
                           companion_gender, companion_age_min, companion_age_max) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (uid, prefs["travel_style"][0], json.dumps(prefs["travel_style"]), prefs["budget"], prefs["pace"], prefs["food"],
                         json.dumps(prefs["languages"]), prefs["companion_gender"], prefs["companion_age_min"],
                         prefs["companion_age_max"]))
        gst = None
        if role == "collaborator":
            gst = gstin.validate(body.collaborator.gstin or "")
            con.execute("""INSERT INTO collaborators (user_id, business_name, business_type, gstin, gstin_check, gstin_state)
                           VALUES (?,?,?,?,?,?)""",
                        (uid, body.collaborator.business_name, body.collaborator.business_type,
                         gst.get("gstin") or (body.collaborator.gstin or None), gst["status"], gst.get("state")))
        prof = _profile(con, uid)
    out = {"token": make_token(uid, role), "user": prof}
    if gst:
        out["gstin_check"] = gst
        out["next_step"] = "Upload a government ID and location-stamped photos of your place. An administrator reviews the ID."
    return out


class Login(BaseModel):
    email: EmailStr
    password: str
    role: Optional[str] = None     # the tab the user chose; must match the account


_attempts: Dict[str, deque] = defaultdict(deque)


@app.post("/api/v2/auth/login")
def login(body: Login):
    email = body.email.lower()
    q = _attempts[email]
    now = time.time()
    while q and now - q[0] > 300:
        q.popleft()
    if len(q) >= 8:
        raise HTTPException(429, "Too many attempts - try again in 5 minutes")
    with db.session() as con:
        u = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not u or not verify_password(body.password, u["password_hash"]):
            q.append(now)
            raise HTTPException(401, "Wrong email or password")
        if body.role and body.role != u["role"] and u["role"] != "admin" and email not in ADMIN_EMAILS:
            raise HTTPException(403, f"This is a {u['role']} account - use the {u['role']} login")
        if email in ADMIN_EMAILS and u["role"] != "admin":
            con.execute("UPDATE users SET role='admin' WHERE id=?", (u["id"],))
            u["role"] = "admin"
        prof = _profile(con, u["id"])
    q.clear()
    return {"token": make_token(u["id"], u["role"]), "user": prof}


@app.get("/api/v2/me")
def me(user: Dict = Depends(current_user)):
    with db.session() as con:
        return _profile(con, user["id"])


@app.put("/api/v2/me/profile")
def update_profile(body: ProfileBasics, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        con.execute("UPDATE users SET age=?, gender=? WHERE id=?",
                    (body.age, body.gender, user["id"]))
        return _profile(con, user["id"])


@app.post("/api/v2/me/avatar")
async def upload_avatar(file: UploadFile = File(...), user: Dict = Depends(current_user)):
    allowed = {
        "image/jpeg": (".jpg", b"\xff\xd8"),
        "image/png": (".png", b"\x89PNG\r\n\x1a\n"),
        "image/webp": (".webp", b"RIFF"),
    }
    image_type = allowed.get(file.content_type or "")
    if not image_type:
        raise HTTPException(415, "Choose a JPG, PNG or WebP image")
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image must be smaller than {MAX_UPLOAD_MB} MB")
    ext, signature = image_type
    if not data.startswith(signature) or (ext == ".webp" and data[8:12] != b"WEBP"):
        raise HTTPException(415, "Image content does not match its file type")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for old in UPLOAD_DIR.glob(f"{user['id']}_avatar.*"):
        old.unlink(missing_ok=True)
    (UPLOAD_DIR / f"{user['id']}_avatar{ext}").write_bytes(data)
    return {"avatar_url": _avatar_url(user["id"])}


@app.get("/api/v2/avatars/{uid}")
def get_avatar(uid: str):
    try:
        uid = str(uuid.UUID(uid))
    except ValueError:
        raise HTTPException(404, "Profile image not found")
    path = _avatar_path(uid)
    if not path:
        raise HTTPException(404, "Profile image not found")
    media_type = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix)
    if not media_type:
        raise HTTPException(404, "Profile image not found")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


@app.put("/api/v2/me/preferences")
def update_prefs(body: Prefs, user: Dict = Depends(require_role("user"))):
    prefs, errs = preferences.validate(body.model_dump())
    if errs:
        raise HTTPException(422, errs)
    with db.session() as con:
        con.execute("""INSERT INTO preferences (user_id, travel_style, travel_styles, budget, pace, food, languages, companion_gender,
                       companion_age_min, companion_age_max) VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(user_id) DO UPDATE SET travel_style=excluded.travel_style, travel_styles=excluded.travel_styles, budget=excluded.budget,
                       pace=excluded.pace, food=excluded.food, languages=excluded.languages,
                       companion_gender=excluded.companion_gender, companion_age_min=excluded.companion_age_min,
                       companion_age_max=excluded.companion_age_max, updated_at=CURRENT_TIMESTAMP""",
                    (user["id"], prefs["travel_style"][0], json.dumps(prefs["travel_style"]), prefs["budget"], prefs["pace"], prefs["food"],
                     json.dumps(prefs["languages"]), prefs["companion_gender"], prefs["companion_age_min"],
                     prefs["companion_age_max"]))
        return _profile(con, user["id"])


# ---------------------------------------------------------------------------
# collaborators
# ---------------------------------------------------------------------------
@app.post("/api/v2/collaborator/documents")
async def upload_document(doc_type: str = Form(..., pattern="^(govt_id|geotagged_photo)$"),
                          file: UploadFile = File(...), latitude: Optional[float] = Form(None), longitude: Optional[float] = Form(None),
                          user: Dict = Depends(require_role("collaborator"))):
    if doc_type == "geotagged_photo" and (latitude is None or longitude is None or not (6 <= latitude <= 37.5 and 68 <= longitude <= 97.5)):
        raise HTTPException(422, "Capture the photo location in India before uploading it")
    if doc_type == "govt_id" and (latitude is not None or longitude is not None):
        raise HTTPException(422, "Government ID uploads do not use a location")
    ext = ALLOWED_DOC_TYPES.get(file.content_type or "")
    if not ext:
        raise HTTPException(415, "Upload a PDF, JPG or PNG")
    if doc_type == "geotagged_photo" and ext not in (".jpg", ".png"):
        raise HTTPException(415, "Location evidence must be a JPG or PNG photo")
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    magic = {".pdf": b"%PDF", ".jpg": b"\xff\xd8", ".png": b"\x89PNG"}[ext]
    if not data.startswith(magic):
        raise HTTPException(415, "File content does not match its type")
    path = UPLOAD_DIR / f"{user['id']}_{doc_type}_{uuid.uuid4().hex[:8]}{ext}"
    path.write_bytes(data)
    with db.session() as con:
        con.execute("INSERT INTO collaborator_documents (id,user_id,doc_type,file_path,latitude,longitude) VALUES (?,?,?,?,?,?)",
                    (str(uuid.uuid4()), user["id"], doc_type, str(path), latitude, longitude))
        if doc_type == "govt_id":
            con.execute("UPDATE collaborators SET document_path=?, document_type=?, verification_status='pending' WHERE user_id=?",
                        (str(path), doc_type, user["id"]))
        return {"status": "pending", "message": "Upload received for review",
                "collaborator": _profile(con, user["id"])["collaborator"]}


@app.get("/api/v2/collaborator/status")
def collaborator_status(user: Dict = Depends(require_role("collaborator"))):
    with db.session() as con:
        prof = _profile(con, user["id"])
        listings = [dict(r) | {"evaluation": json.loads(r["evaluation"])} for r in con.execute(
            "SELECT * FROM listings WHERE collaborator_id=? ORDER BY created_at DESC", (user["id"],))]
        for item in listings:
            item["view_count"] = con.execute("SELECT COUNT(*) AS n FROM listing_views WHERE listing_id=?", (item["id"],)).fetchone()["n"]
            ratings = con.execute("SELECT COUNT(*) AS n, AVG(score) AS avg FROM listing_ratings WHERE listing_id=?", (item["id"],)).fetchone()
            item["rating_count"] = ratings["n"]
            item["average_rating"] = round(float(ratings["avg"]), 1) if ratings["avg"] is not None else None
        docs = [dict(r) for r in con.execute("SELECT id,doc_type,latitude,longitude,created_at FROM collaborator_documents WHERE user_id=? ORDER BY created_at DESC", (user["id"],))]
    return {"collaborator": prof.get("collaborator"), "listings": listings, "documents": docs}


class Verify(BaseModel):
    approve: bool
    notes: Optional[str] = None


@app.get("/api/v2/admin/collaborators")
def admin_list(user: Dict = Depends(require_role("admin"))):
    with db.session() as con:
        rows = con.execute("""SELECT c.*, u.email, u.name FROM collaborators c JOIN users u ON u.id=c.user_id
                              ORDER BY c.verification_status, u.created_at""").fetchall()
        docs = {r["user_id"]: [dict(d) | {"file_path": None} for d in con.execute(
            "SELECT id,doc_type,latitude,longitude,created_at FROM collaborator_documents WHERE user_id=? ORDER BY created_at DESC", (r["user_id"],)).fetchall()] for r in rows}
    return [dict(r) | {"has_document": bool(r["document_path"]), "documents": docs.get(r["user_id"], [])} for r in rows]


@app.post("/api/v2/admin/collaborators/{uid}/verify")
def admin_verify(uid: str, body: Verify, user: Dict = Depends(require_role("admin"))):
    with db.session() as con:
        c = con.execute("SELECT * FROM collaborators WHERE user_id=?", (uid,)).fetchone()
        if not c:
            raise HTTPException(404, "No such collaborator")
        if body.approve and c["document_type"] != "govt_id":
            raise HTTPException(422, "Review a government ID before approving this collaborator")
        status = "verified" if body.approve else "rejected"
        con.execute("UPDATE collaborators SET verification_status=?, reviewed_by=?, reviewed_at=CURRENT_TIMESTAMP, "
                    "review_notes=? WHERE user_id=?", (status, user["id"], body.notes, uid))
        # Publish only after identity verification, model/admin approval and place-photo upload.
        con.execute("UPDATE listings SET published=CASE WHEN ?=1 AND photo_path IS NOT NULL AND review_status IN ('auto_approved','admin_approved') THEN 1 ELSE 0 END WHERE collaborator_id=? AND decision='hidden_gem'",
                    (1 if body.approve else 0, uid))
    return {"user_id": uid, "verification_status": status}


class ListingIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    kind: str = Field(pattern="^(eat|visit)$")
    town: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    category: Optional[str] = Field(default=None, max_length=120)
    description: Optional[str] = Field(default=None, max_length=1500)
    cost_inr: Optional[float] = Field(default=None, ge=0, le=100000)


@app.post("/api/v2/listings")
def create_listing(body: ListingIn, user: Dict = Depends(require_role("collaborator"))):
    import geo_india as gi
    from .collaborators import evaluate
    if body.lat is not None and body.lon is not None:
        if not (6 <= body.lat <= 37.5 and 68 <= body.lon <= 97.5):
            raise HTTPException(422, "Location must be in India")
        lat, lon = body.lat, body.lon
    elif body.town:
        c = gi.town_coords(body.town)
        if not c:
            raise HTTPException(422, f"Unknown town '{body.town}'")
        lat, lon = c
    else:
        raise HTTPException(422, "Give a town or lat/lon")
    city = body.town or ""
    with db.session() as con:
        c = con.execute("SELECT verification_status FROM collaborators WHERE user_id=?", (user["id"],)).fetchone()
        verified = bool(c) and c["verification_status"] == "verified"
    listing = {**body.model_dump(), "lat": lat, "lon": lon, "city": city or body.town or "unknown"}
    ev = evaluate(listing, verified)
    # Listings are only visible once their real place photo has been uploaded.
    if ev["decision"] == "hidden_gem":
        ev["published"] = False
        ev["publish_note"] = "Upload a real place photo to finish publication"
    lid = str(uuid.uuid4())
    with db.session() as con:
        con.execute("""INSERT INTO listings (id, collaborator_id, name, kind, city, lat, lon, category, description,
                       cost_inr, evaluation, decision, published, review_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (lid, user["id"], body.name, body.kind, listing["city"], lat, lon, body.category,
                     body.description, body.cost_inr, json.dumps(ev, default=float), ev["decision"], int(ev["published"]),
                     "auto_approved" if ev["decision"] == "hidden_gem" else "pending_admin"))
    return {"listing_id": lid, **ev}


class ListingReview(BaseModel):
    approve: bool
    notes: Optional[str] = Field(default=None, max_length=1000)


@app.get("/api/v2/admin/overview")
def admin_overview(user: Dict = Depends(require_role("admin"))):
    with db.session() as con:
        counts = {"users": con.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"],
                  "collaborators": con.execute("SELECT COUNT(*) AS n FROM collaborators").fetchone()["n"],
                  "listings": con.execute("SELECT COUNT(*) AS n FROM listings").fetchone()["n"],
                  "pending_collaborators": con.execute("SELECT COUNT(*) AS n FROM collaborators WHERE verification_status='pending'").fetchone()["n"],
                  "pending_listings": con.execute("SELECT COUNT(*) AS n FROM listings WHERE review_status='pending_admin'").fetchone()["n"]}
        queue = con.execute("SELECT l.id,l.name,l.kind,l.city,l.decision,l.evaluation,l.created_at,u.name AS collaborator_name,u.email "
                            "FROM listings l JOIN users u ON u.id=l.collaborator_id WHERE l.review_status='pending_admin' ORDER BY l.created_at").fetchall()
    return {"counts": counts, "listing_queue": [dict(r) | {"evaluation": json.loads(r["evaluation"])} for r in queue]}


@app.get("/api/v2/admin/model-status")
def admin_model_status(user: Dict = Depends(require_role("admin"))):
    """Safe model/database diagnostics for the admin desk; never includes secrets."""
    import recommender
    route_model = {"ready": False, "place_count": 0, "message": "Route model data is unavailable."}
    try:
        route_model["place_count"] = int(len(recommender.load_candidates()))
        route_model["ready"] = route_model["place_count"] > 0
        route_model["message"] = "The trained SideQuest route recommender is ready." if route_model["ready"] else "No route candidates are loaded."
    except Exception as exc:
        route_model["error"] = type(exc).__name__
    from . import nugen
    try:
        with db.engine().connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        database = {"connected": True, "dialect": db.dialect()}
    except Exception as exc:
        database = {"connected": False, "dialect": db.dialect(), "error": type(exc).__name__}
    return {"database": database,
            "route_model": route_model, "nugen": nugen.status()}


class NugenPredictIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=3000)


@app.post("/api/v2/admin/nugen/predict")
def admin_nugen_predict(body: NugenPredictIn, user: Dict = Depends(require_role("admin"))):
    from . import nugen
    try:
        return {"impact": nugen.predict_impact(body.prompt)}
    except nugen.NugenError as exc:
        raise HTTPException(503, str(exc)) from None


@app.post("/api/v2/admin/listings/{listing_id}/review")
def admin_review_listing(listing_id: str, body: ListingReview, user: Dict = Depends(require_role("admin"))):
    with db.session() as con:
        row = con.execute("SELECT * FROM listings WHERE id=?", (listing_id,)).fetchone()
        if not row: raise HTTPException(404, "Listing not found")
        status = "admin_approved" if body.approve else "admin_rejected"
        collaborator = con.execute("SELECT verification_status FROM collaborators WHERE user_id=?", (row["collaborator_id"],)).fetchone()
        published = bool(body.approve and collaborator and collaborator["verification_status"] == "verified" and row["photo_path"])
        ev = json.loads(row["evaluation"]); ev["admin_review"] = {"approved": body.approve, "notes": body.notes}
        con.execute("UPDATE listings SET review_status=?, reviewed_by=?, reviewed_at=CURRENT_TIMESTAMP, published=?, evaluation=? WHERE id=?",
                    (status, user["id"], int(published), json.dumps(ev), listing_id))
        return {"listing_id": listing_id, "review_status": status, "published": published}


@app.post("/api/v2/listings/{listing_id}/view")
def view_listing(listing_id: str, user: Dict = Depends(current_user)):
    with db.session() as con:
        row = con.execute("SELECT published,collaborator_id FROM listings WHERE id=?", (listing_id,)).fetchone()
        if not row or (not row["published"] and row["collaborator_id"] != user["id"]): raise HTTPException(404, "Listing not found")
        if row["collaborator_id"] != user["id"]:
            con.execute("INSERT INTO listing_views (listing_id,viewer_id) VALUES (?,?) ON CONFLICT DO NOTHING", (listing_id,user["id"]))
        return {"recorded": row["collaborator_id"] != user["id"]}


class ListingRating(BaseModel):
    score: int = Field(ge=1, le=5)


@app.post("/api/v2/listings/{listing_id}/rating")
def rate_listing(listing_id: str, body: ListingRating, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        row=con.execute("SELECT published,collaborator_id FROM listings WHERE id=?",(listing_id,)).fetchone()
        if not row or not row["published"]: raise HTTPException(404,"Published place not found")
        if row["collaborator_id"]==user["id"]: raise HTTPException(403,"You cannot rate your own listing")
        con.execute("INSERT INTO listing_ratings (listing_id,user_id,score) VALUES (?,?,?) ON CONFLICT(listing_id,user_id) DO UPDATE SET score=excluded.score,created_at=CURRENT_TIMESTAMP",
                    (listing_id,user["id"],body.score))
        stat=con.execute("SELECT COUNT(*) AS n,AVG(score) AS avg FROM listing_ratings WHERE listing_id=?",(listing_id,)).fetchone()
    return {"rating_count":stat["n"],"average_rating":round(float(stat["avg"]),1)}


@app.get("/api/v2/listings/{listing_id}/photo")
def listing_photo(listing_id: str):
    with db.session() as con:
        row = con.execute("SELECT published,collaborator_id,photo_path FROM listings WHERE id=?", (listing_id,)).fetchone()
        if not row or not row["photo_path"] or not row["published"]:
            raise HTTPException(404, "Listing photo not found")
        path = Path(row["photo_path"])
    return FileResponse(path)


@app.get("/api/v2/admin/collaborator-documents/{document_id}")
def admin_document(document_id: str, user: Dict = Depends(require_role("admin"))):
    with db.session() as con:
        row=con.execute("SELECT file_path FROM collaborator_documents WHERE id=?",(document_id,)).fetchone()
        if not row: raise HTTPException(404,"Document not found")
        path=Path(row["file_path"])
        if not path.exists(): raise HTTPException(404,"Document file is no longer available")
    return FileResponse(path,filename=path.name)


@app.post("/api/v2/collaborator/listings/{listing_id}/photo")
async def upload_listing_photo(listing_id: str, file: UploadFile = File(...), user: Dict = Depends(require_role("collaborator"))):
    ext = {"image/jpeg":".jpg","image/png":".png","image/webp":".webp"}.get(file.content_type or "")
    if not ext: raise HTTPException(415,"Choose a JPG, PNG or WebP photo")
    data=await file.read(MAX_UPLOAD_MB*1024*1024+1)
    if len(data)>MAX_UPLOAD_MB*1024*1024: raise HTTPException(413,f"Photo must be smaller than {MAX_UPLOAD_MB} MB")
    if not data.startswith({".jpg":b"\xff\xd8",".png":b"\x89PNG\r\n\x1a\n",".webp":b"RIFF"}[ext]) or (ext==".webp" and data[8:12]!=b"WEBP"):
        raise HTTPException(415,"Photo content does not match its file type")
    path=UPLOAD_DIR/f"listing_{uuid.uuid4().hex}{ext}";path.write_bytes(data)
    with db.session() as con:
        changed=con.execute("UPDATE listings SET photo_path=? WHERE id=? AND collaborator_id=?",(str(path),listing_id,user["id"]))
        if not changed.rowcount: path.unlink(missing_ok=True); raise HTTPException(404,"Listing not found")
        listing=con.execute("SELECT decision,review_status FROM listings WHERE id=?",(listing_id,)).fetchone()
        host=con.execute("SELECT verification_status FROM collaborators WHERE user_id=?",(user["id"],)).fetchone()
        if listing["review_status"] in ("auto_approved","admin_approved") and listing["decision"]=="hidden_gem" and host and host["verification_status"]=="verified":
            con.execute("UPDATE listings SET published=1 WHERE id=?",(listing_id,))
    return {"photo_url":f"/api/v2/listings/{listing_id}/photo"}


# ---------------------------------------------------------------------------
# trips + solo matchmaking + chat
# ---------------------------------------------------------------------------
class TripIn(BaseModel):
    destination: str = Field(min_length=2, max_length=80)
    start_date: date
    end_date: date
    solo_match: bool = False


@app.post("/api/v2/trips")
def create_trip(body: TripIn, user: Dict = Depends(require_role("user"))):
    import geo_india as gi
    if body.end_date < body.start_date:
        raise HTTPException(422, "End date is before start date")
    if body.end_date < date.today():
        raise HTTPException(422, "Trip is already over")
    if (body.end_date - body.start_date).days > 60:
        raise HTTPException(422, "Trips longer than 60 days are not supported")
    c = gi.town_coords(body.destination)
    if not c:
        raise HTTPException(422, f"Unknown destination '{body.destination}' - try a nearby town")
    tid = str(uuid.uuid4())
    with db.session() as con:
        con.execute("INSERT INTO trips (id, user_id, destination, dest_lat, dest_lon, start_date, end_date, solo_match) "
                    "VALUES (?,?,?,?,?,?,?,?)", (tid, user["id"], body.destination.strip(), c[0], c[1],
                                                 body.start_date.isoformat(), body.end_date.isoformat(), int(body.solo_match)))
        return dict(con.execute("SELECT * FROM trips WHERE id=?", (tid,)).fetchone())


@app.get("/api/v2/trips")
def my_trips(user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        return [dict(r) for r in con.execute("SELECT * FROM trips WHERE user_id=? ORDER BY start_date", (user["id"],))]


class SoloToggle(BaseModel):
    solo_match: bool


@app.patch("/api/v2/trips/{trip_id}")
def toggle_solo(trip_id: str, body: SoloToggle, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        n = con.execute("UPDATE trips SET solo_match=? WHERE id=? AND user_id=?",
                        (int(body.solo_match), trip_id, user["id"])).rowcount
        if not n:
            raise HTTPException(404, "Trip not found")
        return dict(con.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone())


@app.get("/api/v2/solo/candidates")
def solo_candidates(trip_id: str, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        try:
            return matchmaking.candidates(con, user["id"], trip_id)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))


class RequestIn(BaseModel):
    trip_id: str
    to_user: str
    their_trip_id: str


@app.post("/api/v2/solo/requests")
def solo_request(body: RequestIn, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        try:
            return matchmaking.send_request(con, user["id"], body.trip_id, body.to_user, body.their_trip_id)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except (LookupError, ValueError) as e:
            raise HTTPException(422, str(e))


class Respond(BaseModel):
    accept: bool


@app.post("/api/v2/solo/requests/{request_id}/respond")
def solo_respond(request_id: str, body: Respond, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        try:
            return matchmaking.respond(con, user["id"], request_id, body.accept)
        except PermissionError as e:
            raise HTTPException(403, str(e))


@app.get("/api/v2/solo/requests")
def solo_requests(user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        return matchmaking.list_requests(con, user["id"])


@app.get("/api/v2/matches")
def matches(user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        return matchmaking.my_matches(con, user["id"])


class MessageIn(BaseModel):
    body: str = Field(min_length=1, max_length=MESSAGE_MAX_CHARS)


_msg_times: Dict[str, deque] = defaultdict(deque)


@app.get("/api/v2/matches/{match_id}/messages")
def get_messages(match_id: str, after: int = 0, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        if not matchmaking.is_participant(con, match_id, user["id"]):
            raise HTTPException(403, "Chat opens only after both travellers accept")
        rows = con.execute("SELECT id, sender_id, body, created_at FROM messages WHERE match_id=? AND id>? "
                           "ORDER BY id LIMIT 200", (match_id, after)).fetchall()
    return [dict(r) | {"mine": r["sender_id"] == user["id"]} for r in rows]


@app.post("/api/v2/matches/{match_id}/messages")
def post_message(match_id: str, body: MessageIn, user: Dict = Depends(require_role("user"))):
    q = _msg_times[user["id"]]
    now = time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= MESSAGES_PER_MINUTE:
        raise HTTPException(429, "Slow down - too many messages")
    with db.session() as con:
        if not matchmaking.is_participant(con, match_id, user["id"]):
            raise HTTPException(403, "Chat opens only after both travellers accept")
        row = con.execute("INSERT INTO messages (match_id, sender_id, body) VALUES (?,?,?) RETURNING id",
                          (match_id, user["id"], body.body.strip())).fetchone()
        q.append(now)
        return {"id": row["id"], "sent": True}


class UserRef(BaseModel):
    user_id: str
    reason: Optional[str] = Field(default=None, max_length=500)


@app.post("/api/v2/block")
def block(body: UserRef, user: Dict = Depends(require_role("user"))):
    with db.session() as con:
        matchmaking.block(con, user["id"], body.user_id)
    return {"blocked": body.user_id}


@app.post("/api/v2/report")
def report(body: UserRef, user: Dict = Depends(current_user)):
    if not body.reason:
        raise HTTPException(422, "Tell us what happened")
    with db.session() as con:
        con.execute("INSERT INTO reports (id, reporter, reported, reason) VALUES (?,?,?,?)",
                    (str(uuid.uuid4()), user["id"], body.user_id, body.reason))
    return {"reported": body.user_id, "message": "Thanks - our team reviews every report"}


# ---------------------------------------------------------------------------
# dynamic trip planner (add/remove stops, flight-delay re-planning)
# ---------------------------------------------------------------------------
class FixedStopIn(BaseModel):
    place: str = Field(min_length=2, max_length=160)
    time: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
    lat: float = Field(ge=6, le=37.5)
    lon: float = Field(ge=68, le=97.5)
    duration_min: int = Field(default=60, ge=15, le=480)


class ItineraryIn(BaseModel):
    origin: str = Field(min_length=2, max_length=80)
    destination: str = Field(min_length=2, max_length=80)
    depart_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}$")
    arrive_by: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}$")
    transport: Optional[str] = Field(default="car", pattern="^(car|motorcycle|bus)$")
    travellers: int = Field(default=1, ge=1, le=20)
    interests: List[str] = Field(default_factory=list, max_length=12)
    fixed_stops: List[FixedStopIn] = Field(default_factory=list, max_length=30)


def _itin(con, iid: str, uid: str) -> Dict:
    r = con.execute("SELECT * FROM itineraries WHERE id=? AND user_id=?", (iid, uid)).fetchone()
    if not r:
        raise HTTPException(404, "Trip plan not found")
    r["route"] = json.loads(r["route"])
    r["planning_data"] = json.loads(r.get("planning_data") or "{}")
    return r


def _stops(con, iid: str) -> List[Dict]:
    return [json.loads(r["details"]) for r in con.execute(
        "SELECT details FROM itinerary_stops WHERE itinerary_id=? ORDER BY created_at", (iid,)).fetchall()]


def _plan_view(it: Dict, stops: List[Dict], extra: Optional[Dict] = None) -> Dict:
    from . import itinerary as P
    route = it["route"]
    return {"id": it["id"], "origin": it["origin"], "destination": it["destination"],
            "depart_at": it["depart_at"], "arrive_by": it["arrive_by"], "delay_min": it["delay_min"],
            "route": {k: route[k] for k in ("distance_km", "duration_min", "source", "transport", "transport_eta_factor") if k in route},
            "schedule": P.schedule(it, route, stops), "planning_data": it.get("planning_data") or {}, **(extra or {})}


@app.post("/api/v2/itineraries")
def create_itinerary(body: ItineraryIn, user: Dict = Depends(current_user)):
    from . import itinerary as P
    dep, arr = P.parse_local(body.depart_at), P.parse_local(body.arrive_by)
    if arr <= dep:
        raise HTTPException(422, "Arrive-by time must be after departure")
    for fixed in body.fixed_stops:
        appointment = P.parse_local(fixed.time)
        if not dep <= appointment < arr:
            raise HTTPException(422, f"Fixed stop at {fixed.time} must fall between departure and arrival")
    if (arr - dep).days > 3:
        raise HTTPException(422, "Plans longer than 3 days are not supported")
    try:
        route = P.build_route(body.origin, body.destination)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if (arr - dep).total_seconds() / 60 < route["duration_min"]:
        raise HTTPException(422, f"The drive alone takes about {route['duration_min'] / 60:.1f} h - "
                                 "allow more time before your arrive-by time")
    # The router supplies a road-network drive baseline. Adjust only the ETA estimate
    # for the selected road vehicle; the geometry remains the actual road corridor.
    adjustment = {"car": 1.0, "motorcycle": 0.95, "bus": 1.25}[body.transport or "car"]
    route["duration_min"] = round(route["duration_min"] * adjustment, 1)
    route["transport"] = body.transport or "car"
    route["transport_eta_factor"] = adjustment
    iid = str(uuid.uuid4())
    with db.session() as con:
        planning = {"transport":body.transport,"travellers":body.travellers,"interests":body.interests,
                    "fixed_stops":[x.model_dump() for x in body.fixed_stops]}
        con.execute("INSERT INTO itineraries (id, user_id, origin, destination, depart_at, arrive_by, route, planning_data) "
                    "VALUES (?,?,?,?,?,?,?,?)", (iid, user["id"], body.origin, body.destination,
                                               body.depart_at.replace(" ", "T"), body.arrive_by.replace(" ", "T"),
                                               json.dumps(route), json.dumps(planning)))
        it = _itin(con, iid, user["id"])
    return _plan_view(it, [])


@app.get("/api/v2/itineraries")
def list_itineraries(user: Dict = Depends(current_user)):
    with db.session() as con:
        rows = con.execute("SELECT id, origin, destination, depart_at, arrive_by, delay_min FROM itineraries "
                           "WHERE user_id=? ORDER BY depart_at DESC", (user["id"],)).fetchall()
    return rows


@app.get("/api/v2/itineraries/{iid}")
def get_itinerary(iid: str, user: Dict = Depends(current_user)):
    with db.session() as con:
        it = _itin(con, iid, user["id"])
        return _plan_view(it, _stops(con, iid))


class StopIn(BaseModel):
    place_id: str = Field(min_length=3, max_length=200)


@app.post("/api/v2/itineraries/{iid}/stops")
def add_stop(iid: str, body: StopIn, user: Dict = Depends(current_user)):
    from . import itinerary as P
    with db.session() as con:
        it = _itin(con, iid, user["id"])
        try:
            snap = P.place_snapshot(body.place_id, it["route"])
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        con.execute("INSERT INTO itinerary_stops (id, itinerary_id, place_id, details) VALUES (?,?,?,?) "
                    "ON CONFLICT DO NOTHING", (str(uuid.uuid4()), iid, body.place_id, json.dumps(snap)))
        view = _plan_view(it, _stops(con, iid))
    if not view["schedule"]["fits"]:
        view["warning"] = (f"This plan now runs {-view['schedule']['slack_min']} min past your arrive-by time - "
                           "remove a stop or accept arriving later")
    return view


@app.delete("/api/v2/itineraries/{iid}/stops/{place_id:path}")
def remove_stop(iid: str, place_id: str, user: Dict = Depends(current_user)):
    with db.session() as con:
        it = _itin(con, iid, user["id"])
        n = con.execute("DELETE FROM itinerary_stops WHERE itinerary_id=? AND place_id=?", (iid, place_id)).rowcount
        if not n:
            raise HTTPException(404, "That stop is not in this plan")
        return _plan_view(it, _stops(con, iid))


class DelayIn(BaseModel):
    minutes: int = Field(ge=0, le=24 * 60)   # total delay so far (0 = back on time)


class WhatIfIn(BaseModel):
    scenario: str = Field(pattern=r"^(rain|traffic|closure|late_start|custom)$")
    delay_min: int = Field(ge=0, le=12 * 60)
    weather_summary: Optional[str] = Field(default=None, max_length=500)


class WhatIfApplyIn(BaseModel):
    scenario: str = Field(pattern=r"^(rain|traffic|closure|late_start|custom)$")
    delay_min: int = Field(ge=0, le=12 * 60)
    add_place_ids: List[str] = Field(default_factory=list, max_length=5)


@app.post("/api/v2/itineraries/{iid}/delay")
def report_delay(iid: str, body: DelayIn, user: Dict = Depends(current_user)):
    """Flight (or anything before departure) is late: shrink the free time, keep the best
    stops that still fit, drop the rest, and suggest gems that fit the time left."""
    from . import itinerary as P
    with db.session() as con:
        it = _itin(con, iid, user["id"])
        con.execute("UPDATE itineraries SET delay_min=? WHERE id=?", (body.minutes, iid))
        it["delay_min"] = body.minutes
        stops = _stops(con, iid)
        kept, dropped = P.best_fit(it, it["route"], stops)
        for s in dropped:
            con.execute("DELETE FROM itinerary_stops WHERE itinerary_id=? AND place_id=?", (iid, s["place_id"]))
        exclude = {s["place_id"] for s in stops}
        subs = P.replacements(it, it["route"], kept, exclude)
        view = _plan_view(it, kept, {
            "kept": [s["name"] for s in kept], "dropped": dropped, "suggested_replacements": subs})
    sched = view["schedule"]
    if not sched["fits"]:
        view["message"] = (f"Even with no stops you'd arrive {-sched['slack_min']} min after your arrive-by time.")
    elif dropped:
        view["message"] = (f"{len(dropped)} stop(s) no longer fit after a {body.minutes}-min delay; kept the best "
                           f"{len(kept)}. {len(subs)} shorter gem(s) fit the {sched['slack_min']} min left.")
    else:
        view["message"] = f"All stops still fit - {sched['slack_min']} min to spare."
    return view


@app.post("/api/v2/itineraries/{iid}/what-if")
def itinerary_what_if(iid: str, body: WhatIfIn, user: Dict = Depends(require_role("user"))):
    """Simulate a traveller-entered delay without changing the saved itinerary."""
    from . import itinerary as P
    with db.session() as con:
        it = _itin(con, iid, user["id"])
        route = it["route"]
        stops = _stops(con, iid)
    baseline = P.schedule(it, route, stops)
    if body.scenario == "late_start":
        shift = timedelta(minutes=body.delay_min)
        scenario_it = {**it,
                       "depart_at": (P.parse_local(it["depart_at"]) + shift).strftime("%Y-%m-%dT%H:%M"),
                       "arrive_by": (P.parse_local(it["arrive_by"]) + shift).strftime("%Y-%m-%dT%H:%M")}
    else:
        scenario_it = {**it, "delay_min": int(it.get("delay_min") or 0) + body.delay_min}
    kept, dropped = P.best_fit(scenario_it, route, stops)
    scenario = P.schedule(scenario_it, route, kept)
    replacements = P.replacements(scenario_it, route, kept,
                                  {s["place_id"] for s in stops}) if scenario["slack_min"] > 20 else []
    nugen_result = {"available": False, "state": "api_unavailable", "message": "Nugen inference is not available right now."}
    try:
        from . import nugen
        nugen_status = nugen.status()
        if nugen_status.get("aligned_model_ready"):
            model_id = os.environ.get("NUGEN_ALIGNED_MODEL_ID", "").strip()
            facts = {"route": f"{it['origin']} to {it['destination']}",
                     "scenario": body.scenario, "user_assumed_delay_minutes": body.delay_min,
                     "weather_forecast": body.weather_summary or "Unavailable",
                     "saved_arrival": baseline.get("arrive_destination"),
                     "scenario_arrival": scenario.get("arrive_destination"),
                     "deadline": scenario.get("deadline"),
                     "baseline_schedule": {"fits": baseline.get("fits"), "slack_minutes": baseline.get("slack_min"),
                                           "stop_count": len(baseline.get("stops", []))},
                     "scenario_schedule": {"fits": scenario.get("fits"), "slack_minutes": scenario.get("slack_min"),
                                           "stop_count": len(scenario.get("stops", []))},
                     "saved_stop_context": [{"name": s.get("name"),
                                             "safety": (s.get("safety") or {}).get("class", "unknown") if isinstance(s.get("safety"), dict) else "unknown"}
                                            for s in stops[:8]],
                     "stops_removed_by_schedule": [s.get("name") for s in dropped],
                     "safe_replacements": [{"name": s.get("name"), "safety": s.get("safety", {}).get("class", "unknown")}
                                           for s in replacements[:3]]}
            schema = {"accessibility": "likely_open|limited|unknown", "delay_prob": "number 0..1 or null",
                      "demand_shift_pct": "number -100..100 or null", "safety_impact": "low|moderate|high|unknown",
                      "cascades": ["up to three evidence-based itinerary effects"],
                      "recommendation": "continue_with_caution|wait_for_update|consider_alternative|insufficient_data",
                      "confidence": "low|medium|high", "guidance": "concise traveller-facing explanation"}
            prompt = ("You are SideQuest's travel-impact Digital Twin. Return exactly one JSON object matching this schema, "
                      "with no markdown or extra keys: " + json.dumps(schema) + ". Use only the supplied facts. "
                      "Estimate changes for this scenario, not verified real-world conditions. Never claim a live flood, "
                      "closure, open facility, guaranteed safety, or measured visitor-demand change. Use unknown/null and "
                      "low confidence when evidence is insufficient. Accessibility is a cautious scenario estimate, not a "
                      "verified operational status. Keep recommendations and cascades consistent with the calculated schedule. "
                      "Facts: " + json.dumps(facts, ensure_ascii=False))
            impact = nugen.predict_impact(prompt, model_id=model_id)
            nugen_result = {"available": True, "state": "ready", "message": "Nugen returned a validated scenario estimate.", **impact}
        else:
            if nugen_status.get("alignment_processing_count", 0):
                state = "alignment_processing"
            else:
                state = "model_unavailable" if nugen_status.get("configured") and nugen_status.get("reachable") else "api_unavailable"
            nugen_result = {"available": False, "state": state,
                            "message": nugen_status.get("message", "No deployed SideQuest-aligned Nugen model is available.")}
    except Exception as exc:
        # Keep deterministic schedule/weather results usable if Nugen is down.
        nugen_module = sys.modules.get(f"{__package__}.nugen")
        nugen_error = getattr(nugen_module, "NugenError", ())
        message = str(exc) if nugen_error and isinstance(exc, nugen_error) else "Nugen inference could not be reached."
        nugen_result = {"available": False, "state": "inference_unavailable", "message": message}
    return {"scenario": body.scenario, "assumed_delay_min": body.delay_min,
            "baseline": baseline, "scenario_plan": scenario,
            "kept": [s["name"] for s in kept], "dropped": dropped,
            "suggested_replacements": replacements,
            "nugen": nugen_result,
            "notice": "The schedule comparison is calculated from your saved itinerary. Weather is a separate forecast; traffic and road closures are not live feeds."}


@app.post("/api/v2/itineraries/{iid}/what-if/apply")
def apply_what_if(iid: str, body: WhatIfApplyIn, user: Dict = Depends(require_role("user"))):
    """Commit a preview explicitly: delay the plan, remove stops that no longer fit,
    then add only replacements offered by the same route model."""
    from . import itinerary as P
    with db.session() as con:
        it = _itin(con, iid, user["id"])
        stops = _stops(con, iid)
        if body.scenario == "late_start":
            shift = timedelta(minutes=body.delay_min)
            it["depart_at"] = (P.parse_local(it["depart_at"]) + shift).strftime("%Y-%m-%dT%H:%M")
            it["arrive_by"] = (P.parse_local(it["arrive_by"]) + shift).strftime("%Y-%m-%dT%H:%M")
        else:
            it["delay_min"] = int(it.get("delay_min") or 0) + body.delay_min
            if it["delay_min"] > 24 * 60:
                raise HTTPException(422, "The combined delay cannot exceed 24 hours")
        kept, dropped = P.best_fit(it, it["route"], stops)
        replacements = P.replacements(it, it["route"], kept, {s["place_id"] for s in stops})
        offered = {s["place_id"]: s for s in replacements}
        if any(pid not in offered for pid in body.add_place_ids):
            raise HTTPException(422, "One of the selected replacement places is no longer available")
        additions = [offered[pid] for pid in dict.fromkeys(body.add_place_ids)]
        updated = kept + additions
        if not P.schedule(it, it["route"], updated)["fits"]:
            raise HTTPException(422, "Those replacement stops do not fit the updated travel window")
        if body.scenario == "late_start":
            con.execute("UPDATE itineraries SET depart_at=?, arrive_by=? WHERE id=?", (it["depart_at"], it["arrive_by"], iid))
        else:
            con.execute("UPDATE itineraries SET delay_min=? WHERE id=?", (it["delay_min"], iid))
        for stop in dropped:
            con.execute("DELETE FROM itinerary_stops WHERE itinerary_id=? AND place_id=?", (iid, stop["place_id"]))
        for stop in additions:
            con.execute("INSERT INTO itinerary_stops (id, itinerary_id, place_id, details) VALUES (?,?,?,?) ON CONFLICT DO NOTHING",
                        (str(uuid.uuid4()), iid, stop["place_id"], json.dumps(stop)))
        view = _plan_view(it, updated, {"kept": [s["name"] for s in kept], "dropped": dropped,
                                        "suggested_replacements": P.replacements(it, it["route"], updated,
                                            {s["place_id"] for s in updated}),
                                        "message": "Your saved trip and its Hidden Gems stops are updated."})
    return view


@app.get("/api/v2/itineraries/{iid}/weather")
def itinerary_weather(iid: str, user: Dict = Depends(require_role("user"))):
    """Weather outlook sampled along the saved route. No client-side provider key."""
    import requests
    from . import itinerary as P
    with db.session() as con:
        it = _itin(con, iid, user["id"])
    coords = it["route"].get("geometry") or []
    if len(coords) < 2:
        raise HTTPException(422, "This saved route has no weather sampling coordinates")
    dep = P.parse_local(it["depart_at"]) + timedelta(minutes=int(it.get("delay_min") or 0))
    if dep.date() < date.today() or dep.date() > date.today() + timedelta(days=15):
        return {"available": False, "reason": "A reliable hourly forecast is not available for this trip date.",
                "route_date": dep.strftime("%Y-%m-%d"), "samples": [], "later_options": []}
    # Sample start, middle and end of the route; keep provider calls bounded.
    points = [coords[0], coords[len(coords) // 2], coords[-1]]
    seen, samples, forecast_rows = set(), [], []
    for lat, lon in points:
        key = (round(float(lat), 2), round(float(lon), 2))
        if key in seen: continue
        seen.add(key)
        try:
            response = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": key[0], "longitude": key[1], "hourly": "precipitation_probability,precipitation,weather_code",
                "timezone": "auto", "forecast_days": 16}, timeout=8)
            response.raise_for_status()
            data = response.json().get("hourly", {})
            forecast_rows.append(data)
            times = data.get("time", [])
            index = min(range(len(times)), key=lambda i: abs((datetime.fromisoformat(times[i]) - dep).total_seconds())) if times else -1
            if index >= 0 and abs((datetime.fromisoformat(times[index]) - dep).total_seconds()) <= 5400:
                samples.append({"time": times[index], "precipitation_probability": (data.get("precipitation_probability") or [None])[index],
                                "precipitation_mm": (data.get("precipitation") or [None])[index],
                                "weather_code": (data.get("weather_code") or [None])[index]})
        except (requests.RequestException, ValueError, KeyError, IndexError):
            continue
    if not samples:
        return {"available": False, "reason": "The forecast provider did not return route-hour data. Try again later.",
                "route_date": dep.strftime("%Y-%m-%d"), "samples": [], "later_options": []}
    risk = max([int(s["precipitation_probability"] or 0) for s in samples])
    mm = max([float(s["precipitation_mm"] or 0) for s in samples])
    status = "Rain likely" if risk >= 60 or mm >= 4 else ("Some rain possible" if risk >= 35 or mm >= 1 else "No significant rain signal")
    later = []
    for hours in (3, 6):
        future = dep + timedelta(hours=hours)
        matches = []
        for data in forecast_rows:
            ts = data.get("time", [])
            ix = min(range(len(ts)), key=lambda i: abs((datetime.fromisoformat(ts[i]) - future).total_seconds())) if ts else -1
            if ix >= 0 and abs((datetime.fromisoformat(ts[ix]) - future).total_seconds()) <= 5400:
                matches.append(int((data.get("precipitation_probability") or [0])[ix] or 0))
        if matches:
            later.append({"hours_later": hours, "precipitation_probability": max(matches),
                          "recommendation": "Lower rain probability" if max(matches) + 10 < risk else "No clear improvement in the forecast"})
    return {"available": True, "provider": "Open-Meteo", "route_date": dep.strftime("%Y-%m-%d"),
            "departure": dep.strftime("%Y-%m-%d %H:%M"), "status": status, "max_precipitation_probability": risk,
            "max_precipitation_mm": round(mm, 1), "samples": samples, "later_options": later,
            "notice": "Forecast along three points on your route; weather guidance only, not an official road, flood, or travel-safety alert."}


@app.get("/api/v2/itineraries/{iid}/suggestions")
def itinerary_suggestions(iid: str, user: Dict = Depends(current_user)):
    from . import itinerary as P
    with db.session() as con:
        it = _itin(con, iid, user["id"])
        stops = _stops(con, iid)
    return {"slack_min": P.schedule(it, it["route"], stops)["slack_min"],
            "suggestions": P.replacements(it, it["route"], stops, {s["place_id"] for s in stops}, limit=8)}


# ---------------------------------------------------------------------------
# ML: route gems, RAG search, YouTube check (models loaded once)
# ---------------------------------------------------------------------------
class RouteIn(BaseModel):
    from_: str = Field(alias="from")
    to: str
    bufferKm: float = Field(default=10.0, ge=1, le=50)
    youtube: str = Field(default="cache", pattern="^(off|cache|live)$")
    only: Optional[str] = Field(default=None, pattern="^(eat|visit)$")


@app.post("/api/v2/gems/route")
def route_gems(body: RouteIn):
    import recommender
    try:
        kinds = (body.only,) if body.only else ("eat", "visit")
        return recommender.recommend(body.from_, body.to, body.bufferKm, youtube=body.youtube, kinds=kinds)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/v2/gems/city")
def gems_in_city(city: str = "", limit: int = 12, kind: str = ""):
    """Hidden gems in a town (or India's best, one per town) - the SAME rules as the route
    recommender, so the Home, Gems and Routes screens always agree."""
    import numpy as np
    import pandas as pd
    import recommender as rec
    from .rag import city_key
    limit = max(1, min(limit, 40))
    c = rec.load_candidates()
    lc = rec.listing_candidates()
    if len(lc):
        c = pd.concat([c, lc], ignore_index=True)
    if kind in ("eat", "visit"):
        c = c[c["kind"] == kind]
    is_listing = c["id"].str.startswith("listing:")
    enough = (c["n_reviews"] >= c["kind"].map(rec.MIN_REVIEWS)) | is_listing
    not_sight = (c["kind"] == "visit") & c["name"].str.lower().str.contains(rec.NOT_A_SIGHT, regex=True)
    human_approved = c.get("human_approved", pd.Series(False, index=c.index)).fillna(False).astype(bool)
    pool = c[enough & ~not_sight].copy()
    matched_city_keys = set()
    if city:
        ck = city_key(city)
        normalized_cities = pool["city"].map(city_key)
        exact = normalized_cities == ck
        if exact.any():
            matched_city_keys = set(normalized_cities[exact].unique())
        else:
            terms = [part for part in ck.split() if len(part) >= 3]
            partial = normalized_cities.map(lambda name: any(term in name for term in terms))
            matched_city_keys = set(normalized_cities[partial].unique())
        pool = pool[pool["city"].map(city_key).isin(matched_city_keys)]
    g = c[enough & ~not_sight & (((c["quality_rank"] >= rec.GEM_RANK_MIN)
          & (c["exposure_pct"] <= rec.HIDDEN_MAX_EXPOSURE) & (c["chain_outlets"] <= rec.MAX_CHAIN_OUTLETS)) | human_approved)]
    if city:
        g = g[g["city"].map(city_key).isin(matched_city_keys)]
        g = g.sort_values("quality_rank", ascending=False).drop_duplicates("dedupe_key")
        # mix eat + visit when the town has both
        g = pd.concat([g[g["kind"] == "visit"].head(limit // 2), g[g["kind"] == "eat"]]).drop_duplicates("id")
        g = g.sort_values("quality_rank", ascending=False).head(limit)
    else:
        g = (g.sort_values(["quality_rank", "n_reviews"], ascending=False)
             .drop_duplicates("dedupe_key").drop_duplicates("city").head(limit))
    blocks = rec.safety_blocks(g) if len(g) else []
    gems = []
    for (_, r), b in zip(g.iterrows(), blocks):
        gems.append({
            "place_id": r["id"], "name": r["name"], "kind": r["kind"], "city": r["city"],
            "category": str(r["category"]).replace("_", " "), "cuisines": "",
            "rating": None if pd.isna(r["rating"]) else float(r["rating"]),
            "review_count": None if pd.isna(r["n_reviews"]) else int(r["n_reviews"]),
            "hidden_gem_score": None if pd.isna(r["quality_rank"]) else round(float(r["quality_rank"]), 1), "gem_score_10": None if pd.isna(r["quality_rank"]) else round(float(r["quality_rank"]) / 10, 1),
            "human_approved": bool(pd.notna(r.get("human_approved")) and r.get("human_approved")),
            "exposure_pct": round(float(r["exposure_pct"]), 3), "safety_score": b["score"], "safety_class": b["class"],
            "safety_reasons": b["reasons"][:3], "hospital_km": b.get("hospital_km"), "police_km": b.get("police_km"),
            "price_level": None, "lat": float(r["lat"]), "lng": float(r["lon"]),
            "photo_url": None if pd.isna(r.get("photo_url")) else r.get("photo_url"),
            "is_hidden": True, "score_basis": r["basis"]})
    popular = []
    if city and len(pool):
        candidates = pool[(pool["quality_rank"] >= rec.GEM_RANK_MIN)
                          & (pool["exposure_pct"] >= rec.FAMOUS_MIN_EXPOSURE)
                          & (pool["chain_outlets"] <= 3 * rec.MAX_CHAIN_OUTLETS)]
        candidates = candidates.sort_values("quality_rank", ascending=False).drop_duplicates("dedupe_key").head(5)
        popular_blocks = rec.safety_blocks(candidates) if len(candidates) else []
        for (_, r), b in zip(candidates.iterrows(), popular_blocks):
            popular.append({
                "place_id": r["id"], "name": r["name"], "kind": r["kind"], "city": r["city"],
                "category": str(r["category"]).replace("_", " "), "rating": None if pd.isna(r["rating"]) else float(r["rating"]),
                "review_count": None if pd.isna(r["n_reviews"]) else int(r["n_reviews"]),
                "hidden_gem_score": round(float(r["quality_rank"]), 1),
                "gem_score_10": round(float(r["quality_rank"]) / 10, 1), "human_approved": False,
                "exposure_pct": round(float(r["exposure_pct"]), 3), "safety_score": b["score"],
                "safety_class": b["class"], "safety_reasons": b["reasons"][:3],
                "hospital_km": b.get("hospital_km"), "police_km": b.get("police_km"),
                "lat": float(r["lat"]), "lng": float(r["lon"]), "photo_url": None,
                "is_hidden": False, "score_basis": r["basis"]})
    return {"success": True, "city": city or None, "count": len(gems), "gems": gems,
            "hidden_gems": gems[:5] if city else [], "famous_and_good": popular,
            "rule": f"quality rank >= {rec.GEM_RANK_MIN:.0f}, less reviewed than half the town, not a chain"}


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    city: str = ""
    kind: str = Field(default="", pattern="^(|eat|visit)$")
    top_k: int = Field(default=5, ge=1, le=20)
    only_hidden: bool = False


@app.post("/api/v2/rag/search")
def rag_search(body: SearchIn):
    from . import rag
    try:
        return rag.search(body.query, body.top_k, body.city, body.kind, body.only_hidden)
    except Exception as e:  # index not built
        raise HTTPException(503, f"Search index unavailable ({type(e).__name__}) - run python -m backend.rag --build")


class VideoIn(BaseModel):
    url: str = Field(min_length=5, max_length=300)


@app.post("/api/v2/youtube/check")
def youtube_check(body: VideoIn, user: Dict = Depends(current_user)):
    import youtube_verify as yv
    try:
        return yv.check_video(body.url)
    except yv.YouTubeUnavailable as e:
        raise HTTPException(503, str(e))


@app.get("/api/v2/health")
def health():
    ok = {}
    try:
        import recommender
        ok["places"] = int(len(recommender.load_candidates()))
    except Exception as e:
        ok["places_error"] = type(e).__name__
    try:
        from . import rag
        ok["rag_documents"] = rag._load()[0].count()
    except Exception as e:
        ok["rag_error"] = type(e).__name__
    return {"status": "ok", **ok}
