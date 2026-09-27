"""Database layer: PostgreSQL in production, SQLite for local runs and tests.

    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/sidequest   # production
    (unset)      -> sqlite file at data/app/sidequest.db                  # zero-setup local run

The schema is declared once with SQLAlchemy and created with create_all(), so both
databases get identical tables, keys, cascades and CHECK constraints. Queries are plain
SQL that both engines understand (CURRENT_TIMESTAMP, ON CONFLICT, RETURNING); the small
`Con` wrapper lets them use `?` placeholders and returns rows as plain dicts.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import (CheckConstraint, Column, Float, ForeignKey, Index, Integer, MetaData, String, Table, inspect,
                        Text, UniqueConstraint, create_engine, event, func, text)
from sqlalchemy.engine import Engine

from .config import DB_PATH

meta = MetaData()
_NOW = func.now()


def _fk(table: str) -> ForeignKey:
    return ForeignKey(f"{table}.id", ondelete="CASCADE")


users = Table(
    "users", meta,
    Column("id", String(36), primary_key=True),
    Column("email", String(254), nullable=False, unique=True),
    Column("password_hash", String(200), nullable=False),
    Column("role", String(20), nullable=False),
    Column("name", String(80), nullable=False),
    Column("phone", String(20)),
    Column("age", Integer),
    Column("gender", String(20)),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("is_active", Integer, nullable=False, server_default=text("1")),
    CheckConstraint("role IN ('user', 'collaborator', 'admin')", name="ck_users_role"),
    CheckConstraint("age IS NULL OR age >= 18", name="ck_users_age"),
    CheckConstraint("gender IS NULL OR gender IN ('female', 'male', 'non_binary', 'prefer_not')", name="ck_users_gender"),
)

# the five traveller compatibility preferences; companion filters default to any traveller aged 18-99
preferences = Table(
    "preferences", meta,
    Column("user_id", String(36), _fk("users"), primary_key=True),
    Column("travel_style", String(30), nullable=False),
    Column("travel_styles", Text),                          # JSON list; travel_style remains as legacy first choice
    Column("budget", String(30), nullable=False),
    Column("pace", String(30), nullable=False),
    Column("food", String(30), nullable=False),
    Column("languages", Text, nullable=False),                 # JSON list
    Column("companion_gender", String(10), nullable=False, server_default=text("'any'")),
    Column("companion_age_min", Integer, nullable=False, server_default=text("18")),
    Column("companion_age_max", Integer, nullable=False, server_default=text("99")),
    Column("updated_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

collaborators = Table(
    "collaborators", meta,
    Column("user_id", String(36), _fk("users"), primary_key=True),
    Column("business_name", String(120), nullable=False),
    Column("business_type", String(60), nullable=False),
    Column("gstin", String(15)),
    Column("gstin_check", String(20), nullable=False, server_default=text("'not_provided'")),
    Column("gstin_state", String(60)),
    Column("document_path", Text),
    Column("document_type", String(30)),
    Column("verification_status", String(20), nullable=False, server_default=text("'pending'")),
    Column("reviewed_by", String(36)),
    Column("reviewed_at", String(32)),
    Column("review_notes", Text),
    CheckConstraint("verification_status IN ('pending', 'verified', 'rejected')", name="ck_collab_status"),
)

trips = Table(
    "trips", meta,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), _fk("users"), nullable=False),
    Column("destination", String(80), nullable=False),
    Column("dest_lat", Float, nullable=False),
    Column("dest_lon", Float, nullable=False),
    Column("start_date", String(10), nullable=False),
    Column("end_date", String(10), nullable=False),
    Column("solo_match", Integer, nullable=False, server_default=text("0")),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Index("ix_trips_solo", "solo_match", "start_date", "end_date"),
)

match_requests = Table(
    "match_requests", meta,
    Column("id", String(36), primary_key=True),
    Column("from_user", String(36), _fk("users"), nullable=False),
    Column("to_user", String(36), _fk("users"), nullable=False),
    Column("from_trip", String(36), _fk("trips"), nullable=False),
    Column("to_trip", String(36), _fk("trips"), nullable=False),
    Column("status", String(20), nullable=False, server_default=text("'pending'")),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("responded_at", String(32)),
    UniqueConstraint("from_user", "to_user", "from_trip", "to_trip", name="uq_request_pair"),
    CheckConstraint("status IN ('pending', 'accepted', 'declined', 'cancelled')", name="ck_request_status"),
    Index("ix_requests_to", "to_user", "status"),
)

matches = Table(
    "matches", meta,
    Column("id", String(36), primary_key=True),
    Column("user_a", String(36), _fk("users"), nullable=False),
    Column("user_b", String(36), _fk("users"), nullable=False),
    Column("trip_a", String(36), nullable=False),
    Column("trip_b", String(36), nullable=False),
    Column("status", String(20), nullable=False, server_default=text("'active'")),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("status IN ('active', 'closed')", name="ck_match_status"),
)

messages = Table(
    "messages", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", String(36), _fk("matches"), nullable=False),
    Column("sender_id", String(36), _fk("users"), nullable=False),
    Column("body", Text, nullable=False),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Index("ix_messages_match", "match_id", "id"),
)

blocks = Table(
    "blocks", meta,
    Column("blocker", String(36), _fk("users"), primary_key=True),
    Column("blocked", String(36), _fk("users"), primary_key=True),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

reports = Table(
    "reports", meta,
    Column("id", String(36), primary_key=True),
    Column("reporter", String(36), _fk("users"), nullable=False),
    Column("reported", String(36), _fk("users"), nullable=False),
    Column("reason", Text, nullable=False),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

listings = Table(
    "listings", meta,
    Column("id", String(36), primary_key=True),
    Column("collaborator_id", String(36), _fk("users"), nullable=False),
    Column("name", String(120), nullable=False),
    Column("kind", String(10), nullable=False),
    Column("city", String(80), nullable=False),
    Column("lat", Float, nullable=False),
    Column("lon", Float, nullable=False),
    Column("category", String(120)),
    Column("description", Text),
    Column("cost_inr", Float),
    Column("evaluation", Text, nullable=False),        # JSON produced by the model
    Column("decision", String(20), nullable=False),    # hidden_gem | not_a_gem | needs_reviews
    Column("published", Integer, nullable=False, server_default=text("0")),
    Column("review_status", String(20), nullable=False, server_default=text("'model_reviewed'")),
    Column("reviewed_by", String(36)),
    Column("reviewed_at", String(32)),
    Column("photo_path", Text),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("kind IN ('eat', 'visit')", name="ck_listing_kind"),
)

# dynamic trip planner: a road trip with a time window and the stops the user picked
itineraries = Table(
    "itineraries", meta,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), _fk("users"), nullable=False),
    Column("origin", String(80), nullable=False),
    Column("destination", String(80), nullable=False),
    Column("depart_at", String(16), nullable=False),     # local time 'YYYY-MM-DDTHH:MM'
    Column("arrive_by", String(16), nullable=False),
    Column("delay_min", Integer, nullable=False, server_default=text("0")),
    Column("route", Text, nullable=False),               # JSON: distance, duration, source, simplified geometry
    Column("planning_data", Text),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

listing_views = Table(
    "listing_views", meta,
    Column("listing_id", String(36), _fk("listings"), primary_key=True),
    Column("viewer_id", String(36), _fk("users"), primary_key=True),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

listing_ratings = Table(
    "listing_ratings", meta,
    Column("listing_id", String(36), _fk("listings"), primary_key=True),
    Column("user_id", String(36), _fk("users"), primary_key=True),
    Column("score", Integer, nullable=False),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("score BETWEEN 1 AND 5", name="ck_listing_rating_score"),
)

collaborator_documents = Table(
    "collaborator_documents", meta,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), _fk("users"), nullable=False),
    Column("doc_type", String(30), nullable=False),
    Column("file_path", Text, nullable=False),
    Column("latitude", Float),
    Column("longitude", Float),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

itinerary_stops = Table(
    "itinerary_stops", meta,
    Column("id", String(36), primary_key=True),
    Column("itinerary_id", String(36), _fk("itineraries"), nullable=False),
    Column("place_id", String(200), nullable=False),
    Column("details", Text, nullable=False),             # JSON snapshot: name, kind, lat/lon, along_km, visit_min ...
    Column("priority", Integer, nullable=False, server_default=text("0")),
    Column("created_at", String(32), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("itinerary_id", "place_id", name="uq_stop_place"),
)


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def database_url() -> str:
    return os.environ.get("DATABASE_URL") or f"sqlite:///{os.environ.get('SIDEQUEST_DB', DB_PATH)}"


_engines: Dict[str, Engine] = {}


def engine() -> Engine:
    url = database_url()
    if url not in _engines:
        if url.startswith("sqlite"):
            eng = create_engine(url, connect_args={"check_same_thread": False, "timeout": 15})

            @event.listens_for(eng, "connect")
            def _sqlite_pragmas(dbapi_con, _):
                cur = dbapi_con.cursor()
                cur.execute("PRAGMA foreign_keys = ON")
                cur.execute("PRAGMA journal_mode = WAL")
                cur.close()
        else:
            eng = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
        _engines[url] = eng
    return _engines[url]


def dialect() -> str:
    return engine().dialect.name


def init() -> None:
    eng = engine()
    meta.create_all(eng)
    # create_all does not add columns to existing databases; preserve old profiles while
    # adding multi-select trip styles for both SQLite and PostgreSQL installations.
    with eng.begin() as conn:
        columns = {col["name"] for col in inspect(conn).get_columns("preferences")}
        if "travel_styles" not in columns:
            conn.execute(text("ALTER TABLE preferences ADD COLUMN travel_styles TEXT"))
        rows = conn.execute(text("SELECT user_id, travel_style FROM preferences WHERE travel_styles IS NULL")).mappings()
        for row in rows:
            legacy = row["travel_style"]
            try:
                parsed = json.loads(legacy) if legacy else []
                styles = parsed if isinstance(parsed, list) else [legacy]
            except (TypeError, ValueError):
                styles = [legacy] if legacy else []
            conn.execute(text("UPDATE preferences SET travel_styles=:styles WHERE user_id=:uid"),
                         {"styles": json.dumps(styles), "uid": row["user_id"]})
        for table, additions in {
            "itineraries": {"planning_data": "TEXT"},
            "listings": {"review_status": "VARCHAR(20) NOT NULL DEFAULT 'model_reviewed'", "reviewed_by": "VARCHAR(36)",
                         "reviewed_at": "VARCHAR(32)", "photo_path": "TEXT"},
        }.items():
            existing = {col["name"] for col in inspect(conn).get_columns(table)}
            for name, definition in additions.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))


# ---------------------------------------------------------------------------
# thin connection wrapper: '?' placeholders, dict rows, consistent timestamps
# ---------------------------------------------------------------------------
_Q = re.compile(r"\?")


def _clean(v: Any) -> Any:
    if isinstance(v, _dt.datetime):
        return v.isoformat(sep=" ", timespec="seconds")
    return v


class Result:
    def __init__(self, res):
        self._res = res
        self.rowcount = res.rowcount

    def fetchone(self) -> Optional[Dict]:
        r = self._res.mappings().fetchone() if self._res.returns_rows else None
        return None if r is None else {k: _clean(v) for k, v in r.items()}

    def fetchall(self) -> List[Dict]:
        if not self._res.returns_rows:
            return []
        return [{k: _clean(v) for k, v in r.items()} for r in self._res.mappings().fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class Con:
    def __init__(self, conn):
        self._c = conn

    def execute(self, sql: str, params: tuple | list | dict = ()) -> Result:
        if isinstance(params, dict):
            return Result(self._c.execute(text(sql), params))
        names = {}
        n = [0]

        def repl(_m):
            k = f"p{n[0]}"
            names[k] = params[n[0]]
            n[0] += 1
            return f":{k}"
        stmt = _Q.sub(repl, sql)
        if n[0] != len(params):
            raise ValueError(f"expected {n[0]} params, got {len(params)}")
        return Result(self._c.execute(text(stmt), names))


@contextmanager
def session() -> Iterator[Con]:
    with engine().begin() as conn:      # commits on success, rolls back on error
        yield Con(conn)
