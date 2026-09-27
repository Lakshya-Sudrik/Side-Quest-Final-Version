"""Backend settings (all overridable with environment variables)."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(os.environ.get("SIDEQUEST_ROOT", Path(__file__).resolve().parent.parent))
APP_DIR = Path(os.environ.get("SIDEQUEST_APP_DIR", ROOT / "data" / "app"))
APP_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.environ.get("SIDEQUEST_DB", APP_DIR / "sidequest.db"))
UPLOAD_DIR = Path(os.environ.get("SIDEQUEST_UPLOADS", APP_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = 8
ALLOWED_DOC_TYPES = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png"}

# Emails listed here get the admin role at registration (they approve collaborators).
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("SIDEQUEST_ADMIN_EMAILS", "").split(",") if e.strip()}

JWT_TTL_HOURS = int(os.environ.get("SIDEQUEST_JWT_TTL_HOURS", "72"))


def jwt_secret() -> str:
    """From SIDEQUEST_JWT_SECRET, else a random secret stored once in data/app (never committed)."""
    env = os.environ.get("SIDEQUEST_JWT_SECRET")
    if env:
        return env
    f = APP_DIR / ".jwt_secret"
    if not f.exists():
        f.write_text(secrets.token_hex(32), encoding="utf-8")
        try:
            f.chmod(0o600)
        except OSError:
            pass
    return f.read_text(encoding="utf-8").strip()


# Solo matchmaking rules
SAME_DESTINATION_KM = 25.0      # two trips count as "same destination" within this distance
MIN_OVERLAP_DAYS = 1
MESSAGE_MAX_CHARS = 1000
MESSAGES_PER_MINUTE = 20
