"""Passwords (scrypt, salted) and login tokens (JWT, HS256)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from typing import Dict

import jwt

from .config import JWT_TTL_HOURS, jwt_secret

_N, _R, _P = 2 ** 14, 8, 1


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(base64.b64decode(dk)))
        return hmac.compare_digest(got, base64.b64decode(dk))
    except (ValueError, TypeError):
        return False


def password_problem(password: str) -> str | None:
    if len(password) < 8:
        return "Password must be at least 8 characters"
    if password.isdigit() or password.isalpha():
        return "Password must mix letters and numbers"
    return None


def make_token(user_id: str, role: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user_id, "role": role, "iat": now, "exp": now + JWT_TTL_HOURS * 3600},
                      jwt_secret(), algorithm="HS256")


def read_token(token: str) -> Dict:
    return jwt.decode(token, jwt_secret(), algorithms=["HS256"])
