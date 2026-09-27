"""GSTIN check (Goods and Services Tax Identification Number, India).

What this CAN confirm offline: the 15-character structure, a valid state code, the
embedded PAN pattern, and the official mod-36 check digit - which catches typos and
made-up numbers. What it CANNOT confirm: that the GSTIN is currently active and
belongs to this business. That needs the GST portal / a GST Suvidha Provider API, or
an admin checking the uploaded registration certificate - so a valid format only
moves a collaborator to "pending review", never straight to "verified".
"""
from __future__ import annotations

import re
from typing import Dict

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STATE_CODES = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra and Nagar Haveli and Daman and Diu", "27": "Maharashtra", "29": "Karnataka",
    "30": "Goa", "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman and Nicobar Islands", "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
    "97": "Other Territory",
}
PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def check_digit(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        v = CHARS.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return CHARS[(36 - total % 36) % 36]


def validate(gstin: str) -> Dict:
    g = (gstin or "").strip().upper()
    if not g:
        return {"status": "not_provided"}
    if not PATTERN.match(g):
        return {"status": "invalid", "reason": "Must be 15 characters: 2-digit state code, 10-character PAN, "
                                               "entity digit, 'Z', check character"}
    if g[:2] not in STATE_CODES:
        return {"status": "invalid", "reason": f"Unknown state code {g[:2]}"}
    if check_digit(g[:14]) != g[14]:
        return {"status": "invalid", "reason": "Check character does not match - likely a typo"}
    return {"status": "valid_format", "gstin": g, "state": STATE_CODES[g[:2]], "pan": g[2:12],
            "note": "Format and check digit are correct. Registration status still needs admin review "
                    "of the certificate (or a GST API lookup)."}
