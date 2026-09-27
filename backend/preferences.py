"""The 5 sign-up preferences and the solo-matching compatibility score built from them.

The score is a transparent rule, not a trained model: there is no data yet on which
travel pairs worked out. Every point comes with a reason the user can read. Once real
outcomes exist (both accepted, travelled together, rated each other) a model can be
trained on them.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

OPTIONS: Dict[str, Dict] = {
    "travel_style": {"question": "What kind of trips do you enjoy most?",
                     "values": ["adventure", "culture_heritage", "food", "nature_wildlife", "relaxation"],
                     "multi": True, "max": 5},
    "budget": {"question": "Your usual budget per day?",
               "values": ["budget", "mid_range", "premium"]},
    "pace": {"question": "How do you like to travel?",
             "values": ["slow", "balanced", "packed"]},
    "food": {"question": "Food preference?",
             "values": ["vegetarian", "eggetarian", "non_vegetarian", "vegan", "jain"]},
    "languages": {"question": "Languages you can chat in (pick up to 4)",
                  "values": ["english", "hindi", "kannada", "tamil", "telugu", "malayalam", "marathi",
                             "bengali", "gujarati", "punjabi", "odia", "urdu"], "multi": True, "max": 4},
}
COMPANION = {"companion_gender": ["any", "same"], "age_min": 18, "age_max": 99}
_ORDER = {"budget": ["budget", "mid_range", "premium"], "pace": ["slow", "balanced", "packed"]}
_VEG = {"vegetarian", "vegan", "jain", "eggetarian"}
WEIGHTS = {"travel_style": 0.30, "budget": 0.20, "pace": 0.20, "food": 0.15, "languages": 0.15}


def validate(p: Dict) -> Tuple[Dict, List[str]]:
    errs, out = [], {}
    for key, spec in OPTIONS.items():
        v = p.get(key)
        if spec.get("multi"):
            vals = [v] if isinstance(v, str) else [x for x in (v or []) if isinstance(x, str)]
            bad = [x for x in vals if x not in spec["values"]]
            if not vals:
                errs.append(f"{key}: pick at least one")
            elif bad:
                errs.append(f"{key}: unknown value(s) {bad}")
            elif len(vals) > spec["max"]:
                errs.append(f"{key}: at most {spec['max']}")
            out[key] = sorted(set(vals))
        else:
            if v not in spec["values"]:
                errs.append(f"{key}: must be one of {spec['values']}")
            out[key] = v
    cg = p.get("companion_gender", "any")
    if cg not in COMPANION["companion_gender"]:
        errs.append("companion_gender: any | same")
    amin, amax = int(p.get("companion_age_min", 18)), int(p.get("companion_age_max", 99))
    if not (18 <= amin <= amax <= 99):
        errs.append("companion age range must be within 18-99 and min <= max")
    out.update(companion_gender=cg, companion_age_min=amin, companion_age_max=amax)
    return out, errs


def compatibility(a: Dict, b: Dict) -> Tuple[int, List[str]]:
    """0-100 score + human-readable reasons, symmetric in a and b."""
    parts, reasons = {}, []
    def styles(value):
        if isinstance(value, str):
            try:
                decoded = __import__("json").loads(value)
                value = decoded if isinstance(decoded, list) else [value]
            except (ValueError, TypeError):
                value = [value]
        return set(value or [])
    styles_a, styles_b = styles(a["travel_style"]), styles(b["travel_style"])
    shared_styles = sorted(styles_a & styles_b)
    parts["travel_style"] = len(shared_styles) / len(styles_a | styles_b) if styles_a | styles_b else 0.0
    if shared_styles:
        reasons.append(f"Shared trip styles: {', '.join(style.replace('_', ' ') for style in shared_styles)}")
    for k in ("budget", "pace"):
        d = abs(_ORDER[k].index(a[k]) - _ORDER[k].index(b[k]))
        parts[k] = {0: 1.0, 1: 0.5}.get(d, 0.0)
        if d == 0:
            reasons.append(f"Same {k.replace('_', ' ')} ({a[k].replace('_', ' ')})")
        elif d == 2:
            reasons.append(f"Different {k}: {a[k].replace('_', ' ')} vs {b[k].replace('_', ' ')}")
    if a["food"] == b["food"]:
        parts["food"] = 1.0
        reasons.append(f"Same food preference ({a['food'].replace('_', ' ')})")
    elif (a["food"] in _VEG) == (b["food"] in _VEG):
        parts["food"] = 0.7
    else:
        parts["food"] = 0.4
        reasons.append("One vegetarian, one non-vegetarian - worth discussing meals")
    shared = sorted(set(a["languages"]) & set(b["languages"]))
    parts["languages"] = 1.0 if shared else 0.0
    reasons.append(f"Shared language: {', '.join(shared)}" if shared else "No shared language")
    score = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()))
    return int(score), reasons
