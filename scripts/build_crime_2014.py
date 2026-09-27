"""District crime rates (NCRB 2014, official) -> data/safety/district_crime.csv

    python scripts/build_crime_2014.py

Inputs (data/raw/india/):
    ipc_2014.csv                     NCRB "District-wise crimes committed under IPC 2014"
    india-districts-census-2011.csv  population by district (Census 2011)

Rates are per 100,000 people. Police units that are not places (railway
police, CID, crime branch ...) are dropped. Several NCRB units of one city
(e.g. 'Bengaluru Commr.' + 'Bengaluru Rural') are added up per census district.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "india"
OUT = ROOT / "data" / "safety" / "district_crime.csv"

# serious crimes only: minor categories (hurt, petty theft) mostly measure how readily
# a state registers FIRs (Kerala registers far more), not how dangerous it is
VIOLENT = ["Murder", "Attempt to commit Murder", "Rape", "Kidnapping & Abduction_Total", "Robbery",
           "Dacoity"]
WOMEN = ["Rape", "Attempt to commit Rape", "Assault on Women with intent to outrage her Modesty",
         "Insult to the Modesty of Women", "Acid attack"]
PROPERTY = ["Theft", "Robbery", "Criminal Trespass/Burglary"]
# NCRB state label -> (census 2011 state, district-map state)
STATES = {"A&N Islands": ("andaman and nicobar islands", "Andaman and Nicobar Islands"),
          "D&N Haveli": ("dadra and nagar haveli", "Dadra and Nagar Haveli and Daman and Diu"),
          "Daman & Diu": ("daman and diu", "Dadra and Nagar Haveli and Daman and Diu"),
          "Delhi UT": ("nct of delhi", "Delhi"), "Jammu & Kashmir": ("jammu and kashmir", "Jammu and Kashmir"),
          "Odisha": ("orissa", "Odisha"), "Puducherry": ("pondicherry", "Puducherry"),
          "Telangana": ("andhra pradesh", "Telangana")}
NOT_A_PLACE = r"\b(g\.?r\.?p|grp|railway|rly|cid|crime branch|stf|eow|special|traffic|cyber|total|bureau|vigilance)\b"
SUFFIX = r"\b(commr|commissionerate|commissioner|city|urban|rural|dist|district|police)\b"
ALIAS = {"bengaluru": "bangalore", "mysuru": "mysore", "gurugram": "gurgaon", "belagavi": "belgaum",
         "kalaburagi": "gulbarga", "vijayapura": "bijapur", "shivamogga": "shimoga", "tumakuru": "tumkur",
         "ballari": "bellary", "prayagraj": "allahabad", "thiruvananthapuram": "trivandrum",
         "kozhikode": "calicut", "puducherry": "pondicherry", "cuddapah": "ysr", "kadapa": "ysr",
         "hooghly": "hugli", "howrah": "haora", "vizag": "visakhapatnam", "ahmadabad": "ahmedabad"}


def key(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    for a, b in ALIAS.items():
        s = s.replace(a, b)
    s = re.sub(SUFFIX, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def skey(s: str) -> str:
    s = re.sub(r"[^a-z ]", " ", str(s).lower().replace("&", "and"))
    return re.sub(r"\s+", " ", s).strip()


def build() -> pd.DataFrame:
    c = pd.read_csv(RAW / "ipc_2014.csv")
    c = c[~c["District"].astype(str).str.lower().str.contains(NOT_A_PLACE, regex=True)]
    for col in set(VIOLENT + WOMEN + PROPERTY):
        c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0)
    c["violent"] = c[VIOLENT].sum(axis=1)
    c["women"] = c[WOMEN].sum(axis=1)
    c["property"] = c[PROPERTY].sum(axis=1)
    c["st"] = c["States/UTs"].map(lambda x: STATES.get(x, (skey(x), x))[0])
    c["States/UTs"] = c["States/UTs"].map(lambda x: STATES.get(x, (None, x))[1])
    c["dk"] = c["District"].map(key)
    agg = c.groupby(["st", "dk"]).agg(state=("States/UTs", "first"), district=("District", "first"),
                                      violent=("violent", "sum"), women=("women", "sum"),
                                      property=("property", "sum")).reset_index()

    cen = pd.read_csv(RAW / "india-districts-census-2011.csv")
    cen["st"] = cen["State name"].map(skey)
    cen["dk"] = cen["District name"].map(key)
    pop = cen.groupby(["st", "dk"])["Population"].sum()

    def find_pop(st, dk):
        if (st, dk) in pop.index:
            return pop[(st, dk)]
        if st in pop.index.get_level_values(0):
            sub = pop.loc[st]
            for k, v in sub.items():
                if dk and k and (dk in k or k in dk or dk[:5] == k[:5]):
                    return v
        return np.nan

    agg["population"] = [find_pop(s, d) for s, d in zip(agg["st"], agg["dk"])]
    ok = agg["population"].notna() & (agg["population"] > 0)
    for k in ("violent", "women", "property"):
        agg[f"{k}_rate"] = (agg[k] / agg["population"] * 1e5).where(ok)
    # states where no police unit matched a census district (Delhi's units are
    # police zones) -> one state-wide row, so the state still gets real numbers
    state_pop = cen.groupby("st")["Population"].sum()
    extra = []
    for st, g in agg.groupby("st"):
        if ok[g.index].any() or st not in state_pop.index:
            continue
        p = float(state_pop[st])
        extra.append({"state": g["state"].iloc[0], "district": g["state"].iloc[0], "population": p,
                      **{f"{k}_rate": g[k].sum() / p * 1e5 for k in ("violent", "women", "property")}})
    out = agg.loc[ok, ["state", "district", "population", "violent_rate", "property_rate", "women_rate"]]
    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
        print("state-wide rows for:", [e["state"] for e in extra])
    out = out.assign(source="NCRB Crime in India 2014 (district-wise IPC) / Census 2011")
    print(f"NCRB units kept: {len(agg)}  with population match: {int(ok.sum())} ({ok.mean():.0%})")
    return out


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} districts, {df['state'].nunique()} states)")
    sys.path.insert(0, str(ROOT / "src"))
    from safety_index import SafetyIndex
    si = SafetyIndex.load(refresh=True)
    for name, lat, lon in (("Bangalore", 12.97, 77.59), ("Delhi", 28.63, 77.22), ("Manali", 32.24, 77.19),
                           ("Hampi", 15.335, 76.46), ("Munnar", 10.088, 77.06)):
        s = si.score_point(lat, lon, name)
        print(f"  {name:10s} safety={s['safety_score']} level={s['level']} crime={s['components']['crime']} "
              f"district={s['district']} ({s['crime_resolution']})")
