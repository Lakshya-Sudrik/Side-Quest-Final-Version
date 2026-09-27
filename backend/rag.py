"""Retrieval over every restaurant and sight we score, stored in ChromaDB.

    python -m backend.rag --build        # (re)build data/chroma_db (~5 min)
    python -m backend.rag --eval         # retrieval accuracy on auto-generated queries
    python -m backend.rag --q "quiet waterfall near Munnar"

Documents : 70,724 restaurants + 15,239 sights. Each carries the CURRENT honest scores
            (quality rank, how well known, hidden-gem / famous-and-good flags from the same
            rules as the route recommender) and, for sights, two real visitor review snippets.
Embeddings: sentence-transformers all-MiniLM-L6-v2 when it is installed (set
            RAG_EMBEDDER=minilm), otherwise a local LSA embedder (TF-IDF 1-2 grams -> 256-d
            SVD) that needs no download. The index records which one built it; queries
            always use the same one.
Search    : semantic top-N from Chroma (with city / type filters, and the city is also
            detected from the query text) -> re-ranked 70% similarity + 30% quality ->
            safety block for each result -> an answer built ONLY from retrieved facts
            (no text generator, so nothing can be invented).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ROOT

sys.path.insert(0, str(ROOT / "src"))
CHROMA_DIR = Path(os.environ.get("SIDEQUEST_CHROMA", ROOT / "data" / "chroma_db"))
EMBEDDER_PATH = ROOT / "models" / "rag_embedder.pkl"
COLLECTION = "sidequest_places_v2"
DIM = 256


# ---------------------------------------------------------------------------
# embedders
# ---------------------------------------------------------------------------
class LSAEmbedder:
    kind = "lsa_tfidf_svd256"

    def fit(self, texts: List[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(max_features=80000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                   dtype=np.float32)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=DIM, random_state=42).fit(X)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        Z = self.svd.transform(self.vec.transform(texts)).astype(np.float32)
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


class MiniLMEmbedder:
    kind = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("all-MiniLM-L6-v2")

    def fit(self, texts):
        return self

    def encode(self, texts):
        return self.m.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def city_key(c: str) -> str:
    import geo_india as gi
    n = gi.norm(c)
    return gi.norm(gi.CITY_ALIASES.get(n, n))


def _snippets() -> Dict[str, List[str]]:
    """Two real review snippets per sight (longest informative ones, trimmed)."""
    cache = ROOT / "data" / "attractions" / "_reviews_cache.pkl"
    small = ROOT / "data" / "attractions" / "review_snippets.json.gz"
    if not cache.exists():
        # the raw review dump is ~440 MB; a pre-extracted snippet file ships with the code
        if small.exists():
            import gzip
            with gzip.open(small, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        return {}
    r = pd.read_pickle(cache)[["place_id", "Raw_Review", "words"]]
    r = r[(r["words"] >= 15) & (r["words"] <= 120)]
    r = r.sort_values("words", ascending=False).groupby("place_id").head(2)
    r["s"] = r["Raw_Review"].astype(str).str.replace(r"\s+", " ", regex=True).str[:240]
    return r.groupby("place_id")["s"].apply(list).to_dict()


def build_documents() -> pd.DataFrame:
    import recommender as rec
    c = rec.load_candidates().copy()
    c = c[c["lat"].notna()]
    # shops, agencies and tour operators are not places to visit (same rule as the recommender)
    c = c[~((c["kind"] == "visit") & c["name"].str.lower().str.contains(rec.NOT_A_SIGHT, regex=True))]
    enough = c["n_reviews"] >= c["kind"].map(rec.MIN_REVIEWS)
    c["is_hidden"] = (enough & (c["quality_rank"] >= rec.GEM_RANK_MIN) & (c["exposure_pct"] <= rec.HIDDEN_MAX_EXPOSURE)
                      & (c["chain_outlets"] <= rec.MAX_CHAIN_OUTLETS))
    c["is_famous"] = enough & (c["quality_rank"] >= rec.GEM_RANK_MIN) & (c["exposure_pct"] >= rec.FAMOUS_MIN_EXPOSURE)
    food = pd.read_csv(rec.FOOD, usecols=["place_id", "area", "cuisines"])
    food["id"] = "food:" + food["place_id"].astype(str)
    c = c.merge(food[["id", "area", "cuisines"]], on="id", how="left")
    snips = _snippets()
    docs = []
    for r in c.itertuples(index=False):
        status = ("hidden gem" if r.is_hidden else "famous and genuinely good" if r.is_famous else
                  "well rated" if r.quality_rank >= 60 else "")
        if r.kind == "eat":
            text = (f"{r.name}. Restaurant / place to eat in {r.area}, {r.city}. {r.category}. "
                    f"Cuisines: {str(r.cuisines).strip('[]').replace(chr(39), '')}. "
                    f"{'Cost for two about ' + str(int(r.cost_inr)) + ' rupees. ' if pd.notna(r.cost_inr) else ''}"
                    f"{status}")
        else:
            pid = r.id.split(":", 1)[1]
            sn = " ".join(snips.get(pid, []))
            text = f"{r.name}. Place to visit in {r.city}: {str(r.category).replace('_', ' ')}. {status}. {sn}"
        docs.append(text)
    c["document"] = docs
    c["city_key"] = c["city"].map(city_key)
    return c


# ---------------------------------------------------------------------------
# build / load
# ---------------------------------------------------------------------------
def _client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def build() -> Dict:
    t0 = time.time()
    d = build_documents()
    emb = MiniLMEmbedder() if os.environ.get("RAG_EMBEDDER") == "minilm" else LSAEmbedder()
    emb.fit(d["document"].tolist())
    if isinstance(emb, LSAEmbedder):
        with open(EMBEDDER_PATH, "wb") as fh:
            pickle.dump(emb, fh)
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "embedder": emb.kind,
                                                         "built_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    B = 4000
    for s in range(0, len(d), B):
        part = d.iloc[s:s + B]
        vecs = emb.encode(part["document"].tolist())
        col.add(ids=part["id"].tolist(), embeddings=vecs.tolist(), documents=part["document"].tolist(),
                metadatas=[{
                    "name": str(r.name), "kind": r.kind, "city": str(r.city), "city_key": r.city_key,
                    "category": str(r.category), "lat": float(r.lat), "lon": float(r.lon),
                    "rating": float(r.rating) if pd.notna(r.rating) else -1.0,
                    "reviews": int(r.n_reviews) if pd.notna(r.n_reviews) else 0,
                    "quality_rank": round(float(r.quality_rank), 1), "exposure_pct": round(float(r.exposure_pct), 3),
                    "is_hidden": bool(r.is_hidden), "is_famous": bool(r.is_famous),
                    "location_precision": str(r.location_precision),
                } for r in part.itertuples(index=False)])
    return {"documents": len(d), "embedder": emb.kind, "seconds": round(time.time() - t0, 1)}


@lru_cache(maxsize=1)
def _load():
    col = _client().get_collection(COLLECTION)
    kind = (col.metadata or {}).get("embedder", LSAEmbedder.kind)
    if kind == MiniLMEmbedder.kind:
        emb = MiniLMEmbedder()
    else:
        with open(EMBEDDER_PATH, "rb") as fh:
            emb = pickle.load(fh)
    return col, emb


_NAME_STOP = {"the", "and", "of", "in", "at", "a", "restaurant", "cafe", "hotel", "temple", "point", "near",
              "best", "food", "place", "places", "visit", "sri", "shri", "new"}


@lru_cache(maxsize=1)
def _name_index() -> Dict[str, List[str]]:
    """token -> ids of places whose name contains it (for exact-name lookups)."""
    col, _ = _load()
    idx: Dict[str, List[str]] = {}
    n, step = col.count(), 5000
    for off in range(0, n, step):                 # paged: SQLite limits variables per query
        got = col.get(include=["metadatas"], limit=step, offset=off)
        for i, md in zip(got["ids"], got["metadatas"]):
            for t in set(re.findall(r"[a-z0-9]+", md["name"].lower())) - _NAME_STOP:
                if len(t) >= 3:
                    idx.setdefault(t, []).append(i)
    return idx


def _get_many(ids: List[str], include: List[str]) -> Dict[str, list]:
    col, _ = _load()
    out = {"ids": [], **{k: [] for k in include}}
    for s in range(0, len(ids), 500):
        g = col.get(ids=ids[s:s + 500], include=include)
        out["ids"] += g["ids"]
        for k in include:
            out[k] += list(g[k])
    return out


def _name_hits(query: str, city_key: Optional[str], limit: int = 20) -> List[str]:
    """ids of places whose distinctive name words all appear in the query."""
    col, _ = _load()
    q = set(re.findall(r"[a-z0-9]+", query.lower())) - _NAME_STOP
    lists = [set(_name_index().get(t, [])) for t in q if len(t) >= 3]
    cand = set().union(*lists) if lists else set()
    if not cand or len(cand) > 5000:
        return []
    got = _get_many(list(cand), ["metadatas"])
    out = []
    for i, md in zip(got["ids"], got["metadatas"]):
        nt = set(re.findall(r"[a-z0-9]+", md["name"].lower())) - _NAME_STOP
        if nt and nt <= q and (not city_key or md["city_key"] == city_key):
            out.append(i)
    return out[:limit]


@lru_cache(maxsize=1)
def _city_keys() -> Dict[str, str]:
    import recommender as rec
    c = rec.load_candidates()
    keys = {}
    for city in c["city"].dropna().unique():
        k = city_key(city)
        if len(k) >= 4:
            keys[k] = k
    return keys


def detect_city(query: str) -> Optional[str]:
    import geo_india as gi
    q = " " + gi.norm(query) + " "
    best = None
    for k in _city_keys():
        if f" {k} " in q and (best is None or len(k) > len(best)):
            best = k
    for alias, target in gi.CITY_ALIASES.items():
        if f" {gi.norm(alias)} " in q:
            best = gi.norm(target)
    return best


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def search(query: str, top_k: int = 5, city: str = "", kind: str = "", only_hidden: bool = False,
           with_safety: bool = True) -> Dict:
    col, emb = _load()
    ck = city_key(city) if city else detect_city(query)
    conds = []
    if ck:
        conds.append({"city_key": ck})
    if kind in ("eat", "visit"):
        conds.append({"kind": kind})
    if only_hidden:
        conds.append({"is_hidden": True})
    where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
    qv = emb.encode([query])[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=max(top_k * 6, 30), where=where,
                    include=["documents", "metadatas", "distances"])
    # hybrid: add places the query names outright (keyword match), scored with the same formula
    ids0 = set(res["ids"][0])
    extra = [i for i in _name_hits(query, ck) if i not in ids0]
    if kind in ("eat", "visit"):
        extra = [i for i in extra if i.startswith("food:" if kind == "eat" else "visit:")]
    if extra:
        g = _get_many(extra, ["documents", "metadatas", "embeddings"])
        qarr = np.asarray(qv)
        for i, md, doc, e in zip(g["ids"], g["metadatas"], g["documents"], g["embeddings"]):
            if only_hidden and not md.get("is_hidden"):
                continue
            res["ids"][0].append(i); res["metadatas"][0].append(md); res["documents"][0].append(doc)
            res["distances"][0].append(1.0 - float(np.dot(qarr, np.asarray(e))))
    items = []
    qtok = set(re.findall(r"[a-z0-9]+", query.lower()))
    for i, md, doc, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]):
        sim = 1.0 - float(dist)
        ntok = set(re.findall(r"[a-z0-9]+", md["name"].lower()))
        name_match = len(qtok & ntok) / len(ntok) if ntok else 0.0   # the query names this place
        score = 0.55 * sim + 0.25 * name_match + 0.20 * md["quality_rank"] / 100
        items.append({"id": i, "similarity": round(sim, 4), "name_match": round(name_match, 2),
                      "score": round(score, 4), **md, "document": doc})
    items.sort(key=lambda x: -x["score"])
    seen, uniq = set(), []
    for x in items:                      # one result per name (chains have many branches)
        k = re.sub(r"[^a-z0-9]", "", x["name"].lower())
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    items = uniq[:top_k]
    if with_safety and items:
        import recommender as rec
        frame = pd.DataFrame([{"id": x["id"], "name": x["name"], "kind": x["kind"], "city": x["city"],
                               "lat": x["lat"], "lon": x["lon"], "n_reviews": x["reviews"],
                               "tx_unsafe": np.nan, "tx_safe": np.nan} for x in items])
        for x, b in zip(items, rec.safety_blocks(frame)):
            x["safety"] = {"class": b["class"], "score": b["score"], "reasons": b["reasons"][:3]}
    return {"query": query, "city_filter": ck, "kind_filter": kind or None, "results": items,
            "answer": grounded_answer(query, items, ck)}


def grounded_answer(query: str, items: List[Dict], ck: Optional[str]) -> str:
    """A short answer that only restates retrieved facts."""
    if not items:
        return f"No matching places found{' in ' + ck.title() if ck else ''}."
    lines = [f"Top matches for '{query}'{' in ' + ck.title() if ck else ''}:"]
    for i, x in enumerate(items, 1):
        tag = "hidden gem" if x["is_hidden"] else "famous & good" if x["is_famous"] else f"quality {x['quality_rank']:.0f}/100"
        rating = f"{x['rating']:.1f}★ from {x['reviews']} reviews" if x["rating"] > 0 else "no rating yet"
        safe = f", safety: {x['safety']['class']}" if "safety" in x else ""
        lines.append(f"{i}. {x['name']} ({'eat' if x['kind'] == 'eat' else 'visit'}, {x['city']}) - {tag}, {rating}{safe}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# evaluation (auto-generated queries with known answers)
# ---------------------------------------------------------------------------
def evaluate(n: int = 300, seed: int = 7) -> Dict:
    d = build_documents()
    rng = np.random.default_rng(seed)
    out = {}
    # 1. find a known place by name + town
    s = d.sample(n, random_state=seed)
    exact, same_name = [], []
    for r in s.itertuples(index=False):
        res = search(f"{r.name} {r.city}", top_k=5, with_safety=False)["results"]
        exact.append(r.id in [x["id"] for x in res])
        key = re.sub(r"[^a-z0-9]", "", r.name.lower())
        same_name.append(any(re.sub(r"[^a-z0-9]", "", x["name"].lower()) == key and x["city_key"] == r.city_key
                             for x in res))
    out["name_lookup_hit_at_5"] = round(float(np.mean(same_name)), 3)
    out["name_lookup_exact_branch_hit_at_5"] = round(float(np.mean(exact)), 3)
    # 2. "<type> in <town>": share of top-5 in the right town and of the right type
    types = {"temple": "temple", "water": "lake", "fort_palace": "fort", "hill_view": "viewpoint", "museum": "museum"}
    sights = d[(d["kind"] == "visit") & d["category"].isin(types) & (d["city_key"].str.len() >= 4)]
    sights = sights[sights.groupby("city_key")["id"].transform("size") >= 10]
    precs, city_ok = [], []
    for r in sights.sample(min(n, len(sights)), random_state=seed).itertuples(index=False):
        res = search(f"{types[r.category]} in {r.city}", top_k=5, with_safety=False)["results"]
        if res:
            city_ok.append(np.mean([x["city_key"] == r.city_key for x in res]))
            precs.append(np.mean([x["category"] == r.category for x in res]))
    out["type_in_town_precision_at_5"] = round(float(np.mean(precs)), 3)
    out["town_filter_correct"] = round(float(np.mean(city_ok)), 3)
    # 3. cuisine in town (restaurants)
    food = d[(d["kind"] == "eat") & d["city_key"].isin(d["city_key"].value_counts()[lambda v: v >= 200].index)]
    pc = []
    for r in food.sample(min(n, len(food)), random_state=seed).itertuples(index=False):
        cu = str(r.category).lower()
        res = search(f"{cu} food in {r.city}", top_k=5, kind="eat", with_safety=False)["results"]
        if res:
            pc.append(np.mean([cu in x["document"].lower() for x in res]))
    out["cuisine_in_town_precision_at_5"] = round(float(np.mean(pc)), 3)
    out["n_queries_each"] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--q")
    ap.add_argument("--city", default="")
    a = ap.parse_args()
    if a.build:
        print(json.dumps(build(), indent=1))
    if a.eval:
        rep = evaluate()
        print(json.dumps(rep, indent=1))
        mp = ROOT / "reports" / "metrics.json"
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"tasks": {}}
        m.setdefault("rag", {}).update({**rep, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        mp.write_text(json.dumps(m, indent=2, default=float), encoding="utf-8")
    if a.q:
        r = search(a.q, city=a.city)
        print(r["answer"])


if __name__ == "__main__":
    # run through the importable module so the saved embedder is backend.rag.LSAEmbedder
    from backend import rag as _rag
    _rag.main()
