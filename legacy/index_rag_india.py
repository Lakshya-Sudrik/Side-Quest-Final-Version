"""PHASE 6: Re-index India places into the RAG vector store.

Batch-encodes place documents (234 docs/s measured -> ~7 min for 102K)
into offline/chroma_db (persistent). Documents include name/city/category/
rating/price; metadata carries hidden_gem_score + safety_score for RAG-aware
recommendations.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("index_rag")


def build_document(r: pd.Series) -> str:
    parts = [
        f"Name: {r['name']}",
        f"City: {r['city']}",
        f"Area: {r['area']}",
        f"Category: {r['primary_category']}",
        f"Cuisines: {r['categories']}",
        f"Rating: {r['stars']} stars",
        f"Reviews: {int(r['review_count'])}",
        f"Price level: {int(r['price_level'])}",
        f"Hidden gem score: {int(r['hidden_gem_score'])}",
        f"Safety score: {int(r['safety_score'])}",
    ]
    if r.get("cost_inr") and pd.notna(r.get("cost_inr")):
        parts.append(f"Approx cost: {int(r['cost_inr'])} INR")
    return " | ".join(parts)


def main() -> None:
    from rag_pipeline import EmbeddingGenerator, VectorStore

    logger.info("=" * 60)
    logger.info("PHASE 6: RAG INDEXING (INDIA PLACES)")
    logger.info("=" * 60)

    df = pd.read_csv(ROOT / "data" / "india_places.csv")
    logger.info(f"  {len(df)} places loaded")

    vs = VectorStore(str(ROOT / "offline" / "chroma_db"))
    done_ids: set[str] = set()
    if vs.collection is not None:
        try:
            existing = vs.collection.get(include=[])["ids"]
            done_ids = set(existing)
            logger.info(f"  collection has {len(done_ids)} entries (resume mode)")
        except Exception as e:
            logger.warning(f"  could not list: {e}")

    emb = EmbeddingGenerator("all-MiniLM-L6-v2")

    t0 = time.time()
    BATCH = 512
    total = len(df)
    indexed_now = 0
    for start in range(0, total, BATCH):
        chunk = df.iloc[start:start + BATCH]
        chunk = chunk[~chunk["place_id"].astype(str).isin(done_ids)]
        if len(chunk) == 0:
            continue
        docs = [build_document(r) for _, r in chunk.iterrows()]
        embeddings = emb.generate_embeddings(docs)
        places = []
        for i, (_, r) in enumerate(chunk.iterrows()):
            places.append({
                "id": str(r["place_id"]),
                "document": docs[i],
                "embedding": np.asarray(embeddings[i], dtype=np.float32),
                "metadata": {
                    "name": str(r["name"])[:200],
                    "city": str(r["city"]),
                    "state": str(r["state"]),
                    "category": str(r["primary_category"]),
                    "stars": float(r["stars"]),
                    "review_count": int(r["review_count"]),
                    "price_level": float(r["price_level"]),
                    "hidden_gem_score": float(r["hidden_gem_score"]),
                    "safety_score": float(r["safety_score"]),
                    "source": str(r["source"]),
                },
            })
        vs.add_places(places)
        indexed_now += len(places)
        if (start // BATCH) % 10 == 0:
            logger.info(f"  +{indexed_now} this run ({indexed_now/(time.time()-t0):.0f}/s), "
                        f"total {vs.collection.count() if vs.collection else 0}/{total}")

    vs.save()
    n = vs.collection.count() if vs.collection is not None else 0
    logger.info(f"  DONE: {n} places in vector store ({time.time()-t0:.0f}s)")
    logger.info(f"  saved: offline/chroma_db")


if __name__ == "__main__":
    main()
