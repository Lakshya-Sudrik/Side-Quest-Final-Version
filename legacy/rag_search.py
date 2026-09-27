"""RAG search CLI for server integration.

Usage:
  python rag_search.py --query "vegan chinese in Indiranagar" --top-k 5

Prints JSON: {"results": [{id, document, metadata, distance}, ...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--city", default="")
    args = ap.parse_args()

    from rag_pipeline import RAGPipeline

    pipe = RAGPipeline()
    q_emb = pipe.embedding_generator.generate_embedding(args.query)
    results = pipe.vector_store.search(q_emb, top_k=args.top_k * 3)

    # optional city filter
    if args.city:
        results = [r for r in results
                   if str(r["metadata"].get("city", "")).lower() == args.city.lower()]
        results = results[: args.top_k]

    out = []
    for r in results:
        md = r.get("metadata", {})
        out.append({
            "id": r["id"],
            "name": md.get("name", ""),
            "city": md.get("city", ""),
            "category": md.get("category", ""),
            "stars": md.get("stars", 0),
            "hidden_gem_score": md.get("hidden_gem_score", 0),
            "safety_score": md.get("safety_score", 0),
            "document": r.get("document", ""),
            "distance": float(r.get("distance", 0)),
        })
    print(json.dumps({"results": out[: args.top_k]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
