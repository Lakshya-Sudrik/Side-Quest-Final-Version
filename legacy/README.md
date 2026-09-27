# Retired code (kept for reference only - do not run)

| File | Why it was retired | Replaced by |
|---|---|---|
| `trip_suggest.py` | Ranked stops by the old `hidden_gem_score` column, a formula over the model's own features (leaky) | `src/recommender.py` (`/api/v2/gems/route`) |
| `rag_search.py`, `index_rag_india.py` | Old vector index built on the leaky `hidden_gem_score` / `safety_score` columns | `backend/rag.py` (ChromaDB, current scores) |
| `match_india.py` | Matchmaking model whose label was a formula of its own inputs - the 99.6% accuracy was not real | Transparent 5-preference rule in `backend/preferences.py` |
| `solo_matchmaking_candidates.py` | Solo matching over 15 hard-coded JSON users | `backend/matchmaking.py` (database-backed) |
| `api_main_old.py` | Second, older FastAPI app (served the old unified model) | `backend/app.py` |
| `CollaboratorScreen_old.tsx.txt` | Old host screen: a Gemini prompt decided whether a place is a gem and "approvals" lived only in the browser | `src/components/CollaboratorPortal.tsx` + `POST /api/v2/listings` |
