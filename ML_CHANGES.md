# ML fixes – what changed and how to run it

## 1. Serving no longer uses stale or invented numbers
- `src/unified_model.py` loads models against `models/feature_specs.json`. A pickle whose features don't match its spec is **refused** (logged), instead of serving an old leaky model.
- The old leaky pickles moved to `models/legacy/` (`hidden_gem_model_leaky_xgb.pkl`, `safety_score_model.pkl`).
- `src/inference.py`: a score that can't be computed is `null` + a reason, never `0` / `50`. The default crime rates (15 / 8) and invented distances are gone.
- `server.ts`: the collaborator safety score (`96` if the safety text was > 20 characters, else `84`) is gone. Gemini no longer invents a safety number either.
- `api/main.py`: `np.random.uniform(...)` hospital/pharmacy distances and random review stats are gone.
- All hardcoded `D:/sidequest_model` and `C:\Users\manak\...` paths replaced. Set `SIDEQUEST_ROOT` / `PYTHON_PATH` to override.

## 2. Safety is a rule-based index, not a regression
`src/safety_index.py`: district crime (NCRB, national percentile; violent 45% / crimes against women 35% / property 20%) + nearest police station + nearest hospital. Place → district by point-in-polygon. Every score comes with reasons and a `data_coverage` value; missing layers are reported, never filled in.

Build the data:
```bash
python scripts/fetch_safety_data.py                 # districts + NCRB + OSM police/hospitals, state by state
python scripts/fetch_safety_data.py --nhfr hospitals.csv   # optional: data.gov.in / NHFR hospital directory
python src/safety_index.py 12.97 77.59 Bangalore    # check one point
```
Committed now: district boundaries (760 districts) + ~730 police stations / ~11.7k hospitals from the existing cache.

## 3. Hidden gem

### Final held-out results (real data, 2,266 test restaurants never seen in training)
| | R² | Spearman | Gates |
|---|---|---|---|
| Old reported (duplicate restaurants in train + test) | 0.150 | – | – |
| Old label, honest split (true baseline) | 0.082 | – | – |
| **Now (EBM, 5 interactions)** | **0.409** | **0.64** | all pass (gap 0.042, shuffled-label R² 0.007) |

`hidden_gem_class` (top-20% quality): macro-F1 0.71, balanced acc 0.72, minority recall 0.60, PR-AUC 2.7× prevalence – all gates pass.
Precision@top-20%: 0.53 vs 0.20 random.

### What changed
- **Duplicates removed before splitting** (`gem_features.dedupe_listings`): Zomato repeats each restaurant ~4.4× with identical reviews. 102,810 → 70,724 rows. This is why the old 0.150 was inflated – the same restaurant sat in train and test.
- **No real data dropped:** every unique restaurant with ≥ 1 review is labelled (9,062). The old ≥ 15-review rule threw most of them away.
- **Label = review quality only**, 0–100 percentile within the city: 30 % share of detailed (≥ 40-word) positive reviews + 70 % mean review stars, each shrunk toward the training-fold average with 20 pseudo-reviews (so 5/5 on 1 review doesn't beat 4.3 on 50). Constants chosen by grouped CV on the training fold only.
- **"Hidden" is not in the label:** `gem = quality ≥ 60 AND exposure (review-count percentile in city) ≤ 0.5`, applied at serving.
- **New features** (`src/gem_features.py`): `chain_size`, `cost_vs_area`, 25 cuisine flags, `area_cat` (neighbourhood), `rest_type`, `review_count_log`, `has_phone`. Dropped: `competitor_density` (copy of `area_business_count`), and every exposure/label-derived column (`review_count`, `rating_vs_city`, `is_mainstream`, …) – see `criteria.json`.
- **Model:** EBM, 5 pairwise interactions, `min_samples_leaf=50`, `max_interaction_bins=16` (10 interactions scored 0.399 but overfit: gap 0.07).
- **Bugs fixed:** `hidden_gem_class` trained on the leaky legacy `hidden_gem_score` column and on float labels (inverted probabilities); training column order differed from `feature_specs.json` so serving refused the models; gems mode ignored `--city`.

### Caveats
- Labels exist only for **Zomato Bangalore** (≤ 5 reviews per place). Other cities/sources are extrapolated – gems output marks them `in_training_domain: false` and ranks them after in-domain places.
- Review stars feed Zomato's aggregate `stars` rating, which is a feature – a mild overlap. Checked: R² stays 0.26–0.37 across vote bands, so the signal isn't only that overlap.
- `collaboration_auth` fails its gates honestly (macro-F1 0.59, minority recall 0.23). Its old 0.95 accuracy came from the leaky pipeline.


## 4. All-India quality (528 cities, no review text)

`src/india_quality.py` - `python src/india_quality.py` trains, checks and saves `models/india_quality.pkl`.

| Held-out (brand-grouped, 19,125 places) | Value |
|---|---|
| R² | **0.264** (train 0.311, gap 0.047) |
| Spearman | 0.50 |
| Precision@top-20% | 0.46 (random 0.20) |
| Shuffled-label R² | 0.000 |
| **25 cities never seen in training** | R² 0.19, Spearman 0.42 |

Per city (held-out): Pune 0.24, Mumbai 0.22, Kolkata 0.21, Bangalore 0.20, Chennai 0.19, Hyderabad 0.19, Delhi 0.17, Ahmedabad 0.11.

- **Data:** all 70,710 unique rated places (61,302 Swiggy across 528 cities + 9,422 Zomato Bangalore). Nothing dropped except duplicate listings.
- **Label:** the place's rating, shrunk toward its (source, city) average with K=60 pseudo-ratings (K chosen by brand-grouped CV on the training fold), as a 0-100 percentile within (source, city). Swiggy and Zomato are never ranked against each other (different rating scales). Checked the shrinkage isn't turning the label into popularity: correlation with rating count is 0.29 (0.24 with no shrinkage); correlation with the raw rating 0.94.
- **Features (never the place's own rating or count):** cost vs area/city, 40 cuisines (Swiggy/Zomato names unified), chain size in city/India, name words (cafe, dhaba, mess, bhavan…), area density, distance from centre, and **how well the other places within 2 km are rated** (excluding the place itself and its own brand, which also removes the same restaurant listed on both apps).
- **Serving:** `quality = (n·own_rating_pct + 60·model_pct) / (n + 60)`. With 500 ratings the place's own rating counts ~89 %; with 20 ratings 25 %; with none, the model alone (`recommendation = possible_gem_unrated`).
- **App:** gems mode uses the review-text model for Zomato Bangalore and this score everywhere else (`score_basis` = `reviews_model` / `ratings_blend`, plus `own_rating_weight`). Analyze mode adds a `quality_india` block and uses it outside Bangalore. 137 cities currently have at least one hidden gem (4,766 total).
- **Why R² is lower than Bangalore's 0.41:** without review text the own rating has to be the label, so it cannot also be an input. For places that already have many ratings this doesn't matter - their score is mostly their real rating.


## 5. Hidden-gem recommender: 5 hidden gems + 5 famous & good (final)

`python src/recommender.py --from "Bangalore" --to "Mysore"`  ·  API: `POST /api/ml/hidden-gems {from, to}`  ·  UI: AI Route Planner → "Find Hidden Gems Along My Route"

### Models behind it (all measured on data the model never saw)
| Model | Data | Held-out result | Overfit gap | Shuffled-label check |
|---|---|---|---|---|
| Food, Bangalore (review text) | 9,062 Zomato restaurants | **R² 0.409**, Spearman 0.64 | 0.042 | 0.007 |
| Food, all India (ratings) | 70,710 places, 528 cities | **R² 0.264**, unseen cities 0.19 | 0.047 | 0.000 |
| **Sights, all India (new)** | 15,239 places, 1.47M reviews | **R² 0.490** vs 0.444 rating-only baseline, Spearman 0.55 | 0.019 | < 0 |
| Food gem yes/no (Bangalore) | same as above | accuracy 79%, macro-F1 0.71, finds 60% of gems | — | collapses |
| **Sight gem yes/no (new)** | 1,130 less-reviewed test places | **accuracy 71.8%**, precision 57%, AUC 0.72 (baseline 70.2% / 54% / 0.69) | — | — |

**Sights model, how it stays honest:** each place's reviews are split at random in two. The model sees one half (ratings, text, counts) and must predict the average rating of the *other* half, i.e. how the next visitors will rate it. Places are split into train/test by name, exact duplicate reviews are removed first, and the text model is trained out-of-fold. By place size: 4–9 reviews R² 0.13 (baseline 0.04), 10–39 R² 0.33 (0.23), 40–199 R² 0.64 (0.61), 200+ ≈ baseline (the rating alone is already reliable there).

### Pipeline
1. **Route**: OSRM road route; if unreachable, straight line × 1.3 and every distance says "estimated".
2. **Candidates** within 10 km (+5 km for places only known to town level): 70,724 restaurants (`data/processed/food_scored.csv.gz`) + 15,239 sights (`data/attractions/attractions_scored.csv.gz`).
3. **Classify**: `quality_rank` = percentile within its type. Hidden gem = rank ≥ 70 and less reviewed than most places in its town (exposure ≤ 0.5), not a chain (> 5 outlets). Famous & good = rank ≥ 70 and exposure ≥ 0.8. Needs ≥ 20 ratings (food) / ≥ 8 reviews (sights). Shops, agencies and tour operators are excluded.
4. **YouTube** (`src/youtube_verify.py`): searches the towns on the route, checks each video (channel age/size, like and comment rates, view spikes, copy-paste comments, "I visited" comments, declared paid promotion, declared AI content, recorded location) and matches it to a known place by name + town. Only videos judged *genuine* count; they add at most +5 rank points. Videos about unknown places are listed as `unverified_new_places` and never shown as gems.
5. **Safety block under every option**: official district crime (NCRB 2014, per 100k via Census 2011, 578 districts) 50% + what reviewers report (theft, scams, harassment, hazards / "felt safe") 30% + hospital & police distance 20%. Class from evidence: *avoid* = worst 10% crime or ≥ 6% of reviews report problems; *caution* = worst 35%, ≥ 3% complaints, hospital > 15 km, or combined score < 50; else *safe*. Missing parts are listed, never filled in.
6. **Pick**: sort by safety class, then gem score; *avoid* only if nothing else is left; ≥ 2 eat + ≥ 2 visit per list when the route has them; stops spread along the route; no brand twice.

### Build / refresh
```bash
python src/geo_india.py                 # town + site coordinates from GeoNames IN.txt
python scripts/build_crime_2014.py      # district crime rates
python src/attractions_model.py         # train + score sights (~8 min)
python src/build_food_scores.py         # score restaurants
python src/recommender.py --from Kochi --to Munnar --youtube live   # YOUTUBE_API_KEY in .env
python -m backend.rag --build           # search index (~2 min)
python -m pytest tests -q               # 40 tests
```

### Known limits
- Sights have no coordinates in the source data: 97% get a town location from GeoNames (3% an exact site). Distances for those are to the town centre and say so.
- Official crime data is from 2014 (the latest district-wise file publicly available without login). States that register more complaints (e.g. Kerala) look worse than they are; the safety block says so.
- YouTube and OSRM run on your machine: this build environment could not reach either, so the YouTube checks were tested with simulated API responses. There is no accuracy figure for the YouTube check yet: that needs labelled real/fake videos.


## 6. Final models (XGBoost) - all numbers are on data the model never saw

| Model | Held-out result | Train–test gap | Shuffled labels (held-out) | vs. previous |
|---|---|---|---|---|
| **Sights hidden-gem** (15,239 places, 1.47M reviews) | **R² 0.501**, Spearman 0.56; gem yes/no accuracy 71.5%, AUC 0.73 | 0.034 | −0.007 | LightGBM 0.490 (validation 0.487 vs XGBoost 0.502) |
| **Food, Bangalore** (review text) | **R² 0.413**, Spearman 0.64, precision@top-20% 0.53 | 0.033 | −0.03 | EBM 0.409 |
| **Food gem yes/no, Bangalore** | macro-F1 0.708, balanced acc 0.742, finds 68% of gems, PR-AUC 2.7× chance | – | collapses | EBM 0.706 / 0.721 / 60% |
| **Food, all India** (528 cities) | **R² 0.258**, Spearman 0.49; unseen cities 0.19 | 0.040 | 0.000 | LightGBM 0.264 (validation 0.280 vs XGBoost 0.285) |
| Collaborator authenticity | macro-F1 0.59 - **fails its gates, not used for decisions** | – | – | – |

- XGBoost was chosen on the **validation** split, never on test; LightGBM is trained on the identical split and reported next to it in `reports/metrics.json` (`algorithm_comparison`). For the Bangalore model 4 regularisation settings were tried; the chosen one is the most accurate that keeps the gap ≤ 0.05, so read 0.41 as ±0.01.
- **Bug fixed in the sanity check:** the shuffled-label test used to score the model on the same rows it was trained on (that measures memorisation, not leakage). It now scores on held-out rows with the true labels, where a leak-free model must be ≈ 0. All models above pass the corrected check.

## 7. Backend (Python FastAPI + Node/Express) - `backend/`

`npm run dev` starts Express on :3000 and, if needed, the Python backend on :8000 (`python -m uvicorn backend.app:app`). Express forwards `/api/v2/*` untouched (JSON and file uploads). Models load once, so a route request takes ~0.4 s instead of ~3 s.

| Area | Endpoints | Rules |
|---|---|---|
| Accounts | `POST /auth/register`, `POST /auth/login`, `GET /me`, `PUT /me/preferences` | role user / collaborator (admin via `SIDEQUEST_ADMIN_EMAILS`); scrypt password hashes; JWT (72 h); 8 failed logins → 5 min lock; users must be 18+ |
| 5 sign-up preferences | `GET /preferences/options` | travel style, budget, pace, food, languages (+ "only same-gender companions") |
| Solo matching | `POST/GET/PATCH /trips`, `GET /solo/candidates`, `POST /solo/requests`, `POST /solo/requests/{id}/respond` | same destination ≤ 25 km, ≥ 1 overlapping day, solo toggle on, both people's companion settings respected, blocks respected; everyone who qualifies is listed (score only orders); only first name, age band, preferences and overlap dates are shown |
| Chat | `GET /matches`, `GET/POST /matches/{id}/messages` | only after BOTH accept; 1,000 chars, 20 msgs/min; block closes it for both; report |
| Collaborators | `POST /collaborator/documents`, `GET /collaborator/status`, `GET /admin/collaborators`, `POST /admin/collaborators/{id}/verify` | GSTIN checked offline (format, state code, mod-36 check digit) → still needs admin review of an uploaded GST certificate / FSSAI / shop licence / Udyam (PDF/JPG/PNG, content-checked, 8 MB) |
| Listings | `POST /listings` | the model decides: existing place → measured scores; new restaurant → all-India model prediction (no ratings used, labelled "predicted"); new sight → "needs reviews". Self-reported ratings are ignored. Published (and shown on route results) only if hidden gem AND collaborator verified |
| ML | `POST /gems/route`, `POST /rag/search`, `POST /youtube/check`, `GET /health` | – |

Storage: SQLite at `data/app/sidequest.db` (no server needed; plain SQL, moving to Postgres = change `backend/db.py:connect`). Uploads in `data/app/uploads/`. Both git-ignored.

## 8. RAG on ChromaDB - `backend/rag.py`

- **Corpus (current data):** 84,107 documents = 70,724 restaurants + 13,383 sights (shops/agencies removed), each with its current honest quality rank, how well known it is, hidden-gem / famous-and-good flag, and for sights two real review snippets. The old index used the leaky `hidden_gem_score`/`safety_score` columns - replaced.
- **ChromaDB:** persistent at `data/chroma_db`, collection `sidequest_places_v2`, cosine space; metadata filters for town, eat/visit, hidden-only. Build: `python -m backend.rag --build` (~2 min).
- **Embeddings:** local LSA (TF-IDF 1–2 grams → 256-d SVD), no download needed. On a machine with `sentence-transformers`, `RAG_EMBEDDER=minilm python -m backend.rag --build` switches to all-MiniLM-L6-v2; queries automatically use whatever built the index.
- **Search:** town detected from the query → semantic top-N + keyword name match (hybrid) → score = 0.55 similarity + 0.25 name match + 0.20 quality → one result per name → safety block per result → answer text built only from retrieved facts (no generator, nothing invented).
- **Measured (300 auto-generated queries each):** named place found in top 5: **93%** (exact branch 75%); cuisine-in-town precision@5 **92%**; town filter **100%** correct; type-of-sight-in-town precision@5 54%.

Retrain (needs the gitignored raw data):
```bash
python data_pipeline_india.py        # rebuild india_places.csv with dedupe + new features
python src/training_honest.py        # hidden_gem, hidden_gem_class, collaboration_auth
python src/india_quality.py          # all-India quality (~8 min)
python -m pytest tests -q            # 14 tests, synthetic data
```
