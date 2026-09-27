# FINAL_PILLAI_PRODUCT — project guide

This folder contains the backend assembled from the user-provided 32-part archive and the React UI copied from the earlier SideQuest UI project. It was built separately in `Downloads`; no Git metadata or Python/model/data files from the earlier UI checkout were copied here. The archive's own `legacy/` directory is retained as backend documentation/code, and the duplicate `jash_sq/` snapshot was excluded.

## What arrived

- FastAPI API in `backend/`, Express/Vite app server in `server.ts`, and the typed `/api/v2` client at `src/services/backend.ts`.
- Trained model files in `models/` (including the RAG embedder), scored place/safety/geo datasets in `data/`, and a prebuilt ChromaDB index in `data/chroma_db/` (~645 MB).
- `data/app/sidequest.db` is a SQLite database shipped inside the archive. It currently has 2 user rows, 2 trips, 1 match and 2 messages; there are no collaborator/listing rows. This is the SQLite dataset that arrived in the archive, not a PostgreSQL database/server. The service also supports PostgreSQL if you provide a separate server and `DATABASE_URL`.
- The final directory intentionally has no `.git`, `node_modules`, `.venv`, or `.env` secrets.

## UI connections made

- Sign-in and account creation now call the backend, store its bearer token, collect the required user preferences, and route collaborator accounts into the host portal.
- Home/Hidden Gems load backend-ranked city gems; semantic search uses `/api/v2/rag/search` and route recommendations use `/api/v2/gems/route`.
- Solo Match creates a trip, loads backend compatibility candidates, and sends a double-opt-in request; the UI no longer calls the retired/fabricated ML score endpoint.
- Chats load accepted matches and messages, send messages, and call block/report endpoints. The old simulated replies and fictional threads are removed.
- Collaborator uploads go through `/api/v2/collaborator/documents`; listing submissions go through `/api/v2/listings` and show the real review decision.
- Existing UI palette, typography and landing/intro assets were retained.

## UI work still needed

- Add user controls for reviewing incoming/outgoing solo requests and accepting/declining requests. The backend supports this; the current screen only creates trips, shows candidates, and sends requests.
- Add a real itinerary planner UI for `/api/v2/itineraries`: create plan, add/remove stops, report a flight delay, and fetch time-aware suggestions. Current route screen is connected for gem recommendations; its manual stop/analyze controls still have local demo behavior.
- Add an admin-only collaborator review queue for `/api/v2/admin/collaborators` and `/api/v2/admin/collaborators/{uid}/verify`.
- Add editable saved preferences and account/profile controls (`/api/v2/me`, `/api/v2/me/preferences`).
- Wire authenticated YouTube verification (`/api/v2/youtube/check`) into listing review if desired; YouTube API key is optional but required for live checks.
- Remove or replace remaining static demo content in Home/Gems and local-only saved-gem state. Google/Apple sign-in, password reset, payments/pricing and notification actions have no corresponding backend implementation and remain presentation-only.

## Local run

Requirements: Node.js, Python 3.13 for the archived ML stack (Python 3.14 is too new for some pinned packages). Install Python 3.13 with the [official Python Install Manager](https://www.python.org/downloads/windows/) if it is not available on your machine, then run `py install 3.13`.

```powershell
cd "$env:USERPROFILE\Downloads\FINAL_PILLAI_PRODUCT"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
npm install
npm run dev
```

Open `http://localhost:3000`. The Express server starts the Python API on port 8000. SQLite tables are initialized automatically. The included RAG embedder and Chroma index mean the first run should not need `python -m backend.rag --build`; rebuild only if the index is missing or stale. Add `YOUTUBE_API_KEY` for live video checks. Set `SIDEQUEST_ADMIN_EMAILS` before registration to grant admin role. Leave `DATABASE_URL` blank to use SQLite; set a valid PostgreSQL URL only when a PostgreSQL server is already installed and running.

## Verification performed

- The 32 split ZIP segments were concatenated and verified against the supplied SHA-256 file before extraction.
- `npm install` completed.
- `npm run build` completed (Vite production output, Express bundle and service worker).
- `npm run lint` completed (`tsc --noEmit`).
- Python API tests were not run here; backend Python dependencies are not preinstalled.
- Replace the old four-proof mock cards with the backend's supported document types (`gst_certificate`, `fssai`, `shop_licence`, `udyam`, `other`) and show per-document review state. The current screen sends real uploads, but the backend stores one document path/type per collaborator, so the four-card state is not yet a faithful document manager.

## Nugen SideQuest alignment

- What If compares the saved schedule using SideQuest's itinerary logic, then asks a deployed SideQuest-aligned Nugen model to explain the same supplied facts. If the provider cannot answer, the screen labels that state and keeps the schedule result visible.
- The Nugen client uses the current `/api/v3/alignment-projects/*`, `/models/aligned`, `/models/base`, and `/inference/chat/completions` routes. It selects only a deployed aligned model; raw API credentials stay in server-side `.env`.
- `backend/nugen_sidequest_alignment.md` contains the synthetic travel-scenario corpus. It contains no saved itinerary or traveller account data. `scripts/nugen_sidequest_alignment.py` can start and poll an alignment and request deployment; provider state is stored locally under ignored `data/app/`.
- This machine has started the SideQuest alignment with Nugen. Nugen reported training complete and the model deployed. Nugen inference has been intermittent: a short chat request returned text once, but repeated calls ended with an upstream timeout or premature stream close. What If displays provider failure honestly and keeps the SideQuest schedule comparison available. End-to-end Nugen inference must be confirmed again when the provider responds consistently.
- After provider inference is healthy, restart the server if `.env` changed, then run a scenario from What If. Admin > Model and data health reports provider reachability and alignment state. No prediction accuracy claim is made until an evaluation benchmark has been run.
- Never copy `.env` into a ZIP or commit it. `.env.example` is the blank template.
