import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import { execFile, spawn, ChildProcess } from "child_process";
import { promisify } from "util";
import fs from "fs";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3000);

// ---------------------------------------------------------------------------
// Python backend (FastAPI, backend/app.py): accounts, solo matchmaking, chat,
// collaborators, route gems, RAG. It keeps the models loaded, so requests are
// fast. /api/v2/* is forwarded to it untouched (JSON and file uploads alike).
// ---------------------------------------------------------------------------
const PY_API_URL = process.env.PY_API_URL || "http://127.0.0.1:8000";
let pyProc: ChildProcess | null = null;

async function pyHealthy(): Promise<boolean> {
  try {
    const r = await fetch(`${PY_API_URL}/api/v2/health`, { signal: AbortSignal.timeout(3000) });
    return r.ok;
  } catch {
    return false;
  }
}

async function py(pathAndQuery: string, body?: unknown, timeoutMs = 60000): Promise<{ status: number; data: any }> {
  const r = await fetch(`${PY_API_URL}/api/v2${pathAndQuery}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  return { status: r.status, data: await r.json().catch(() => ({})) };
}

function retired(res: any, replacement: string, why: string) {
  res.status(410).json({ success: false, error: `This endpoint is retired: ${why}`, use: replacement });
}

async function ensurePythonBackend(pythonExe: string, root: string): Promise<void> {
  if (await pyHealthy()) {
    console.log(`Python backend already running at ${PY_API_URL}`);
    return;
  }
  if (process.env.PY_AUTOSTART === "0") {
    console.warn(`Python backend not reachable at ${PY_API_URL} (PY_AUTOSTART=0, not starting it)`);
    return;
  }
  const port = new URL(PY_API_URL).port || "8000";
  pyProc = spawn(pythonExe, ["-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", port],
    { cwd: root, stdio: ["ignore", "inherit", "inherit"], env: process.env });
  pyProc.on("exit", (code) => { console.warn(`Python backend exited (${code})`); pyProc = null; });
  for (let i = 0; i < 60; i++) {           // models take a few seconds to load
    await new Promise((r) => setTimeout(r, 1000));
    if (await pyHealthy()) {
      console.log(`Python backend started at ${PY_API_URL}`);
      return;
    }
  }
  console.error("Python backend did not become healthy in 60 s - /api/v2 will return 503");
}
for (const sig of ["SIGINT", "SIGTERM", "exit"] as const) {
  process.on(sig, () => { pyProc?.kill(); if (sig !== "exit") process.exit(0); });
}

// forward /api/v2/* BEFORE the JSON body parser so uploads stream through unchanged
app.use("/api/v2", async (req, res) => {
  try {
    const headers: Record<string, string> = {};
    for (const h of ["authorization", "content-type", "content-length", "accept"]) {
      const v = req.headers[h];
      if (typeof v === "string") headers[h] = v;
    }
    const hasBody = !["GET", "HEAD"].includes(req.method);
    const upstream = await fetch(`${PY_API_URL}/api/v2${req.url}`, {
      method: req.method,
      headers,
      body: hasBody ? (req as any) : undefined,
      // @ts-ignore - required by Node's fetch when streaming a request body
      duplex: hasBody ? "half" : undefined,
      signal: AbortSignal.timeout(120000),
    });
    res.status(upstream.status);
    const ct = upstream.headers.get("content-type");
    if (ct) res.setHeader("content-type", ct);
    res.send(Buffer.from(await upstream.arrayBuffer()));
  } catch (err) {
    res.status(503).json({ detail: "Python backend unavailable - start it with: python -m uvicorn backend.app:app --port 8000",
                           error: String(err) });
  }
});

app.use(express.json({ limit: "10mb" }));

// Google Places requests stay on the server so API credentials never enter the bundle.
const googlePlaces = (url: string, init: RequestInit = {}) => {
  const key = process.env.GOOGLE_PLACES_API_KEY || process.env.GOOGLE_MAPS_API_KEY;
  if (!key) throw new Error("Google Places is not configured. Add GOOGLE_PLACES_API_KEY to .env.");
  const headers = new Headers(init.headers);
  headers.set("X-Goog-Api-Key", key);
  return fetch(url, { ...init, headers, signal: AbortSignal.timeout(12000) });
};

const normalizePlaceText = (value: string) => value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

app.post("/api/places/enrich", async (req, res) => {
  if (!process.env.GOOGLE_PLACES_API_KEY && !process.env.GOOGLE_MAPS_API_KEY) return res.json({ results: [] });
  const places = Array.isArray(req.body?.places) ? req.body.places.slice(0, 10) : [];
  try {
    const results = await Promise.all(places.map(async (item: any) => {
      const name = String(item?.name || "").slice(0, 120);
      const city = String(item?.city || "India").slice(0, 80);
      const search = await googlePlaces("https://places.googleapis.com/v1/places:searchText", {
        method: "POST", headers: { "Content-Type": "application/json", "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.photos,places.primaryTypeDisplayName,places.types" },
        body: JSON.stringify({ textQuery: `${name}, ${city}, India`, maxResultCount: 5 }),
      });
      if (!search.ok) return { name, photoUrl: "", description: "", mapsUrl: "" };
      const data: any = await search.json();
      const candidates: any[] = data.places || [];
      if (!candidates.length) return { name, photoUrl: "", description: "", mapsUrl: "" };
      const wanted = new Set(normalizePlaceText(name).split(" ").filter((word) => word.length > 2));
      const ranked = candidates.map((place) => {
        const words = new Set(normalizePlaceText(place.displayName?.text || "").split(" "));
        const overlap = [...wanted].filter((word) => words.has(word)).length / Math.max(1, wanted.size);
        return { place, overlap, score: overlap + (place.photos?.length ? 0.14 : 0) };
      }).sort((a, b) => b.score - a.score);
      const closestMatch = ranked[0];
      const best = ranked.find((candidate) => candidate.place.photos?.length && candidate.overlap >= closestMatch.overlap - 0.15)?.place || closestMatch.place;
      const placeId = String(best.id || "");
      if (!/^[A-Za-z0-9_-]{8,}$/.test(placeId)) return { name, photoUrl: "", description: "", mapsUrl: "" };
      const detail = await googlePlaces(`https://places.googleapis.com/v1/places/${placeId}`, {
        headers: { "X-Goog-FieldMask": "displayName,formattedAddress,photos,editorialSummary,primaryTypeDisplayName,types,googleMapsUri" },
      });
      const d: any = detail.ok ? await detail.json() : best;
      const photo = d.photos?.[0]?.name || best.photos?.[0]?.name;
      const result = {
        photoUrl: photo ? `/api/places/photo?name=${encodeURIComponent(photo)}` : "",
        photoAttributions: (d.photos?.[0]?.authorAttributions || best.photos?.[0]?.authorAttributions || []).map((credit: any) => ({ name: String(credit.displayName || ""), url: String(credit.uri || "") })).filter((credit: any) => credit.name),
        description: String(d.editorialSummary?.text || best.editorialSummary?.text || "").trim(),
        mapsUrl: String(d.googleMapsUri || best.googleMapsUri || ""),
        verifiedName: String(d.displayName?.text || best.displayName?.text || name),
        verifiedType: String(d.primaryTypeDisplayName?.text || best.primaryTypeDisplayName?.text || ""),
        address: String(d.formattedAddress || best.formattedAddress || ""),
      };
      return { name, ...result };
    }));

    const needsGemini = results.map((result: any, index: number) => ({ result, index })).filter(({ result }) => !result.description && result.verifiedName);
    if (needsGemini.length && needsGemini.length <= 10) {
      const ai = getGenAI();
      if (ai) {
        const facts = needsGemini.map(({ result, index }) => ({ index, name: result.verifiedName, area: result.address, type: result.verifiedType }));
        const prompt = `Write one specific, factual 1-2 sentence travel introduction for each place using only the confirmed place name, address, type and reliable general knowledge. Do not repeat rankings or scores. Do not claim current opening hours, price, safety, or operational status. If uncertain, return an empty description. Return only a JSON array with objects {"index": number, "description": string}. Places: ${JSON.stringify(facts)}`;
        let text = "";
        for (const model of ["gemini-3.8-flash", "gemini-3.6-flash"]) {
          try { const generated = await ai.models.generateContent({ model, contents: prompt }); text = String(generated.text || ""); if (text) break; }
          catch { /* model availability varies; keep verified Place data usable */ }
        }
        try {
          const json = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
          const descriptions = JSON.parse(json);
          for (const entry of Array.isArray(descriptions) ? descriptions : []) {
            const index = Number(entry.index);
            const description = String(entry.description || "").trim();
            if (!Number.isInteger(index) || index < 0 || index >= results.length || description.length < 30 || description.length > 420) continue;
            results[index].description = description;
          }
        } catch { /* Ignore malformed model output instead of showing made-up copy. */ }
      }
    }
    res.json({ results });
  } catch { res.status(502).json({ error: "Google Places enrichment is temporarily unavailable" }); }
});

app.get("/api/places/photo", async (req, res) => {
  const name = String(req.query.name || "");
  if (!/^places\/[^/]+\/photos\/[^/]+$/.test(name)) return res.status(400).end();
  try {
    const key = process.env.GOOGLE_PLACES_API_KEY || process.env.GOOGLE_MAPS_API_KEY;
    if (!key) return res.status(503).end();
    const r = await fetch(`https://places.googleapis.com/v1/${name}/media?maxWidthPx=1000&key=${encodeURIComponent(key)}`, { redirect: "follow", signal: AbortSignal.timeout(12000) });
    if (!r.ok) return res.status(r.status).end();
    res.setHeader("Content-Type", r.headers.get("content-type") || "image/jpeg");
    res.setHeader("Cache-Control", "public, max-age=86400");
    res.send(Buffer.from(await r.arrayBuffer()));
  } catch { res.status(502).end(); }
});

app.post("/api/places/operators", async (req, res) => {
  const place = String(req.body?.place || "").slice(0, 120);
  const area = String(req.body?.area || "").slice(0, 120);
  const lat = Number(req.body?.lat);
  const lng = Number(req.body?.lng);
  const hasLocation = Number.isFinite(lat) && Number.isFinite(lng);
  if (!place || !area) return res.status(400).json({ error: "Place and area are required" });
  try {
    // Search Google Places directly so temporary Gemini outages never disable phone lookup.
    const locality = hasLocation ? place : `${place} in ${area}`;
    const queries = [`taxi service near ${locality}, India`, `cab service near ${locality}, India`, `local tour operator near ${locality}, India`];
    const searches = await Promise.all(queries.map((textQuery) => googlePlaces("https://places.googleapis.com/v1/places:searchText", {
      method: "POST", headers: { "Content-Type": "application/json", "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.location" },
      body: JSON.stringify({ textQuery, maxResultCount: 5, ...(hasLocation ? { locationBias: { circle: { center: { latitude: lat, longitude: lng }, radius: 20000 } } } : {}) }),
    }).then(async (response) => response.ok ? response.json() : null).catch(() => null)));
    const candidates = new Map<string, any>();
    for (const result of searches) for (const item of result?.places || []) if (item.id) candidates.set(item.id, item);
    const operators = await Promise.all([...candidates.values()].slice(0, 10).map(async (p: any) => {
      let details = p;
      if (!p.nationalPhoneNumber && !p.internationalPhoneNumber) {
        const response = await googlePlaces(`https://places.googleapis.com/v1/places/${p.id}`, { headers: { "X-Goog-FieldMask": "displayName,formattedAddress,nationalPhoneNumber,internationalPhoneNumber,websiteUri,rating" } }).catch(() => null);
        if (response?.ok) details = await response.json().catch(() => p);
      }
      const phone = details.nationalPhoneNumber || details.internationalPhoneNumber || null;
      if (!phone) return null;
      let distanceKm: number | null = null;
      const operatorLat = Number(details.location?.latitude ?? p.location?.latitude);
      const operatorLng = Number(details.location?.longitude ?? p.location?.longitude);
      if (Number.isFinite(lat) && Number.isFinite(lng) && Number.isFinite(operatorLat) && Number.isFinite(operatorLng)) {
        const radians = (value: number) => value * Math.PI / 180;
        const dLat = radians(operatorLat - lat), dLng = radians(operatorLng - lng);
        const a = Math.sin(dLat / 2) ** 2 + Math.cos(radians(lat)) * Math.cos(radians(operatorLat)) * Math.sin(dLng / 2) ** 2;
        distanceKm = 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      }
      return { name: details.displayName?.text || p.displayName?.text, address: details.formattedAddress || p.formattedAddress, phone, website: details.websiteUri || p.websiteUri || null, rating: details.rating ?? p.rating ?? null, distanceKm: distanceKm == null ? null : Number(distanceKm.toFixed(1)) };
    }));
    const published = operators.filter((operator: any) => operator?.name && operator.phone && (operator.distanceKm == null || operator.distanceKm <= 50))
      .sort((a: any, b: any) => a.distanceKm != null && b.distanceKm != null ? a.distanceKm - b.distanceKm : Number(b.rating || 0) - Number(a.rating || 0)).slice(0, 3);
    res.json({ operators: published, message: published.length ? "Contacts shown only where a public phone number is listed." : "No local operator with a public phone number was found." });
  } catch (e) { res.status(502).json({ error: "Could not search Google Places for public operator contacts", detail: String(e) }); }
});

app.post("/api/places/resolve", async (req, res) => {
  const place = String(req.body?.place || "").slice(0, 140);
  const area = String(req.body?.area || "").slice(0, 100);
  if (!place || !area) return res.status(400).json({ error: "Place and route area are required" });
  try {
    const r = await googlePlaces("https://places.googleapis.com/v1/places:searchText", {
      method: "POST", headers: { "Content-Type": "application/json", "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location" },
      body: JSON.stringify({ textQuery: `${place}, ${area}, India`, maxResultCount: 1 }),
    });
    const d: any = await r.json();
    if (!r.ok || !d.places?.[0]?.location) return res.status(404).json({ error: "Google Places could not locate that stop. Add a nearby landmark or town." });
    const p = d.places[0];
    res.json({ name: p.displayName?.text || place, address: p.formattedAddress || "", lat: p.location.latitude, lon: p.location.longitude });
  } catch (e) { res.status(502).json({ error: "Place lookup is unavailable", detail: String(e) }); }
});

// Python executable path (override with PYTHON_PATH; WindowsApps alias is machine-dependent)
const PYTHON =
  process.env.PYTHON_PATH || (process.platform === "win32" ? "python" : "python3");
// Project root: override with SIDEQUEST_ROOT, otherwise the folder the server runs from.
const PROJECT_ROOT = process.env.SIDEQUEST_ROOT || process.cwd();
const MODELS_DIR = process.env.ML_MODELS_DIR || path.join(PROJECT_ROOT, "models");
const SRC_DIR = path.join(PROJECT_ROOT, "src");

const execFileAsync = promisify(execFile);

// Lazy initialization of Gemini SDK
let aiClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI | null {
  if (!aiClient && process.env.GEMINI_API_KEY) {
    aiClient = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// API: Health check
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    hasGeminiKey: Boolean(process.env.GEMINI_API_KEY),
    hasModels: fs.existsSync(path.join(MODELS_DIR, "hidden_gem_model.pkl")),
  });
});

// Helper: Run Python ML inference script
async function runMLInference(script: string, args: string[], fromRoot = false): Promise<any> {
  try {
    const scriptPath = fromRoot
      ? path.join(PROJECT_ROOT, script)
      : path.join(SRC_DIR, script);
    const { stdout, stderr } = await execFileAsync(PYTHON, ["-X", "utf8", scriptPath, ...args], {
      timeout: 60000,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    });
    if (stderr) console.error(`ML stderr:`, stderr.slice(0, 500));
    return JSON.parse(stdout);
  } catch (err: any) {
    console.error(`ML inference error (${script}):`, err.message);
    throw err;
  }
}

// API: Analyze a place using trained ML models
app.post("/api/ml/analyze-place", async (req, res) => {
  try {
    const { name, category, latitude, longitude, reviewCount, rating, priceRange, distanceKm, city } = req.body;

    const result = await runMLInference("inference.py", [
      "--mode", "analyze",
      "--name", name || "Unknown",
      "--category", category || "",
      "--city", city || "",
      "--lat", String(latitude || 19.07),
      "--lng", String(longitude || 72.87),
      "--reviews", String(reviewCount || 50),
      "--rating", String(rating || 4.5),
      "--price", String(priceRange || 2),
      "--distance", String(distanceKm || 10),
    ]);

    res.json(result);
  } catch (err) {
    res.status(500).json({ error: "ML analysis failed", details: err });
  }
});

// API: Verify a collaborator's place
app.post("/api/ml/verify-collaborator", (_req, res) => retired(res, "POST /api/v2/auth/register (role collaborator) + /api/v2/collaborator/documents", "the collaborator-authenticity model failed its quality checks (macro-F1 0.59); businesses are verified by GSTIN + document review"));

// API: Get hidden gems from the trained model
app.get("/api/ml/hidden-gems", async (req, res) => {
  // same rules as the route recommender (backend /api/v2/gems/city)
  try {
    const { limit, city, kind } = req.query;
    const q = new URLSearchParams({ limit: String(limit || 12), city: String(city || ""), kind: String(kind || "") });
    const r = await py(`/gems/city?${q}`);
    res.status(r.status).json(r.data);
  } catch (err) {
    res.status(503).json({ success: false, error: "Python backend unavailable", details: String(err) });
  }
});

// API: RAG semantic search over 102,810 India places (vector store)
app.post("/api/rag/search", async (req, res) => {
  try {
    const { query, city, topK } = req.body;
    if (!query) {
      res.status(400).json({ error: "query is required" });
      return;
    }
    // ChromaDB index with current data + honest scores (backend/rag.py)
    const r = await py("/rag/search", { query: String(query), city: String(city || ""), top_k: Number(topK || 5) }, 30000);
    res.status(r.status).json(r.status === 200 ? { success: true, ...r.data } : { success: false, error: r.data?.detail });
  } catch (err) {
    res.status(500).json({ error: "RAG search failed", details: err });
  }
});

// API: Solo matchmaking (hybrid model on 59K Indianized profiles)
app.post("/api/ml/match", (_req, res) => retired(res, "GET /api/v2/solo/candidates", "the old matchmaking model was trained on labels built from its own inputs (its 99.6% was not real); compatibility is now a transparent rule over the 5 preferences"));

// API: Hidden gems along a route - 5 hidden gems + 5 famous-and-good places,
// each with a safety block and distances (src/recommender.py)
app.post("/api/ml/hidden-gems", async (req, res) => {
  try {
    const { from, to, bufferKm, youtube, only } = req.body || {};
    if (!from || !to) {
      res.status(400).json({ success: false, error: "Required: from, to (town name or 'lat,lon')" });
      return;
    }
    const args = ["--from", String(from), "--to", String(to),
                  "--buffer-km", String(bufferKm ?? 10),
                  "--youtube", ["off", "cache", "live"].includes(youtube) ? youtube : "cache"];
    if (only === "eat" || only === "visit") args.push("--only", only);
    // fast path: the always-on Python backend (models already loaded)
    try {
      const r = await fetch(`${PY_API_URL}/api/v2/gems/route`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ from, to, bufferKm: bufferKm ?? 10, youtube: youtube ?? "cache", only }),
        signal: AbortSignal.timeout(60000),
      });
      if (r.status !== 503) {
        const data = await r.json();
        res.status(r.ok ? 200 : 422).json(r.ok ? data : { success: false, error: data?.detail ?? "failed" });
        return;
      }
    } catch { /* fall back to running the script */ }
    const result = await runMLInference("recommender.py", args);
    res.status(result?.success === false ? 422 : 200).json(result);
  } catch (err) {
    console.error("Hidden gems error:", err);
    res.status(500).json({ success: false, error: "Hidden gem recommender failed", details: String(err) });
  }
});

// API: Trip suggestions (hidden gems + popular along route)
app.post("/api/ml/trip-suggest", (_req, res) => retired(res, "POST /api/ml/hidden-gems or POST /api/v2/gems/route",
  "it ranked stops by the old leaky hidden_gem_score column"));

// API: Solo matchmaking candidates (transparent list + sort by compatibility)
app.get("/api/ml/solo-candidates", (_req, res) => retired(res, "GET /api/v2/solo/candidates", "solo matching moved to the database-backed backend"));

// API: Express interest in a candidate (double opt-in)
app.post("/api/ml/solo-interest", (_req, res) => retired(res, "POST /api/v2/solo/requests", "solo matching moved to the database-backed backend"));

// API: Get mutual matches (chat unlocked)
app.get("/api/ml/solo-matches", (_req, res) => retired(res, "GET /api/v2/matches", "solo matching moved to the database-backed backend"));

// API: Candidate pool stats (admin/debug)
app.get("/api/ml/solo-stats", (_req, res) => retired(res, "GET /api/v2/solo/candidates", "solo matching moved to the database-backed backend"));

// API: Detect fake reviews
app.post("/api/ml/detect-fake-reviews", async (req, res) => {
  try {
    const { reviewText, rating, userReviewCount, businessReviewCount } = req.body;

    const result = await runMLInference("inference.py", [
      "--mode", "fake-detect",
      "--text", reviewText || "",
      "--rating", String(rating || 5),
      "--user-reviews", String(userReviewCount || 10),
      "--biz-reviews", String(businessReviewCount || 100),
    ]);

    res.json(result);
  } catch (err) {
    res.status(500).json({ error: "ML fake detection failed", details: err });
  }
});

// API: Combined analysis (ML + RAG context + Gemini availability)
app.post("/api/analyze", async (req, res) => {
  try {
    const { name, category, description, latitude, longitude, reviewCount, rating, priceRange, distanceKm, city } = req.body;

    // Run ML analysis and RAG retrieval in parallel
    const mlPromise = runMLInference("inference.py", [
      "--mode", "analyze",
      "--name", name || "Unknown",
      "--category", category || "",
      "--city", city || "",
      "--lat", String(latitude || 19.07),
      "--lng", String(longitude || 72.87),
      "--reviews", String(reviewCount || 50),
      "--rating", String(rating || 4.5),
      "--price", String(priceRange || 2),
      "--distance", String(distanceKm || 10),
    ]).catch((e) => ({ success: false, error: String(e?.message || e) }));

    const ragQuery = [name, category, description].filter(Boolean).join(" ");
    const ragPromise = ragQuery
      ? py("/rag/search", { query: ragQuery, city: city || "", top_k: 5 }, 30000)
          .then((r) => (r.status === 200 ? r.data : { results: [] }))
          .catch(() => ({ results: [] }))
      : Promise.resolve({ results: [] });

    const [mlResult, ragResult] = await Promise.all([mlPromise, ragPromise]);

    res.json({
      success: true,
      mlAnalysis: mlResult,
      ragContext: ragResult,
      geminiAvailable: Boolean(process.env.GEMINI_API_KEY),
    });
  } catch (err) {
    res.status(500).json({ error: "Combined analysis failed" });
  }
});

// Safety index for a point (src/safety_index.py via inference.py --mode safety).
// Returns nulls + a note when location or data is missing - never a guessed score.
async function computeSafety(latitude: unknown, longitude: unknown, city: unknown) {
  const lat = Number(latitude);
  const lng = Number(longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return {
      safetyScore: null,
      safetyLevel: "unknown",
      womenSafe: null,
      safetyReasons: [],
      safetyNote: "Add the exact location (latitude, longitude) to compute the safety index.",
    };
  }
  try {
    const r = await runMLInference("inference.py", [
      "--mode", "safety", "--lat", String(lat), "--lng", String(lng), "--city", String(city || ""),
    ]);
    return {
      safetyScore: r.safety_score,
      safetyLevel: r.level,
      womenSafe: r.women_safe,
      safetyReasons: r.reasons || [],
      safetyNote: r.safety_score == null ? "No police, hospital or crime data for this area yet." : undefined,
    };
  } catch {
    return {
      safetyScore: null,
      safetyLevel: "unknown",
      womenSafe: null,
      safetyReasons: [],
      safetyNote: "Safety index unavailable right now.",
    };
  }
}

// API: AI Hidden Gem Evaluation for Collaborator / Host submissions
app.post("/api/evaluate-gem", (_req, res) => retired(res, "POST /api/v2/listings",
  "a language model was deciding whether a place is a hidden gem with no data behind it; listings are now scored by the trained hidden-gem models and published only for verified collaborators"));

// Vite middleware setup
async function startServer() {
  await ensurePythonBackend(PYTHON, PROJECT_ROOT);
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`SideQuest Full-Stack Server running on port ${PORT}`);
  });
}

startServer();
