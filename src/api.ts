// Compatibility adapters for the original screens. All data now comes from /api/v2.
import { api } from './services/backend';
import type { HiddenGem } from './types';
const colors: HiddenGem['categoryColor'][] = ['primary', 'secondary', 'tertiary'];

function toUiGem(g: any, idx: number): HiddenGem {
  const kind = g.kind === 'visit' || String(g.type || '').toLowerCase().includes('visit') ? 'visit' : 'eat';
  return {
    id: String(g.place_id ?? g.id), name: g.name, category: String(g.category || 'Local find'),
    categoryIcon: kind === 'visit' ? 'landscape' : 'restaurant', categoryColor: colors[idx % colors.length],
    // The model's `why` field explains the ranking, not the place itself.
    // Keep it out of the visitor-facing introduction; enrich with factual place details separately.
    description: String(g.description || g.summary || ''),
    image: String(g.photo_url || ''), distanceKm: Number(g.distance?.from_route_km ?? 0),
    gemScore: Number(g.gem_score_10 ?? g.hidden_gem_score / 10 ?? 0), safetyScore: g.safety?.score ?? g.safety_score ?? null,
    womenSafe: null, costLabel: g.price_level ? '₹'.repeat(Math.max(1, Math.min(3, Number(g.price_level)))) : 'Local find',
    hikeDurationOrFeature: g.city || 'Along your route', badgeLabel: g.human_approved ? 'Human-reviewed' : g.is_hidden ? 'Hidden gem' : 'Curated find',
    tags: [g.city, g.category, kind].filter(Boolean), lat: g.lat, lng: g.lng ?? g.lon,
    kind, rating: g.rating, reviews: g.review_count ?? g.reviews, city: g.city,
    safetyClass: g.safety?.class ?? g.safety_class, safetyReasons: g.safety?.reasons ?? g.safety_reasons,
    nearestHospitalKm: g.safety?.hospital_km ?? g.hospital_km,
    nearestPoliceKm: g.safety?.police_km ?? g.police_km,
    googleMapsUri: g.google_maps_uri || g.googleMapsUri || '',
    photoAttributions: Array.isArray(g.photo_attributions) ? g.photo_attributions : [],
    detourMin: Number(g.distance?.detour_min_est ?? 0) || undefined,
    whyReasons: Array.isArray(g.why) ? g.why : [g.why, g.score_basis].filter(Boolean),
  };
}

export async function fetchHiddenGems(limit = 12): Promise<HiddenGem[]> {
  const data = await fetch(`/api/v2/gems/city?limit=${limit}`).then(async r => { const j = await r.json(); if (!r.ok) throw new Error(j.detail || 'Could not load gems'); return j; });
  return (data.gems || []).map(toUiGem);
}

export async function ragSearch(query: string, topK = 6, city = ''): Promise<HiddenGem[]> {
  const result = await api.ragSearch(query, city);
  return (result.results || []).slice(0, topK).map(toUiGem);
}

export async function fetchRouteGems(from: string, to: string, youtube: 'off' | 'cache' | 'live' = 'cache') {
  const r = await fetch('/api/v2/gems/route', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ from, to, youtube }) });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || 'Route recommendations are unavailable');
  const maps = (place: any) => place.google_maps_uri || place.googleMapsUri ||
    `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${place.name}, ${place.city || ''}`)}`;
  return { hidden: (d.hidden_gems || []).map((g: any, i: number) => ({...toUiGem(g, i),googleMapsUri:maps(g),badgeLabel:g.human_approved?'Human-reviewed':'Hidden gem'})), famous: (d.famous_and_good || []).map((g: any, i: number) => ({...toUiGem(g, i + 4),googleMapsUri:maps(g),badgeLabel:'Popular place'})), routeKm: d.route?.distance_km || 0, routeSource: d.route?.source || '' };
}

export async function fetchExploreGems(city: string) {
  const params = new URLSearchParams({ city, limit: '5' });
  const response = await fetch(`/api/v2/gems/city?${params}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Nearby place suggestions are unavailable');
  const map = (rows: any[], badgeLabel: string) => (rows || []).slice(0, 5).map((place, index) => {
    const gem = toUiGem(place, index);
    return { ...gem, badgeLabel, googleMapsUri: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${place.name}, ${place.city}`)}` };
  });
  return { hidden: map(data.hidden_gems || data.gems || [], 'Hidden gem'), famous: map(data.famous_and_good || [], 'Popular place') };
}

export { api } from './services/backend';
