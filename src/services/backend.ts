// Client for the SideQuest backend (/api/v2, served by backend/app.py via the Express proxy).

export type Role = 'user' | 'collaborator' | 'admin';

export interface Preferences {
  travel_style: string[];
  budget: string;
  pace: string;
  food: string;
  languages: string[];
  companion_gender: 'any' | 'same';
  companion_age_min: number;
  companion_age_max: number;
}

export interface Me {
  id: string;
  email: string;
  role: Role;
  name: string;
  avatar_url?: string | null;
  age?: number | null;
  gender?: string | null;
  preferences?: Preferences | null;
  collaborator?: {
    business_name: string;
    business_type: string;
    gstin?: string | null;
    gstin_check: string;
    gstin_state?: string | null;
    document_type?: string | null;
    verification_status: 'pending' | 'verified' | 'rejected';
    review_notes?: string | null;
  } | null;
}

export interface Trip {
  id: string;
  destination: string;
  start_date: string;
  end_date: string;
  solo_match: number;
}

export interface Candidate {
  user_id: string;
  trip_id: string;
  first_name: string;
  avatar_url?: string | null;
  age_band: string;
  gender: string;
  preferences: Pick<Preferences, 'travel_style' | 'budget' | 'pace' | 'food' | 'languages'>;
  destination: string;
  overlap: { from: string; to: string; days: number };
  compatibility: number;
  reasons: string[];
  request_state: 'none' | 'request_sent' | 'request_received' | 'accepted' | 'declined' | 'cancelled';
  request_id: string | null;
}

export interface MatchSummary {
  match_id: string;
  with_user: string;
  with_name: string;
  avatar_url?: string | null;
  destination: string;
  overlap: { from: string; to: string; days: number } | null;
  last_message: { body: string; created_at: string; sender_id: string } | null;
}

export interface ChatMessage {
  id: number;
  sender_id: string;
  body: string;
  created_at: string;
  mine: boolean;
}

export interface PlanStop {
  place_id: string;
  name: string;
  kind: 'eat' | 'visit';
  category: string;
  city: string;
  gem_score: number;
  off_route_km: number;
  along_km: number;
  detour_min: number;
  visit_min: number;
  arrive?: string;
  leave?: string;
  rating?: number | null;
  safety?: { class?: string; score?: number | null; reasons?: string[] };
}

export interface Plan {
  id: string;
  origin: string;
  destination: string;
  depart_at: string;
  arrive_by: string;
  delay_min: number;
  route: { distance_km: number; duration_min: number; source: string };
  schedule: {
    start: string;
    arrive_destination: string;
    deadline: string;
    slack_min: number;
    fits: boolean;
    free_time_min: number;
    stops: PlanStop[];
    distance_note: string;
  };
  warning?: string;
  message?: string;
  dropped?: PlanStop[];
  suggested_replacements?: PlanStop[];
  planning_data?: { transport?: string; travellers?: number; interests?: string[]; fixed_stops?: {time:string;place:string}[] };
}

export interface ItinerarySummary {
  id: string;
  origin: string;
  destination: string;
  depart_at: string;
  arrive_by: string;
  delay_min: number;
}

export interface WhatIfResult {
  scenario: 'rain' | 'traffic' | 'closure' | 'late_start' | 'custom';
  assumed_delay_min: number;
  baseline: Plan['schedule'];
  scenario_plan: Plan['schedule'];
  kept: string[];
  dropped: PlanStop[];
  suggested_replacements: PlanStop[];
  notice: string;
  nugen?: {
    available: boolean;
    state?: 'ready' | 'alignment_processing' | 'model_unavailable' | 'api_unavailable' | 'inference_unavailable';
    message: string;
    guidance?: string;
    accessibility?: 'likely_open' | 'limited' | 'unknown';
    delay_prob?: number | null;
    demand_shift_pct?: number | null;
    safety_impact?: 'low' | 'moderate' | 'high' | 'unknown';
    cascades?: string[];
    recommendation?: 'continue_with_caution' | 'wait_for_update' | 'consider_alternative' | 'insufficient_data';
    confidence?: 'low' | 'medium' | 'high';
  };
}

export interface RouteWeather {
  available: boolean;
  reason?: string;
  provider?: string;
  route_date: string;
  departure?: string;
  status?: string;
  max_precipitation_probability?: number;
  max_precipitation_mm?: number;
  later_options: { hours_later: number; precipitation_probability: number; recommendation: string }[];
  notice?: string;
}

// The token lives in memory; localStorage only remembers it across reloads when available.
let token: string | null = null;
try {
  token = localStorage.getItem('sq_token') || sessionStorage.getItem('sq_token');
} catch {
  token = null;
}

export function setToken(t: string | null, remember = true) {
  token = t;
  try {
    localStorage.removeItem('sq_token'); sessionStorage.removeItem('sq_token');
    if (t) (remember ? localStorage : sessionStorage).setItem('sq_token', t);
  } catch {
    /* storage unavailable - session stays in memory */
  }
}

export const isLoggedIn = () => Boolean(token);

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (init.body && !(init.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const res = await fetch(`/api/v2${path}`, { ...init, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401) setToken(null);
    const d = (data as any)?.detail;
    const msg = Array.isArray(d)
      ? d.map((x: any) => (typeof x === 'string' ? x : `${(x.loc || []).slice(-1)[0]}: ${x.msg}`)).join('; ')
      : d || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data as T;
}

const post = <T>(path: string, body: unknown) => call<T>(path, { method: 'POST', body: JSON.stringify(body) });

export const api = {
  options: () => call<{ questions: Record<string, { question: string; values: string[]; multi?: boolean; max?: number }> }>(
    '/preferences/options'),
  register: async (body: Record<string, unknown>) => {
    const r = await post<{ token: string; user: Me; gstin_check?: any; next_step?: string }>('/auth/register', body);
    setToken(r.token);
    return r;
  },
  login: async (email: string, password: string, role?: Role, remember = true) => {
    const r = await post<{ token: string; user: Me }>('/auth/login', { email, password, role });
    setToken(r.token, remember);
    return r;
  },
  logout: () => setToken(null),
  me: () => call<Me>('/me'),
  updateProfile: (profile: { age: number; gender: string }) =>
    call<Me>('/me/profile', { method: 'PUT', body: JSON.stringify(profile) }),
  updatePreferences: (p: Preferences) => call<Me>('/me/preferences', { method: 'PUT', body: JSON.stringify(p) }),
  uploadAvatar: (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    return call<{ avatar_url: string }>('/me/avatar', { method: 'POST', body: fd });
  },

  trips: () => call<Trip[]>('/trips'),
  createTrip: (destination: string, start_date: string, end_date: string, solo_match: boolean) =>
    post<Trip>('/trips', { destination, start_date, end_date, solo_match }),
  setSolo: (id: string, solo_match: boolean) =>
    call<Trip>(`/trips/${id}`, { method: 'PATCH', body: JSON.stringify({ solo_match }) }),
  candidates: (tripId: string) =>
    call<{ candidates: Candidate[]; note: string }>(`/solo/candidates?trip_id=${encodeURIComponent(tripId)}`),
  sendRequest: (trip_id: string, to_user: string, their_trip_id: string) =>
    post<{ request_id: string; status: string; match_id?: string }>('/solo/requests', { trip_id, to_user, their_trip_id }),
  respond: (requestId: string, accept: boolean) =>
    post<{ status: string; match_id?: string }>(`/solo/requests/${requestId}/respond`, { accept }),
  requests: () => call<{ incoming: any[]; outgoing: any[] }>('/solo/requests'),
  matches: () => call<MatchSummary[]>('/matches'),
  messages: (matchId: string, after = 0) => call<ChatMessage[]>(`/matches/${matchId}/messages?after=${after}`),
  send: (matchId: string, body: string) => post<{ id: number }>(`/matches/${matchId}/messages`, { body }),
  block: (user_id: string) => post('/block', { user_id }),
  report: (user_id: string, reason: string) => post('/report', { user_id, reason }),

  collaboratorStatus: () => call<{ collaborator: Me['collaborator']; listings: any[] }>('/collaborator/status'),
  uploadDocument: (docType: string, file: File, latitude?:number, longitude?:number) => {
    const fd = new FormData();
    fd.append('doc_type', docType);
    fd.append('file', file);
    if(latitude!==undefined) fd.append('latitude',String(latitude));
    if(longitude!==undefined) fd.append('longitude',String(longitude));
    return call<{ status: string; message: string }>('/collaborator/documents', { method: 'POST', body: fd });
  },
  createListing: (body: Record<string, unknown>) => post<any>('/listings', body),
  uploadListingPhoto: (listingId: string, file: File) => {
    const fd = new FormData(); fd.append('file', file);
    return call<{photo_url:string}>(`/collaborator/listings/${encodeURIComponent(listingId)}/photo`, {method:'POST',body:fd});
  },
  adminOverview: () => call<any>('/admin/overview'),
  adminModelStatus: () => call<any>('/admin/model-status'),
  adminNugenPredict: (prompt: string) => post<{ impact: Omit<NonNullable<WhatIfResult['nugen']>, 'available' | 'state' | 'message'> }>('/admin/nugen/predict', { prompt }),
  adminReviewListing: (id:string, approve:boolean, notes?:string) => post<any>(`/admin/listings/${encodeURIComponent(id)}/review`,{approve,notes}),
  downloadAdminDocument: async (id:string) => {
    const res=await fetch(`/api/v2/admin/collaborator-documents/${encodeURIComponent(id)}`,{headers:token?{Authorization:`Bearer ${token}`}:{}});
    if(!res.ok) throw new Error('Document download was denied or unavailable.');
    const blob=await res.blob(); const url=URL.createObjectURL(blob); const a=document.createElement('a');a.href=url;a.download='collaborator-document';a.click();window.setTimeout(()=>URL.revokeObjectURL(url),30000);
  },
  recordListingView: (id:string) => post<any>(`/listings/${encodeURIComponent(id)}/view`,{}),
  rateListing: (id:string,score:number) => post<any>(`/listings/${encodeURIComponent(id)}/rating`,{score}),
  adminCollaborators: () => call<any[]>('/admin/collaborators'),
  adminVerify: (uid: string, approve: boolean, notes?: string) =>
    post(`/admin/collaborators/${uid}/verify`, { approve, notes }),
  ragSearch: (query: string, city = '', kind = '') => post<any>('/rag/search', { query, city, kind, top_k: 5 }),

  // dynamic trip planner
  createPlan: (origin: string, destination: string, depart_at: string, arrive_by: string, planning_data: Record<string, unknown> = {}) =>
    post<Plan>('/itineraries', { origin, destination, depart_at, arrive_by, ...planning_data }),
  itineraries: () => call<ItinerarySummary[]>('/itineraries'),
  getPlan: (id: string) => call<Plan>(`/itineraries/${id}`),
  whatIf: (id: string, scenario: WhatIfResult['scenario'], delay_min: number, weather_summary?: string) =>
    post<WhatIfResult>(`/itineraries/${encodeURIComponent(id)}/what-if`, { scenario, delay_min, weather_summary }),
  itineraryWeather: (id: string) => call<RouteWeather>(`/itineraries/${encodeURIComponent(id)}/weather`),
  applyWhatIf: (id: string, scenario: WhatIfResult['scenario'], delay_min: number, add_place_ids: string[] = []) =>
    post<Plan>(`/itineraries/${encodeURIComponent(id)}/what-if/apply`, { scenario, delay_min, add_place_ids }),
  addStop: (id: string, place_id: string) => post<Plan>(`/itineraries/${id}/stops`, { place_id }),
  removeStop: (id: string, place_id: string) =>
    call<Plan>(`/itineraries/${id}/stops/${encodeURIComponent(place_id)}`, { method: 'DELETE' }),
  delay: (id: string, minutes: number) => post<Plan>(`/itineraries/${id}/delay`, { minutes }),
  planSuggestions: (id: string) => call<{ slack_min: number; suggestions: PlanStop[] }>(`/itineraries/${id}/suggestions`),
};

export const label = (s: string) => s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
