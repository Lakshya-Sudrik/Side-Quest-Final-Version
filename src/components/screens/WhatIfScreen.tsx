import React, { useEffect, useMemo, useState } from 'react';
import type { AppScreen } from '../../types';
import { api, isLoggedIn, type ItinerarySummary, type Plan, type RouteWeather, type WhatIfResult } from '../../services/backend';

interface Props { onNavigate: (screen: AppScreen) => void; onShowToast: (message: string, icon?: string) => void }
type Scenario = WhatIfResult['scenario'];

const scenarios: { id: Scenario; title: string }[] = [
  { id: 'rain', title: 'Heavy rain / possible disruption' },
  { id: 'traffic', title: 'Traffic delay' },
  { id: 'closure', title: 'Road disruption' },
  { id: 'late_start', title: 'Leave later and shift the plan' },
];

const field = 'w-full rounded-xl border border-[#d8d2bc] bg-[#f8f5eb] px-4 py-3 text-[15px] text-[#343723] outline-none focus:border-[#898861] focus:ring-2 focus:ring-[#898861]/15';
const panel = 'rounded-[22px] border border-[#d8d2bc] bg-[#fbf9f1]';

export const WhatIfScreen: React.FC<Props> = ({ onNavigate, onShowToast }) => {
  const [trips, setTrips] = useState<ItinerarySummary[]>([]);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planId, setPlanId] = useState('');
  const [scenario, setScenario] = useState<Scenario>('rain');
  const [delay, setDelay] = useState(30);
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [weather, setWeather] = useState<RouteWeather | null>(null);
  const [selectedReplacements, setSelectedReplacements] = useState<string[]>([]);
  const [applying, setApplying] = useState(false);
  const [planRevision, setPlanRevision] = useState(0);
  const [photo, setPhoto] = useState<{ url: string; credit?: string } | null>(null);
  const [loadingTrips, setLoadingTrips] = useState(true);
  const [loadingPlan, setLoadingPlan] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    if (!isLoggedIn()) { setLoadingTrips(false); return; }
    api.itineraries().then(async (rows) => {
      if (!active) return;
      setTrips(rows);
      let preferredId = '';
      try { preferredId = localStorage.getItem('sq_active_itinerary') || ''; } catch { /* use most recent trip */ }
      const first = rows.find((trip) => trip.id === preferredId) || rows[0];
      if (!first) return;
      setPlanId(first.id);
      try { localStorage.setItem('sq_active_itinerary', first.id); } catch { /* optional preference */ }
      setLoadingPlan(true);
      try { const loaded = await api.getPlan(first.id); if (active) setPlan(loaded); }
      catch (e) { if (active) setError(e instanceof Error ? e.message : 'Could not load this saved trip.'); }
      finally { if (active) setLoadingPlan(false); }
    }).catch((e) => {
      if (active) setError(e instanceof Error ? e.message : 'Your saved trips could not be loaded.');
    }).finally(() => { if (active) setLoadingTrips(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    setPhoto(null);
    if (!plan?.destination) return;
    const controller = new AbortController();
    fetch('/api/places/enrich', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ places: [{ name: plan.destination, city: '' }] }),
      signal: controller.signal,
    }).then((r) => r.ok ? r.json() : null).then((data) => {
      if (!active) return;
      const place = data?.results?.[0];
      if (place?.photoUrl?.startsWith('/api/')) setPhoto({ url: place.photoUrl, credit: place.photoAttributions?.[0]?.name });
    }).catch(() => { /* Image enrichment is optional; the route graphic remains. */ });
    return () => { active = false; controller.abort(); };
  }, [plan?.destination]);

  const selectedTrip = useMemo(() => trips.find((trip) => trip.id === planId), [trips, planId]);
  const laterForecast = scenario === 'late_start' && weather?.available
    ? weather.later_options.find((option) => option.hours_later === Math.round(delay / 60)) : undefined;
  const weatherDelayAssumption = weather?.available
    ? (Number(weather.max_precipitation_probability || 0) >= 60 || Number(weather.max_precipitation_mm || 0) >= 4 ? 30
      : Number(weather.max_precipitation_probability || 0) >= 35 || Number(weather.max_precipitation_mm || 0) >= 1 ? 15 : 0)
    : null;
  const weatherSummary = weather?.available
    ? `${weather.status}; maximum precipitation probability ${weather.max_precipitation_probability}%, up to ${weather.max_precipitation_mm} mm/hour along sampled route points.`
    : undefined;
  const changeTrip = async (id: string) => {
    setPlanId(id); setPlan(null); setResult(null); setError('');
    try { if (id) localStorage.setItem('sq_active_itinerary', id); } catch { /* optional preference */ }
    if (!id) return;
    setLoadingPlan(true);
    try { setPlan(await api.getPlan(id)); }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not load this saved trip.'); }
    finally { setLoadingPlan(false); }
  };
  const runScenario = async () => {
    if (!plan) return;
    setRunning(true); setError('');
    try { setResult(await api.whatIf(plan.id, scenario, delay, weatherSummary)); }
    catch (e) { setError(e instanceof Error ? e.message : 'The route scenario could not be calculated.'); }
    finally { setRunning(false); }
  };

  useEffect(() => {
    if (!plan || !isLoggedIn()) return;
    let active = true;
    setWeather(null);
    api.itineraryWeather(plan.id).then((value) => { if (active) setWeather(value); })
      .catch(() => { if (active) setWeather({ available: false, route_date: plan.depart_at.slice(0, 10),
        reason: 'Weather data could not be loaded for this route.', later_options: [] }); });
    return () => { active = false; };
  }, [plan?.id, plan?.depart_at]);

  // Recalculate as soon as the saved trip, scenario, or assumed delay changes.
  useEffect(() => {
    if (!plan || !isLoggedIn()) return;
    let active = true;
    setRunning(true);
    api.whatIf(plan.id, scenario, delay, weatherSummary).then((next) => {
      if (active) { setResult(next); setSelectedReplacements([]); }
    }).catch((e) => { if (active) setError(e instanceof Error ? e.message : 'The route scenario could not be calculated.'); })
      .finally(() => { if (active) setRunning(false); });
    return () => { active = false; };
  }, [plan?.id, plan?.delay_min, plan?.depart_at, scenario, delay, planRevision, weatherSummary]);

  const applyScenario = async () => {
    if (!plan || !result) return;
    setApplying(true); setError('');
    try {
      const updated = await api.applyWhatIf(plan.id, scenario, delay, selectedReplacements);
      setPlan(updated);
      setDelay(0);
      setPlanRevision((value) => value + 1);
      setTrips((current) => current.map((trip) => trip.id === updated.id ? {
        ...trip, depart_at: updated.depart_at, arrive_by: updated.arrive_by, delay_min: updated.delay_min,
      } : trip));
      onShowToast('Saved trip and Hidden Gems updated.', 'check_circle');
    } catch (e) { setError(e instanceof Error ? e.message : 'The updated trip could not be saved.'); }
    finally { setApplying(false); }
  };

  if (!isLoggedIn()) return <main className="mx-auto max-w-5xl px-5 py-20 md:px-10"><p className="text-xs font-semibold uppercase tracking-[.2em] text-[#7d2826]">SideQuest / For travellers</p><h1 className="mt-3 font-headline text-4xl text-[#343723] md:text-6xl">Think a few turns ahead.</h1><p className="mt-4 max-w-xl text-lg leading-7 text-[#595a4a]">Sign in to explore how a change could affect one of your saved trips.</p><button onClick={() => onNavigate('login')} className="mt-7 rounded-full bg-[#343723] px-6 py-3 font-semibold text-[#f8f5eb]">Sign in to continue</button></main>;

  return <main className="mx-auto w-full max-w-7xl px-5 pb-20 pt-8 md:px-10">
    <header className="mb-8 grid gap-6 border-b border-[#d8d2bc] pb-7 lg:grid-cols-[1fr_360px] lg:items-end">
      <div><p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-[#7d2826]">SideQuest / Trip scenarios</p><h1 className="font-headline text-4xl leading-[1.04] tracking-tight text-[#343723] md:text-6xl">Plans change.<br /><span className="text-[#898861]">See what shifts.</span></h1><p className="mt-4 max-w-2xl text-base leading-7 text-[#595a4a] md:text-lg">Try a delay against a saved route. SideQuest recalculates arrival time and checks which of your stops still fit.</p></div>
      <div className="relative min-h-[190px] overflow-hidden rounded-[22px] border border-[#d8d2bc] bg-[#e6e2d3]">
        {photo && <img src={photo.url} alt={`View near ${plan?.destination || 'your destination'}`} className="absolute inset-0 h-full w-full object-cover" onError={() => setPhoto(null)} />}
        {photo && <div className="absolute inset-0 bg-gradient-to-t from-[#25271d]/70 via-[#25271d]/10 to-transparent" />}
        <RouteArtwork origin={plan?.origin || 'Your start'} destination={plan?.destination || 'Your destination'} stops={plan?.schedule.stops.length || 0} photo={Boolean(photo)} />
        <div className={`absolute bottom-4 left-5 right-5 ${photo ? 'text-[#fffaf0]' : 'text-[#343723]'}`}><p className="text-[10px] font-semibold uppercase tracking-[.18em] opacity-80">{plan ? `${plan.origin} to ${plan.destination}` : 'A route with room to adapt'}</p><p className="mt-1 font-headline text-2xl">{plan ? `${Math.round(plan.route.distance_km)} km, at your pace` : 'The road is never just one plan.'}</p>{photo?.credit && <p className="mt-1 text-[10px] opacity-80">Photo: {photo.credit}</p>}</div>
      </div>
    </header>

    <div className="grid items-start gap-6 lg:grid-cols-[minmax(340px,.86fr)_minmax(0,1.14fr)]">
      <section className={`${panel} p-5 md:p-7`} aria-labelledby="scenario-title">
        <div className="flex items-start justify-between gap-3 border-b border-[#e0dccb] pb-5"><div><p className="text-[10px] font-semibold uppercase tracking-[.16em] text-[#898861]">Test a change</p><h2 id="scenario-title" className="mt-1 font-headline text-2xl text-[#343723]">What might the day look like?</h2></div><span className="material-symbols-outlined text-3xl text-[#898861]">alt_route</span></div>
        {loadingTrips ? <div className="mt-5 space-y-3" aria-busy="true" aria-label="Loading saved trips"><div className="h-12 rounded-xl bg-[#eeeadd]" /><div className="h-28 rounded-xl bg-[#eeeadd]" /></div> : trips.length === 0 ? <div className="py-7"><p className="font-headline text-xl">No saved route yet</p><p className="mt-2 text-sm leading-6 text-[#595a4a]">Save a trip in Route Planner first. Your saved route and stops will appear here for a preview you can apply to the trip.</p><button onClick={() => onNavigate('routes')} className="mt-5 rounded-full bg-[#343723] px-5 py-3 text-sm font-semibold text-[#f8f5eb]">Open Route Planner <span aria-hidden="true">↗</span></button></div> : <>
          <label className="mt-5 block text-sm font-semibold text-[#595a4a]" htmlFor="what-if-trip">Saved trip<select id="what-if-trip" className={`${field} mt-2`} value={planId} onChange={(e) => void changeTrip(e.target.value)}>{trips.map((trip) => <option key={trip.id} value={trip.id}>{trip.origin} → {trip.destination} · {trip.depart_at.slice(0, 10)}</option>)}</select></label>
          <label className="mt-6 block text-sm font-semibold text-[#595a4a]" htmlFor="what-if-scenario">What changed?<select id="what-if-scenario" className={`${field} mt-2`} value={scenario} onChange={(e) => { setScenario(e.target.value as Scenario); setResult(null); }}>{scenarios.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}<option value="custom">Other delay</option></select></label>
          <div className="mt-4 rounded-xl border border-[#ded9c7] bg-[#f8f5eb] p-4" aria-live="polite"><p className="text-xs font-semibold uppercase tracking-[.12em] text-[#898861]">Weather along this route · {weather?.route_date || plan?.depart_at.slice(0,10) || ''}</p>{weather?.available ? <><p className="mt-1 font-headline text-xl text-[#343723]">{laterForecast ? laterForecast.recommendation : weather.status}</p><p className="mt-1 text-sm text-[#595a4a]">{laterForecast ? `At ${laterForecast.hours_later}h later: up to ${laterForecast.precipitation_probability}% rain probability along sampled route points.` : `At planned departure: highest forecast rain probability ${weather.max_precipitation_probability}% · up to ${weather.max_precipitation_mm} mm/hour.`}</p>{scenario === 'rain' && weatherDelayAssumption !== null && <button type="button" onClick={() => setDelay(weatherDelayAssumption)} className="mt-3 rounded-full bg-[#e9e4d3] px-3 py-1.5 text-xs font-semibold text-[#343723]">Use a {weatherDelayAssumption}-minute planning buffer from this rain outlook</button>}{weather.later_options.length > 0 && <div className="mt-3 flex flex-wrap gap-2">{weather.later_options.map((option) => <button key={option.hours_later} type="button" onClick={() => { setScenario('late_start'); setDelay(option.hours_later * 60); }} className="rounded-full border border-[#aaa78c] px-3 py-1.5 text-xs font-semibold text-[#343723]">Check leaving {option.hours_later}h later · {option.precipitation_probability}%</button>)}</div>}<p className="mt-3 text-[11px] leading-5 text-[#6e6e5c]">{weather.notice} Source: <a href="https://open-meteo.com/" target="_blank" rel="noreferrer" className="underline underline-offset-2">Open-Meteo</a>.</p></> : <p className="mt-1 text-sm text-[#595a4a]">{weather?.reason || 'Checking the forecast…'}</p>}</div>
          <div className="mt-6 rounded-xl bg-[#eeeadb] p-4"><div className="flex items-end justify-between gap-3"><label htmlFor="what-if-delay" className="text-sm font-semibold text-[#343723]">{scenario === 'late_start' ? 'Leave later by' : 'Assumed extra travel time'}</label><span className="font-headline text-2xl text-[#7d2826]">{delay < 60 ? `${delay} min` : `${Math.floor(delay / 60)}h${delay % 60 ? ` ${delay % 60}m` : ''}`}</span></div><input id="what-if-delay" type="range" min="0" max="360" step="15" value={delay} onChange={(e) => { setDelay(Number(e.target.value)); setResult(null); }} className="mt-4 w-full accent-[#7d2826]"/><div className="mt-1 flex justify-between text-[11px] text-[#6e6e5c]"><span>No change</span><span>6 hours</span></div><p className="mt-3 text-xs leading-5 text-[#62634c]">{scenario === 'late_start' ? 'This shifts both departure and arrive-by times, then checks your fixed stops.' : 'Enter a planning assumption. Traffic and road closure are not live feeds, and the route is not rerouted around closures.'}</p></div>
          <p className="mt-4 text-xs leading-5 text-[#62634c]">Weather supports rain and timing decisions only. Traffic and road disruption are manual what-if scenarios; no live closure or flood-warning feed is connected.</p>
          <button type="button" disabled={!plan || loadingPlan || running} onClick={() => void runScenario()} className="mt-5 flex min-h-12 w-full items-center justify-center gap-2 rounded-xl bg-[#343723] px-5 text-sm font-semibold text-[#f8f5eb] disabled:cursor-not-allowed disabled:opacity-55">{running ? 'Updating this trip…' : 'Refresh this comparison'}</button>
        </>}
        {error && <p role="alert" className="mt-4 rounded-lg border border-[#c98578] bg-[#f7e7e1] px-4 py-3 text-sm text-[#7d2826]">{error}</p>}
      </section>

      <section aria-live="polite" className="min-w-0">
        {!result ? <div className={`${panel} min-h-[390px] overflow-hidden`}><div className="relative flex min-h-[230px] items-center justify-center bg-[#e7e3d5] px-6"><RouteArtwork origin={plan?.origin || 'Start'} destination={plan?.destination || 'Finish'} stops={plan?.schedule.stops.length || 2} /><div className="absolute bottom-5 left-6"><p className="text-[10px] font-semibold uppercase tracking-[.18em] text-[#898861]">A more considered route</p><p className="mt-1 font-headline text-2xl text-[#343723]">Keep the parts that matter.</p></div></div><div className="p-6 md:p-8"><p className="text-xs font-semibold uppercase tracking-[.16em] text-[#7d2826]">Your trip, recalculated</p><h2 className="mt-2 font-headline text-3xl text-[#343723]">A clearer plan B.</h2><p className="mt-3 max-w-xl text-sm leading-6 text-[#595a4a]">Choose a saved trip and a delay to see the revised arrival, stops that still fit, and alternatives along the same route.</p><div className="mt-6 flex flex-wrap gap-x-6 gap-y-2 border-t border-[#e0dccb] pt-4 text-xs text-[#666753]"><span>Saved route timings</span><span>Schedule-aware stop selection</span><span>Preview first, apply when ready</span></div></div></div> : <div className={`${panel} overflow-hidden`}>
          <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[#e0dccb] p-5 md:p-7"><div><p className="text-[10px] font-semibold uppercase tracking-[.16em] text-[#7d2826]">SideQuest schedule simulation</p><h2 className="mt-1 font-headline text-2xl text-[#343723] md:text-3xl">{selectedTrip?.origin} to {selectedTrip?.destination}</h2><p className="mt-2 text-sm text-[#656553]">{scenarioLabel(result.scenario)} · {result.assumed_delay_min} min added</p></div><span className="rounded-full bg-[#e9e6d7] px-3 py-1.5 text-xs font-semibold text-[#5c6044]">Schedule model</span></div>
          <div className="grid sm:grid-cols-2"><ArrivalColumn title="Your saved plan" time={result.baseline.arrive_destination} deadline={result.baseline.deadline} fits={result.baseline.fits} slack={result.baseline.slack_min} stops={result.baseline.stops.length} /><ArrivalColumn title="With this change" time={result.scenario_plan.arrive_destination} deadline={result.scenario_plan.deadline} fits={result.scenario_plan.fits} slack={result.scenario_plan.slack_min} stops={result.scenario_plan.stops.length} emphasized /></div>
          <div className="border-t border-[#e0dccb] p-5 md:p-7"><div className="flex items-start gap-3"><span className="material-symbols-outlined mt-0.5 text-[#898861]">route</span><div><h3 className="font-semibold text-[#343723]">Stops that still fit</h3>{result.kept.length ? <p className="mt-1 text-sm leading-6 text-[#595a4a]">{result.kept.join(' · ')}</p> : <p className="mt-1 text-sm text-[#595a4a]">No saved detours fit this time window; the direct route remains.</p>}</div></div>
            <NugenImpactPanel result={result.nugen} />
            {result.dropped.length > 0 && <div className="mt-5 rounded-xl border border-[#d7b9a8] bg-[#f5e8df] p-4"><p className="text-sm font-semibold text-[#743b31]">These stops no longer fit</p><ul className="mt-2 space-y-1 text-sm text-[#595a4a]">{result.dropped.map((stop) => <li key={stop.place_id} className="flex items-center justify-between gap-3"><span>{stop.name}</span><span className="shrink-0 text-xs">{stop.detour_min + stop.visit_min} min</span></li>)}</ul></div>}
            {result.suggested_replacements.length > 0 && <div className="mt-5"><p className="text-sm font-semibold text-[#343723]">Add a shorter Hidden Gem to the revised trip</p><p className="mt-1 text-xs text-[#6e6e5c]">The safety model filters out places it marks “avoid”; other classes are evidence-based context, not a guarantee.</p><div className="mt-2 divide-y divide-[#e0dccb]">{result.suggested_replacements.slice(0, 5).map((stop) => <label key={stop.place_id} className="flex cursor-pointer flex-wrap items-center justify-between gap-2 py-3"><span className="flex items-center gap-3"><input type="checkbox" checked={selectedReplacements.includes(stop.place_id)} onChange={(e) => setSelectedReplacements((ids) => e.target.checked ? [...ids, stop.place_id] : ids.filter((id) => id !== stop.place_id))} className="h-4 w-4 accent-[#7d2826]"/><span><span className="block font-headline text-lg text-[#343723]">{stop.name}</span><span className="text-xs text-[#6e6e5c]">{stop.city} · {stop.detour_min + stop.visit_min} min total · Safety: {stop.safety?.class || 'not rated'}</span></span></span><span className="text-xs font-semibold text-[#898861]">SQ {stop.gem_score.toFixed(1)}</span></label>)}</div></div>}
            <div className="mt-5 border-t border-[#e0dccb] pt-4"><p className="text-xs leading-5 text-[#6e6e5c]">{result.notice} Schedule precision also depends on the saved route’s ETA quality.</p><button type="button" disabled={applying || running} onClick={() => void applyScenario()} className="mt-4 min-h-12 w-full rounded-xl bg-[#7d2826] px-5 text-sm font-semibold text-white disabled:opacity-55">{applying ? 'Updating saved trip…' : 'Apply changes to my travel plan'}</button><p className="mt-2 text-center text-[11px] text-[#6e6e5c]">This updates the saved itinerary and its Hidden Gems list.</p></div>
          </div>
        </div>}
      </section>
    </div>
  </main>;
};

function NugenImpactPanel({ result }: { result: WhatIfResult['nugen'] }) {
  const available = Boolean(result?.available);
  const label = result?.state === 'alignment_processing' ? 'Model alignment in progress'
    : result?.state === 'model_unavailable' ? 'No deployed model'
      : result?.state === 'inference_unavailable' ? 'Inference unavailable'
        : result?.state === 'api_unavailable' ? 'Nugen API unavailable'
          : 'SideQuest-aligned model estimate';
  const recommendation = result?.recommendation?.replaceAll('_', ' ');
  const metrics = available ? [
    ['Access outlook', result?.accessibility?.replaceAll('_', ' ') || 'unknown'],
    ['Safety impact', result?.safety_impact || 'unknown'],
    ['Added-delay likelihood', result?.delay_prob == null ? 'Insufficient data' : `${Math.round(result.delay_prob * 100)}%`],
    ['Visitor-demand estimate', result?.demand_shift_pct == null ? 'Insufficient data' : `${result.demand_shift_pct > 0 ? '+' : ''}${Math.round(result.demand_shift_pct)}%`],
  ] as const : [];
  return <section className={`mt-5 rounded-xl border p-4 ${available ? 'border-[#c9c8aa] bg-[#f3f0e4]' : 'border-[#d9aaa0] bg-[#f7e8e2]'}`} aria-live="polite" aria-labelledby="nugen-impact-title">
    <div className="flex flex-wrap items-center justify-between gap-2"><div><p id="nugen-impact-title" className="text-xs font-semibold uppercase tracking-[.15em] text-[#343723]">Nugen travel impact</p><p className={`mt-1 text-xs font-semibold ${available ? 'text-[#596143]' : 'text-[#8d302c]'}`}>{label}</p></div>{available && result?.confidence && <span className="rounded-full bg-white/70 px-2.5 py-1 text-[11px] font-medium capitalize text-[#5c6044]">{result.confidence} confidence</span>}</div>
    {available ? <>
      <dl className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-[#ddd8c5] bg-[#ddd8c5] sm:grid-cols-4">{metrics.map(([title, value]) => <div key={title} className="bg-[#fbf9f1] px-3 py-3"><dt className="text-[10px] font-semibold uppercase tracking-[.1em] text-[#777760]">{title}</dt><dd className="mt-1 text-sm font-semibold capitalize text-[#343723]">{value}</dd></div>)}</dl>
      {recommendation && <p className="mt-3 text-sm font-semibold capitalize text-[#343723]">Suggested response: {recommendation}</p>}
      {result?.guidance && <p className="mt-2 text-sm leading-6 text-[#454735]">{result.guidance}</p>}
      {!!result?.cascades?.length && <div className="mt-3 border-t border-[#d9d5c4] pt-3"><p className="text-[10px] font-semibold uppercase tracking-[.12em] text-[#777760]">What changes in your plan</p><ul className="mt-2 space-y-1.5 text-sm leading-5 text-[#454735]">{result.cascades.map((item, index) => <li key={`${index}-${item}`} className="flex gap-2"><span aria-hidden="true" className="text-[#8c332e]">•</span><span>{item}</span></li>)}</ul></div>}
      <p className="mt-3 text-[11px] leading-5 text-[#777760]">Scenario estimates only; not a calibrated forecast, live closure alert, or safety guarantee. Verify conditions with local authorities.</p>
    </> : <p className="mt-2 text-sm leading-6 text-[#69433c]">{result?.message || 'Nugen did not return an answer.'} SideQuest still shows the schedule comparison calculated from your saved trip.</p>}
  </section>;
}

function ArrivalColumn({ title, time, deadline, fits, slack, stops, emphasized = false }: { title: string; time: string; deadline: string; fits: boolean; slack: number; stops: number; emphasized?: boolean }) {
  return <div className={`p-5 md:p-7 ${emphasized ? 'bg-[#eeeadb]' : ''}`}><p className="text-[10px] font-semibold uppercase tracking-[.15em] text-[#898861]">{title}</p><p className="mt-2 font-headline text-3xl text-[#343723]">{clock(time)}</p><p className="mt-1 text-xs text-[#6e6e5c]">Arrive by {clock(deadline)}</p><div className="mt-4 flex flex-wrap items-center gap-2"><span className={`rounded-full px-3 py-1 text-xs font-semibold ${fits ? 'bg-[#e2e7d5] text-[#454b32]' : 'bg-[#f1ddd4] text-[#7d2826]'}`}>{fits ? `${Math.max(slack, 0)} min spare` : `${Math.abs(slack)} min past deadline`}</span><span className="rounded-full bg-white/65 px-3 py-1 text-xs text-[#595a4a]">{stops} {stops === 1 ? 'stop' : 'stops'} fit</span></div></div>;
}

function RouteArtwork({ origin, destination, stops, photo = false }: { origin: string; destination: string; stops: number; photo?: boolean }) {
  return <div aria-hidden="true" className={`absolute inset-0 overflow-hidden ${photo ? 'opacity-65' : ''}`}><svg viewBox="0 0 520 220" preserveAspectRatio="xMidYMid slice" className={`h-full w-full ${photo ? 'opacity-40' : ''}`}><path d="M-15 155 C80 150 70 65 160 82 S255 175 330 116 420 50 535 78" fill="none" stroke={photo ? '#f8f5eb' : '#aaa78c'} strokeWidth="2" strokeDasharray="5 8"/><path d="M-15 155 C80 150 70 65 160 82 S255 175 330 116 420 50 535 78" fill="none" stroke={photo ? '#f8f5eb' : '#7d2826'} strokeWidth="2.5" strokeDasharray="120 600" strokeLinecap="round"/><circle cx="38" cy="149" r="6" fill={photo ? '#f8f5eb' : '#7d2826'}/><circle cx="481" cy="72" r="6" fill={photo ? '#f8f5eb' : '#343723'}/>{Array.from({ length: Math.max(0, Math.min(stops, 4)) }).map((_, i) => <circle key={i} cx={130 + i * 72} cy={84 + (i % 2) * 50} r="4" fill={photo ? '#f8f5eb' : '#898861'}/>)}</svg><span className={`absolute left-5 top-5 text-[9px] font-semibold uppercase tracking-[.16em] ${photo ? 'text-white' : 'text-[#6e6e5c]'}`}>{origin}</span><span className={`absolute right-5 top-5 text-[9px] font-semibold uppercase tracking-[.16em] ${photo ? 'text-white' : 'text-[#6e6e5c]'}`}>{destination}</span></div>;
}

function clock(value: string) {
  const parsed = new Date(value.replace(' ', 'T') + (value.length === 16 ? ':00' : ''));
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function scenarioLabel(id: Scenario) { return scenarios.find((item) => item.id === id)?.title || 'Custom change'; }
