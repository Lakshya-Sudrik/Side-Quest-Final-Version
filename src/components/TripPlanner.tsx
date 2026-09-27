import React, { forwardRef, useImperativeHandle, useState } from 'react';
import { api, isLoggedIn, label, Plan, PlanStop } from '../services/backend';

export interface TripPlannerHandle {
  hasPlan: () => boolean;
  add: (placeId: string, name: string) => Promise<void>;
}

interface Props {
  from: string;
  to: string;
  onShowToast: (msg: string, icon?: string) => void;
  onNavigate: (screen: any) => void;
}

const card = 'bg-white rounded-2xl p-4 border border-[#e7eeff] shadow-[0_2px_14px_rgba(0,0,0,0.04)] space-y-2';
const field = 'w-full bg-[#f0f3ff] rounded-xl px-3 py-2 text-xs text-[#111c2d] border border-[#dee8ff] focus:outline-none focus:ring-2 focus:ring-[#00685f]';
// tomorrow at the given hour, in the user's local time (datetime-local format)
const tomorrowAt = (hour: number) => {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  d.setHours(hour, 0, 0, 0);
  return new Date(d.getTime() - d.getTimezoneOffset() * 60e3).toISOString().slice(0, 16);
};
const hm = (min: number) => `${Math.floor(Math.abs(min) / 60)}h ${Math.abs(min) % 60}m`;

/** Dynamic trip plan: add/remove stops, live timeline, flight-delay re-planning. */
export const TripPlanner = forwardRef<TripPlannerHandle, Props>(({ from, to, onShowToast, onNavigate }, ref) => {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [depart, setDepart] = useState(tomorrowAt(8));
  const [arrive, setArrive] = useState(tomorrowAt(20));
  const [delay, setDelay] = useState('');
  const [suggest, setSuggest] = useState<PlanStop[]>([]);
  const [busy, setBusy] = useState(false);

  const fail = (e: unknown) => onShowToast((e as Error).message, 'error');

  useImperativeHandle(ref, () => ({
    hasPlan: () => plan !== null,
    add: async (placeId: string, name: string) => {
      if (!plan) return;
      try {
        const p = await api.addStop(plan.id, placeId);
        setPlan(p);
        setSuggest((s) => s.filter((x) => x.place_id !== placeId));
        onShowToast(p.warning ?? `Added "${name}" to your plan`, p.warning ? 'warning' : 'check_circle');
      } catch (e) { fail(e); }
    },
  }), [plan]);

  if (!isLoggedIn()) {
    return (
      <div className={card}>
        <p className="text-sm font-extrabold text-[#111c2d]">Plan your stops</p>
        <p className="text-xs text-[#3d4947]">Log in to build a timed plan, add or remove stops, and re-plan if your flight is delayed.</p>
        <button onClick={() => onNavigate('login')} className="px-4 py-2 bg-[#00685f] text-white rounded-xl text-xs font-bold">Log in</button>
      </div>
    );
  }

  const create = async () => {
    if (!from.trim() || !to.trim()) return onShowToast('Enter From and To above first', 'warning');
    setBusy(true);
    try {
      const p = await api.createPlan(from.trim(), to.trim(), depart, arrive);
      setPlan(p);
      setSuggest((await api.planSuggestions(p.id)).suggestions);
      onShowToast(`Plan created: ${hm(p.schedule.free_time_min)} free for stops`, 'schedule');
    } catch (e) { fail(e); } finally { setBusy(false); }
  };

  const remove = async (s: PlanStop) => {
    try {
      const p = await api.removeStop(plan!.id, s.place_id);
      setPlan(p);
      setSuggest((await api.planSuggestions(p.id)).suggestions);
      onShowToast(`Removed "${s.name}"`, 'remove_circle');
    } catch (e) { fail(e); }
  };

  const applyDelay = async () => {
    const m = Number(delay);
    if (!Number.isFinite(m) || m < 0) return onShowToast('Enter the delay in minutes', 'warning');
    setBusy(true);
    try {
      const p = await api.delay(plan!.id, m);
      setPlan(p);
      setSuggest(p.suggested_replacements ?? []);
      onShowToast(p.message ?? 'Plan updated', 'flight');
    } catch (e) { fail(e); } finally { setBusy(false); }
  };

  const addSuggested = async (s: PlanStop) => {
    try {
      const p = await api.addStop(plan!.id, s.place_id);
      setPlan(p);
      setSuggest((await api.planSuggestions(p.id)).suggestions);
      onShowToast(p.warning ?? `Added "${s.name}"`, p.warning ? 'warning' : 'check_circle');
    } catch (e) { fail(e); }
  };

  if (!plan) {
    return (
      <div className={card}>
        <p className="text-sm font-extrabold text-[#111c2d]">Plan your stops ({from || 'From'} → {to || 'To'})</p>
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[10px] font-bold text-[#3d4947]">Leaving at
            <input type="datetime-local" value={depart} onChange={(e) => setDepart(e.target.value)} className={field} /></label>
          <label className="text-[10px] font-bold text-[#3d4947]">Must arrive by
            <input type="datetime-local" value={arrive} onChange={(e) => setArrive(e.target.value)} className={field} /></label>
        </div>
        <button onClick={create} disabled={busy} className="w-full py-2.5 bg-[#00685f] text-white rounded-xl text-xs font-bold disabled:opacity-60">
          {busy ? 'Planning…' : 'Create timed plan'}
        </button>
      </div>
    );
  }

  const sch = plan.schedule;
  return (
    <div className={card}>
      <div className="flex items-center justify-between">
        <p className="text-sm font-extrabold text-[#111c2d]">{plan.origin} → {plan.destination}</p>
        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${sch.fits ? 'bg-[#dcfce7] text-[#14532d]' : 'bg-[#fee2e2] text-[#7f1d1d]'}`}>
          {sch.fits ? `${hm(sch.slack_min)} spare` : `${hm(sch.slack_min)} late`}
        </span>
      </div>
      <p className="text-[11px] text-[#3d4947]">
        Leave {sch.start.slice(11)}{plan.delay_min ? ` (after ${plan.delay_min} min delay)` : ''} · {plan.route.distance_km} km ·
        arrive {sch.arrive_destination.slice(11)} (by {sch.deadline.slice(11)})
      </p>
      <p className="text-[10px] text-[#6d7a77]">{sch.distance_note}</p>

      <ol className="space-y-1.5">
        {sch.stops.length === 0 && <li className="text-xs text-[#3d4947]">No stops yet - tap "Take this Gem" on a suggestion above, or add one below.</li>}
        {sch.stops.map((s) => (
          <li key={s.place_id} className="flex items-center justify-between gap-2 bg-[#f8faff] rounded-xl px-3 py-2 text-xs">
            <span>
              <b>{s.arrive?.slice(11)}–{s.leave?.slice(11)}</b> · {s.name}
              <span className="block text-[10px] text-[#6d7a77]">{s.kind === 'eat' ? 'Eat' : 'Visit'} · {label(s.category)} · {s.visit_min} min + {s.detour_min} min detour</span>
            </span>
            <button onClick={() => remove(s)} className="text-[11px] font-bold text-[#b91c1c]">Remove</button>
          </li>
        ))}
      </ol>

      <div className="flex items-end gap-2 pt-1">
        <label className="flex-1 text-[10px] font-bold text-[#3d4947]">Flight / departure delayed by (minutes)
          <input type="number" min={0} max={1440} value={delay} onChange={(e) => setDelay(e.target.value)} placeholder="e.g. 120" className={field} /></label>
        <button onClick={applyDelay} disabled={busy} className="px-3 py-2 bg-[#ac3400] text-white rounded-xl text-xs font-bold disabled:opacity-60">Re-plan</button>
      </div>
      {plan.message && <p className="text-xs text-[#111c2d] bg-[#fffbeb] border border-[#fcd34d] rounded-lg px-3 py-2">{plan.message}</p>}
      {!!plan.dropped?.length && (
        <p className="text-[11px] text-[#7f1d1d]">Dropped: {plan.dropped.map((d) => d.name).join(', ')}</p>
      )}

      {suggest.length > 0 && (
        <div className="space-y-1 pt-1">
          <p className="text-xs font-extrabold text-[#111c2d]">Gems that fit your remaining time</p>
          {suggest.map((s) => (
            <div key={s.place_id} className="flex items-center justify-between gap-2 text-xs">
              <span>{s.name} <span className="text-[10px] text-[#6d7a77]">({s.kind === 'eat' ? 'eat' : 'visit'}, {s.city}, {s.visit_min + s.detour_min} min)</span></span>
              <button onClick={() => addSuggested(s)} className="px-2 py-1 bg-[#4648d4] text-white rounded-lg text-[11px] font-bold">Add</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});
TripPlanner.displayName = 'TripPlanner';
