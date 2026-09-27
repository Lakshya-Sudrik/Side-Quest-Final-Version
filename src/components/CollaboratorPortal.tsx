import React, { useEffect, useState } from 'react';
import { api, isLoggedIn, label, Me } from '../services/backend';

interface Props {
  onShowToast: (msg: string, icon?: string) => void;
  onNavigate: (screen: any) => void;
}

const card = 'bg-white rounded-2xl p-4 border border-[#e7eeff] shadow-[0_2px_14px_rgba(0,0,0,0.04)] space-y-2';
const field = 'w-full bg-[#f0f3ff] rounded-xl px-3 py-2 text-xs text-[#111c2d] border border-[#dee8ff] focus:outline-none focus:ring-2 focus:ring-[#ac3400]';
const DECISION: Record<string, { text: string; cls: string }> = {
  hidden_gem: { text: 'Hidden gem', cls: 'bg-[#dcfce7] text-[#14532d] border-[#86efac]' },
  not_a_gem: { text: 'Not a hidden gem', cls: 'bg-[#f1f5f9] text-[#334155] border-[#cbd5e1]' },
  needs_reviews: { text: 'Needs visitor reviews', cls: 'bg-[#fffbeb] text-[#78350f] border-[#fcd34d]' },
};

/** Real collaborator workflow: verification + model-evaluated listings (backend /api/v2). */
export const CollaboratorPortal: React.FC<Props> = ({ onShowToast, onNavigate }) => {
  const [me, setMe] = useState<Me | null>(null);
  const [listings, setListings] = useState<any[]>([]);
  const [docType, setDocType] = useState('gst_certificate');
  const [file, setFile] = useState<File | null>(null);
  const [form, setForm] = useState({ name: '', kind: 'eat', town: '', category: '', cost_inr: '', description: '' });
  const [result, setResult] = useState<any | null>(null);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<any[]>([]);

  const fail = (e: unknown) => onShowToast((e as Error).message, 'error');

  const load = async () => {
    try {
      const m = await api.me();
      setMe(m);
      if (m.role === 'collaborator') setListings((await api.collaboratorStatus()).listings);
      if (m.role === 'admin') setPending(await api.adminCollaborators());
    } catch (e) { fail(e); }
  };
  useEffect(() => { if (isLoggedIn()) load(); }, []);

  if (!isLoggedIn()) {
    return (
      <div className={`${card} border-[#ffb59d]`}>
        <p className="text-sm font-extrabold text-[#111c2d]">Host / collaborator portal</p>
        <p className="text-xs text-[#3d4947]">Log in with a collaborator account to verify your business and list places.</p>
        <div className="flex gap-2">
          <button onClick={() => onNavigate('login')} className="px-4 py-2 bg-[#ac3400] text-white rounded-xl text-xs font-bold">Log in</button>
          <button onClick={() => onNavigate('register')} className="px-4 py-2 bg-white border border-[#ffb59d] rounded-xl text-xs font-bold">Register as host</button>
        </div>
      </div>
    );
  }
  if (!me) return null;

  if (me.role === 'admin') {
    return (
      <div className={card}>
        <p className="text-sm font-extrabold text-[#111c2d]">Admin: collaborator verification</p>
        {pending.map((c) => (
          <div key={c.user_id} className="flex items-center justify-between gap-2 text-xs py-1.5 border-b border-[#f0f3ff]">
            <span><b>{c.business_name}</b> ({label(c.business_type)}) · GSTIN {c.gstin || '-'} [{c.gstin_check}] ·
              {c.has_document ? ` document: ${label(c.document_type)}` : ' no document'} · <i>{c.verification_status}</i></span>
            <span className="flex gap-1">
              <button onClick={async () => { try { await api.adminVerify(c.user_id, true); load(); } catch (e) { fail(e); } }}
                className="px-2 py-1 bg-[#15803d] text-white rounded font-bold">Approve</button>
              <button onClick={async () => { try { await api.adminVerify(c.user_id, false); load(); } catch (e) { fail(e); } }}
                className="px-2 py-1 bg-white border rounded font-bold">Reject</button>
            </span>
          </div>
        ))}
        {!pending.length && <p className="text-xs text-[#3d4947]">No collaborators yet.</p>}
      </div>
    );
  }
  if (me.role !== 'collaborator' || !me.collaborator) return null;
  const c = me.collaborator;

  const upload = async () => {
    if (!file) return onShowToast('Choose a PDF, JPG or PNG first', 'warning');
    setBusy(true);
    try {
      const r = await api.uploadDocument(docType, file);
      onShowToast(r.message, 'upload_file');
      await load();
    } catch (e) { fail(e); } finally { setBusy(false); }
  };

  const submitListing = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setResult(null);
    try {
      const r = await api.createListing({ ...form, cost_inr: form.cost_inr ? Number(form.cost_inr) : undefined });
      setResult(r);
      onShowToast(`${form.name}: ${DECISION[r.decision]?.text ?? r.decision}`, 'diamond');
      await load();
    } catch (err) { fail(err); } finally { setBusy(false); }
  };

  const status = c.verification_status;
  return (
    <div className="space-y-3">
      <div className={`${card} ${status === 'verified' ? 'border-[#86efac]' : 'border-[#ffb59d]'}`}>
        <div className="flex items-center justify-between">
          <p className="text-sm font-extrabold text-[#111c2d]">{c.business_name}</p>
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${status === 'verified' ? 'bg-[#15803d] text-white'
            : status === 'rejected' ? 'bg-[#b91c1c] text-white' : 'bg-[#fcd34d] text-[#78350f]'}`}>{label(status)}</span>
        </div>
        <p className="text-xs text-[#3d4947]">
          GSTIN: {c.gstin || 'not given'} · {c.gstin_check === 'valid_format' ? `format & check digit valid (${c.gstin_state})`
            : c.gstin_check === 'invalid' ? 'invalid - please correct it' : 'not provided'}
        </p>
        <p className="text-xs text-[#3d4947]">Document: {c.document_type ? `${label(c.document_type)} uploaded - under review` : 'none yet'}</p>
        {c.review_notes && <p className="text-xs text-[#3d4947]">Reviewer note: {c.review_notes}</p>}
        {status !== 'verified' && (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <select value={docType} onChange={(e) => setDocType(e.target.value)} className={`${field} w-auto`}>
              {['gst_certificate', 'fssai', 'shop_licence', 'udyam', 'other'].map((d) => <option key={d} value={d}>{label(d)}</option>)}
            </select>
            <input type="file" accept="application/pdf,image/jpeg,image/png" onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="text-xs" />
            <button onClick={upload} disabled={busy} className="px-3 py-2 bg-[#ac3400] text-white rounded-xl text-xs font-bold disabled:opacity-60">
              Upload for review
            </button>
          </div>
        )}
      </div>

      <form onSubmit={submitListing} className={card}>
        <p className="text-sm font-extrabold text-[#111c2d]">List a place - the hidden-gem model decides</p>
        <div className="grid grid-cols-2 gap-2">
          <input required placeholder="Place name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className={field} />
          <select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value })} className={field}>
            <option value="eat">Place to eat</option>
            <option value="visit">Place to visit</option>
          </select>
          <input required placeholder="Town, e.g. Munnar" value={form.town} onChange={(e) => setForm({ ...form, town: e.target.value })} className={field} />
          <input placeholder={form.kind === 'eat' ? 'Cuisines, e.g. Kerala, Seafood' : 'Type, e.g. waterfall'} value={form.category}
            onChange={(e) => setForm({ ...form, category: e.target.value })} className={field} />
          {form.kind === 'eat' && <input type="number" min={0} placeholder="Cost for two (₹)" value={form.cost_inr}
            onChange={(e) => setForm({ ...form, cost_inr: e.target.value })} className={field} />}
        </div>
        <textarea placeholder="Short description" maxLength={1500} value={form.description}
          onChange={(e) => setForm({ ...form, description: e.target.value })} className={`${field} h-16`} />
        <button disabled={busy} className="w-full py-2.5 bg-[#ac3400] text-white rounded-xl text-xs font-bold disabled:opacity-60">
          {busy ? 'Scoring…' : 'Evaluate & submit'}
        </button>
        {result && (
          <div className={`rounded-xl border px-3 py-2 text-xs space-y-1 ${DECISION[result.decision]?.cls}`}>
            <p className="font-extrabold">{DECISION[result.decision]?.text} {result.published ? '· published' : ''}</p>
            <p>{result.reason}</p>
            {result.quality_rank != null && <p>Quality rank: {result.quality_rank}/100 · evidence: {label(result.evidence)}</p>}
            {result.model_accuracy && <p className="opacity-80">Model: {result.model_accuracy}</p>}
            {result.publish_note && <p className="font-bold">{result.publish_note}</p>}
            {result.safety && <p>Safety: {label(result.safety.class)}{result.safety.score != null && ` (${result.safety.score}/100)`} - {result.safety.reasons?.slice(0, 2).join('; ')}</p>}
          </div>
        )}
      </form>

      {listings.length > 0 && (
        <div className={card}>
          <p className="text-sm font-extrabold text-[#111c2d]">Your listings</p>
          {listings.map((l) => (
            <div key={l.id} className="flex items-center justify-between text-xs py-1 border-b border-[#f0f3ff]">
              <span><b>{l.name}</b> · {l.city} · {l.kind === 'eat' ? 'eat' : 'visit'}</span>
              <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${DECISION[l.decision]?.cls}`}>
                {DECISION[l.decision]?.text}{l.published ? ' · live' : ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
