import React, { useEffect, useRef, useState } from 'react';
import type { AppScreen, ChatMessage } from '../../types';
import { api, type MatchSummary } from '../../services/backend';

interface ChatsScreenProps {
  onNavigate: (screen: AppScreen) => void;
  onShowToast: (msg: string, icon?: string) => void;
  initialPeerId?: string;
}

export const ChatsScreen: React.FC<ChatsScreenProps> = ({ onNavigate, onShowToast, initialPeerId }) => {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [activeId, setActiveId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const active = matches.find(m => m.match_id === activeId);

  const loadMatches = async () => {
    try {
      const rows = await api.matches();
      setMatches(rows);
      if (rows.length && !rows.some(m => m.match_id === activeId)) {
        const target = rows.find(m => m.with_user === initialPeerId) || rows[0];
        setActiveId(target.match_id);
      }
    } catch (e) {
      onShowToast(e instanceof Error ? e.message : 'Sign in to view conversations', 'info');
    } finally { setLoading(false); }
  };

  useEffect(() => { void loadMatches(); }, []);
  useEffect(() => {
    if (!activeId) return;
    let live = true;
    const load = async () => {
      try {
        const rows = await api.messages(activeId);
        if (live) setMessages(rows.map(m => ({ id: String(m.id), sender: m.mine ? 'user' : 'peer', text: m.body, time: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) })));
      } catch (e) { if (live) onShowToast(e instanceof Error ? e.message : 'Messages unavailable', 'error'); }
    };
    void load();
    const interval = window.setInterval(load, 8000);
    return () => { live = false; window.clearInterval(interval); };
  }, [activeId]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const sendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const body = input.trim();
    if (!body || !activeId || sending) return;
    setSending(true);
    try { await api.send(activeId, body); setInput(''); const rows = await api.messages(activeId); setMessages(rows.map(m => ({ id: String(m.id), sender: m.mine ? 'user' : 'peer', text: m.body, time: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }))); }
    catch (error) { onShowToast(error instanceof Error ? error.message : 'Message could not be sent', 'error'); }
    finally { setSending(false); }
  };

  const safetyAction = async (type: 'block' | 'report') => {
    if (!active) return;
    try {
      if (type === 'block') { await api.block(active.with_user); setMatches(rows => rows.filter(m => m.match_id !== activeId)); setActiveId(''); }
      else await api.report(active.with_user, 'Reported from the conversation screen');
      onShowToast(type === 'block' ? 'Traveller blocked' : 'Report sent to the safety team', 'verified_user');
    } catch (error) { onShowToast(error instanceof Error ? error.message : 'Safety action failed', 'error'); }
  };

  if (loading) return <div className="mx-auto grid min-h-[55vh] max-w-3xl place-items-center text-sm text-[#595a4a]">Opening your conversations…</div>;
  if (!active) return <div className="mx-auto max-w-2xl px-6 py-24 text-center"><span className="material-symbols-outlined text-5xl text-[#898861]">forum</span><h1 className="mt-4 font-headline text-3xl font-bold text-[#343723]">Your conversations start with a match</h1><p className="mt-3 text-sm text-[#595a4a]">Private chat opens once both travellers accept a solo-match request.</p><button onClick={() => onNavigate('solo-match')} className="mt-6 rounded-full bg-[#343723] px-6 py-3 text-sm font-bold text-[#EFE9D7]">Find a travel companion</button></div>;

  return <div className="mx-auto flex h-[calc(100vh-80px)] w-full max-w-5xl flex-col overflow-hidden px-3 pb-16 pt-3 sm:px-6">
    <header className="mb-3 flex items-center justify-between gap-4 rounded-2xl border border-[#dedac8] bg-[#f5f2e7] px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">{active.avatar_url ? <img alt={`${active.with_name}'s profile`} src={active.avatar_url} className="h-11 w-11 rounded-full object-cover"/> : <span aria-hidden="true" className="grid h-11 w-11 place-items-center rounded-full bg-[#dedac8] font-semibold text-[#343723]">{active.with_name.slice(0, 1).toUpperCase()}</span>}<div className="min-w-0"><p className="truncate font-headline text-lg font-bold text-[#343723]">{active.with_name}</p><p className="truncate text-xs text-[#595a4a]">Shared trip · {active.destination}</p></div></div>
      <div className="flex shrink-0 gap-2"><button onClick={() => void safetyAction('report')} className="rounded-full border border-[#dedac8] px-3 py-2 text-xs font-bold text-[#595a4a] hover:bg-[#e8e4d4]">Report</button><button onClick={() => void safetyAction('block')} className="rounded-full border border-[#e5c9be] px-3 py-2 text-xs font-bold text-[#7d2826] hover:bg-[#e5c9be]/50">Block</button></div>
    </header>
    {matches.length > 1 && <nav aria-label="Conversations" className="mb-3 flex gap-2 overflow-x-auto">{matches.map(m => <button key={m.match_id} onClick={() => setActiveId(m.match_id)} className={`shrink-0 rounded-full px-4 py-2 text-xs font-bold ${m.match_id === activeId ? 'bg-[#343723] text-[#EFE9D7]' : 'bg-[#f5f2e7] text-[#595a4a]'}`}>{m.with_name}</button>)}</nav>}
    <section aria-label="Messages" className="flex-1 space-y-3 overflow-y-auto rounded-3xl border border-[#dedac8] bg-[#f5f2e7] p-4 sm:p-6">{active.overlap && <p className="mx-auto w-fit rounded-full bg-[#e8e4d4] px-4 py-2 text-center text-[11px] font-semibold text-[#595a4a]">{active.overlap.days} overlapping travel days · {active.overlap.from} to {active.overlap.to}</p>}{messages.length === 0 && <div className="py-20 text-center text-sm text-[#898861]">Say hello and start planning your shared detour.</div>}{messages.map(msg => <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[82%] rounded-2xl px-4 py-3 ${msg.sender === 'user' ? 'rounded-br-sm bg-[#343723] text-[#EFE9D7]' : 'rounded-bl-sm border border-[#dedac8] bg-[#EFE9D7] text-[#343723]'}`}><p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.text}</p><time className={`mt-1 block text-right text-[10px] ${msg.sender === 'user' ? 'text-[#dedac8]' : 'text-[#898861]'}`}>{msg.time}</time></div></div>)}<div ref={endRef}/></section>
    <form onSubmit={sendMessage} className="mt-3 flex gap-2 rounded-2xl border border-[#dedac8] bg-[#EFE9D7] p-2"><input value={input} onChange={e => setInput(e.target.value)} placeholder="Message your travel companion…" aria-label="Message" className="min-w-0 flex-1 bg-transparent px-3 text-sm text-[#343723] outline-none"/><button disabled={sending || !input.trim()} className="rounded-xl bg-[#343723] px-5 py-3 text-sm font-bold text-[#EFE9D7] disabled:opacity-50">{sending ? 'Sending…' : 'Send'}</button></form>
  </div>;
};
