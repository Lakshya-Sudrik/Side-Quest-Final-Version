import React, { useEffect, useState } from 'react';
import { api, type MatchSummary } from '../services/backend';

interface NotificationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateToChat: () => void;
}

export const NotificationModal: React.FC<NotificationModalProps> = ({ isOpen, onClose, onNavigateToChat }) => {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    let active = true;
    setLoading(true);
    api.matches().then((data) => { if (active) setMatches(data); }).catch(() => { if (active) setMatches([]); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-end bg-[#343723]/40 p-4 backdrop-blur-xs sm:p-6" onClick={onClose}>
      <section role="dialog" aria-modal="true" aria-labelledby="recent-matches-title" className="mt-14 flex max-h-[80vh] w-full max-w-sm flex-col overflow-hidden rounded-2xl border border-[#e8e4d4] bg-[#EFE9D7] shadow-2xl sm:mt-16" onClick={(event) => event.stopPropagation()}>
        <header className="flex items-center justify-between border-b border-[#e8e4d4] bg-[#f5f2e7] p-4">
          <div className="flex items-center gap-2"><span className="material-symbols-outlined text-[20px] text-[#343723]">forum</span><h2 id="recent-matches-title" className="font-headline text-sm font-bold text-[#343723]">Your matches</h2></div>
          <button type="button" aria-label="Close" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full text-[#595a4a] hover:bg-[#dedac8]"><span className="material-symbols-outlined text-[18px]">close</span></button>
        </header>
        <div className="flex-1 overflow-y-auto divide-y divide-[#e8e4d4]">
          {loading ? <p className="p-5 text-sm text-[#595a4a]">Loading your matches…</p> : matches.length ? matches.map((match) => (
            <button key={match.match_id} type="button" onClick={() => { onNavigateToChat(); onClose(); }} className="flex w-full items-start gap-3 p-4 text-left transition-colors hover:bg-[#f5f2e7]">
              {match.avatar_url ? <img src={match.avatar_url} alt="" className="h-10 w-10 rounded-full object-cover" /> : <span aria-hidden="true" className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-[#dedac8] font-semibold text-[#343723]">{match.with_name.slice(0, 1).toUpperCase()}</span>}
              <span className="min-w-0 flex-1"><span className="block truncate text-sm font-bold text-[#343723]">You matched with {match.with_name}</span><span className="mt-1 block truncate text-xs text-[#595a4a]">{match.last_message?.body || `Trip to ${match.destination}`}</span></span>
              <span className="material-symbols-outlined text-[18px] text-[#898861]">arrow_forward</span>
            </button>
          )) : <div className="px-5 py-10 text-center"><span className="material-symbols-outlined text-3xl text-[#898861]">forum</span><p className="mt-2 text-sm font-semibold text-[#343723]">No matches yet</p><p className="mt-1 text-xs text-[#595a4a]">When a traveller accepts your request, they’ll appear here.</p></div>}
        </div>
        <footer className="border-t border-[#e8e4d4] bg-[#f5f2e7] p-3 text-center"><button type="button" onClick={() => { onNavigateToChat(); onClose(); }} className="text-xs font-semibold text-[#343723] hover:underline">Open chats</button></footer>
      </section>
    </div>
  );
};
