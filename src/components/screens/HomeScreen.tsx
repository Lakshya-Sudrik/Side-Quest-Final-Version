import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { fetchHiddenGems } from '../../api';
import { GsapTextHighlight } from '../GsapTextHighlight';
import { GsapInteractiveText } from '../GsapInteractiveText';
import { GsapCounter } from '../GsapCounter';
import { SpotlightCard } from '../ui/SpotlightCard';
import { ShimmerButton } from '../ui/ShimmerButton';
import type { AppScreen } from '../../types';
import { api, type MatchSummary, type Me, type Trip } from '../../services/backend';

interface HomeScreenProps {
  onNavigate: (screen: AppScreen) => void;
  onOpenGemPreview?: (gemId: string) => void;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({
  onNavigate,
  onOpenGemPreview,
}) => {
  const [selectedGemModal, setSelectedGemModal] = useState<any | null>(null);
  const [trendingGems, setTrendingGems] = useState<any[]>([]);
  const [profile, setProfile] = useState<Me | null>(null);
  const [recentMatches, setRecentMatches] = useState<MatchSummary[]>([]);
  const [activeTrip, setActiveTrip] = useState<Trip | null>(null);
  const [tripLoaded, setTripLoaded] = useState(false);
  const [compatibleCount, setCompatibleCount] = useState<number | null>(null);

  // Suggestions come from the trained India model.
  useEffect(() => {
    let cancelled = false;
    fetchHiddenGems(6)
      .then(async (gems) => {
        try {
          const r=await fetch('/api/places/enrich',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({places:gems.map(g=>({name:g.name,city:g.city}))})});
          if(r.ok){const d=await r.json();const map=new Map<string, any>((d.results||[]).map((x:any)=>[String(x.name).toLowerCase(),x]));gems=gems.map(g=>{const place=map.get(g.name.toLowerCase());return {...g,image:place?.photoUrl||g.image||'',description:place?.description||g.description,googleMapsUri:place?.mapsUrl||g.googleMapsUri,photoAttributions:place?.photoAttributions||g.photoAttributions}})}
        } catch { /* keep the actual place records when photo search is unavailable */ }
        if (!cancelled) setTrendingGems(gems);
      })
      .catch(() => { if (!cancelled) setTrendingGems([]); });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.me().then((value) => { if (!cancelled) setProfile(value); }).catch(() => {});
    api.matches().then((value) => { if (!cancelled) setRecentMatches(value.slice(0, 3)); }).catch(() => {});
    api.trips().then(async (value) => { if (cancelled) return; const active=value[0]||null;setActiveTrip(active);if(active?.solo_match){try{const candidates=await api.candidates(active.id);if(!cancelled)setCompatibleCount(candidates.candidates.length)}catch{if(!cancelled)setCompatibleCount(0)}} }).catch(() => {}).finally(() => { if (!cancelled) setTripLoaded(true); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="flex flex-col w-full max-w-md md:max-w-2xl lg:max-w-4xl mx-auto px-4 space-y-5 pt-2 pb-24">
      {/* Framer Motion Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col pt-1"
      >
        <div className="flex items-center justify-between">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1, duration: 0.3 }}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#e8e4d4] text-[#595a4a] text-xs font-semibold"
          >
            <span className="w-2 h-2 rounded-full bg-[#343723] animate-pulse"></span>
            <span>{profile?.role === 'user' ? 'TRAVELLER PROFILE' : 'SIDEQUEST'}</span>
          </motion.div>

          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.15, duration: 0.3 }}
            className="flex items-center gap-1 px-3 py-1 rounded-full bg-[#dedac8] text-[#343723] text-xs font-semibold"
          >
            <span className="material-symbols-outlined text-[16px] text-[#7d2826]">person</span>
            <span>{profile?.role ? profile.role.replace('_', ' ') : 'Explore at your pace'}</span>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          className="mt-3 flex items-start justify-between gap-2"
        >
          <div>
            <h1 className="font-headline text-2xl sm:text-3xl font-extrabold text-[#343723] tracking-tight leading-tight">
              {profile?.name ? `Hey, ${profile.name.split(' ')[0]}` : 'Your next detour'} 🌄<br />
              Ready for your{' '}
              <GsapInteractiveText
                scaleHover={1.12}
                className="text-[#343723]"
              >
                next detour?
              </GsapInteractiveText>
            </h1>
            <p className="mt-1.5 text-xs sm:text-sm text-[#595a4a]">
              Explore <span className="font-bold text-[#343723]"><GsapCounter value={trendingGems.length} /> hidden spots</span> or plan your next trail.
            </p>
          </div>

          {/* Floating Framer Motion Compass Badge */}
          <motion.div
            animate={{
              y: [0, -8, 0],
              rotate: [-2, 3, -2],
            }}
            transition={{
              duration: 3.5,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-[#343723] to-[#898861] p-0.5 shadow-md flex-shrink-0 cursor-pointer active:scale-90 transition-transform"
            title="Interactive Compass (Tap to inspect)"
            onClick={() => onNavigate('routes')}
          >
            <div className="w-full h-full bg-[#efe9d7] rounded-[14px] flex items-center justify-center text-[#898861]">
              <span className="material-symbols-outlined text-[26px]">explore</span>
            </div>
          </motion.div>
        </motion.div>
      </motion.section>

      {tripLoaded && (activeTrip ? (
        <motion.section initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="rounded-3xl border border-[#e8e4d4] bg-[#efe9d7] p-5 shadow-sm">
          <div className="flex items-start justify-between gap-4">
            <div><p className="text-xs font-bold uppercase tracking-[0.12em] text-[#7d2826]">Your trip</p><h2 className="mt-2 font-headline text-xl font-bold text-[#343723]">{activeTrip.destination}</h2><p className="mt-1 text-sm text-[#595a4a]">{activeTrip.start_date} – {activeTrip.end_date}</p></div>
            <span className="rounded-full bg-[#e8e4d4] px-3 py-1.5 text-xs font-semibold text-[#595a4a]">{activeTrip.solo_match ? 'Matching on' : 'Saved'}</span>
          </div>
          <button type="button" onClick={() => onNavigate('routes')} className="mt-5 inline-flex min-h-11 items-center gap-2 rounded-full bg-[#343723] px-5 text-sm font-semibold text-[#efe9d7]">Open trip planner <span aria-hidden="true">→</span></button>
        </motion.section>
      ) : (
        <div className="rounded-3xl border border-dashed border-[#d8d3be] bg-[#f5f2e7] px-5 py-6"><p className="font-headline text-lg font-bold text-[#343723]">Your trips start here</p><p className="mt-1 text-sm text-[#595a4a]">Save a destination and dates to keep your plans together.</p><button type="button" onClick={() => onNavigate('routes')} className="mt-4 rounded-full bg-[#343723] px-5 py-2.5 text-sm font-semibold text-[#EFE9D7]">Plan a trip</button></div>
      ))}
      {/* Quick Action Navigation Trio */}
      <section className="grid grid-cols-3 gap-2.5">
        {/* Card 1: Plan Route */}
        <button
          type="button"
          onClick={() => onNavigate('routes')}
          className="group flex flex-col items-center text-center p-3 rounded-2xl bg-gradient-to-b from-[#f5f2e7] to-[#e8e4d4] hover:shadow-md transition-all border border-[#d8d3be]/50 active:scale-95"
        >
          <div className="w-12 h-12 rounded-xl bg-[#343723] text-[#EFE9D7] flex items-center justify-center shadow-sm group-hover:scale-110 transition-transform">
            <span className="material-symbols-outlined text-[24px]">alt_route</span>
          </div>
          <span className="mt-2 font-headline text-xs font-bold text-[#343723]">Plan Route</span>
          <span className="mt-0.5 text-[10px] text-[#595a4a] truncate">AI Detour</span>
        </button>

        {/* Card 2: Find Hidden Gems */}
        <button
          type="button"
          onClick={() => onNavigate('gems')}
          className="group flex flex-col items-center text-center p-3 rounded-2xl bg-gradient-to-b from-[#e5c9be]/30 to-[#e5c9be]/80 hover:shadow-md transition-all border border-[#c98d7f]/50 active:scale-95"
        >
          <div className="w-12 h-12 rounded-xl bg-[#7d2826] text-[#EFE9D7] flex items-center justify-center shadow-sm group-hover:scale-110 transition-transform">
            <span className="material-symbols-outlined text-[24px]">diamond</span>
          </div>
          <span className="mt-2 font-headline text-xs font-bold text-[#343723]">Hidden Gems</span>
          <span className="mt-0.5 text-[10px] text-[#62201f] truncate">Explore suggestions</span>
        </button>

        {/* Card 3: Solo Match */}
        <button
          type="button"
          onClick={() => onNavigate('solo-match')}
          className="group flex flex-col items-center text-center p-3 rounded-2xl bg-gradient-to-b from-[#e8e4d3]/40 to-[#d5d1b9]/80 hover:shadow-md transition-all border border-[#d5d1b9]/50 active:scale-95"
        >
          <div className="w-12 h-12 rounded-xl bg-[#898861] text-[#EFE9D7] flex items-center justify-center shadow-sm group-hover:scale-110 transition-transform">
            <span className="material-symbols-outlined text-[24px]">diversity_1</span>
          </div>
          <span className="mt-2 font-headline text-xs font-bold text-[#343723]">Solo Match</span>
          <span className="mt-0.5 text-[10px] text-[#6e704d] truncate">Compatibility</span>
        </button>
      </section>

      {/* Solo Travel Radar / Affinity Beacon (21st.dev SpotlightCard) */}
      <SpotlightCard
        spotlightColor="rgba(70, 72, 212, 0.12)"
        onClick={() => onNavigate('solo-match')}
        className="p-4 flex items-center gap-3.5 border-[#d8d3be] bg-[#f5f2e7] cursor-pointer hover:shadow-md transition-all group"
      >
        <div className="relative flex-shrink-0">
          <div className="w-12 h-12 rounded-full bg-[#898861]/15 flex items-center justify-center text-[#898861] group-hover:scale-105 transition-transform">
            <span className="material-symbols-outlined text-[26px]">radar</span>
          </div>
          <span className="absolute top-0 right-0 w-3 h-3 rounded-full bg-[#9b453e] ring-2 ring-[#EFE9D7] animate-ping"></span>
          <span className="absolute top-0 right-0 w-3 h-3 rounded-full bg-[#9b453e] ring-2 ring-[#EFE9D7]"></span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1 text-[#898861] text-[10px] font-bold uppercase tracking-wider">
            <span>Affinity Beacon</span>
          </div>
          <h3 className="font-headline text-xs sm:text-sm font-bold text-[#343723] truncate">
            <GsapInteractiveText scaleHover={1.06}>
              {compatibleCount===null?'Explore compatible travellers':`${compatibleCount} compatible ${compatibleCount===1?'traveller':'travellers'}`}
            </GsapInteractiveText>
          </h3>
          <p className="text-[11px] text-[#595a4a] truncate">{activeTrip?.solo_match?`${activeTrip.destination} · ${activeTrip.start_date} – ${activeTrip.end_date}`:'Create a trip to check destination and date overlap.'}</p>
        </div>
        <div className="w-9 h-9 rounded-full bg-[#EFE9D7] text-[#343723] flex items-center justify-center shadow-xs flex-shrink-0 group-hover:translate-x-0.5 transition-transform">
          <span className="material-symbols-outlined text-[20px]">chevron_right</span>
        </div>
      </SpotlightCard>

      {/* Horizontal Scroll Section: Trending Hidden Gems */}
      <section className="space-y-3 pt-1">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-headline text-base sm:text-lg font-bold text-[#343723]">
              <GsapInteractiveText scaleHover={1.08}>
                Places ranked by SideQuest
              </GsapInteractiveText>
            </h2>
            <p className="text-xs text-[#595a4a]">Suggestions from your SideQuest place index</p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('gems')}
            className="text-xs font-bold text-[#343723] hover:underline flex items-center gap-0.5"
          >
            <span>See all</span>
            <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
          </button>
        </div>

        {/* Carousel Container */}
        <div className="flex gap-3 overflow-x-auto pb-2 -mx-4 px-4 no-scrollbar snap-x snap-mandatory">
          {trendingGems.map((gem) => (
            <div
              key={gem.id}
              className="snap-start min-w-[260px] w-[260px] rounded-2xl bg-[#EFE9D7] overflow-hidden shadow-[0_2px_16px_rgba(0,0,0,0.06)] border border-[#e8e4d4] flex flex-col hover:shadow-lg transition-shadow"
            >
              <div className="relative h-36 w-full overflow-hidden">
                  {gem.image ? <img
                  className="w-full h-full object-cover transition-transform duration-500 hover:scale-105"
                  alt={gem.name}
                  src={gem.image}
                  loading="lazy"
                  onError={() => setTrendingGems((current) => current.map((item) => item.id === gem.id ? { ...item, image: '' } : item))}
                  /> : <div aria-label={gem.city||'Place'} className="flex h-full items-center justify-between bg-[#e9e5d7] px-4"><span className="font-headline text-sm text-[#5a5a43]">{gem.city||'India'}</span><span aria-hidden="true" className="font-headline text-3xl text-[#c3bea5]">{String(gem.name||'S').slice(0,1).toUpperCase()}</span></div>}
                {gem.image&&<div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent pointer-events-none"></div>}

                {/* Badges overlay */}
                <div className="absolute top-2.5 left-2.5 flex items-center gap-1 bg-[#EFE9D7]/95 backdrop-blur-md px-2 py-0.5 rounded-full text-[11px] text-[#7d2826] font-bold shadow-xs">
                  <span className="material-symbols-outlined text-[13px] text-[#7d2826]">star</span>
                  <span>{gem.badgeLabel==='Human-reviewed'?'Community reviewed':`SQ score ${Number(gem.gemScore).toFixed(1)}`}</span>
                </div>
                {gem.safetyClass&&<div className="absolute top-2.5 right-2.5 flex items-center gap-1 bg-[#343723]/90 text-[#EFE9D7] backdrop-blur-md px-2 py-0.5 rounded-full text-[11px] shadow-xs font-semibold">
                  <span className="material-symbols-outlined text-[13px]">shield</span>
                  <span>Safety: {gem.safetyClass}</span>
                </div>}

                <span className="absolute bottom-2 left-2.5 text-[#EFE9D7] text-[11px] font-medium bg-black/50 px-2 py-0.5 rounded backdrop-blur-xs">
                  {gem.category} • {gem.city||'India'}
                </span>
              </div>

              <div className="p-3.5 flex flex-col flex-1 justify-between">
                <div>
                  <h4 className="font-headline text-sm font-bold text-[#343723] leading-snug">
                    <GsapInteractiveText scaleHover={1.06}>
                      {gem.name}
                    </GsapInteractiveText>
                  </h4>
                  <p className="text-xs text-[#595a4a] line-clamp-2 mt-1 leading-relaxed">
                    {gem.description}
                  </p>
                </div>

                <div className="mt-3 pt-2 flex items-center justify-between border-t border-[#f5f2e7]">
                  <span className="text-[11px] text-[#595a4a] flex items-center gap-1">
                    <span className="material-symbols-outlined text-[14px] text-[#343723]">
                      {gem.categoryIcon}
                    </span>
                    {gem.hikeDurationOrFeature}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      if (onOpenGemPreview) onOpenGemPreview(gem.id);
                      else onNavigate('gems');
                    }}
                    className="text-xs text-[#343723] font-bold hover:underline"
                  >
                    View Trail
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-[#d8d2bc] bg-[#343723] p-6 text-[#f3efdf] md:p-8">
        <p className="text-xs uppercase tracking-[.18em] text-[#c6c49f]">Your next route</p>
        <div className="mt-3 flex flex-col justify-between gap-6 md:flex-row md:items-end"><div><h2 className="font-headline text-3xl md:text-4xl">Leave room for the detour.</h2><p className="mt-2 max-w-xl text-sm leading-6 text-[#e1deca]">Add your start, destination and timing. SideQuest will surface places along the way and show the route evidence we have.</p></div><button type="button" onClick={()=>onNavigate('routes')} className="rounded-xl bg-[#7d2826] px-5 py-3 text-sm font-semibold text-white">Open route planner</button></div>
      </section>

      {/* Recent Chats Teaser Snippet */}
      <section className="space-y-2.5 pt-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="font-headline text-base sm:text-lg font-bold text-[#343723]">
              <GsapInteractiveText scaleHover={1.08}>
                Recent Trail Chats
              </GsapInteractiveText>
            </h2>
            <span className="w-5 h-5 rounded-full bg-[#9b453e] text-[#EFE9D7] text-xs flex items-center justify-center font-bold">
              {recentMatches.length}
            </span>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('chats')}
            className="text-xs font-bold text-[#343723] hover:underline"
          >
            Open inbox
          </button>
        </div>

        <div className="space-y-2">
          {recentMatches.length ? recentMatches.map((match) => (
            <button key={match.match_id} type="button" onClick={() => onNavigate('chats')} className="flex w-full items-center gap-3 rounded-2xl border border-[#e8e4d4] bg-[#EFE9D7] p-3 text-left shadow-xs transition-colors hover:bg-[#f5f2e7]">
              {match.avatar_url ? <img className="h-11 w-11 rounded-full object-cover" alt={`${match.with_name}'s profile`} src={match.avatar_url} /> : <span aria-hidden="true" className="grid h-11 w-11 place-items-center rounded-full bg-[#dedac8] font-semibold text-[#343723]">{match.with_name.slice(0, 1).toUpperCase()}</span>}
              <span className="min-w-0 flex-1"><span className="block truncate font-headline text-sm font-bold text-[#343723]">{match.with_name}</span><span className="mt-0.5 block truncate text-xs text-[#595a4a]">{match.last_message?.body || `Matched for ${match.destination}`}</span></span>
              <span className="text-xs font-semibold text-[#898861]">{match.destination}</span>
            </button>
          )) : <div className="rounded-2xl border border-dashed border-[#d8d3be] px-4 py-5 text-sm text-[#595a4a]">Your trail conversations will appear here after a traveller match.</div>}
        </div>
      </section>
    </div>
  );
};
