import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'motion/react';
import confetti from 'canvas-confetti';
import { AppScreen, SoloPeer } from '../../types';
import { api } from '../../services/backend';
import type { Candidate, Trip } from '../../services/backend';
import { GsapInteractiveText } from '../GsapInteractiveText';
import { GsapCounter } from '../GsapCounter';
import { RadarScanner } from '../ui/RadarScanner';
import { SpotlightCard } from '../ui/SpotlightCard';
import { ShimmerButton } from '../ui/ShimmerButton';

interface SoloMatchScreenProps {
  onNavigate: (screen: AppScreen) => void;
  onShowToast: (msg: string, icon?: string) => void;
  onSelectPeerForChat?: (peerId: string) => void;
}

export const SoloMatchScreen: React.FC<SoloMatchScreenProps> = ({
  onNavigate,
  onShowToast,
  onSelectPeerForChat,
}) => {
  const [peerIndex, setPeerIndex] = useState(0);
  const [matchedPeer, setMatchedPeer] = useState<SoloPeer | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [showRadar, setShowRadar] = useState(false);
  const [trip, setTrip] = useState<Trip | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [destination, setDestination] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [tripBusy, setTripBusy] = useState(false);
  const [myName, setMyName] = useState('');
  const [myAvatar, setMyAvatar] = useState<string | null>(null);

  useEffect(() => {
    api.me().then((me) => { setMyName(me.name); setMyAvatar(me.avatar_url || null); }).catch(() => {});
    api.trips().then(async trips => {
      const active = trips.find(t => t.solo_match) || trips[0];
      if (active) { setTrip(active); setDestination(active.destination); const result = await api.candidates(active.id); setCandidates(result.candidates); }
    }).catch(() => { /* Signed-out visitors can set up after logging in. */ });
  }, []);

  const currentCandidate = candidates[peerIndex];
  const currentPeer: SoloPeer | null = currentCandidate ? {
    id: currentCandidate.user_id, name: currentCandidate.first_name, age: Number(currentCandidate.age_band.match(/\d+/)?.[0] || 0), location: currentCandidate.destination,
    image: currentCandidate.avatar_url || '', verifiedBadge: 'Travel profile', matchScore: currentCandidate.compatibility,
    travelDates: `${currentCandidate.overlap.from} – ${currentCandidate.overlap.to}`, overlapDaysText: `${currentCandidate.overlap.days} days overlap`,
    sharedPassions: currentCandidate.reasons, statsText: 'Compatibility calculated from shared travel preferences and overlapping dates',
  } : null;

  const createTrip = async (e: React.FormEvent) => {
    e.preventDefault(); setTripBusy(true);
    try { const created = await api.createTrip(destination.trim(), startDate, endDate, true); setTrip(created); const result = await api.candidates(created.id); setCandidates(result.candidates); setPeerIndex(0); onShowToast(result.note || `${result.candidates.length} possible travel companions found`, 'diversity_1'); }
    catch (error) { onShowToast(error instanceof Error ? error.message : 'Could not create trip', 'error'); }
    finally { setTripBusy(false); }
  };

  const handleConnect = async (peer: SoloPeer) => {
    // Fire confetti celebration
    try {
      confetti({
        particleCount: 80,
        spread: 70,
        origin: { y: 0.6 },
        colors: ['#343723', '#898861', '#9b453e', '#898861'],
      });
    } catch {
      // ignore
    }

    try {
      const candidate = candidates.find(c => c.user_id === peer.id);
      if (!trip || !candidate) throw new Error('Create a trip to send a match request.');
      await api.sendRequest(trip.id, candidate.user_id, candidate.trip_id);
      setMatchedPeer(peer);
      onShowToast(`Travel request sent to ${peer.name}`, 'favorite');
    } catch (error) { onShowToast(error instanceof Error ? error.message : 'Could not send request', 'error'); }
  };

  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-14, 14]);
  const likeOpacity = useTransform(x, [25, 110], [0, 1]);
  const passOpacity = useTransform(x, [-25, -110], [0, 1]);

  const handleSkip = () => {
    setPeerIndex((prev) => prev + 1);
    x.set(0);
    onShowToast('Next explorer queued', 'arrow_forward');
  };

  const handleScanRadar = () => {
    setShowRadar(true);
    if (!trip) onShowToast('Add your destination and dates to find compatible travellers.', 'info');
    else api.candidates(trip.id).then(r => { setCandidates(r.candidates); setPeerIndex(0); onShowToast(r.note, 'radar'); }).catch(e => onShowToast(e.message, 'error'));
  };

  return (
    <div className="flex flex-col w-full max-w-md md:max-w-2xl lg:max-w-4xl mx-auto px-4 pt-3 pb-24 space-y-4">
      {/* Header */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[11px] uppercase tracking-wider text-[#898861] font-bold flex items-center gap-1">
            <span className="material-symbols-outlined text-[15px]">diversity_1</span>
            Safety-First Affinity Match
          </span>
          <button
            type="button"
            onClick={handleScanRadar}
            disabled={isScanning}
            className="flex items-center gap-1 px-3 py-1 rounded-full bg-[#dedac8] text-[#898861] text-xs font-bold hover:bg-[#d5d1b9] active:scale-95 transition-all"
          >
            <span
              className={`material-symbols-outlined text-[16px] ${isScanning ? 'animate-spin' : ''}`}
            >
              sensors
            </span>
            <span>{isScanning ? 'Sweeping...' : showRadar ? 'Refresh Radar' : 'Open Radar'}</span>
          </button>
        </div>

        <h1 className="font-headline text-2xl sm:text-3xl font-extrabold text-[#343723] tracking-tight">
          <GsapInteractiveText
            scaleHover={1.1}
            className="text-[#343723]"
          >
            Solo Travel Match
          </GsapInteractiveText>
        </h1>
        <p className="text-xs sm:text-sm text-[#595a4a] leading-relaxed">
          Connect with verified solo travellers matching your itinerary, safety criteria, and travel pace.
        </p>
      </div>

      <form onSubmit={createTrip} className="grid grid-cols-1 sm:grid-cols-[1.5fr_1fr_1fr_auto] gap-2 rounded-2xl border border-[#dedac8] bg-[#f5f2e7] p-3">
        <input required minLength={2} value={destination} onChange={e => setDestination(e.target.value)} placeholder="Where are you headed?" className="min-w-0 rounded-xl border border-[#dedac8] bg-[#EFE9D7] px-3 py-2.5 text-sm" aria-label="Trip destination" />
        <input required type="date" value={startDate} onChange={e => setStartDate(e.target.value)} aria-label="Trip start date" className="rounded-xl border border-[#dedac8] bg-[#EFE9D7] px-3 py-2.5 text-sm" />
        <input required type="date" value={endDate} onChange={e => setEndDate(e.target.value)} aria-label="Trip end date" className="rounded-xl border border-[#dedac8] bg-[#EFE9D7] px-3 py-2.5 text-sm" />
        <button disabled={tripBusy} className="rounded-xl bg-[#343723] px-4 py-2.5 text-sm font-bold text-[#EFE9D7] disabled:opacity-60">{tripBusy ? 'Finding…' : 'Find travellers'}</button>
      </form>

      {/* Safety & Protocol Badges */}
      <div className="grid grid-cols-3 gap-2 bg-[#f5f2e7] p-3 rounded-2xl border border-[#dedac8] text-center">
        <div className="flex flex-col items-center">
          <span className="material-symbols-outlined text-[#343723] text-[20px]">verified_user</span>
            <span className="text-[10px] font-bold text-[#343723] mt-1">Profile details</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="material-symbols-outlined text-[#7d2826] text-[20px]">route</span>
          <span className="text-[10px] font-bold text-[#343723] mt-1">Corridor Overlap</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="material-symbols-outlined text-[#898861] text-[20px]">shield_with_heart</span>
            <span className="text-[10px] font-bold text-[#343723] mt-1">Mutual consent</span>
        </div>
      </div>

      {/* Trip overlap visualization (uses count returned by the backend) */}
      <AnimatePresence>
        {showRadar && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <RadarScanner activeCount={candidates.length} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Interactive Swipe Hint */}
      <div className="flex items-center justify-center gap-1.5 text-[11px] text-[#595a4a] font-semibold">
        <span className="material-symbols-outlined text-[15px] text-[#343723]">swipe</span>
        <span>Drag card left to skip • Drag right to connect</span>
      </div>

      {!currentCandidate ? <div className="rounded-3xl border border-[#dedac8] bg-[#f5f2e7] px-6 py-14 text-center"><span className="material-symbols-outlined text-4xl text-[#898861]">travel_explore</span><h2 className="mt-3 font-headline text-xl font-bold text-[#343723]">{trip ? 'No compatible travellers yet' : 'Add a trip to begin'}</h2><p className="mt-2 text-sm text-[#595a4a]">{trip ? 'We’ll show profiles when their destination and dates overlap yours. Chat unlocks only after both travellers accept.' : 'Choose a destination and travel dates above. Compatibility uses your saved preferences.'}</p></div> : <>
      {/* Main Swipeable Candidate Card with backend-ranked candidates */}
      <motion.article
        key={currentPeer!.id}
        style={{ x, rotate }}
        drag="x"
        dragConstraints={{ left: 0, right: 0 }}
        dragElastic={0.65}
        onDragEnd={(_, info) => {
          if (info.offset.x > 85) {
            handleConnect(currentPeer!);
          } else if (info.offset.x < -85) {
            handleSkip();
          }
        }}
        initial={{ scale: 0.94, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.94, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 350, damping: 25 }}
        className="cursor-grab active:cursor-grabbing select-none"
      >
        <SpotlightCard
          spotlightColor="rgba(70, 72, 212, 0.14)"
          className="relative w-full bg-[#EFE9D7] rounded-3xl overflow-hidden shadow-xl border-[#e8e4d4] flex flex-col p-0"
        >
          {/* Visual Drag Feedback Stamps */}
          <motion.div
            style={{ opacity: likeOpacity }}
            className="absolute top-16 right-6 z-30 pointer-events-none px-4 py-2 border-4 border-[#343723] rounded-2xl bg-[#343723]/90 text-[#EFE9D7] font-extrabold text-lg uppercase tracking-wider rotate-12 shadow-xl"
          >
            CONNECT ✨
          </motion.div>
          <motion.div
            style={{ opacity: passOpacity }}
            className="absolute top-16 left-6 z-30 pointer-events-none px-4 py-2 border-4 border-[#7d2826] rounded-2xl bg-[#7d2826]/90 text-[#EFE9D7] font-extrabold text-lg uppercase tracking-wider -rotate-12 shadow-xl"
          >
            PASS ✕
          </motion.div>

          <div className="relative w-full h-72 sm:h-80 overflow-hidden">
            {currentPeer!.image ? <img
              className="w-full h-full object-cover pointer-events-none"
              alt={`${currentPeer!.name}'s profile`}
              src={currentPeer!.image}
            /> : <div aria-hidden="true" className="grid h-full w-full place-items-center bg-[#dedac8] text-8xl font-headline text-[#898861]">{currentPeer!.name.slice(0, 1).toUpperCase()}</div>}
            <div className="absolute inset-0 bg-gradient-to-t from-[#343723]/90 via-[#343723]/25 to-transparent pointer-events-none"></div>

            {/* Top Chips */}
            <div className="absolute top-3 left-3 flex items-center gap-1 bg-[#EFE9D7]/90 backdrop-blur-md px-2.5 py-1 rounded-full text-xs font-bold text-[#343723] shadow-sm">
              <span className="material-symbols-outlined text-[16px]">verified</span>
              <span>{currentPeer!.verifiedBadge}</span>
            </div>

            <div className="absolute top-3 right-3 bg-[#898861] text-[#EFE9D7] px-3 py-1 rounded-full text-xs font-bold flex items-center gap-1 shadow-md">
              <span className="material-symbols-outlined text-[15px]">favorite</span>
              <span>
                <GsapCounter value={currentPeer!.matchScore} suffix="%" /> Match
              </span>
            </div>

            {/* Overlay Info at bottom of image */}
            <div className="absolute bottom-3 left-3 right-3 text-[#EFE9D7]">
              <div className="flex items-baseline gap-2">
                <h2 className="font-headline text-xl sm:text-2xl font-bold">
                  <GsapInteractiveText
                    scaleHover={1.08}
                    className="text-[#EFE9D7]"
                  >
                    {`${currentPeer!.name}, ${currentPeer!.age}`}
                  </GsapInteractiveText>
                </h2>
              </div>
              <p className="text-xs sm:text-sm text-[#f5f2e7] flex items-center gap-1 mt-0.5">
                <span className="material-symbols-outlined text-[16px] text-[#898861]">
                  location_on
                </span>
                {currentPeer!.location}
              </p>
            </div>
          </div>

          {/* Card Details */}
          <div className="p-4 sm:p-5 flex flex-col gap-3">
            {/* Direct Route Overlap */}
            <div className="p-3 rounded-2xl bg-[#e8e4d4] flex items-center justify-between border border-[#dedac8]">
              <div className="flex items-center gap-2">
                <span className="w-8 h-8 rounded-full bg-[#343723] text-[#EFE9D7] flex items-center justify-center flex-shrink-0">
                  <span className="material-symbols-outlined text-[18px]">calendar_month</span>
                </span>
                <div>
                  <span className="text-[11px] font-bold text-[#343723] uppercase tracking-wider block">
                    {currentPeer!.overlapDaysText}
                  </span>
                  <span className="text-xs font-bold text-[#343723]">{currentPeer!.travelDates}</span>
                </div>
              </div>
              <span className="text-[11px] text-[#595a4a] font-semibold bg-[#EFE9D7] px-2 py-0.5 rounded-full">
                Same Schedule
              </span>
            </div>

            {/* Passions & Tags */}
            <div>
              <label className="text-xs font-bold text-[#343723] block mb-1.5">
                Shared Passions &amp; Vibe
              </label>
              <div className="flex flex-wrap gap-1.5">
                {currentPeer!.sharedPassions.map((tag) => (
                  <span
                    key={tag}
                    className="px-2.5 py-1 rounded-full bg-[#f5f2e7] text-xs font-semibold text-[#595a4a] border border-[#e8e4d4]"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {/* Safety & History Stat */}
            <div className="pt-2 flex items-center gap-2 text-xs text-[#595a4a] border-t border-[#f5f2e7]">
              <span className="material-symbols-outlined text-[18px] text-[#343723]">
                shield
              </span>
              <span>{currentPeer!.statsText}</span>
            </div>

            {/* Action Buttons Row with 21st.dev Shimmer Button */}
            <div className="pt-2 flex items-center gap-3">
              <button
                type="button"
                aria-label="Skip to next candidate"
                onClick={handleSkip}
                className="w-14 h-14 rounded-2xl bg-[#f5f2e7] text-[#595a4a] hover:bg-[#dedac8] flex items-center justify-center flex-shrink-0 transition-transform active:scale-95"
              >
                <span className="material-symbols-outlined text-[26px]">close</span>
              </button>

              <div className="flex-1">
                <ShimmerButton
                onClick={() => handleConnect(currentPeer!)}
                  className="w-full h-14"
                >
                  <span className="material-symbols-outlined text-[22px]">chat</span>
                  <span>Connect &amp; Convoy</span>
                </ShimmerButton>
              </div>

              <button
                type="button"
                aria-label="Favorite"
                onClick={() => onShowToast(`Saved ${currentPeer!.name} to your shortlist`, 'favorite')}
                className="w-14 h-14 rounded-2xl bg-[#e5c9be] text-[#7d2826] hover:bg-[#c98d7f] flex items-center justify-center flex-shrink-0 transition-transform active:scale-95"
              >
                <span className="material-symbols-outlined text-[24px]">favorite</span>
              </button>
            </div>
          </div>
        </SpotlightCard>
      </motion.article>

      </>}

      {/* Request confirmation */}
      <AnimatePresence>
        {matchedPeer && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className="bg-[#EFE9D7] rounded-3xl max-w-sm w-full p-6 text-center shadow-2xl border border-[#e8e4d4] flex flex-col items-center gap-3"
            >
              <div className="w-16 h-16 rounded-full bg-[#898861] text-[#efe9d7] flex items-center justify-center text-3xl font-bold shadow-md">
                ✨
              </div>

              <h3 className="font-headline text-xl font-extrabold text-[#343723]">
                Request sent
              </h3>
              <p className="text-xs text-[#595a4a] leading-relaxed">
                Your request is with <span className="font-bold text-[#343723]">{matchedPeer.name}</span>. Private chat opens after they accept.
              </p>

              <div className="flex items-center justify-center -space-x-3 my-2">
                {myAvatar ? <img alt={`${myName}'s profile`} src={myAvatar} className="h-14 w-14 rounded-full object-cover ring-4 ring-[#EFE9D7]" /> : <span aria-label={myName || 'Your profile'} className="grid h-14 w-14 place-items-center rounded-full bg-[#dedac8] font-semibold text-[#343723] ring-4 ring-[#EFE9D7]">{myName.slice(0, 1).toUpperCase() || 'Y'}</span>}
                {matchedPeer.image ? <img alt={`${matchedPeer.name}'s profile`} src={matchedPeer.image} className="h-14 w-14 rounded-full object-cover ring-4 ring-[#EFE9D7]" /> : <span aria-label={matchedPeer.name} className="grid h-14 w-14 place-items-center rounded-full bg-[#dedac8] font-semibold text-[#343723] ring-4 ring-[#EFE9D7]">{matchedPeer.name.slice(0, 1).toUpperCase()}</span>}
              </div>

              <div className="w-full flex flex-col gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => {
                    setMatchedPeer(null);
                    setMatchedPeer(null);
                    onShowToast('You can chat once the other traveller accepts.', 'info');
                  }}
                  className="w-full py-3.5 bg-[#343723] hover:bg-[#51533c] text-[#EFE9D7] text-xs sm:text-sm font-bold rounded-xl shadow-md flex items-center justify-center gap-2"
                >
                  <span className="material-symbols-outlined text-[20px]">check</span>
                  <span>Got it</span>
                </button>

                <button
                  type="button"
                  onClick={() => setMatchedPeer(null)}
                  className="w-full py-2.5 text-xs font-semibold text-[#595a4a] hover:bg-[#f5f2e7] rounded-xl"
                >
                  Keep Browsing Explorers
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};
