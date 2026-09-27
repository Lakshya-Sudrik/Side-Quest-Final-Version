import React, { useState, useRef } from 'react';
import { AppScreen } from '../../types';
import { GsapTextHighlight } from '../GsapTextHighlight';

interface TermsScreenProps {
  onNavigate: (screen: AppScreen) => void;
  onShowToast: (msg: string, icon?: string) => void;
}

export const TermsScreen: React.FC<TermsScreenProps> = ({
  onNavigate,
  onShowToast,
}) => {
  const [hasAgreed, setHasAgreed] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const sections = [
    { id: 'sec-1', label: '1. Acceptance', title: '1. Acceptance of Terms' },
    { id: 'sec-2', label: '2. Solo-Safety', title: '2. Solo-Traveler Safety Protocols' },
    { id: 'sec-3', label: '3. Verification', title: '3. Identity Verification & Conduct' },
    { id: 'sec-4', label: '4. Routes', title: '4. Uncharted Routes & Off-Grid Navigation' },
    { id: 'sec-5', label: '5. Liability', title: '5. Limitation of Liability' },
    { id: 'sec-6', label: '6. Data Privacy', title: '6. Privacy & Location Masking' },
  ];

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const distance = scrollHeight - clientHeight;
    const progress = distance > 0 ? Math.min(100, Math.round((scrollTop / distance) * 100)) : 100;
    setScrollProgress(progress);
  };

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleAcceptAndContinue = () => {
    onShowToast('You’re all set. Returning to SideQuest.', 'verified_user');
    onNavigate('home');
  };

  return (
    <div className="mx-auto flex h-[calc(100dvh-64px)] min-h-0 w-full max-w-md flex-col px-4 pt-3 md:max-w-2xl lg:max-w-4xl">
      <div ref={containerRef} onScroll={handleScroll} className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-contain pb-5">
      <div className="sticky top-0 z-30 bg-[#efe9d7]/95 pt-1 pb-2 backdrop-blur-md">
        <div className="flex items-center justify-between text-[11px] font-bold text-[#595a4a] mb-1">
          <span>Reading Progress</span>
          <span className="text-[#343723]">{scrollProgress}%</span>
        </div>
        <div className="w-full bg-[#e8e4d4] h-1.5 rounded-full overflow-hidden">
          <div
            className="bg-[#343723] h-full transition-all duration-150 rounded-full"
            style={{ width: `${scrollProgress}%` }}
          />
        </div>
      </div>

      {/* Header */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <span className="px-2.5 py-0.5 rounded-full bg-[#f5f2e7] text-[#343723] text-xs font-bold uppercase tracking-wider">
            Legal &amp; Compliance
          </span>
        </div>
        <h1 className="font-headline text-2xl sm:text-3xl font-extrabold text-[#343723] tracking-tight">
          <GsapTextHighlight>
            Terms &amp; Conditions
          </GsapTextHighlight>
        </h1>
        <p className="text-xs text-[#595a4a]">
          Safety protocols, liability guidelines, and community expectations for all SideQuest members.
        </p>
      </div>

      {/* Jump to Section Pills */}
      <div className="flex gap-2 overflow-x-auto no-scrollbar py-1">
        {sections.map((sec) => (
          <button
            key={sec.id}
            type="button"
            onClick={() => scrollToSection(sec.id)}
            className="px-3 py-1.5 rounded-full bg-[#EFE9D7] text-[#595a4a] hover:bg-[#e8e4d4] hover:text-[#343723] text-xs font-bold whitespace-nowrap border border-[#e8e4d4] shadow-xs active:scale-95 transition-all"
          >
            {sec.label}
          </button>
        ))}
      </div>

      {/* Legal Clauses */}
      <div className="space-y-4 bg-[#EFE9D7] p-5 sm:p-6 rounded-3xl border border-[#e8e4d4] shadow-sm text-xs sm:text-sm text-[#595a4a] leading-relaxed">
        {/* Section 1 */}
        <section id="sec-1" className="space-y-2 pt-1">
          <h2 className="font-headline text-base font-bold text-[#343723] flex items-center gap-1.5">
            <span className="w-6 h-6 rounded-full bg-[#e8e4d4] text-[#343723] text-xs flex items-center justify-center font-bold">
              1
            </span>
            <span>1. Acceptance of Terms</span>
          </h2>
          <p>
            By accessing or utilizing the SideQuest application, AI route generation services, or community solo matching radar, you confirm that you are at least 18 years of age and agree to be bound by these Terms of Service.
          </p>
        </section>

        <hr className="border-[#f5f2e7]" />

        {/* Section 2 */}
        <section id="sec-2" className="space-y-2">
          <h2 className="font-headline text-base font-bold text-[#343723] flex items-center gap-1.5">
            <span className="w-6 h-6 rounded-full bg-[#898861] text-[#efe9d7] text-xs flex items-center justify-center font-bold">
              2
            </span>
            <span>2. Solo-Traveler Safety Protocols</span>
          </h2>
          <p>
            SideQuest provides safety-weighted algorithmic route planning and solo explorer connectivity. Members agree to utilize masked audio/text communications and must adhere to our emergency contact dispatch protocols when traveling off-grid.
          </p>
          <div className="p-3 bg-[#e8e4d4] rounded-xl border border-[#898861] text-xs text-[#efe9d7] space-y-1">
            <span className="font-bold flex items-center gap-1">
              <span className="material-symbols-outlined text-[16px] text-[#343723]">
                verified_user
              </span>
              SOS Safe Connect Escort:
            </span>
            <p>
              In regions without cell reception, you may store offline GPX topography. SideQuest's automated safety ping alerts designated guardians if a check-in is overdue by &gt;4 hours.
            </p>
          </div>
        </section>

        <hr className="border-[#f5f2e7]" />

        {/* Section 3 */}
        <section id="sec-3" className="space-y-2">
          <h2 className="font-headline text-base font-bold text-[#343723] flex items-center gap-1.5">
            <span className="w-6 h-6 rounded-full bg-[#dedac8] text-[#898861] text-xs flex items-center justify-center font-bold">
              3
            </span>
            <span>3. Identity Verification &amp; Member Conduct</span>
          </h2>
          <p>
            To use the Solo Match affinity beacon, all users must submit authentic government ID documents. Impersonation, harassment, or abusive conduct results in immediate permanent expulsion and report to appropriate authorities.
          </p>
        </section>

        <hr className="border-[#f5f2e7]" />

        {/* Section 4 */}
        <section id="sec-4" className="space-y-2">
          <h2 className="font-headline text-base font-bold text-[#343723] flex items-center gap-1.5">
            <span className="w-6 h-6 rounded-full bg-[#e5c9be] text-[#7d2826] text-xs flex items-center justify-center font-bold">
              4
            </span>
            <span>4. Uncharted Routes &amp; Off-Grid Navigation</span>
          </h2>
          <p>
            Offbeat tracks, river crossings, and mountain ridgelines can be subject to rapid monsoon changes, landslides, and flash flooding. Travelers assume personal responsibility for vehicle capability (e.g. 4x4 vs scooter) and physical condition.
          </p>
        </section>

        <hr className="border-[#f5f2e7]" />

        {/* Section 5 & 6 */}
        <section id="sec-5" className="space-y-2">
          <h2 className="font-headline text-base font-bold text-[#343723] flex items-center gap-1.5">
            <span className="w-6 h-6 rounded-full bg-[#e8e4d3] text-[#343723] text-xs flex items-center justify-center font-bold">
              5
            </span>
            <span>5. Limitation of Liability &amp; Disclaimers</span>
          </h2>
          <p>
            SideQuest operates as a discovery and matching platform. We do not operate independent vehicles or host accommodations. All travel activities are undertaken at the member's own risk.
          </p>
        </section>

        <section id="sec-6" className="space-y-2 pt-2">
          <h2 className="font-headline text-base font-bold text-[#343723] flex items-center gap-1.5">
            <span className="w-6 h-6 rounded-full bg-[#dedac8] text-[#343723] text-xs flex items-center justify-center font-bold">
              6
            </span>
            <span>6. Privacy &amp; Location Masking</span>
          </h2>
          <p>
            We do not sell real-time GPS trails to third parties. Precise live locations are only broadcast to fellow convoy members when explicitly authorized via the "Send Live Location" action.
          </p>
        </section>
      </div>

      </div>
      <div className="z-10 shrink-0 border-t border-[#d8d3be] bg-[#efe9d7] px-1 pb-[calc(0.75rem+env(safe-area-inset-bottom))] pt-3">
        <div className="mx-auto flex max-w-3xl flex-col gap-3 rounded-2xl border border-[#dedac8] bg-[#f5f2e7] p-3 shadow-[0_-8px_24px_rgba(52,55,35,0.06)] sm:p-4">
        <label className="flex items-start gap-3 text-sm leading-snug font-medium text-[#343723] cursor-pointer">
          <input
            type="checkbox"
            checked={hasAgreed}
            onChange={(e) => setHasAgreed(e.target.checked)}
            className="mt-0.5 h-5 w-5 shrink-0 rounded accent-[#343723] focus:ring-[#343723]"
          />
          <span>I have read, understood, and agree to the Terms and Safety Protocol</span>
        </label>

        <div className="flex">
          <button
            type="button"
            disabled={!hasAgreed}
            onClick={handleAcceptAndContinue}
            className="min-h-12 w-full rounded-xl bg-[#343723] px-4 py-3 text-sm font-bold text-[#EFE9D7] shadow-md transition-all hover:bg-[#51533c] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9b453e] focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 flex items-center justify-center gap-2 active:scale-[0.99]"
          >
            <span className="material-symbols-outlined text-[18px]">verified</span>
            <span>Accept and Continue</span>
          </button>
        </div>
        </div>
      </div>
    </div>
  );
};
