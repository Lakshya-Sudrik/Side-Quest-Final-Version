import React from 'react';
import { motion } from 'motion/react';

interface RadarScannerProps {
  activeCount?: number;
  className?: string;
  onSelectPeer?: (id: string) => void;
}

export const RadarScanner: React.FC<RadarScannerProps> = ({
  activeCount = 0,
  className = '',
}) => {

  return (
    <div className={`relative flex flex-col items-center justify-center overflow-hidden rounded-3xl bg-[#efe9d7] text-[#EFE9D7] p-6 ${className}`}>
      {/* Background Dot Grid */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(circle, #898861 1px, transparent 1px)',
          backgroundSize: '18px 18px',
        }}
      />

      {/* Concentric Radar Rings */}
      <div className="relative w-48 h-48 sm:w-56 sm:h-56 flex items-center justify-center">
        {/* Outer Ring */}
        <div className="absolute inset-0 rounded-full border border-[#343723]/40" />
        {/* Middle Ring */}
        <div className="absolute inset-8 rounded-full border border-[#343723]/60" />
        {/* Inner Ring */}
        <div className="absolute inset-16 rounded-full border border-[#898861]/40" />

        {/* Crosshair lines */}
        <div className="absolute inset-x-0 top-1/2 h-px bg-[#343723]/40" />
        <div className="absolute inset-y-0 left-1/2 w-px bg-[#343723]/40" />

        {/* Rotating Radar Sweep Beam (21st.dev style) */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          className="absolute inset-0 rounded-full pointer-events-none"
          style={{
            background: 'conic-gradient(from 0deg, transparent 270deg, rgba(137, 245, 231, 0.4) 360deg)',
          }}
        />

        {/* Center Beacon (You) */}
        <div className="relative z-10 w-4 h-4 rounded-full bg-[#898861] shadow-[0_0_16px_#898861] flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-[#efe9d7]" />
          <span className="absolute -bottom-5 text-[10px] font-bold text-[#898861] whitespace-nowrap">
            Your trip
          </span>
        </div>

        {/* Location is intentionally not plotted: the match service only shares trip overlap. */}
      </div>

      {/* Radar Stats */}
      <div className="mt-5 text-center relative z-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#51533c]/60 border border-[#898861]/30 text-xs text-[#898861] font-semibold">
          <span className="w-2 h-2 rounded-full bg-[#898861] animate-pulse" />
          <span>{activeCount} compatible traveller{activeCount === 1 ? '' : 's'} for your trip dates</span>
        </div>
      </div>
    </div>
  );
};
