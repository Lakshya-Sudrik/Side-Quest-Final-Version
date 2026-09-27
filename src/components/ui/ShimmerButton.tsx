import React from 'react';
import { motion } from 'motion/react';

interface ShimmerButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  className?: string;
  shimmerColor?: string;
  shimmerDuration?: string;
  variant?: 'primary' | 'secondary' | 'outline' | 'amber';
}

export const ShimmerButton: React.FC<ShimmerButtonProps> = ({
  children,
  className = '',
  shimmerColor = 'rgba(255, 255, 255, 0.35)',
  shimmerDuration = '2.5s',
  variant = 'primary',
  disabled,
  onClick,
  type = 'button',
  ...props
}) => {
  const getVariantStyles = () => {
    switch (variant) {
      case 'secondary':
        return 'bg-[#7d2826] text-[#EFE9D7] hover:bg-[#62201f] shadow-[0_4px_16px_rgba(172,52,0,0.25)]';
      case 'amber':
        return 'bg-gradient-to-r from-[#7d2826] to-[#9b453e] text-[#EFE9D7] shadow-[0_4px_16px_rgba(253,107,54,0.25)]';
      case 'outline':
        return 'bg-[#EFE9D7] text-[#343723] border border-[#dedac8] hover:bg-[#f5f2e7] shadow-xs';
      case 'primary':
      default:
        return 'bg-[#343723] text-[#EFE9D7] hover:bg-[#51533c] shadow-[0_4px_16px_rgba(0,104,95,0.25)]';
    }
  };

  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.01 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`group relative overflow-hidden rounded-xl font-headline font-bold text-xs sm:text-sm py-3 px-5 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 ${getVariantStyles()} ${className}`}
      {...(props as any)}
    >
      {/* 21st.dev Shimmer light sweep */}
      <span
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"
        style={{
          background: `linear-gradient(90deg, transparent, ${shimmerColor}, transparent)`,
        }}
      />
      <span className="relative z-10 flex items-center gap-2">{children}</span>
    </motion.button>
  );
};
