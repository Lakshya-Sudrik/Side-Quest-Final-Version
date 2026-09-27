import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';

interface GsapCounterProps {
  value: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
  decimals?: number;
}

/**
 * GSAP Animated Counter:
 * Smoothly interpolates numbers with custom GSAP easing curves.
 */
export const GsapCounter: React.FC<GsapCounterProps> = ({
  value,
  duration = 1.4,
  prefix = '',
  suffix = '',
  className = '',
  decimals = 0,
}) => {
  const spanRef = useRef<HTMLSpanElement | null>(null);
  const countObj = useRef({ val: 0 });

  useEffect(() => {
    const el = spanRef.current;
    if (!el) return;

    countObj.current.val = 0;

    const tween = gsap.to(countObj.current, {
      val: value,
      duration,
      ease: 'power3.out',
      onUpdate: () => {
        if (el) {
          const formatted = decimals > 0
            ? countObj.current.val.toFixed(decimals)
            : Math.round(countObj.current.val).toLocaleString();
          el.textContent = `${prefix}${formatted}${suffix}`;
        }
      },
    });

    return () => {
      tween.kill();
    };
  }, [value, duration, prefix, suffix, decimals]);

  return (
    <span ref={spanRef} className={`inline-block font-mono tracking-tight ${className}`}>
      {prefix}{value}{suffix}
    </span>
  );
};
