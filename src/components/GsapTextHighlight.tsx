import React, { useRef, useEffect } from 'react';
import gsap from 'gsap';

interface GsapTextHighlightProps {
  children: React.ReactNode;
  className?: string;
  highlightColor?: string;
  as?: 'span' | 'h1' | 'h2' | 'h3' | 'p' | 'div';
}

/**
 * Text emphasis component with clean GSAP scaling physics:
 * Pure text zoom/scale without any background highlighter tint or bar.
 */
export const GsapTextHighlight: React.FC<GsapTextHighlightProps> = ({
  children,
  className = '',
  as = 'span',
}) => {
  const containerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onMouseEnter = () => {
      gsap.to(el, {
        scale: 1.05,
        duration: 0.25,
        ease: 'power2.out',
        transformOrigin: 'left center',
        overwrite: 'auto',
      });
    };

    const onMouseLeave = () => {
      gsap.to(el, {
        scale: 1,
        duration: 0.25,
        ease: 'power2.out',
        transformOrigin: 'left center',
        overwrite: 'auto',
      });
    };

    el.addEventListener('mouseenter', onMouseEnter);
    el.addEventListener('mouseleave', onMouseLeave);

    return () => {
      el.removeEventListener('mouseenter', onMouseEnter);
      el.removeEventListener('mouseleave', onMouseLeave);
      gsap.killTweensOf(el);
    };
  }, []);

  const Component = as;

  return (
    <Component
      ref={containerRef as any}
      className={`inline-block cursor-pointer select-text origin-left ${className}`}
      style={{ willChange: 'transform' }}
    >
      {children}
    </Component>
  );
};

