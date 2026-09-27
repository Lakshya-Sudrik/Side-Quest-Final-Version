import React, { useRef, useEffect } from 'react';
import gsap from 'gsap';

interface GsapInteractiveTextProps {
  children: string;
  className?: string;
  scaleHover?: number;
  as?: 'h1' | 'h2' | 'h3' | 'span' | 'div' | 'p';
  enableKineticStagger?: boolean;
}

/**
 * Pure GSAP scaling interactive text:
 * - On hover / focus / tap: the text cleanly expands/scales larger with GSAP physics.
 * - Absolutely NO background highlight, box, tint, or glow.
 * - Smooth spring physics on scale up and reset.
 */
export const GsapInteractiveText: React.FC<GsapInteractiveTextProps> = ({
  children,
  className = '',
  scaleHover = 1.08,
  as = 'span',
  enableKineticStagger = false,
}) => {
  const containerRef = useRef<HTMLElement | null>(null);
  const wordsRef = useRef<(HTMLSpanElement | null)[]>([]);

  const words = children.split(' ');

  useEffect(() => {
    const el = containerRef.current;
    const wordEls = wordsRef.current.filter(Boolean) as HTMLSpanElement[];

    if (!el) return;

    // Optional subtle word entrance on mount
    if (enableKineticStagger && wordEls.length > 0) {
      gsap.fromTo(
        wordEls,
        {
          y: 10,
          opacity: 0,
          scale: 0.95,
        },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 0.5,
          stagger: 0.04,
          ease: 'power2.out',
        }
      );
    }

    // Hover / Pointer: Text cleanly gets bigger with GSAP spring physics (no highlight)
    const onPointerEnter = () => {
      gsap.to(el, {
        scale: scaleHover,
        duration: 0.3,
        ease: 'power2.out',
        transformOrigin: 'left center',
        overwrite: 'auto',
      });

      // Subtle natural spring across words
      if (wordEls.length > 1) {
        gsap.to(wordEls, {
          y: -1.5,
          duration: 0.25,
          stagger: 0.02,
          ease: 'power1.out',
        });
      }
    };

    const onPointerLeave = () => {
      gsap.to(el, {
        scale: 1,
        duration: 0.3,
        ease: 'power2.out',
        transformOrigin: 'left center',
        overwrite: 'auto',
      });

      if (wordEls.length > 1) {
        gsap.to(wordEls, {
          y: 0,
          duration: 0.25,
          stagger: 0.02,
          ease: 'power1.out',
        });
      }
    };

    const onClick = () => {
      // Crisp click bounce
      gsap.timeline()
        .to(el, { scale: scaleHover * 1.04, duration: 0.1, ease: 'power1.out' })
        .to(el, { scale: scaleHover, duration: 0.25, ease: 'power2.out' });
    };

    el.addEventListener('mouseenter', onPointerEnter);
    el.addEventListener('mouseleave', onPointerLeave);
    el.addEventListener('click', onClick);

    return () => {
      el.removeEventListener('mouseenter', onPointerEnter);
      el.removeEventListener('mouseleave', onPointerLeave);
      el.removeEventListener('click', onClick);
      gsap.killTweensOf([el, ...wordEls]);
    };
  }, [children, scaleHover, enableKineticStagger]);

  const Component = as;

  return (
    <Component
      ref={containerRef as any}
      className={`inline-flex flex-wrap items-center gap-x-1 cursor-pointer select-none origin-left ${className}`}
      style={{ willChange: 'transform' }}
    >
      {words.map((word, idx) => (
        <span
          key={`${word}-${idx}`}
          ref={(node) => {
            wordsRef.current[idx] = node;
          }}
          className="inline-block transform-gpu"
        >
          {word}
        </span>
      ))}
    </Component>
  );
};

