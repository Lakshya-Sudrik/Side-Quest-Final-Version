import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { AppScreen } from '../types';
import SideQuestHome from './SideQuestHome';

interface LandingPageProps {
  onNavigate: (screen: AppScreen) => void;
}

export default function LandingPage({ onNavigate }: LandingPageProps) {
  const [homePrepared, setHomePrepared] = useState(false);
  const [introComplete, setIntroComplete] = useState(false);
  const introFrameRef = useRef<HTMLIFrameElement | null>(null);
  const introLoadedRef = useRef(false);
  const skipQueuedRef = useRef(false);

  useEffect(() => {
    const handleIntroMessage = (event: MessageEvent<unknown>) => {
      if (event.origin !== window.location.origin || event.source !== introFrameRef.current?.contentWindow) return;
      if (typeof event.data !== 'object' || event.data === null || !('type' in event.data)) return;

      if (event.data.type === 'sidequest:intro:prepare-home') setHomePrepared(true);
      if (event.data.type === 'sidequest:intro:complete') {
        setHomePrepared(true);
        setIntroComplete(true);
      }
    };

    window.addEventListener('message', handleIntroMessage);
    return () => window.removeEventListener('message', handleIntroMessage);
  }, []);

  useLayoutEffect(() => {
    if (homePrepared && !introComplete) {
      introFrameRef.current?.contentWindow?.postMessage({ type: 'sidequest:intro:home-ready' }, window.location.origin);
    }
  }, [homePrepared, introComplete]);

  const skipIntro = () => {
    skipQueuedRef.current = true;
    if (introLoadedRef.current) {
      introFrameRef.current?.contentWindow?.postMessage({ type: 'sidequest:intro:skip' }, window.location.origin);
    }
  };

  return (
    <div className={`sq-landing${homePrepared ? ' has-home' : ''}${introComplete ? ' home-visible' : ''}`}>
      {homePrepared && <SideQuestHome onNavigate={onNavigate} />}

      <div
        className={`sq-intro-transition${introComplete ? ' is-complete' : ''}`}
        aria-label="SideQuest introduction animation"
        aria-hidden={introComplete}
      >
        <iframe
          ref={introFrameRef}
          src="/sidequest-intro.html"
          title="SideQuest illustrated introduction"
          tabIndex={introComplete ? -1 : 0}
          onLoad={() => {
            introLoadedRef.current = true;
            if (skipQueuedRef.current) {
              introFrameRef.current?.contentWindow?.postMessage({ type: 'sidequest:intro:skip' }, window.location.origin);
            }
          }}
        />
        {!introComplete && (
          <button type="button" className="sq-intro-skip" onClick={skipIntro}>
            Skip intro <span aria-hidden="true">↗</span>
          </button>
        )}
      </div>
    </div>
  );
}
