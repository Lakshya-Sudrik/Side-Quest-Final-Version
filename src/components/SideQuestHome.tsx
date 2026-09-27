import { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import type { AppScreen } from '../types';
import { LANDING_MEDIA } from '../data/siteMedia';
import SideQuestFilm from './SideQuestFilm';

interface SideQuestHomeProps {
  onNavigate: (screen: AppScreen) => void;
}

export default function SideQuestHome({ onNavigate }: SideQuestHomeProps) {
  const [navOverFilm, setNavOverFilm] = useState(true);
  const reduceMotion = useReducedMotion();
  const rise = reduceMotion ? false : { opacity: 1, y: 0 };
  const fadeFromBelow = reduceMotion ? false : { opacity: 0, y: 28 };

  useEffect(() => {
    const filmStory = document.querySelector<HTMLElement>('#home-film');
    if (!filmStory) return;

    const observer = new IntersectionObserver(([entry]) => {
      setNavOverFilm(entry.isIntersecting);
    }, { rootMargin: '-72px 0px 0px 0px', threshold: 0 });
    observer.observe(filmStory);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="sq-home-experience" id="sidequest-home">
      <header className={`sq-home-nav${navOverFilm ? ' is-over-film' : ''}`}>
        <a className="sq-home-brand" href="#sidequest-home" aria-label="SideQuest home">
          <span className="sq-home-brand-mark" aria-hidden="true">SQ</span>
          <span className="sq-home-brand-name">SideQuest<span>.</span></span>
        </a>
        <nav className="sq-home-links" aria-label="Main navigation">
          <a href="#sidequest-home">Home</a>
          <a href="#home-gems">Hidden gems</a>
          <a href="#home-routes">Routes</a>
          <a href="#home-travelers">Solo travel</a>
        </nav>
        <div className="sq-home-nav-actions">
          <button className="sq-home-sign-in" type="button" onClick={() => onNavigate('login')}>Sign in</button>
          <button className="sq-home-nav-cta" type="button" onClick={() => onNavigate('register')}>
            Join the quest <span aria-hidden="true">↗</span>
          </button>
        </div>
      </header>

      <SideQuestFilm />

      <main className="sq-home-main">
        <section className="sq-gems-section" id="home-gems">
          <div className="sq-section-head">
            <motion.div
              initial={fadeFromBelow}
              whileInView={rise}
              viewport={{ once: true, amount: 0.35 }}
              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            >
              <p className="sq-eyebrow"><span>01</span> THE PLACES BETWEEN</p>
              <h2>There’s more<br />just <em>around the bend.</em></h2>
            </motion.div>
            <motion.p
              className="sq-section-intro"
              initial={fadeFromBelow}
              whileInView={rise}
              viewport={{ once: true, amount: 0.35 }}
              transition={{ duration: 0.8, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
            >
              Find the small trails, local favourites and quiet corners that make a familiar route feel like your own.
            </motion.p>
          </div>

          <div className="sq-gem-feature">
            <motion.figure
              className="sq-gem-image"
              initial={fadeFromBelow}
              whileInView={rise}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
            >
              <img src={LANDING_MEDIA.hiddenGems} alt="A quiet valley framed by snow-covered mountain peaks" loading="lazy" />
              <figcaption><span>OFF THE BEATEN PATH</span><span>MADE FOR CURIOUS FEET</span></figcaption>
            </motion.figure>
            <motion.div
              className="sq-gem-copy"
              initial={fadeFromBelow}
              whileInView={rise}
              viewport={{ once: true, amount: 0.25 }}
              transition={{ duration: 0.8, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
            >
              <span className="sq-gem-ornament" aria-hidden="true">✳</span>
              <p className="sq-eyebrow">A PLACE WORTH A PAUSE</p>
              <h3>Less checklist.<br /><em>More discovery.</em></h3>
              <p className="sq-feature-copy">Browse hidden gems with the context that matters: what makes a place special, what to expect when you get there, and how it fits your kind of day.</p>
              <ul className="sq-feature-points">
                <li><span>01</span> Discover local finds and lesser-known stops</li>
                <li><span>02</span> See useful notes before you set off</li>
                <li><span>03</span> Save places for the next time you roam</li>
              </ul>
              <button className="sq-text-link" type="button" onClick={() => onNavigate('gems')}>
                Explore hidden gems <span aria-hidden="true">↗</span>
              </button>
            </motion.div>
          </div>
          <div className="sq-section-rule"><span>TAKE THE SCENIC ROUTE</span><span aria-hidden="true">✳</span><span>STAY A LITTLE LONGER</span></div>
        </section>

        <section className="sq-route-section" id="home-routes">
          <motion.div
            className="sq-route-copy"
            initial={fadeFromBelow}
            whileInView={rise}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className="sq-eyebrow sq-eyebrow-light"><span>02</span> ROUTES WITH ROOM TO WANDER</p>
            <h2>Let the way<br /><em>there surprise you.</em></h2>
            <p>Tell SideQuest where you’re going and what you’re in the mood for. AI-assisted route planning helps turn the drive itself into part of the story.</p>
            <div className="sq-route-steps" aria-label="How route planning works">
              <div><span>01</span><p>Choose your starting point</p></div>
              <div><span>02</span><p>Shape a trip around your interests</p></div>
              <div><span>03</span><p>Leave space for a good detour</p></div>
            </div>
            <button className="sq-light-link" type="button" onClick={() => onNavigate('routes')}>
              Plan a route <span aria-hidden="true">↗</span>
            </button>
          </motion.div>
          <motion.figure
            className="sq-route-image"
            initial={reduceMotion ? false : { opacity: 0, x: 25 }}
            whileInView={reduceMotion ? undefined : { opacity: 1, x: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
          >
            <img src={LANDING_MEDIA.routes} alt="A winding road crossing a green mountain range" loading="lazy" />
            <figcaption><span>TAKE YOUR TIME</span><span>THE LONG WAY HAS ITS REASONS</span></figcaption>
          </motion.figure>
          <span className="sq-route-index" aria-hidden="true">02</span>
        </section>

        <section className="sq-traveler-section" id="home-travelers">
          <motion.figure
            className="sq-traveler-image"
            initial={fadeFromBelow}
            whileInView={rise}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
          >
            <img src={LANDING_MEDIA.travelers} alt="Travellers crossing a mountain lake by wooden boat" loading="lazy" />
            <figcaption>GO AT YOUR OWN PACE</figcaption>
          </motion.figure>
          <motion.div
            className="sq-traveler-copy"
            initial={fadeFromBelow}
            whileInView={rise}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.8, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className="sq-eyebrow"><span>03</span> SOLO, IF YOU LIKE</p>
            <h2>Go alone.<br />Feel <em>less on your own.</em></h2>
            <p className="sq-feature-copy">Find other travellers whose plans overlap with yours. Take a look at their interests, connect when it feels right, and keep the freedom to travel your way.</p>
            <div className="sq-traveler-note">
              <span className="sq-note-mark" aria-hidden="true">SQ</span>
              <p><strong>Connection, on your terms.</strong><span>Discover. Match. Start a conversation.</span></p>
            </div>
            <button className="sq-text-link" type="button" onClick={() => onNavigate('solo-match')}>
              Explore solo matching <span aria-hidden="true">↗</span>
            </button>
          </motion.div>
        </section>

        <section className="sq-local-section" id="home-local">
          <motion.figure
            className="sq-local-image"
            initial={fadeFromBelow}
            whileInView={rise}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
          >
            <img src={LANDING_MEDIA.locals} alt="A local craftsperson at work in a workshop" loading="lazy" />
          </motion.figure>
          <motion.div
            className="sq-local-copy"
            initial={fadeFromBelow}
            whileInView={rise}
            viewport={{ once: true, amount: 0.25 }}
            transition={{ duration: 0.8, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className="sq-eyebrow sq-eyebrow-light"><span>04</span> FOR THE PEOPLE WHO KNOW</p>
            <h2>Some places are<br />better <em>when shared.</em></h2>
            <p>Have a little-known place or local experience worth passing along? SideQuest gives hosts and local collaborators a way to introduce it to curious travellers.</p>
            <button className="sq-light-link" type="button" onClick={() => onNavigate('collaborator')}>
              Share your place <span aria-hidden="true">↗</span>
            </button>
          </motion.div>
        </section>

        <section className="sq-final-cta">
          <p className="sq-eyebrow"><span>YOUR NEXT CHAPTER</span></p>
          <motion.h2
            initial={fadeFromBelow}
            whileInView={rise}
            viewport={{ once: true, amount: 0.4 }}
            transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          >
            There’s a good<br /><em>detour out there.</em>
          </motion.h2>
          <p>Take the long way. See what finds you.</p>
          <button type="button" onClick={() => onNavigate('register')}>
            Find your next detour <span aria-hidden="true">↗</span>
          </button>
        </section>
      </main>

      <footer className="sq-home-footer">
        <a className="sq-home-brand" href="#sidequest-home" aria-label="SideQuest home">
          <span className="sq-home-brand-mark" aria-hidden="true">SQ</span>
          <span className="sq-home-brand-name">SideQuest<span>.</span></span>
        </a>
        <p>Take the long way. Find your somewhere.</p>
        <nav aria-label="Footer navigation">
          <a href="#home-gems">Hidden gems</a>
          <a href="#home-routes">Routes</a>
          <a href="#home-travelers">Solo travel</a>
          <button type="button" onClick={() => onNavigate('login')}>Sign in <span aria-hidden="true">↗</span></button>
        </nav>
        <span className="sq-home-copyright">© SIDEQUEST</span>
      </footer>
    </div>
  );
}
