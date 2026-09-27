import { useRef } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { HOME_REEL, HOME_REEL_POSTER } from '../data/siteMedia';

const chapters = [
  {
    number: '01',
    label: 'THE GOOD DETOUR',
    title: <>Take the road<br />that <em>takes you somewhere.</em></>,
    copy: 'The best part of a trip is often the place you didn’t plan to find.',
    link: '#home-gems',
    action: 'Find your next detour',
    align: 'left',
  },
  {
    number: '02',
    label: 'A BETTER WAY THERE',
    title: <>Make the journey<br /><em>part of the plan.</em></>,
    copy: 'Build a route around the things you love, with room for the unexpected along the way.',
    link: '#home-routes',
    action: 'See how it works',
    align: 'right',
  },
  {
    number: '03',
    label: 'GOOD MILES, SHARED',
    title: <>Go your own way.<br /><em>Meet along it.</em></>,
    copy: 'For solo travellers who want the freedom of going alone and the option of finding their people.',
    link: '#home-travelers',
    action: 'Find your kind of people',
    align: 'left',
  },
];

export default function SideQuestFilm() {
  const storyRef = useRef<HTMLElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const reduceMotion = useReducedMotion();

  return (
    <section className="sq-film-story" id="home-film" ref={storyRef} aria-label="The SideQuest story">
      <div className="sq-film-stage" aria-hidden="true">
        <video
          ref={videoRef}
          className="sq-film-video"
          src={HOME_REEL}
          poster={HOME_REEL_POSTER}
          autoPlay={!reduceMotion}
          loop
          muted
          playsInline
          preload="auto"
          tabIndex={-1}
        />
        <div className="sq-film-shade" />
        <span className="sq-film-caption">A LITTLE FURTHER FROM THE USUAL</span>
      </div>

      <div className="sq-film-chapters">
        {chapters.map((chapter) => {
          const Heading = chapter.number === '01' ? 'h1' : 'h2';
          return (
          <motion.article
            key={chapter.number}
            className={`sq-film-chapter is-${chapter.align}`}
            initial={reduceMotion ? false : { opacity: 0, y: 24 }}
            whileInView={reduceMotion ? undefined : { opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.35 }}
            transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="sq-film-copy">
              <p className="sq-film-kicker"><span aria-hidden="true" />{chapter.label}</p>
              <span className="sq-film-number">{chapter.number} <i>/ 03</i></span>
              <Heading>{chapter.title}</Heading>
              <p className="sq-film-note">{chapter.copy}</p>
              <a className="sq-film-link" href={chapter.link}>
                {chapter.action} <span aria-hidden="true">↗</span>
              </a>
            </div>
          </motion.article>
          );
        })}
      </div>

      <a className="sq-film-scroll-cue" href="#home-gems">
        SCROLL TO WANDER <span aria-hidden="true">↓</span>
      </a>
    </section>
  );
}
