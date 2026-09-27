import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { api, type Preferences } from '../../services/backend';
import type { AppScreen } from '../../types';

type PreferenceKey = 'travel_style' | 'budget' | 'pace' | 'food' | 'languages';
type Question = { question: string; values: string[]; multi?: boolean; max?: number };
type Answers = Pick<Preferences, PreferenceKey>;
type Step = { id: string; key?: PreferenceKey; eyebrow: string; title: string; detail: string };

const questionOrder: PreferenceKey[] = ['travel_style', 'budget', 'pace', 'food', 'languages'];
const labels: Record<string, { title: string; note: string }> = {
  adventure: { title: 'Adventure', note: 'Trails, climbs and the less predictable road' },
  culture_heritage: { title: 'Culture & heritage', note: 'Local history, craft and living traditions' },
  food: { title: 'Food-led', note: 'Markets, regional kitchens and new flavours' },
  nature_wildlife: { title: 'Nature & wildlife', note: 'Forests, quiet trails and open landscapes' },
  relaxation: { title: 'Slow escapes', note: 'Room to pause, rest and take it in' },
  budget: { title: 'Keep it simple', note: 'Thoughtful trips with a lighter daily spend' },
  mid_range: { title: 'A little of both', note: 'Comfort where it counts, local where possible' },
  premium: { title: 'Comfort first', note: 'More room for stays and experiences' },
  slow: { title: 'Unhurried', note: 'Fewer stops, longer stays' },
  balanced: { title: 'A good balance', note: 'A plan with breathing room' },
  packed: { title: 'Full days', note: 'Make the most of every stop' },
  vegetarian: { title: 'Vegetarian', note: 'No meat or fish' },
  eggetarian: { title: 'Eggetarian', note: 'Vegetarian, with eggs' },
  non_vegetarian: { title: 'No restrictions', note: 'I eat meat and vegetarian food' },
  vegan: { title: 'Vegan', note: 'Plant-based meals' },
  jain: { title: 'Jain', note: 'Jain dietary preference' },
  english: { title: 'English', note: ' ' }, hindi: { title: 'Hindi', note: ' ' },
  kannada: { title: 'Kannada', note: ' ' }, tamil: { title: 'Tamil', note: ' ' },
  telugu: { title: 'Telugu', note: ' ' }, malayalam: { title: 'Malayalam', note: ' ' },
  marathi: { title: 'Marathi', note: ' ' }, bengali: { title: 'Bengali', note: ' ' },
  gujarati: { title: 'Gujarati', note: ' ' }, punjabi: { title: 'Punjabi', note: ' ' },
  odia: { title: 'Odia', note: ' ' }, urdu: { title: 'Urdu', note: ' ' },
};

const fallback = (value: string) => ({
  title: value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()),
  note: '',
});

interface PreferencesScreenProps {
  onNavigate: (screen: AppScreen) => void;
  onShowToast: (message: string, icon?: string) => void;
}

export const PreferencesScreen = ({ onNavigate, onShowToast }: PreferencesScreenProps) => {
  const reduceMotion = useReducedMotion();
  const [questions, setQuestions] = useState<Record<string, Question> | null>(null);
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState(1);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [hasSavedProfile, setHasSavedProfile] = useState(false);
  const [answers, setAnswers] = useState<Answers>({
    travel_style: [], budget: '', pace: '', food: '', languages: [],
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    Promise.all([api.options(), api.me()])
      .then(([options, profile]) => {
        if (!active) return;
        setQuestions(options.questions);
        if (profile.age) setAge(String(profile.age));
        if (profile.gender) setGender(profile.gender);
        if (profile.preferences) {
          setHasSavedProfile(profile.age != null && Boolean(profile.gender));
          setAnswers({
            travel_style: Array.isArray(profile.preferences.travel_style) ? profile.preferences.travel_style : [profile.preferences.travel_style].filter(Boolean),
            budget: profile.preferences.budget,
            pace: profile.preferences.pace,
            food: profile.preferences.food,
            languages: profile.preferences.languages || [],
          });
        }
      })
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : 'Could not load your travel setup.'))
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  const steps = useMemo<Step[]>(() => [
    { id: 'about', eyebrow: 'A little about you', title: 'First, the basics.', detail: 'SideQuest is for travellers aged 18 and over. Gender is used to support traveller matching, and “Prefer not to say” is always an option.' },
    ...questionOrder.map((key) => ({
      id: key,
      key,
      eyebrow: ({ travel_style: 'Your kind of trip', budget: 'Daily spend', pace: 'Your trip rhythm', food: 'At the table', languages: 'Talk along the way' } as Record<PreferenceKey, string>)[key],
      title: questions?.[key]?.question || 'Your travel style',
      detail: key === 'languages' ? `Choose up to ${questions?.[key]?.max || 4} languages you can comfortably chat in.` : questions?.[key]?.multi ? 'Select every option that sounds like your kind of trip.' : 'Choose the answer that feels most like you.',
    })),
  ], [questions]);

  const activeStep = steps[step];
  const move = (next: number) => {
    setError('');
    setDirection(next > step ? 1 : -1);
    setStep(next);
  };

  const choose = (key: PreferenceKey, value: string) => {
    setError('');
    if (key === 'languages' || questions?.[key]?.multi) {
      setAnswers((current) => {
        const previous = current[key] as string[];
        const max = questions?.[key]?.max || questions?.[key]?.values.length || 1;
        const selected = previous.includes(value)
          ? previous.filter((item) => item !== value)
          : previous.length < max ? [...previous, value] : previous;
        return { ...current, [key]: selected };
      });
      return;
    }
    setAnswers((current) => ({ ...current, [key]: value }));
  };

  const continueStep = async () => {
    setError('');
    if (step === 0) {
      const parsedAge = Number(age);
      if (!Number.isInteger(parsedAge) || parsedAge < 18 || parsedAge > 99) {
        setError('Enter an age from 18 to 99 to continue.');
        return;
      }
      if (!gender) {
        setError('Choose an option for gender to continue.');
        return;
      }
      move(step + 1);
      return;
    }

    const key = activeStep.key!;
    const value = answers[key];
    if (Array.isArray(value) ? value.length === 0 : !value) {
      setError(Array.isArray(value) ? 'Choose at least one option to continue.' : 'Choose one option to continue.');
      return;
    }
    if (step < steps.length - 1) {
      move(step + 1);
      return;
    }

    setSaving(true);
    try {
      await api.updateProfile({ age: Number(age), gender });
      await api.updatePreferences({
        ...answers,
        companion_gender: 'any',
        companion_age_min: 18,
        companion_age_max: 99,
      });
      onShowToast('Your travel profile is ready.', 'check_circle');
      onNavigate('routes');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'We could not save your answers. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const renderChoices = (key: PreferenceKey) => {
    const question = questions?.[key];
    if (!question) return null;
    const value = answers[key];
    const selected = Array.isArray(value) ? value : [value];
    return (
      <div className={`grid gap-2.5 ${key === 'languages' ? 'grid-cols-2 sm:grid-cols-3' : 'grid-cols-1 sm:grid-cols-2'}`}>
        {question.values.map((option, index) => {
          const copy = labels[option] || fallback(option);
          const isSelected = selected.includes(option);
          return (
            <motion.button
              key={option}
              type="button"
              aria-pressed={isSelected}
              onClick={() => choose(key, option)}
              initial={reduceMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: reduceMotion ? 0 : 0.24, delay: reduceMotion ? 0 : index * 0.035 }}
              className={`group flex min-h-[62px] items-center justify-between gap-3 rounded-xl border px-4 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9b453e] focus-visible:ring-offset-2 ${
                isSelected
                  ? 'border-[#9b453e] bg-[#f1ddd5] text-[#343723]'
                  : 'border-[#dedac8] bg-[#fbf9f1] text-[#343723] hover:border-[#898861] hover:bg-white'
              }`}
            >
              <span className="min-w-0">
                <span className="block text-[15px] font-semibold leading-tight">{copy.title}</span>
                {key !== 'languages' && copy.note && <span className="mt-1 block text-[13px] leading-snug text-[#595a4a]">{copy.note}</span>}
              </span>
              <span className={`grid h-5 w-5 shrink-0 place-items-center rounded-full border text-[11px] ${isSelected ? 'border-[#9b453e] bg-[#9b453e] text-white' : 'border-[#b7b29c] text-transparent'}`} aria-hidden="true">✓</span>
            </motion.button>
          );
        })}
      </div>
    );
  };

  return (
    <section className="min-h-[100dvh] bg-[#f5f2e7] px-4 py-7 sm:px-6 sm:py-10 lg:py-12" aria-label="Travel profile setup">
      <div className="mx-auto grid w-full max-w-5xl items-center gap-7 lg:grid-cols-1 lg:gap-10">
        <div className="mx-auto w-full max-w-[760px]">
          <div className="mb-4 flex items-center justify-between px-1 text-[13px] font-semibold text-[#595a4a]">
            <span className="uppercase tracking-[0.12em]">Travel profile</span>
            {!loading && <span>{String(step + 1).padStart(2, '0')} <span className="text-[#898861]">/ {String(steps.length).padStart(2, '0')}</span></span>}
          </div>
          <div className="mb-4 grid grid-cols-6 gap-1.5" aria-hidden="true">
            {steps.map((item, index) => (
              <span key={item.id} className={`h-1 rounded-full transition-colors duration-300 ${index <= step ? 'bg-[#9b453e]' : 'bg-[#dedac8]'}`} />
            ))}
          </div>

          <div className="overflow-hidden rounded-[24px] border border-[#dedac8] bg-[#efe9d7] shadow-[0_18px_50px_rgba(52,55,35,0.08)] sm:rounded-[28px]">
            {loading ? (
              <div className="grid min-h-[430px] place-items-center p-8 text-[15px] text-[#595a4a]" role="status">Opening your travel profile…</div>
            ) : error && !questions ? (
              <div className="grid min-h-[430px] place-items-center p-8 text-center">
                <div><p className="text-lg font-semibold text-[#343723]">Your setup could not load</p><p className="mt-2 text-[15px] text-[#595a4a]">{error}</p><button onClick={() => window.location.reload()} className="mt-5 rounded-full bg-[#343723] px-5 py-3 text-[15px] font-semibold text-[#efe9d7]">Try again</button></div>
              </div>
            ) : (
              <>
                <div className="min-h-[400px] px-5 pb-6 pt-6 sm:min-h-[440px] sm:px-8 sm:pb-8 sm:pt-8">
                  <AnimatePresence mode="wait" initial={false} custom={direction}>
                    <motion.div
                      key={activeStep.id}
                      custom={direction}
                      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, x: direction * 26 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={reduceMotion ? { opacity: 0 } : { opacity: 0, x: direction * -18 }}
                      transition={{ duration: reduceMotion ? 0.12 : 0.28, ease: [0.22, 1, 0.36, 1] }}
                    >
                      <p className="text-[13px] font-semibold uppercase tracking-[0.16em] text-[#7d2826]">{activeStep.eyebrow}</p>
                      <h2 className="mt-3 max-w-[18ch] font-headline text-[34px] font-semibold leading-[1.05] text-[#343723] sm:text-[42px]">{activeStep.title}</h2>
                      <p className="mt-3 max-w-[52ch] text-[15px] leading-relaxed text-[#595a4a]">{activeStep.detail}</p>

                      <div className="mt-7">
                        {activeStep.id === 'about' ? (
                          <div className="space-y-6">
                            <div>
                              <label htmlFor="traveller-age" className="mb-2 block text-[14px] font-semibold text-[#343723]">Age</label>
                              <div className="flex items-center gap-3">
                                <input id="traveller-age" type="number" min={18} max={99} inputMode="numeric" value={age} onChange={(event) => { setAge(event.target.value); setError(''); }} placeholder="18+" className="h-12 w-full max-w-[180px] rounded-xl border border-[#dedac8] bg-[#fbf9f1] px-4 text-[16px] text-[#343723] outline-none transition focus:border-[#9b453e] focus:ring-2 focus:ring-[#9b453e]/20" />
                                <span className="text-[14px] text-[#595a4a]">You must be 18 or older</span>
                              </div>
                            </div>
                            <fieldset>
                              <legend className="mb-2.5 text-[14px] font-semibold text-[#343723]">Gender for traveller matching</legend>
                              <div className="grid grid-cols-2 gap-2.5">
                                {[
                                  ['female', 'Woman'], ['male', 'Man'], ['non_binary', 'Non-binary'], ['prefer_not', 'Prefer not to say'],
                                ].map(([value, label]) => {
                                  const selected = gender === value;
                                  return <button key={value} type="button" aria-pressed={selected} onClick={() => { setGender(value); setError(''); }} className={`min-h-12 rounded-xl border px-3 py-2.5 text-left text-[14px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9b453e] ${selected ? 'border-[#9b453e] bg-[#f1ddd5]' : 'border-[#dedac8] bg-[#fbf9f1] hover:border-[#898861]'}`}>{label}</button>;
                                })}
                              </div>
                            </fieldset>
                          </div>
                        ) : activeStep.key ? renderChoices(activeStep.key) : null}
                      </div>

                      {error && <p role="alert" className="mt-4 rounded-lg bg-[#f1ddd5] px-3.5 py-2.5 text-[14px] font-medium text-[#7d2826]">{error}</p>}
                    </motion.div>
                  </AnimatePresence>
                </div>

                <div className="flex items-center justify-between gap-3 border-t border-[#dedac8] bg-[#e8e4d4]/65 px-5 py-4 sm:px-8">
                  <div className="flex items-center gap-1">
                    <button type="button" onClick={() => move(Math.max(0, step - 1))} disabled={step === 0 || saving} className="min-h-11 rounded-full px-4 text-[14px] font-semibold text-[#595a4a] transition hover:bg-[#dedac8] disabled:pointer-events-none disabled:opacity-0">Back</button>
                    {hasSavedProfile && <button type="button" onClick={() => onNavigate('routes')} disabled={saving} className="min-h-11 rounded-full px-3 text-[13px] font-semibold text-[#595a4a] underline decoration-[#b7b29c] underline-offset-4 transition hover:text-[#343723] disabled:opacity-50">Use saved profile</button>}
                  </div>
                  <button type="button" onClick={continueStep} disabled={saving} className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-[#343723] px-6 text-[15px] font-semibold text-[#efe9d7] shadow-sm transition hover:bg-[#51533c] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#9b453e] focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-70">
                    {saving ? 'Saving your profile…' : step === steps.length - 1 ? 'Save travel profile' : 'Next'}
                    {!saving && <span aria-hidden="true">{step === steps.length - 1 ? '✓' : '→'}</span>}
                  </button>
                </div>
              </>
            )}
          </div>
          <p className="mt-4 text-center text-[13px] leading-relaxed text-[#595a4a]">Your profile details stay attached to your SideQuest account.</p>
        </div>
      </div>
    </section>
  );
};
