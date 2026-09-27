import React, { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import type { AppScreen, HiddenGem } from '../../types';
import { api, type Plan } from '../../services/backend';
import { fetchExploreGems, fetchRouteGems } from '../../api';

interface Props { onNavigate: (screen: AppScreen) => void; onShowToast: (msg: string, icon?: string) => void }
const today = new Date();
const dateValue = `${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}-${String(today.getDate()).padStart(2,'0')}`;
const timeValue = (mins: number) => `${String(Math.floor(mins/60)).padStart(2,'0')}:${String(mins%60).padStart(2,'0')}`;
const field = 'w-full rounded-xl border border-[#d8d2bc] bg-[#f8f5eb] px-4 py-3 text-[15px] text-[#343723] placeholder:text-[#858573] outline-none transition focus:border-[#898861] focus:ring-2 focus:ring-[#898861]/15';

export const RoutesScreen: React.FC<Props> = ({ onNavigate, onShowToast }) => {
  const [mode, setMode] = useState<'route'|'explore'>('route');
  const [exploreLocation, setExploreLocation] = useState('');
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [date, setDate] = useState(dateValue);
  const [arrivalDate, setArrivalDate] = useState(dateValue);
  const [depart, setDepart] = useState('08:00');
  const [arrive, setArrive] = useState('20:00');
  const [transport, setTransport] = useState('car');
  const [travellers, setTravellers] = useState('1');
  const [interests, setInterests] = useState<string[]>([]);
  const [mix, setMix] = useState(true);
  const [agenda, setAgenda] = useState<{time:string;place:string;lat:number;lon:number;duration_min:number}[]>([]);
  const [agendaTime, setAgendaTime] = useState(`${dateValue}T12:00`);
  const [agendaPlace, setAgendaPlace] = useState('');
  const [gems, setGems] = useState<{hidden:HiddenGem[];famous:HiddenGem[];routeKm:number}|null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [plan, setPlan] = useState<Plan|null>(null);

  const visibleGems = useMemo(() => gems ? (mix ? [...gems.hidden.slice(0,5), ...gems.famous.slice(0,5)] : gems.hidden.slice(0,5)) : [], [gems,mix]);
  const search = async () => {
    if (mode === 'explore' && exploreLocation.trim().length < 2) { onShowToast('Enter a city or town to explore.', 'info'); return; }
    if (mode === 'route' && (!origin.trim() || !destination.trim())) { onShowToast('Add both the starting point and destination.', 'info'); return; }
    setBusy(true); setPlan(null); setGems(null); setSelected(new Set());
    try {
      const found = mode === 'explore'
        ? await fetchExploreGems(exploreLocation.trim())
        : await fetchRouteGems(origin.trim(), destination.trim());
      try {
        const area = mode === 'explore' ? exploreLocation.trim() : destination.trim();
        const enrich = await fetch('/api/places/enrich', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({places:[...found.hidden,...found.famous].map(g=>({name:g.name,city:g.city||area}))})});
        if (enrich.ok) {
          const places = await enrich.json();
          const byName = new Map<string, any>((places.results||[]).map((p:any)=>[String(p.name).toLowerCase(),p]));
          const apply = (g:HiddenGem) => {
            const place = byName.get(g.name.toLowerCase());
            return {...g,image:place?.photoUrl||g.image||'',description:place?.description||g.description,googleMapsUri:place?.mapsUrl||g.googleMapsUri,photoAttributions:place?.photoAttributions||g.photoAttributions};
          };
          found.hidden = found.hidden.map(apply); found.famous = found.famous.map(apply);
        }
      } catch { /* photos are supplemental; recommendations stay usable */ }
      const routeKm = 'routeKm' in found ? found.routeKm : 0;
      setGems({hidden:found.hidden.slice(0,5),famous:found.famous.slice(0,5),routeKm});
      onShowToast(mode==='explore'
        ? `${Math.min(found.hidden.length,5)} hidden gems and ${Math.min(found.famous.length,5)} popular places found around ${exploreLocation.trim()}.`
        : `${Math.min(found.hidden.length,5)} hidden spots found along ${routeKm || 'your'} km route.`, 'check_circle');
    } catch (e) { onShowToast(e instanceof Error ? e.message : 'Route suggestions could not be loaded.', 'error'); }
    finally { setBusy(false); }
  };
  const toggleInterest = (value:string) => setInterests(old => old.includes(value) ? old.filter(x=>x!==value) : [...old,value]);
  const toggleSpot = (id:string) => setSelected(old => { const next=new Set(old); next.has(id)?next.delete(id):next.add(id); return next; });
  const addAgenda = async () => {
    if (!agendaPlace.trim()) return;
    setBusy(true);
    try {
      const response=await fetch('/api/places/resolve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({place:agendaPlace.trim(),area:origin.trim()||destination.trim()})});
      const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not locate this itinerary stop.');
      setAgenda(old => [...old,{time:agendaTime,place:data.name,lat:data.lat,lon:data.lon,duration_min:60}].sort((a,b)=>a.time.localeCompare(b.time)));
      setAgendaPlace('');
    } catch(e) { onShowToast(e instanceof Error?e.message:'Could not locate this itinerary stop.','error'); }
    finally {setBusy(false)}
  };
  const savePlan = async () => {
    if (!gems || mode === 'explore') return;
    setBusy(true);
    try {
      const created = await api.createPlan(origin, destination, `${date}T${depart}`, `${arrivalDate}T${arrive}`, {
        transport, travellers:Number(travellers), interests, fixed_stops:agenda,
      });
      let updated = created;
      for (const gem of visibleGems.filter(g=>selected.has(g.id))) updated = await api.addStop(created.id, gem.id);
      setPlan(updated);
      onShowToast('Your route and selected stops are saved.', 'check_circle');
    } catch(e) { onShowToast(e instanceof Error ? e.message : 'Could not save this route.', 'error'); }
    finally { setBusy(false); }
  };
  const startMatching=async()=>{
    setBusy(true);
    try{const current=await api.trips();const same=current.find(t=>t.destination.toLowerCase()===destination.trim().toLowerCase()&&t.start_date===date&&t.end_date===arrivalDate);if(same){if(!same.solo_match)await api.setSolo(same.id,true)}else await api.createTrip(destination.trim(),date,arrivalDate,true);onNavigate('solo-match')}
    catch(e){onShowToast(e instanceof Error?e.message:'Could not enable solo matching for this trip.','error')}
    finally{setBusy(false)}
  };

  return <div className="mx-auto w-full max-w-7xl px-5 pb-16 pt-8 md:px-10">
    <motion.header initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} transition={{duration:.45}} className="mb-8 grid gap-5 border-b border-[#d8d2bc] pb-7 lg:grid-cols-[1fr_auto] lg:items-end">
      <div><p className="mb-3 text-xs font-semibold uppercase tracking-[.2em] text-[#7d2826]">SideQuest / Plan a journey</p><h1 className="font-headline text-4xl leading-tight tracking-tight text-[#343723] md:text-6xl">Take the long way.<br/><span className="text-[#898861]">Make it yours.</span></h1><p className="mt-4 max-w-2xl text-base leading-7 text-[#595a4a] md:text-lg">Build a route around the places you care about. We’ll find thoughtful detours and show safety context where our data has it.</p></div>
      <div className="flex items-center gap-3 text-sm text-[#595a4a]"><span className="grid h-11 w-11 place-items-center rounded-full border border-[#b9b69b] text-[#7d2826]">01</span><span>Route details<br/><b className="text-[#343723]">then places</b></span></div>
    </motion.header>

    <section className="grid gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(300px,.9fr)]">
      <div className="rounded-[26px] border border-[#d8d2bc] bg-[#f8f5eb] p-5 shadow-[0_12px_35px_rgba(52,55,35,.05)] md:p-8">
        <div className="mb-7 flex items-start justify-between gap-4"><div><p className="text-xs uppercase tracking-[.16em] text-[#898861]">Your journey</p><h2 className="mt-1 font-headline text-2xl text-[#343723] md:text-3xl">Where are you headed?</h2></div><span className="material-symbols-outlined text-3xl text-[#898861]">route</span></div>
        <div className="mb-6 grid grid-cols-2 gap-2 rounded-xl bg-[#eeeadb] p-1.5"><button type="button" aria-pressed={mode==='route'} onClick={()=>{setMode('route');setGems(null);setPlan(null);setSelected(new Set())}} className={`rounded-lg px-3 py-2.5 text-sm font-semibold ${mode==='route'?'bg-[#343723] text-white':'text-[#595a4a]'}`}>Plan between places</button><button type="button" aria-pressed={mode==='explore'} onClick={()=>{setMode('explore');setGems(null);setPlan(null);setSelected(new Set())}} className={`rounded-lg px-3 py-2.5 text-sm font-semibold ${mode==='explore'?'bg-[#343723] text-white':'text-[#595a4a]'}`}>Explore around a place</button></div>
        {mode==='explore' ? <div className="mb-3"><label className="space-y-2 text-sm font-semibold text-[#595a4a]">City or town<input className={field} value={exploreLocation} onChange={e=>setExploreLocation(e.target.value)} onKeyDown={e=>e.key==='Enter'&&void search()} placeholder="Try Mumbai, Pune or Jaipur" autoComplete="address-level2"/></label><p className="mt-2 text-xs leading-5 text-[#6e6e5c]">Get up to five model-ranked hidden gems and five well-known places in the selected city.</p></div> : <>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="space-y-2 text-sm font-semibold text-[#595a4a]">Starting point<input className={field} value={origin} onChange={e=>setOrigin(e.target.value)} placeholder="City, station or address" autoComplete="street-address"/></label>
          <label className="space-y-2 text-sm font-semibold text-[#595a4a]">Final destination<input className={field} value={destination} onChange={e=>setDestination(e.target.value)} placeholder="Where the journey ends"/></label>
          <label className="space-y-2 text-sm font-semibold text-[#595a4a]">Travel date<input type="date" min={dateValue} className={field} value={date} onChange={e=>{setDate(e.target.value);if(arrivalDate<e.target.value)setArrivalDate(e.target.value)}}/></label>
          <label className="space-y-2 text-sm font-semibold text-[#595a4a]">Arrival date<input type="date" min={date||dateValue} className={field} value={arrivalDate} onChange={e=>setArrivalDate(e.target.value)}/></label>
          <label className="space-y-2 text-sm font-semibold text-[#595a4a]">Travellers<input type="number" min="1" max="20" className={field} value={travellers} onChange={e=>setTravellers(e.target.value)}/></label>
          <label className="space-y-2 text-sm font-semibold text-[#595a4a]">Leave around<input type="time" className={field} value={depart} onChange={e=>setDepart(e.target.value)}/></label>
          <label className="space-y-2 text-sm font-semibold text-[#595a4a]">Arrive by<input type="time" className={field} value={arrive} onChange={e=>setArrive(e.target.value)}/></label>
          <label className="space-y-2 text-sm font-semibold text-[#595a4a] sm:col-span-2">Mode of transport<select className={field} value={transport} onChange={e=>setTransport(e.target.value)}><option value="car">Car</option><option value="motorcycle">Motorcycle</option><option value="bus">Bus</option></select><span className="block text-xs font-normal text-[#6e6e5c]">Route distance follows the road network. Bus and motorcycle timing is an estimate.</span></label>
        </div>
        <div className="mt-7 border-t border-[#dedac8] pt-6"><p className="text-sm font-semibold text-[#343723]">What would make the day feel right?</p><div className="mt-3 flex flex-wrap gap-2">{['Nature','Food','Culture','Adventure','Quiet places','Local craft'].map(x=><button key={x} type="button" aria-pressed={interests.includes(x)} onClick={()=>toggleInterest(x)} className={`rounded-full border px-4 py-2 text-sm transition ${interests.includes(x)?'border-[#343723] bg-[#343723] text-[#f8f5eb]':'border-[#cfcab5] bg-transparent text-[#595a4a] hover:border-[#898861]'}`}>{x}</button>)}</div></div>
        <div className="mt-7 border-t border-[#dedac8] pt-6">
          <div className="flex items-start justify-between gap-4"><div><p className="text-sm font-semibold text-[#343723]">Already have a plan?</p><p className="mt-1 text-sm leading-6 text-[#6e6e5c]">Add a place and arrival time. It will be treated as a fixed stop when the route schedule checks detours.</p></div><button type="button" aria-pressed={agenda.length>0} onClick={()=>document.getElementById('fixed-stop-place')?.focus()} className="shrink-0 rounded-full border border-[#cfcab5] px-3 py-2 text-xs font-semibold text-[#7d2826]">Add a time stop</button></div>
          <div className="mt-4 grid gap-3 sm:grid-cols-[210px_1fr_auto]"><label className="sr-only" htmlFor="fixed-stop-time">Stop date and time</label><input id="fixed-stop-time" className={field} type="datetime-local" min={`${date}T00:00`} max={`${arrivalDate}T23:59`} value={agendaTime} onChange={e=>setAgendaTime(e.target.value)}/><label className="sr-only" htmlFor="fixed-stop-place">Place you need to visit</label><input id="fixed-stop-place" className={field} value={agendaPlace} onChange={e=>setAgendaPlace(e.target.value)} onKeyDown={e=>e.key==='Enter'&&(e.preventDefault(),void addAgenda())} placeholder="Place you need to be at"/><button type="button" onClick={()=>void addAgenda()} disabled={busy} className="rounded-xl border border-[#aaa78c] px-4 text-sm font-semibold text-[#343723] hover:bg-[#eeeadb] disabled:opacity-50">{busy?'Finding…':'Add'}</button></div>
          {agenda.length>0&&<ul className="mt-3 divide-y divide-[#e2ddca]">{agenda.map((a,i)=><li key={`${a.time}-${a.place}`} className="flex items-center justify-between py-2 text-sm text-[#595a4a]"><span><b className="mr-3 text-[#7d2826]">{a.time.replace('T',' ')}</b>{a.place}</span><button type="button" aria-label={`Remove ${a.place}`} onClick={()=>setAgenda(old=>old.filter((_,n)=>n!==i))} className="text-[#7d2826]">Remove</button></li>)}</ul>}
        </div>
        </>}
        <button type="button" onClick={search} disabled={busy} className="mt-8 flex min-h-14 w-full items-center justify-center gap-3 rounded-xl bg-[#343723] px-5 text-base font-semibold text-[#f8f5eb] transition hover:bg-[#4a4c34] disabled:opacity-60"><span className="material-symbols-outlined">{busy?'progress_activity':'search'}</span>{busy?(mode==='explore'?'Finding places nearby…':'Finding your route…'):(mode==='explore'?'Explore this place':'Find places along my route')}</button>
      </div>

      <aside className="self-start overflow-hidden rounded-[24px] bg-[#343723] text-[#f3efdf] lg:sticky lg:top-6">
        <div className="border-b border-[#f3efdf]/15 px-6 py-5 md:px-7">
          <p className="text-[11px] font-semibold uppercase tracking-[.18em] text-[#c6c49f]">{mode==='explore'?'A place at a glance':'Route at a glance'}</p>
          <p className="mt-2 font-headline text-2xl leading-tight">{mode==='explore'?'A local day, found one good stop at a time.':'A little room for the unexpected.'}</p>
        </div>
        <div className="px-6 py-5 md:px-7">
          {mode==='route' ? <div className="grid grid-cols-[18px_1fr] gap-x-3">
            <div className="flex flex-col items-center pt-1"><span className="h-2.5 w-2.5 rounded-full border-2 border-[#d58c7e]"/><span className="my-1 min-h-8 w-px flex-1 bg-[#c6c49f]/50"/><span className="h-2.5 w-2.5 rounded-full bg-[#c6c49f]"/></div>
            <div className="flex min-h-[58px] flex-col justify-between pb-2"><span className="text-[10px] uppercase tracking-[.14em] text-[#c6c49f]">Starting point</span><span className="truncate text-sm font-medium">{origin.trim()||'Add where you’re setting out'}</span></div>
            <div/><div className="flex min-h-[58px] flex-col justify-between"><span className="text-[10px] uppercase tracking-[.14em] text-[#c6c49f]">Final destination</span><span className="truncate text-sm font-medium">{destination.trim()||'Choose where the day ends'}</span></div>
          </div> : <div className="border-t border-[#f3efdf]/15 pt-4"><span className="text-[10px] uppercase tracking-[.14em] text-[#c6c49f]">Exploring around</span><p className="mt-2 text-lg font-medium">{exploreLocation.trim()||'Choose a city or town'}</p><p className="mt-2 text-xs leading-5 text-white/70">Place rankings, local popularity and safety context come from SideQuest’s available data.</p></div>}
          <div className="mt-5 grid grid-cols-3 divide-x divide-[#f3efdf]/15 border-t border-[#f3efdf]/15 pt-4 text-center">
            <div><span className="block font-headline text-2xl">{gems?`${gems.routeKm||'—'}`:'—'}</span><span className="text-[9px] uppercase tracking-[.12em] text-[#c6c49f]">km route</span></div>
            <div><span className="block font-headline text-2xl">{gems?visibleGems.length:'—'}</span><span className="text-[9px] uppercase tracking-[.12em] text-[#c6c49f]">suggestions</span></div>
            <div><span className="block font-headline text-2xl">{selected.size}</span><span className="text-[9px] uppercase tracking-[.12em] text-[#c6c49f]">you picked</span></div>
          </div>
        </div>
        <div className="flex items-center justify-between bg-[#2c2e20] px-6 py-3 md:px-7"><span className="text-[10px] uppercase tracking-[.16em] text-[#c6c49f]">SideQuest / India</span><span className="font-headline text-xl">SQ</span></div>
      </aside>
    </section>

    <AnimatePresence>{gems&&<motion.section initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} exit={{opacity:0}} className="mt-14" id="route-suggestions">
      <div className="mb-6 flex flex-col gap-4 border-b border-[#d8d2bc] pb-5 md:flex-row md:items-end md:justify-between"><div><p className="text-xs uppercase tracking-[.16em] text-[#7d2826]">{mode==='explore'?`Around ${exploreLocation}`:`Along your journey · ${gems.routeKm} km`}</p><h2 className="mt-1 font-headline text-3xl text-[#343723] md:text-4xl">{mode==='explore'?'Places worth knowing':'Choose your detours'}</h2><p className="mt-2 text-sm text-[#595a4a]">{mode==='explore'?'SideQuest ranks local hidden gems and well-known places using its place-quality and safety data.':'Suggestions are ranked by the SideQuest place model. Add the ones you want to your itinerary.'}</p></div><label className="flex items-center gap-3 text-sm text-[#595a4a]"><input type="checkbox" checked={mix} onChange={e=>setMix(e.target.checked)} className="h-4 w-4 accent-[#7d2826]"/> Include popular places</label></div>
      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">{visibleGems.map((gem,index)=><SpotCard key={gem.id} gem={gem} index={index} selected={selected.has(gem.id)} canSelect={mode==='route'} onToggle={()=>toggleSpot(gem.id)}/>)}</div>
      {visibleGems.length===0&&<p className="rounded-2xl border border-[#d8d2bc] p-8 text-[#595a4a]">No places were found {mode==='explore'?`around ${exploreLocation}`:'for this route'} yet. Try a nearby city or broader location.</p>}
      {mode==='route'&&<div className="mt-8 flex flex-col gap-4 rounded-2xl border border-[#d8d2bc] bg-[#f8f5eb] p-5 md:flex-row md:items-center md:justify-between"><div><p className="font-semibold text-[#343723]">{selected.size} {selected.size===1?'place':'places'} selected</p><p className="mt-1 text-sm text-[#6e6e5c]">Save your route to see the itinerary and timing fit.</p></div><button type="button" onClick={savePlan} disabled={busy||selected.size===0} className="rounded-xl bg-[#7d2826] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[#67211f] disabled:cursor-not-allowed disabled:opacity-45">{busy?'Saving…':'Save selected route'}</button></div>}
      {plan&&<div className="mt-6 rounded-2xl bg-[#e8e4d4] p-5"><p className="font-semibold text-[#343723]">Route saved · {plan.schedule.fits?'Your stops fit the time window':'Your itinerary cannot meet every timed stop'}</p><p className="mt-2 text-sm text-[#595a4a]">{plan.schedule.stops.map(s=>`${s.arrive||''} ${s.name}`).join(' · ')||'No stops selected.'}</p>{!plan.schedule.fits&&<p className="mt-2 text-sm text-[#7d2826]">Adjust the time window or itinerary so fixed appointments and selected detours fit.</p>}{Number(travellers)===1&&<button type="button" disabled={busy} onClick={()=>void startMatching()} className="mt-5 flex w-full items-center justify-between rounded-xl border border-[#c8c2aa] bg-[#f8f5eb] px-4 py-3 text-left transition hover:border-[#7d2826] hover:bg-white disabled:opacity-50"><span><span className="block text-sm font-semibold text-[#343723]">Travelling solo?</span><span className="mt-0.5 block text-xs text-[#6e6e5c]">See compatible travellers with overlapping plans</span></span><span className="material-symbols-outlined text-[#7d2826]">arrow_forward</span></button>}</div>}
    </motion.section>}</AnimatePresence>
  </div>;
};

const SpotCard: React.FC<{gem:HiddenGem;index:number;selected:boolean;canSelect:boolean;onToggle:()=>void}> = ({gem,index,selected,canSelect,onToggle}) => {
  const [imageFailed,setImageFailed]=useState(false);
  const [operators,setOperators]=useState<any[]|null>(null);
  const [operatorBusy,setOperatorBusy]=useState(false);
  const [operatorMessage,setOperatorMessage]=useState('');
  const [operatorFailed,setOperatorFailed]=useState(false);
  const findOperator=async()=>{
    setOperatorBusy(true);setOperatorMessage('');setOperatorFailed(false);
    try{
      const r=await fetch('/api/places/operators',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({place:gem.name,area:gem.city||'the route area',lat:gem.lat,lng:gem.lng})});
      const d=await r.json();if(!r.ok)throw new Error(d.error||'Local operator lookup is unavailable.');
      const publicContacts=(d.operators||[]).filter((item:any)=>item.name&&item.phone);
      setOperators(publicContacts);
      if(!publicContacts.length)setOperatorMessage('No nearby operator with a published phone number was found.');
    }catch(e){setOperators(null);setOperatorFailed(true);setOperatorMessage(e instanceof Error?e.message:'Could not look up local operators.');}
    finally{setOperatorBusy(false)}
  };
  const hasPhoto=Boolean(!imageFailed&&gem.image&&gem.image.startsWith('/api/'));
  return <motion.article initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} transition={{delay:Math.min(index*.04,.28)}} className="group overflow-hidden rounded-[24px] border border-[#d8d2bc] bg-[#fbf9f1] shadow-[0_8px_26px_rgba(52,55,35,.045)] transition duration-300 hover:-translate-y-1 hover:shadow-[0_18px_38px_rgba(52,55,35,.1)]">
    {hasPhoto&&<div className="relative aspect-[16/9] overflow-hidden bg-[#e9e5d7]">
      <img src={gem.image} alt={`${gem.name}${gem.city?`, ${gem.city}`:''}`} loading="lazy" className="h-full w-full object-cover transition duration-700 group-hover:scale-[1.035]" onError={()=>setImageFailed(true)}/>
      <div className="absolute inset-0 bg-gradient-to-t from-[#201f18]/55 via-transparent to-transparent"/>
      <span className="absolute bottom-3 left-4 rounded-full border border-white/50 bg-[#f8f5eb]/95 px-3 py-1 text-[10px] font-semibold uppercase tracking-[.13em] text-[#7d2826]">{gem.badgeLabel||gem.category}</span>
      <div className="absolute bottom-2 right-4 flex flex-col items-end text-white"><span className="text-xs font-medium">{gem.city}</span>{gem.photoAttributions?.map((credit,i)=><a key={`${credit.name}-${i}`} href={credit.url||gem.googleMapsUri} target="_blank" rel="noreferrer" className="max-w-40 truncate text-[9px] opacity-90">Photo: {credit.name}</a>)}</div>
    </div>}
    <div className="flex h-full flex-col p-5 sm:p-6">
      <div className="flex items-start justify-between gap-3"><div className="min-w-0"><p className="text-[10px] font-semibold uppercase tracking-[.16em] text-[#898861]">{gem.category}</p><h3 className="mt-1 font-headline text-[22px] leading-tight text-[#343723]">{gem.name}</h3></div>{gem.badgeLabel==='Human-reviewed'?<span className="shrink-0 rounded-full bg-[#e6e3d1] px-3 py-1.5 text-xs font-semibold text-[#4d5037]">Community verified</span>:Number.isFinite(gem.gemScore)&&gem.gemScore>0?<span className="shrink-0 rounded-full bg-[#eeeadb] px-3 py-1.5 text-xs font-semibold text-[#4e5039]">SQ {gem.gemScore.toFixed(1)}</span>:null}</div>
      {gem.description&&<p className="mt-3 text-[15px] leading-6 text-[#515343]">{gem.description}</p>}
      {(gem.detourMin!=null||gem.safetyClass||gem.nearestHospitalKm!=null||gem.nearestPoliceKm!=null)&&<div className="mt-5 grid grid-cols-2 gap-y-4 border-y border-[#e0dccb] py-4">{gem.detourMin!=null&&<Metric label="Route detour" value={`${gem.detourMin} min`} />}{gem.safetyClass&&<Metric label="Safety context" value={gem.safetyClass[0].toUpperCase()+gem.safetyClass.slice(1)} />}{gem.nearestHospitalKm!=null&&<Metric label="Nearest hospital" value={`${gem.nearestHospitalKm.toFixed(1)} km`} />}{gem.nearestPoliceKm!=null&&<Metric label="Nearest police" value={`${gem.nearestPoliceKm.toFixed(1)} km`} />}</div>}
      <div className="mt-4 flex min-h-7 items-center justify-between gap-3">{gem.googleMapsUri?<a className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#62634c] hover:text-[#7d2826]" href={gem.googleMapsUri} target="_blank" rel="noreferrer"><span className="material-symbols-outlined text-base">location_on</span>Open in Maps</a>:<span/>}{operators&&operators.length>0?<button type="button" onClick={()=>setOperators(null)} className="text-xs font-semibold text-[#7d2826]">Hide local contacts</button>:operators===null?<button type="button" onClick={()=>void findOperator()} disabled={operatorBusy} className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#7d2826] disabled:opacity-60"><span className="material-symbols-outlined text-base">{operatorBusy?'progress_activity':'call'}</span>{operatorBusy?'Looking for published numbers…':'Find local operator numbers'}</button>:null}</div>
      {operatorMessage&&<p role="status" className="mt-2 text-xs leading-5 text-[#666753]">{operatorMessage}{operatorFailed&&<button type="button" onClick={()=>void findOperator()} className="ml-2 font-semibold text-[#7d2826] underline">Try again</button>}</p>}
      {operators&&operators.length>0&&<div className="mt-3 divide-y divide-[#e0dccb] rounded-xl bg-[#f1eddf] px-3">{operators.map((o:any,i)=><div key={`${o.name}-${i}`} className="flex flex-wrap items-center justify-between gap-2 py-3 text-xs text-[#595a4a]"><span className="min-w-0"><b className="block truncate text-sm text-[#343723]">{o.name}</b><span className="mt-0.5 block truncate">{o.address}</span>{o.distanceKm!=null&&<span className="mt-1 block">About {o.distanceKm} km away</span>}{o.rating&&<span className="mt-1 block">{o.rating} ★ on Google</span>}</span><a className="shrink-0 rounded-lg bg-[#343723] px-3 py-2 font-semibold text-[#f8f5eb]" href={`tel:${o.phone}`}>Call {o.phone}</a></div>)}</div>}
      <div className={`mt-auto grid gap-2 pt-5 ${canSelect?'sm:grid-cols-[1fr_1.2fr]':''}`}>
        {canSelect&&<a href={gem.googleMapsUri||'#'} target="_blank" rel="noreferrer" className="flex min-h-12 items-center justify-center gap-2 rounded-xl border border-[#c4bea5] px-3 text-sm font-semibold text-[#343723] hover:border-[#898861] hover:bg-[#f3f0e4]"><span className="material-symbols-outlined text-lg">map</span>Explore on Maps</a>}
        {canSelect?<button type="button" aria-pressed={selected} onClick={onToggle} className={`flex min-h-12 items-center justify-center gap-2 rounded-xl px-3 text-sm font-semibold ${selected?'bg-[#343723] text-[#f8f5eb]':'border border-[#c4bea5] text-[#343723] hover:border-[#898861] hover:bg-[#f3f0e4]'}`}><span className="material-symbols-outlined text-lg">{selected?'check':'add'}</span>{selected?'Added to route':'Add to route'}</button>:<a href={gem.googleMapsUri||'#'} target="_blank" rel="noreferrer" className="flex min-h-12 w-full items-center justify-center gap-2 rounded-xl border border-[#c4bea5] px-4 text-sm font-semibold text-[#343723] hover:border-[#898861] hover:bg-[#f3f0e4]"><span className="material-symbols-outlined text-lg">map</span>Explore on Maps</a>}
      </div>
    </div>
  </motion.article>;
};

function Metric({label,value}:{label:string;value:string}){
  return <div><p className="text-[10px] font-medium uppercase tracking-[.13em] text-[#86836a]">{label}</p><p className="mt-1 text-sm font-semibold text-[#343723]">{value}</p></div>;
}
