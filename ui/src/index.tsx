import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { AnimatePresence, motion } from 'motion/react';
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpDown, Atom, BookOpen, Box, Check, ChevronDown,
  ChevronLeft, ChevronRight, CircleAlert, CircleCheck, CircleDot, Clock3, Code2, Copy, DoorOpen,
  Download, FlaskConical, GitBranch, GripVertical, History, KeyRound, Layers3, LockKeyhole,
  Maximize2, Minus, MoreHorizontal, Network, Pause, Play, Plus, RotateCcw, Settings2,
  ShieldAlert, ShieldCheck, Shuffle, Sparkles, SquareArrowOutUpRight, TableProperties, Trash2,
  TrendingUp, WandSparkles, X, Zap
} from 'lucide-react';
import './style.css';

// All semantic transitions are computed on the Python side by the trained model.
// This module is only a renderer, an input editor and a history viewer.
type Entity = {id: string; stock: number; capacity: number; kind?: string | null; properties?: Record<string, unknown>};
type Relation = {predicate: string; source: string; target: string};
type World = {entities: Entity[]; relations: Relation[]};
type AtomRule = {predicate: string; left: string; right: string; required?: boolean};
type GateRule = {table: number[]; left: Rule; right: Rule};
type Rule = AtomRule | GateRule;
type Law = {id: string; enabled: boolean; priority: number; condition: Rule; mode: 'consume'|'produce'|'transfer'; amount: number; effect_predicate: string; relation_add: boolean};
type Action = {actor: string; verb: string; target: string; instrument: string | null};
type Subset = {subset_id: number; description: string; effects_applied: string[]; base_v79_energy: number; joint_v710_residual_energy: number; total_consistency_energy: number; is_winner: boolean; conflict_detected: boolean; diagnostics: Record<string, number|boolean>; selected_candidates: number[]};
type Candidate = {candidate_id: number; law_id: string; binding: Record<string,string|null>; effect_mode: string; effect_amount: number; effect_predicate: string; relation_add: boolean};
type StepResponse = {success: boolean; step_log: {action_normalized: string; evaluated_subsets: Subset[]; candidates: Candidate[]; winner_subset_id: number; warnings: string[]; energy_interpretation: string}; new_world_state: World};
type StepRecord = {number: number; action: Action; world_state_before: World; world_state_after: World; laws: Law[]; response: StepResponse};
type Props = {revision: number; presetIndex: number; presets: {name:string;subtitle:string}[];world_state: World;laws: Law[];action: Action;history: StepRecord[];error: string|null;model:string;seed:number};
type CompArgs = {data: Props; parentElement: ShadowRoot | HTMLElement; setTriggerValue: (name: string, value: string)=>void};

const copy = <T,>(a: T): T => JSON.parse(JSON.stringify(a)) as T;
const num = (v:number) => Number.isFinite(v) ? v.toFixed(2) : '—';
const hasGate = (r: Rule): r is GateRule => 'table' in r;
const KIND: Record<string, {label:string;icon: typeof Box; hue:string}> = {
  character:{label:'Персонаж',icon:Atom,hue:'mint'},
  door:{label:'Дверь',icon:DoorOpen,hue:'amber'},
  key:{label:'Ключ',icon:KeyRound,hue:'sky'},
  room:{label:'Комната',icon:Box,hue:'violet'},
  object:{label:'Объект',icon:Box,hue:'sky'},
};
const kindInfo = (s?: string|null) => KIND[s || ''] || KIND.object;
const MODE: Record<string,string> = {consume:'Расход',produce:'Пополнение',transfer:'Перенос'};
const WARNING: Record<string,string> = {
 'Neural selection produced invalid stocks; state was NOT clamped or repaired.':'Модель выбрала недопустимый запас. Значение сохранено без исправления.',
 'Neural selection made opposite writes to the same relation; serialization NOT arbitrated.':'Модель выбрала противоположные изменения одной связи. Сериализатор не разрешал конфликт.',
 'Entity kind/properties are UI metadata, NOT model input features.':'Типы и визуальные свойства объектов не участвуют в нейронном инференсе.',
 'Instrument is encoded as a fact; a law must explicitly reference it (role w).':'Инструмент закодирован входным фактом. Закон должен явно ссылаться на него через роль w.',
 "Raw energy scales are not calibrated probabilities; 'conflict_detected' is post-hoc diagnostic only.":'Энергии не являются вероятностями. Конфликты обозначены диагностикой после выбора, не фильтром модели.'
};
const exprText = (r: Rule): string => hasGate(r) ? `(${exprText(r.left)} ${r.table.join('')==='0001'?'И':r.table.join('')==='0111'?'ИЛИ':`[${r.table.join('')}]`} ${exprText(r.right)})` : `${r.required===false?'¬':''}${r.predicate}(${r.left}, ${r.right})`;
const short = (s:string, n=23) => s.length > n ? `${s.slice(0,n-1)}…` : s;
const validStock = (e:Entity) => e.stock>=0 && e.stock<=e.capacity;
const LABEL: Record<string,string> = {__twm_action__:'наблюдение действия'};
const prettifyEffect = (text:string) => text.replace('resource(', 'запас(').replace('relation(', 'связь(').replace(') = add', ') — добавить').replace(') = remove', ') — удалить');

function IconBadge({kind, size='regular'}:{kind?: string|null;size?:string}) {
 const info=kindInfo(kind); const I=info.icon;
 return <span className={`object-icon ${info.hue} ${size==='lg'?'large':''}`}><I size={size==='lg'?24:18} strokeWidth={1.7}/></span>;
}
function Tag({children,accent='neutral'}:{children:React.ReactNode;accent?:string}) {return <span className={`tag tag-${accent}`}>{children}</span>}
function Button({children,onClick,variant='ghost',title,disabled=false,className=''}:{children:React.ReactNode;onClick?:()=>void;variant?:string;title?:string;disabled?:boolean;className?:string}) {
 return <button disabled={disabled} type="button" title={title} className={`btn btn-${variant} ${className}`} onClick={onClick}>{children}</button>;
}
function Header({step,presetIndex,presets,onPreset,onReset}:{step:number;presetIndex:number;presets:Props['presets'];onPreset:(i:number)=>void;onReset:()=>void}) {
 return <header className="topbar">
   <div className="brand-symbol"><Atom size={26} strokeWidth={1.6}/><span className="brand-spark"/></div>
   <div className="brand-text"><div className="brand-name">TWM <span>WORLD SANDBOX</span><b className="beta">BETA</b></div><div className="brand-caption">ЛАБОРАТОРИЯ НЕЙРОСИМВОЛЬНОЙ МОДЕЛИ МИРА</div></div>
   <div className="topbar-center"><span className="tiny-label">ТЕКУЩИЙ СЦЕНАРИЙ</span><select aria-label="Выбрать сценарий" value={presetIndex} onChange={e=>onPreset(Number(e.target.value))}>{presets.map((p,i)=><option key={i} value={i}>{p.name}</option>)}</select><ChevronDown size={14}/></div>
   <div className="topbar-right"><div className="online"><span className="pulse-dot"/>МОДЕЛЬ ЗАГРУЖЕНА</div><div className="top-separator"/><div className="model-tag">v7.10 <span>·</span> Seed 801</div><div className="top-separator"/><span className="step-pill">Шаг <strong>{String(step).padStart(2,'0')}</strong></span><Button title="Сбросить лабораторию" onClick={onReset}><RotateCcw size={16}/></Button></div>
 </header>
}

function EntityEditor({world,setWorld,selected,setSelected}:{world:World;setWorld:(w:World)=>void;selected:string|null;setSelected:(s:string|null)=>void}) {
 const [adding,setAdding] = useState(false); const [id,setId] = useState('новый_объект'); const [kind,setKind]=useState('object');
 const update=(eid:string,field:'stock'|'capacity',v:number) => setWorld({...world, entities:world.entities.map(e=>e.id===eid?{...e,[field]:Number.isFinite(v)?v:0}:e)});
 const remove=(eid:string)=>{if(world.entities.length<=3)return;setWorld({entities:world.entities.filter(e=>e.id!==eid),relations:world.relations.filter(r=>r.source!==eid&&r.target!==eid)}); if(selected===eid)setSelected(null)};
 const add=()=>{const x=id.trim();if(!x||world.entities.some(e=>e.id===x)||world.entities.length>=32)return;setWorld({...world,entities:[...world.entities,{id:x,kind,stock:2,capacity:10,properties:{}}]});setSelected(x);setAdding(false)};
 return <section className="panel-section entities">
  <div className="section-title"><span><Box size={15}/> СУЩНОСТИ <small>{world.entities.length}</small></span><Button title="Добавить сущность" disabled={world.entities.length>=32} onClick={()=>setAdding(!adding)}><Plus size={15}/></Button></div>
  <div className="entity-list">{world.entities.map(e=>{const info=kindInfo(e.kind); const isSelected=selected===e.id; return <div key={e.id} className={`entity ${isSelected?'selected':''}`}>
    <button className="entity-head" onClick={()=>setSelected(isSelected?null:e.id)}><IconBadge kind={e.kind}/><span className="entity-ident"><b title={e.id}>{short(e.id,22)}</b><small>{info.label}</small></span><span className={`mini-value ${validStock(e)?'':'danger-text'}`}>{e.stock}<span>/{e.capacity}</span></span><ChevronDown size={13} className={isSelected?'flip':''}/></button>
    <div className="resource-track"><motion.span key={`${e.stock}-${e.capacity}`} initial={{width:0}} animate={{width:`${Math.max(0,Math.min(100,e.capacity?e.stock/e.capacity*100:0))}%`}} transition={{duration:.65}} className={validStock(e)?'':'bad'}/></div>
    <AnimatePresence initial={false}>{isSelected && <motion.div initial={{height:0,opacity:0}} animate={{height:'auto',opacity:1}} exit={{height:0,opacity:0}} className="entity-detail"><label>Запас <input type="number" value={e.stock} onChange={ev=>update(e.id,'stock',Number(ev.target.value))}/></label><label>Вместимость <input type="number" min={1} value={e.capacity} onChange={ev=>update(e.id,'capacity',Number(ev.target.value))}/></label><button className="text-danger" disabled={world.entities.length<=3} onClick={()=>remove(e.id)}><Trash2 size={12}/> Удалить объект</button></motion.div>}</AnimatePresence>
  </div>})}</div>
  {adding&& <div className="add-form"><input value={id} aria-label="Идентификатор новой сущности" onChange={e=>setId(e.target.value)} placeholder="Идентификатор"/><select value={kind} onChange={e=>setKind(e.target.value)}><option value="object">Объект</option><option value="character">Персонаж</option><option value="door">Дверь</option><option value="key">Ключ</option><option value="room">Комната</option></select><Button variant="accent" onClick={add}>Добавить</Button></div>}
 </section>
}

function RelationEditor({world,setWorld}:{world:World;setWorld:(v:World)=>void}) {
 const [opened,setOpened]=useState(false);const [p,setP]=useState('подходит');const [a,setA]=useState('');const [b,setB]=useState('');
 const add=()=>{const u=a||world.entities[0]?.id,v=b||world.entities[1]?.id;if(!u||!v||!p.trim()||p.startsWith('__twm_action__/'))return;if(world.relations.some(r=>r.predicate===p.trim()&&r.source===u&&r.target===v))return;setWorld({...world,relations:[...world.relations,{predicate:p.trim(),source:u,target:v}]});setOpened(false)};
 return <section className="panel-section relations"><div className="section-title"><span><Network size={15}/> СВЯЗИ <small>{world.relations.length}</small></span><Button onClick={()=>setOpened(!opened)} title="Добавить связь"><Plus size={15}/></Button></div>
  {world.relations.length===0&&<p className="micro-muted">В этом состоянии нет отношений.</p>}
  <div className="relation-list">{world.relations.map((r,i)=><div className="relation-row" key={i}><span className="relation-pred" title={r.predicate}>{short(r.predicate,22)}</span><span className="relation-flow"><b title={r.source}>{short(r.source,14)}</b><ArrowRight size={12}/><b title={r.target}>{short(r.target,14)}</b></span><button onClick={()=>setWorld({...world,relations:world.relations.filter((_,j)=>j!==i)})} title="Удалить связь"><X size={12}/></button></div>)}</div>
  {opened&&<div className="add-form"><input value={p} onChange={e=>setP(e.target.value)} placeholder="Предикат" aria-label="Предикат"/><div className="form-two"><select value={a} onChange={e=>setA(e.target.value)} aria-label="Источник связи"><option value="">От кого</option>{world.entities.map(e=><option key={e.id} value={e.id}>{e.id}</option>)}</select><select aria-label="Цель связи" value={b} onChange={e=>setB(e.target.value)}><option value="">К кому</option>{world.entities.map(e=><option key={e.id} value={e.id}>{e.id}</option>)}</select></div><Button variant="accent" onClick={add}>Добавить факт</Button></div>}
 </section>
}

function LawEditor({laws,setLaws}:{laws:Law[];setLaws:(l:Law[])=>void}) {
 const [expanded,setExpanded] = useState<string|null>(null);const [code,setCode]=useState('');const [codeError,setCodeError]=useState('');
 const set=(id:string,part:Partial<Law>)=>setLaws(laws.map(l=>l.id===id?{...l,...part}:l));
 const shift=(index:number,d:number)=>{const j=index+d;if(j<0||j>=laws.length)return;const next=[...laws];[next[index],next[j]]=[next[j],next[index]];setLaws(next.map((l,i)=>({...l,priority:20-i})));};
 const open=(l:Law)=>{if(expanded===l.id){setExpanded(null);return;}setExpanded(l.id);setCode(JSON.stringify(l.condition,null,2));setCodeError('')};
 const saveRule=(id:string)=>{try{const x=JSON.parse(code) as Rule;if(!x||typeof x!=='object'||!('predicate' in x)&&!('table' in x))throw new Error('Ожидается дерево атомов / логических таблиц');set(id,{condition:x});setCodeError('');}catch(e){setCodeError((e as Error).message)}};
 const addLaw=()=>{if(laws.length>=5)return;const source=laws[0];if(!source)return;const i=laws.length+1;setLaws([...laws,{...copy(source),id:`закон_${i}_${Math.random().toString(16).slice(2,5)}`,priority:20-i,enabled:false}]);};
 return <section className="panel-section laws"><div className="section-title"><span><Layers3 size={15}/> ЗАКОНЫ МИРА <small>{laws.filter(x=>x.enabled).length} активны</small></span><Button title="Добавить закон (копия)" disabled={laws.length>=5} onClick={addLaw}><Plus size={15}/></Button></div>
 <p className="micro-muted intro">Приоритеты входят в признаки обученной модели, но не исполняются программно.</p>
 <div className="laws-list">{laws.map((l,i)=><div className={`law ${l.enabled?'':'dimmed'}`} key={l.id}>
   <div className="law-top"><span className="law-index">{String(i+1).padStart(2,'0')}</span><span className="law-name" title={l.id}>{short(l.id.replaceAll('_',' '),23)}</span><button className={`switch ${l.enabled?'active':''}`} role="switch" aria-checked={l.enabled} aria-label={`Включить закон ${l.id}`} onClick={()=>set(l.id,{enabled:!l.enabled})}><span/></button></div>
   <p className="law-brief">{MODE[l.mode]} {l.amount} · {l.relation_add?'добавить':'удалить'} «{short(l.effect_predicate,17)}»</p>
   <div className="law-actions"><Tag accent="neutral">приоритет {l.priority}</Tag><div className="law-tools"><button title="Выше" disabled={i===0} onClick={()=>shift(i,-1)}><ChevronLeft size={14} className="up-arrow"/></button><button title="Ниже" disabled={i===laws.length-1} onClick={()=>shift(i,1)}><ChevronRight size={14} className="up-arrow"/></button><button title="Настроить закон" onClick={()=>open(l)}><Settings2 size={15}/></button>{laws.length>1&&<button title="Удалить закон" onClick={()=>{setLaws(laws.filter(x=>x.id!==l.id));if(expanded===l.id)setExpanded(null)}}><Trash2 size={13}/></button>}</div></div>
  {expanded===l.id&&<div className="law-expanded"><div className="form-two"><label>Эффект<select value={l.mode} onChange={e=>set(l.id,{mode:e.target.value as Law['mode']})}><option value="consume">Расход</option><option value="produce">Пополнение</option><option value="transfer">Перенос</option></select></label><label>Количество<input type="number" min={1} value={l.amount} onChange={e=>set(l.id,{amount:Number(e.target.value)})}/></label></div><label>Предикат эффекта<input value={l.effect_predicate} onChange={e=>set(l.id,{effect_predicate:e.target.value})}/></label><label>Реляционная операция<select value={l.relation_add?'add':'remove'} onChange={e=>set(l.id,{relation_add:e.target.value==='add'})}><option value="add">Добавить отношение</option><option value="remove">Удалить отношение</option></select></label><label>AST условия — JSON<textarea spellCheck={false} rows={7} value={code} onChange={e=>setCode(e.target.value)}/></label>{codeError&&<div className="field-error">{codeError}</div>}<Button variant="accent" onClick={()=>saveRule(l.id)}>Применить условие</Button></div>}
 </div>)}</div>
 </section>
}

function ActionConsole({world,action,setAction,onRun,busy,readonly}:{world:World;action:Action;setAction:(a:Action)=>void;onRun:()=>void;busy:boolean;readonly:boolean}) {
 const set=(o:Partial<Action>)=>setAction({...action,...o});
 return <section className="action-console"><div className="console-top"><span className="console-live"><span/> ПУЛЬТ ДЕЙСТВИЙ</span><Zap size={16} color="#5addb3"/></div>
  <div className="action-grid"><label>Кто действует<select value={action.actor} onChange={e=>set({actor:e.target.value})}>{world.entities.map(e=><option key={e.id} value={e.id}>{e.id}</option>)}</select></label><label>Что делает<input aria-label="Глагол действия" value={action.verb} onChange={e=>set({verb:e.target.value})}/></label><label>Цель<select value={action.target} onChange={e=>set({target:e.target.value})}>{world.entities.map(e=><option key={e.id} value={e.id}>{e.id}</option>)}</select></label><label>Инструмент<select value={action.instrument??''} onChange={e=>set({instrument:e.target.value||null})}><option value="">Без инструмента</option>{world.entities.map(e=><option key={e.id} value={e.id}>{e.id}</option>)}</select></label></div>
  <Button variant="primary" onClick={onRun} disabled={busy||readonly} className="run-button">{busy?<><Activity size={18} className="spinner"/> Модель вычисляет энергии…</>:<><Play size={18} fill="currentColor"/> Запустить действие <ArrowRight size={18}/></>}</Button>
  <div className="console-foot"><ShieldCheck size={13}/> Только нейронный выбор · без скрытого эталона</div>
 </section>
}

const POSITIONS: Record<string, [number,number][]> = {
 character:[[19,33],[26,64],[12,78]],door:[[73,36],[82,65],[66,16]],key:[[45,20],[39,55],[61,55]],room:[[52,76],[78,78],[29,83]],object:[[42,41],[61,43],[16,53]],
};
function positions(entities:Entity[]) {
 const counters:Record<string,number>={};
 return Object.fromEntries(entities.map((e,i)=>{const k=e.kind||'object';const n=counters[k]||0;counters[k]=n+1;const pool=POSITIONS[k]||POSITIONS.object;return [e.id,n<pool.length?pool[n]:[15+(i*19)%72,20+(i*31)%65]];})) as Record<string,[number,number]>;
}
function WorldArena({world,before,revision,selected,setSelected,inspect}:{world:World;before?:World;revision:number;selected:string|null;setSelected:(x:string|null)=>void;inspect:boolean}) {
 const pos=useMemo(()=>positions(world.entities),[world.entities]);
 const baseline = before||world;
 const beforeFacts=new Set(baseline.relations.map(r=>`${r.predicate}/${r.source}/${r.target}`));
 const newFacts=new Set(world.relations.filter(r=>!beforeFacts.has(`${r.predicate}/${r.source}/${r.target}`)).map(r=>`${r.predicate}/${r.source}/${r.target}`));
 const changed = new Set(world.entities.filter(e=>baseline.entities.find(b=>b.id===e.id)?.stock!==e.stock).map(e=>e.id));
 const doorUnlocked = new Set(world.relations.filter(r=>r.predicate.includes('отперто')||r.predicate.includes('открыто')).map(r=>r.target));
 return <div className="arena"><div className="arena-grid"/><div className="arena-vignette"/>
  <div className="arena-top-left"><span className="micro-box"><CircleDot size={12}/> ТОПОГРАФИЯ / ГРАФ ОТНОШЕНИЙ</span><div className="arena-sub">{world.entities.length} объектов <span>·</span> {world.relations.length} связей</div></div>
  <div className="arena-top-right"><Tag accent={inspect?'amber':'mint'}>{inspect?'ПРОСМОТР ИСТОРИИ':'ТЕКУЩЕЕ СОСТОЯНИЕ'}</Tag></div>
  <svg className="links-canvas" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Граф отношений между объектами">
    <defs><filter id="glow"><feGaussianBlur stdDeviation=".4"/></filter></defs>
    {world.relations.map((r,i)=>{const a=pos[r.source],b=pos[r.target];if(!a||!b)return null;const isAdded=newFacts.has(`${r.predicate}/${r.source}/${r.target}`);return <g key={`${r.source}-${r.predicate}-${r.target}-${i}`}><motion.path initial={{pathLength:0,opacity:0}} animate={{pathLength:1,opacity:isAdded?1:.65}} transition={{duration:1.3}} d={`M ${a[0]} ${a[1]} Q ${(a[0]+b[0])/2} ${Math.min(a[1],b[1])-7} ${b[0]} ${b[1]}`} fill="none" stroke={isAdded?'#67efbd':'#356875'} strokeWidth={isAdded?.28:.18} strokeDasharray={isAdded?'0':'1.2 .9'} vectorEffect="non-scaling-stroke"/><circle cx={(a[0]+b[0])/2} cy={(a[1]+b[1])/2-4} r="0.55" fill={isAdded?'#6af0bf':'#4f8590'}/></g>})}
  </svg>
  {world.relations.map((r,i)=>{const a=pos[r.source],b=pos[r.target];if(!a||!b)return null;return <span key={`r-${i}`} className="edge-label" style={{left:`${(a[0]+b[0])/2}%`,top:`${(a[1]+b[1])/2-6}%`}} title={`${r.predicate}(${r.source},${r.target})`}>{short(r.predicate,17)}</span>})}
  {world.entities.map((e,i)=>{const [x,y]=pos[e.id],isChanged=changed.has(e.id);return <motion.button layout key={e.id} initial={{opacity:0,scale:.75,y:10}} animate={{opacity:1,scale:1,y:0}} transition={{delay:i*.07,type:'spring',stiffness:155,damping:17}} whileHover={{scale:1.055,y:-5}} onClick={()=>setSelected(selected===e.id?null:e.id)} className={`arena-node ${selected===e.id?'is-selected':''} ${isChanged?'is-changed':''} ${doorUnlocked.has(e.id)?'is-unlocked':''}`} style={{left:`${x}%`,top:`${y}%`}} title={`Открыть инспектор: ${e.id}`}>
    <span className="node-accent"/><span className="node-inner"><IconBadge kind={e.kind}/><span className="node-names"><b>{short(e.id,20)}</b><small>{kindInfo(e.kind).label}</small></span>{doorUnlocked.has(e.id)&&<span className="node-unlocked"><Check size={12}/></span>}</span><span className="node-bottom"><span>РЕСУРС {e.stock} <i>/ {e.capacity}</i></span>{isChanged&&<span className="node-delta"><Zap size={12}/> Δ</span>}</span>
  </motion.button>})}
  <div className="arena-bottom-left"><div className="legend"><span><i className="legend-circle mint"/>Сущности</span><span><i className="legend-line"/>Отношения</span><span><i className="legend-circle changed"/>Изменения</span></div></div>
  <div className="arena-bottom-right"><div className="coord">X: 48.211 &nbsp; Y: 17.024<br/>ПРОЕКЦИЯ: СХЕМА ФАКТОВ</div></div>
 </div>
}

function EnergyDisplay({subsets,selected,setSelected,showAll,setShowAll}:{subsets:Subset[];selected:number|null;setSelected:(v:number|null)=>void;showAll:boolean;setShowAll:(v:boolean)=>void}) {
 const winner=subsets.find(s=>s.is_winner);
 const sorted=[...subsets].sort((a,b)=>b.total_consistency_energy-a.total_consistency_energy);
 const shown=showAll?sorted:sorted.slice(0,6);
 const min=Math.min(...subsets.map(s=>s.total_consistency_energy)),max=Math.max(...subsets.map(s=>s.total_consistency_energy));
 const trackWidth=(s:Subset)=> Math.max(5,((s.total_consistency_energy-min)/(max-min||1)*92)+8);
 return <div className="multiverse"><div className="block-header"><div><span className="eyebrow"><GitBranch size={15}/> ПРОСТРАНСТВО ВАРИАНТОВ</span><h3>Мультивселенная <span>исходов</span></h3></div><Tag accent="mint">{subsets.length} сценариев</Tag></div>
 <div className="score-method"><Activity size={15}/><span>Нейронная энергия JointConsistencyEnergy<br/><small>Энергия = базовая v7.9 + совместная поправка v7.10</small></span></div>
 {winner&&<motion.div key={winner.subset_id+'winner'} initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} className="winner-hero"><div className="winner-ring"><Check size={20}/></div><div><div className="winner-label">ВЫБРАН МОДЕЛЬЮ · ARGMAX</div><b>Сценарий №{winner.subset_id}</b></div><span className="winner-score">{num(winner.total_consistency_energy)}</span></motion.div>}
 <div className="multiverse-list">{shown.map((s,i)=><motion.button key={s.subset_id} initial={{opacity:0,x:24,scale:.95}} animate={{opacity:1,x:0,scale:1}} transition={{type:'spring',stiffness:175,damping:21,delay:i*.055}} className={`universe-card ${s.is_winner?'victor':''} ${selected===s.subset_id?'universe-selected':''}`} onClick={()=>setSelected(selected===s.subset_id?null:s.subset_id)}>
    <div className="universe-row"><span className="universe-id"><span className={`universe-status ${s.is_winner?'':'loss'}`}>{s.is_winner?<Check size={12}/>:<CircleDot size={10}/>}</span>Вариант {String(s.subset_id).padStart(2,'0')}</span><span className="universe-total">{num(s.total_consistency_energy)}</span></div>
    <div className="bar-back"><motion.div className={`energy-fill ${s.is_winner?'winning':''}`} initial={{width:0}} animate={{width:`${trackWidth(s)}%`}} transition={{duration:.8,delay:i*.06+.15}}/></div>
    <div className="energy-meta"><span>Основа: {num(s.base_v79_energy)}</span><span>Δ Joint: {num(s.joint_v710_residual_energy)}</span></div>
    {s.conflict_detected&&<span className="danger-pill"><CircleAlert size={11}/> Диагностика конфликта</span>}
    <AnimatePresence>{selected===s.subset_id&&<motion.div className="universe-detail" initial={{height:0,opacity:0}} animate={{height:'auto',opacity:1}} exit={{height:0,opacity:0}}><div className="detail-label">ЭФФЕКТЫ ПОДМНОЖЕСТВА</div>{s.effects_applied.length?s.effects_applied.map((e,j)=><div key={j} className="effect-line">{prettifyEffect(e)}</div>):<div className="effect-line">Изменения не выбраны</div>}<div className="diag-inline">Недопустимый ресурс: {String(s.diagnostics.invalid_numeric??0)} · Противоположные записи: {String(s.diagnostics.opposed_relations??0)}</div></motion.div>}</AnimatePresence>
  </motion.button>)}</div>
 {sorted.length>6&&<button className="show-more" onClick={()=>setShowAll(!showAll)}>{showAll?'Свернуть варианты':`Показать все ${subsets.length} вариантов`} <ChevronDown size={13} className={showAll?'flip':''}/></button>}
 <div className="truth-foot"><ShieldAlert size={14}/> Конфликты помечены после выбора. Не участвуют в фильтрации argmax.</div>
 </div>
}

function HistoryPanel({history,selected,setSelected,onBranch}:{history:StepRecord[];selected:number|null;setSelected:(n:number|null)=>void;onBranch:(n:number)=>void}) {
 return <div className="history-panel"><div className="block-header"><div><span className="eyebrow"><History size={15}/> ВРЕМЕННАЯ ШКАЛА</span><h3>История <span>мира</span></h3></div><Tag accent="neutral">{history.length} шагов</Tag></div>
 <p className="history-intro">Каждый шаг хранит настоящий ответ модели, оценки всех вариантов и полученный мир.</p>
 <div className="timeline"><button className={`timeline-item ${selected===0?'chosen':''}`} onClick={()=>setSelected(selected===0?null:0)}><span className="timeline-axis"/><span className="timeline-number"><Box size={14}/></span><span className="timeline-text"><b>Начальный мир</b><small>Состояние до первого действия</small></span><ChevronRight size={13}/></button>
 {history.map(h=><button key={h.number} className={`timeline-item ${selected===h.number?'chosen':''}`} onClick={()=>setSelected(selected===h.number?null:h.number)}><span className="timeline-axis"/><span className="timeline-number">{h.number}</span><span className="timeline-text"><b>{h.action.verb} · {h.action.target.replaceAll('_',' ')}</b><small>{h.action.actor} · победил #{h.response.step_log.winner_subset_id}</small></span><ChevronRight size={13}/></button>)}</div>
 {selected!==null&&<div className="timeline-inspect"><Tag accent="amber">СНИМОК №{selected}</Tag><p>На карте показано состояние после выбранного шага. Для продолжения из этой точки создайте новую ветку.</p><Button variant="accent" onClick={()=>onBranch(selected)}><GitBranch size={14}/> Продолжить отсюда</Button></div>}
 </div>
}

function InsightStrip({world,before,log,history}:{world:World;before?:World;log?:StepResponse;history:number}) {
 const changed=before?world.entities.filter(e=>e.stock!==before.entities.find(x=>x.id===e.id)?.stock).length:0;
 const oldFacts=new Set((before||world).relations.map(r=>`${r.predicate}/${r.source}/${r.target}`));
 const currFacts=new Set(world.relations.map(r=>`${r.predicate}/${r.source}/${r.target}`));
 const relDelta=[...currFacts].filter(x=>!oldFacts.has(x)).length+[...oldFacts].filter(x=>!currFacts.has(x)).length;
 const N=log?.step_log.evaluated_subsets.length??0;
 return <div className="insight-strip"><div className="insight"><div className="insight-icon"><Box size={18}/></div><div><span>СУЩНОСТЕЙ</span><b>{world.entities.length}</b></div></div><div className="insight"><div className="insight-icon"><Network size={18}/></div><div><span>ОТНОШЕНИЙ</span><b>{world.relations.length}</b></div></div><div className="insight"><div className="insight-icon highlight"><GitBranch size={18}/></div><div><span>ОЦЕНЕНО ВАРИАНТОВ</span><b>{N||'—'}</b></div></div><div className="insight"><div className="insight-icon"><Activity size={18}/></div><div><span>ИЗМЕНЕНИЙ ЗА ШАГ</span><b>{changed+relDelta}</b></div></div></div>
}
function ResultSummary({record}:{record?:StepRecord}) {
 if(!record)return <div className="empty-result"><div className="empty-glow"><FlaskConical size={31}/></div><h3>Мир готов к эксперименту</h3><p>Измените исходные факты или законы, выберите действие и запустите реальный инференс TWM v7.10.</p><div className="empty-chain"><span>МИР</span><ArrowRight size={14}/><span>ЗАКОНЫ</span><ArrowRight size={14}/><span>НЕЙРОСЕТЬ</span><ArrowRight size={14}/><span>БУДУЩЕЕ</span></div></div>;
 const w=record.response.step_log.evaluated_subsets.find(s=>s.is_winner);
 return <div className="run-summary"><span className="eyebrow"><Sparkles size={14}/> РЕЗУЛЬТАТ НЕЙРОННОГО ПЕРЕХОДА</span><div className="summary-row"><div><h3>{record.action.verb}<span>({record.action.actor}, {record.action.target})</span></h3><div className="summary-muted">Выбрано подмножество №{record.response.step_log.winner_subset_id} из {record.response.step_log.evaluated_subsets.length}</div></div><span className="summary-energy">E = {w?num(w.total_consistency_energy):'—'}</span></div><div className="effect-chips">{w?.effects_applied.length?w.effects_applied.map((t,i)=><span key={i}><CircleCheck size={13}/>{prettifyEffect(t)}</span>):<span><CircleDot size={12}/> Модель не выбрала эффектов</span>}</div>
 {w?.conflict_detected&&<div className="summary-warning"><CircleAlert size={14}/> Выбран конфликтный исход. Результат не исправлен программно.</div>}
 </div>
}

function App({data,dispatch}:{data:Props;dispatch:(x:unknown)=>void}) {
 const [world,setWorld] = useState<World>(()=>copy(data.world_state));
 const [laws,setLaws] = useState<Law[]>(()=>copy(data.laws));
 const [action,setAction] = useState<Action>(()=>copy(data.action));
 const [selectedEntity,setSelectedEntity] = useState<string|null>(null);
 const [selectedHistory,setSelectedHistory] = useState<number|null>(null);
 const [tab,setTab]=useState<'multiverse'|'history'>('multiverse');
 const [selectedSubset,setSelectedSubset]=useState<number|null>(null);
 const [showAll,setShowAll]=useState(false);
 const [busy,setBusy]=useState(false);
 const [clientError,setClientError]=useState<string|null>(null);
 const lastRevision=useRef(data.revision);
 const prevCount=useRef(data.history.length);
 useEffect(()=>{
  if(data.revision!==lastRevision.current){
    setWorld(copy(data.world_state));setLaws(copy(data.laws));setAction(copy(data.action));
    setBusy(false);setClientError(null);setSelectedEntity(null);setSelectedHistory(null);setSelectedSubset(null);setShowAll(false);
    if(data.history.length>prevCount.current)setTab('multiverse');
    prevCount.current=data.history.length;lastRevision.current=data.revision;
  }
 },[data.revision,data.world_state,data.laws,data.action,data.history.length]);
 const history=data.history;
 const viewing=selectedHistory!==null;
 const targetRecord=viewing&&selectedHistory!>0?history[selectedHistory!-1]:!viewing&&history.length?history[history.length-1]:undefined;
 const showWorld=viewing?(selectedHistory===0?(history[0]?.world_state_before??data.world_state):(history[selectedHistory!-1]?.world_state_after??world)):world;
 const before=viewing?(selectedHistory===0?undefined:history[selectedHistory!-1]?.world_state_before):targetRecord?.world_state_before;
 const run=()=>{
   const validLaw=laws.filter(l=>l.enabled);const ids=new Set(world.entities.map(e=>e.id));
   if(!validLaw.length||validLaw.length>5){setClientError('Для инференса нужно от 1 до 5 включённых законов.');return;}
   if(world.entities.some(e=>!validStock(e)||e.capacity<=0)){setClientError('Начальные запасы должны находиться между нулём и вместимостью. Нейросетевые ошибки после прогноза не исправляются.');return;}
   if(!ids.has(action.actor)||!ids.has(action.target)||(action.instrument&&!ids.has(action.instrument))){setClientError('Действие ссылается на отсутствующую сущность.');return;}
   setClientError(null);setBusy(true);
   dispatch({type:'step',request:{world_state:world,laws,action}});
 };
 const reset=()=>{if(busy)return;setBusy(true);dispatch({type:'reset'})};
 const preset=(i:number)=>{if(busy)return;setBusy(true);dispatch({type:'preset',index:i})};
 const branch=(n:number)=>{if(busy)return;setBusy(true);dispatch({type:'branch',number:n})};
 const share=()=>{const payload=JSON.stringify({world_state:world,laws,action},null,2);const blob=new Blob([payload],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='twm_eksperiment.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2500)};
 return <main className="twm-app"><Header step={history.length} presetIndex={data.presetIndex} presets={data.presets} onPreset={preset} onReset={reset}/>
 <div className="workspace">
 <aside className="left-panel"><div className="panel-heading"><div className="eyebrow"><span className="section-mark"/> ПАНЕЛЬ УПРАВЛЕНИЯ</div><h2>Конструктор <span>мира</span></h2><p>Редактируйте факты, ресурсы и законы. Модель сама выбирает последствия.</p></div>
  <div className="left-scroll"><EntityEditor world={world} setWorld={setWorld} selected={selectedEntity} setSelected={setSelectedEntity}/><RelationEditor world={world} setWorld={setWorld}/><LawEditor laws={laws} setLaws={setLaws}/></div>
  <div className="left-footer"><Button onClick={share}><Download size={14}/> Экспортировать JSON</Button></div>
 </aside>
 <section className="center-panel"><div className="center-head"><div className="center-breadcrumb"><span className="crumb-active">ПРОСТРАНСТВО МИРА</span><ChevronRight size={13}/><span>СИМУЛЯЦИЯ / SEED 801</span></div><div className="center-head-actions"><span className="status-text"><span/> НЕЙРОННЫЙ ИНФЕРЕНС</span><Tag accent="neutral">CPU</Tag></div></div>
  <InsightStrip world={showWorld} before={before} log={targetRecord?.response} history={history.length}/>
  <div className="arena-container"><WorldArena world={showWorld} before={before} revision={data.revision} selected={selectedEntity} setSelected={setSelectedEntity} inspect={viewing}/></div>
  <div className="center-lower"><ResultSummary record={targetRecord}/></div>
  {(data.error||clientError)&&<div className="error-banner"><CircleAlert size={17}/><div><b>Действие не выполнено</b><p>{clientError||data.error}</p></div><button onClick={()=>setClientError(null)}><X size={16}/></button></div>}
  {viewing&&<div className="preview-notice"><Clock3 size={16}/> Вы просматриваете снимок №{selectedHistory}. Рабочее состояние не изменено.<Button variant="accent" onClick={()=>branch(selectedHistory!)}>Создать ветку отсюда</Button><Button onClick={()=>setSelectedHistory(null)}>К текущему миру</Button></div>}
  <ActionConsole world={world} action={action} setAction={setAction} onRun={run} busy={busy} readonly={viewing}/>
 </section>
 <aside className="right-panel"><div className="right-tabs"><button className={tab==='multiverse'?'active':''} onClick={()=>setTab('multiverse')}><GitBranch size={15}/> Варианты <span>{targetRecord?.response.step_log.evaluated_subsets.length??0}</span></button><button className={tab==='history'?'active':''} onClick={()=>setTab('history')}><History size={15}/> История <span>{history.length}</span></button></div>
  <div className="right-scroll"><AnimatePresence mode="wait">{tab==='multiverse'?<motion.div key="u" initial={{opacity:0,x:10}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-10}}><>{targetRecord?<EnergyDisplay subsets={targetRecord.response.step_log.evaluated_subsets} selected={selectedSubset} setSelected={setSelectedSubset} showAll={showAll} setShowAll={setShowAll}/>:<div className="multiverse-placeholder"><div className="placeholder-orbits"><GitBranch size={37}/></div><span className="eyebrow">МУЛЬТИВСЕЛЕННАЯ</span><h3>Варианты ещё<br/>не вычислены</h3><p>Когда вы запустите действие, здесь появятся все подмножества эффектов с настоящими энергиями сети.</p><div className="placeholder-equation">E₍S₎ = E₇.₉ + ΔEⱼₒᵢₙₜ</div></div>}</></motion.div>:<motion.div key="h" initial={{opacity:0,x:10}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-10}}><HistoryPanel history={history} selected={selectedHistory} setSelected={setSelectedHistory} onBranch={branch}/></motion.div>}</AnimatePresence></div>
  <div className="right-footer"><div className="integrity-icon"><ShieldCheck size={18}/></div><div><strong>Без программной посткоррекции</strong><span>Все оценки получены из модели. Эталон не выполняется при прогнозе.</span></div></div>
 </aside>
 </div>
 <footer className="bottom-bar"><div><span className="tiny-blink"/> TWM v7.10 · исследовательская грамматика · до 5 законов · до 8 предложенных эффектов</div><div>Типы и имена объектов — метаданные UI, а не языковое понимание модели <span className="footer-bar-sep">|</span> Протокол: строгий <span>argmax</span></div></footer>
 </main>
}

const roots = new WeakMap<Element,Root>();
export default function mount(component:CompArgs) {
 const container=component.parentElement.querySelector('#twm-sandbox-react-root');
 if(!container)return;
 let root=roots.get(container);
 if(!root){root=createRoot(container);roots.set(container,root);}
 // Trigger values are transient and cause exactly one backend operation.
 root.render(<App data={component.data} dispatch={(command)=>component.setTriggerValue('command',JSON.stringify(command))}/>);
 return ()=>{ /* Keeping the root in place across Streamlit reruns prevents flicker. */ };
}
