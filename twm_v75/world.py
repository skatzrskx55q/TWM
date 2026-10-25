"""Experimental rule AST world. Reference evaluation is training/evaluation ONLY.

An unknown Boolean operator is supplied by its full 2-input truth table in the law.
This is NOT natural language understanding nor invention of operator semantics.
"""
from dataclasses import dataclass, replace
from itertools import permutations
import random
import torch

ROLES=('actor','u','v','w')
MODES=('transfer','consume','produce')
AND=(0.,0.,0.,1.)
HELDOUT_TABLES=((1.,1.,1.,0.),(1.,0.,0.,0.),(1.,0.,0.,1.),(0.,0.,1.,0.)) # NAND NOR XNOR A&!B
TABLES=tuple(tuple(float((bits>>k)&1) for k in range(4)) for bits in range(16))
TRAIN_TABLES=tuple(x for x in TABLES if x not in HELDOUT_TABLES)

@dataclass(frozen=True)
class Atom:
    predicate:str
    left:str
    right:str
    required:bool=True

@dataclass(frozen=True)
class Gate:
    table:tuple
    left:object
    right:object

@dataclass(frozen=True)
class Law:
    expr:object
    mode:str
    amount:int
    effect_predicate:str
    relation_add:bool

@dataclass(frozen=True)
class State:
    entities:tuple
    facts:frozenset
    stocks:tuple
    capacities:tuple

@dataclass(frozen=True)
class Case:
    actor:str
    state:State
    law:Law
    domain:str='toy'


def leaves(expr):
    if isinstance(expr,Atom): return (expr,)
    return leaves(expr.left)+leaves(expr.right)


def gates(expr):
    if isinstance(expr,Atom):return ()
    return (expr.table,)+gates(expr.left)+gates(expr.right)


def uses_w(case):
    return any(a.left=='w' or a.right=='w' for a in leaves(case.law.expr))


def validate(case):
    s=case.state;l=case.law
    if l.mode not in MODES or not isinstance(l.amount,int) or l.amount<1:raise ValueError('Invalid numerical law')
    if len(s.entities)<3 or len(s.stocks)!=len(s.entities) or len(s.capacities)!=len(s.entities):raise ValueError('Invalid entities')
    if case.actor in s.entities or len(set(s.entities))!=len(s.entities):raise ValueError('Duplicate entities')
    if any(x<0 or y<x for x,y in zip(s.stocks,s.capacities)):raise ValueError('Invalid resource initial state')
    if len(leaves(l.expr))>10:raise ValueError('Expression too large')
    for a in leaves(l.expr):
        if a.left not in ROLES or a.right not in ROLES:raise ValueError('Unknown role')
    for t in gates(l.expr):
        if len(t)!=4 or any(v not in (0.,1.,0,1) for v in t):raise ValueError('Binary truth table required')


def enumerate_bindings(case):
    es=case.state.entities
    for u in es:
        for v in es:
            if u==v:continue
            witnesses=(x for x in es if x!=u and x!=v) if uses_w(case) else (None,)
            for w in witnesses:yield u,v,w


def truth(expr,case,u,v,w):
    if isinstance(expr,Atom):
        names=dict(actor=case.actor,u=u,v=v,w=w)
        return int(((expr.predicate,names[expr.left],names[expr.right]) in case.state.facts)==expr.required)
    a=truth(expr.left,case,u,v,w);b=truth(expr.right,case,u,v,w)
    return int(expr.table[a*2+b])


def numeric_legal(case,u,v):
    p={x:i for i,x in enumerate(case.state.entities)};i=p[u];j=p[v];s=case.state;l=case.law
    return bool((l.mode=='produce' or s.stocks[i]>=l.amount) and
                (l.mode=='consume' or s.stocks[j]+l.amount<=s.capacities[j]))


def reference_transition(case):
    """True rules, kept separate from neural inference (teacher only)."""
    validate(case);s=case.state;l=case.law;p={e:i for i,e in enumerate(s.entities)}
    stock=list(s.stocks);facts=set(s.facts);done=set();accepted=False
    for u,v,w in enumerate_bindings(case):
        if (u,v) in done or not truth(l.expr,case,u,v,w) or not numeric_legal(case,u,v):continue
        done.add((u,v));accepted=True
        key=(l.effect_predicate,u,v)
        if l.relation_add:facts.add(key)
        else:facts.discard(key)
        if l.mode in ('transfer','consume'):stock[p[u]]-=l.amount
        if l.mode in ('transfer','produce'):stock[p[v]]+=l.amount
    return State(s.entities,frozenset(facts),tuple(stock),s.capacities),accepted


def encode(case):
    """Syntax-only graph lookup (not a semantic clause/guard executor).

    Binding enumeration and exact raw relation membership are infrastructure.
    Neither polarity comparison, Boolean operator, nor numeric legality is
    computed by this adapter.
    """
    validate(case);s=case.state;l=case.law;es=s.entities;p={x:i for i,x in enumerate(es)}
    ls=leaves(l.expr);bindings=list(enumerate_bindings(case));raw=[];nums=[]
    for u,v,w in bindings:
        names=dict(actor=case.actor,u=u,v=v,w=w)
        raw.append([[float((a.predicate,names[a.left],names[a.right]) in s.facts),float(a.required)] for a in ls])
        i,j=p[u],p[v]
        nums.append([s.stocks[i]/16.,s.stocks[j]/16.,s.capacities[i]/16.,s.capacities[j]/16.,l.amount/16.,
                     *[float(l.mode==m) for m in MODES]])
    return (torch.tensor(raw,dtype=torch.float32),torch.tensor(nums,dtype=torch.float32),bindings)


def predict(model,case):
    """Typed effects use neural accept decisions only; no oracle or numerical clamp."""
    raw,numbers,bindings=encode(case)
    with torch.no_grad(): scores=model(raw,numbers,case.law.expr).flatten().tolist()
    s=case.state;l=case.law;stocks=list(s.stocks);facts=set(s.facts);p={e:i for i,e in enumerate(s.entities)}
    edits=set();accepted=False
    for (u,v,w),score in zip(bindings,scores):
        if score<.5 or (u,v) in edits:continue
        edits.add((u,v));accepted=True
        key=(l.effect_predicate,u,v)
        if l.relation_add:facts.add(key)
        else:facts.discard(key)
        if l.mode in ('transfer','consume'):stocks[p[u]]-=l.amount
        if l.mode in ('transfer','produce'):stocks[p[v]]+=l.amount
    return State(s.entities,frozenset(facts),tuple(stocks),s.capacities),accepted


def make_case(seed,domain='rooms',n=6,depth=1,witness=False,heldout=False,amount=3,mode=None,plant=True,noise=.19):
    """Deterministic controlled generator; all heldout variations are explicit."""
    r=random.Random(seed)
    actor=f'{domain}:actor:{seed}'
    es=tuple(f'{domain}:item:{seed}:{i}' for i in range(n))
    u,v=r.sample(es,2);w=r.choice([x for x in es if x not in (u,v)])
    preds=[f'{domain}:p{k}' for k in range(5)]
    facts=set()
    for p in preds:
        for a,b in permutations((actor,*es),2):
            if r.random()<noise:facts.add((p,a,b))
    # Explicit syntactic selection facts, not a hidden selector in prediction.
    sel_u=Atom(f'{domain}:targetU','actor','u')
    sel_v=Atom(f'{domain}:targetV','actor','v')
    facts.add((sel_u.predicate,actor,u));facts.add((sel_v.predicate,actor,v))
    roles=[('actor','u'),('actor','v'),('u','v'),('v','u')]
    if witness:roles+= [('w','u'),('v','w'),('actor','w')]
    table_pool=HELDOUT_TABLES if heldout else TRAIN_TABLES
    table=lambda:r.choice(table_pool)
    def expression(d,idx=0):
        if d<=0:
            p=preds[r.randrange(len(preds))]
            aa,bb=r.choice(roles[4:] if witness and idx==0 else roles)
            return Atom(p,aa,bb,bool(r.randrange(2)))
        return Gate(table(),expression(d-1,idx*2),expression(d-1,idx*2+1))
    body=expression(depth)
    # A mandatory graph-grounded target gate prevents effects at unrelated pairs.
    tree=Gate(AND,sel_u,Gate(AND,sel_v,body))
    cap=[r.randrange(8,16) for _ in es];st=[r.randrange(0,c+1) for c in cap]
    m=mode or r.choice(MODES)
    if plant:
        i,j=es.index(u),es.index(v)
        # Make effect feasibility likely while keeping hard boundaries in negatives.
        if m in ('consume','transfer'):st[i]=r.randint(min(amount,cap[i]),cap[i])
        if m in ('produce','transfer'):st[j]=r.randint(0,max(0,cap[j]-amount))
    law=Law(tree,m,amount,f'{domain}:output',bool(r.randrange(2)))
    return Case(actor,State(es,frozenset(facts),tuple(st),tuple(cap)),law,domain)


def labels(case):
    raw,nums,bindings=encode(case)
    ys=[];truths=[];legals=[]
    for u,v,w in bindings:
        t=truth(case.law.expr,case,u,v,w);g=int(numeric_legal(case,u,v))
        ys.append(float(t*g));truths.append(float(t));legals.append(float(g))
    return (raw,nums,case.law.expr,torch.tensor(ys).view(-1,1),torch.tensor(truths).view(-1,1),torch.tensor(legals).view(-1,1))
