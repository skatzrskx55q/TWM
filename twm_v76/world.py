"""Synthetic multi-law worlds. The oracle below is TRAINING/TEST ONLY.

Semantics: at most ONE atomic action (and its related numeric and relational
change) commits per tick. Of individually applicable proposals, highest
priority wins. A rule may have several distinct bindings, but synthetic worlds
use unique selector facts; ties in priority are rejected at validation.

The structured effect type/amount and atom lookup are declared infrastructure;
they are NOT learned. No oracle is called by inference.py.
"""
from dataclasses import dataclass, replace
from random import Random
from itertools import permutations
from twm_v75.world import (Atom,Gate,Law,State,Case,AND,MODES,TRAIN_TABLES,
    HELDOUT_TABLES,truth,numeric_legal,enumerate_bindings)

@dataclass(frozen=True)
class Proposal:
    law:Law
    priority:int
    identity:str

@dataclass(frozen=True)
class MultiCase:
    actor:str
    state:State
    proposals:tuple
    domain:str='conflict'

def single_cases(m):
    return tuple(Case(m.actor,m.state,p.law,m.domain) for p in m.proposals)

def validate(m):
    if len(m.proposals)<2 or len(m.proposals)>5:raise ValueError('2..5 proposals required')
    ps=[p.priority for p in m.proposals]
    if len(set(ps))!=len(ps):raise ValueError('Unique priorities required')
    if any(not isinstance(p.priority,int) or p.priority<0 or p.priority>20 for p in m.proposals):raise ValueError('Bad priority')
    if any(p.law.mode not in MODES or p.law.amount<1 for p in m.proposals):raise ValueError('Bad law')
    if len(m.state.entities)<3:raise ValueError('Bad world')
    if len(m.state.entities)!=len(m.state.stocks) or len(m.state.entities)!=len(m.state.capacities):raise ValueError('Shape')
    if any(x<0 or x>c for x,c in zip(m.state.stocks,m.state.capacities)):raise ValueError('Initial resources')

def reference_winner(m):
    """Oracle arbitration, isolated from neural inference.

    Returns (proposal_index,binding_index), or None for no-op.
    """
    validate(m)
    options=[]
    for i,c in enumerate(single_cases(m)):
        for k,(u,v,w) in enumerate(enumerate_bindings(c)):
            if truth(c.law.expr,c,u,v,w) and numeric_legal(c,u,v):
                options.append((m.proposals[i].priority,i,k))
    if not options:return None
    _,i,k=max(options)
    return i,k

def apply_choice(m,choice):
    """Typed write-back only. Given a chosen proposal, applies its declared
    effects; DOES NOT run truth, priority, or legality checks. A wrong neural
    selection can create an invalid resource state and is never clamped.
    """
    if choice is None:return m.state,False
    i,k=choice
    c=single_cases(m)[i];u,v,w=list(enumerate_bindings(c))[k]
    law=c.law;s=m.state;stock=list(s.stocks);facts=set(s.facts)
    a=s.entities.index(u);b=s.entities.index(v)
    if law.mode in ('transfer','consume'):stock[a]-=law.amount
    if law.mode in ('transfer','produce'):stock[b]+=law.amount
    key=(law.effect_predicate,u,v)
    if law.relation_add:facts.add(key)
    else:facts.discard(key)
    return State(s.entities,frozenset(facts),tuple(stock),s.capacities),True

def reference_transition(m):
    return apply_choice(m,reference_winner(m))

def make_case(seed,domain='conflict',n=6,laws=3,heldout=False,amount_ood=False,
              all_invalid=False, priority_ood=False):
    r=Random(seed);actor=f'{domain}:actor:{seed}'
    es=tuple(f'{domain}:entity:{seed}:{i}' for i in range(n))
    caps=[r.randint(9,16) for _ in es]
    stocks=[r.randint(0,x) for x in caps]
    facts=set();base_preds=[f'{domain}:noise:{j}' for j in range(3)]
    for pred in base_preds:
        for a,b in permutations((actor,*es),2):
            if r.random()<.08:facts.add((pred,a,b))
    priorities=r.sample(range(5,5+laws) if priority_ood else range(laws),laws)
    shared_pair=r.sample(range(n),2)
    proposals=[];plans=[]
    for j in range(laws):
        # Disambiguating atoms are raw graph facts, not rule decisions.
        if j and r.random()<.55:
            a,b=shared_pair
        else:a,b=r.sample(range(n),2)
        if j==0:shared_pair=(a,b)
        selu=f'{domain}:selector:{j}:u';selv=f'{domain}:selector:{j}:v'
        facts.add((selu,actor,es[a]));facts.add((selv,actor,es[b]))
        # Create a balanced mix of true and false arbitrary operator outputs;
        # each law uses independent facts, making conflict labels inspectable.
        table=r.choice(HELDOUT_TABLES if heldout else TRAIN_TABLES)
        desired=not all_invalid and r.random()<.75
        outcomes=[(p,q) for p in (0,1) for q in (0,1) if bool(table[p*2+q])==desired]
        if not outcomes:
            desired=not desired
            outcomes=[(p,q) for p in (0,1) for q in (0,1) if bool(table[p*2+q])==desired]
        aa,bb=r.choice(outcomes)
        roles=[('actor','u'),('actor','v'),('u','v'),('v','u')]
        pred1=f'{domain}:condition:{j}:A';pred2=f'{domain}:condition:{j}:B'
        pair1=r.choice(roles);pair2=r.choice(roles)
        neg1=bool(r.randrange(2));neg2=bool(r.randrange(2))
        names={'actor':actor,'u':es[a],'v':es[b]}
        for pred,roles_pair,required,out in [(pred1,pair1,neg1,aa),(pred2,pair2,neg2,bb)]:
            present=bool(out)==required
            if present:facts.add((pred,names[roles_pair[0]],names[roles_pair[1]]))
        body=Gate(tuple(table),Atom(pred1,*pair1,neg1),Atom(pred2,*pair2,neg2))
        expr=Gate(AND,Atom(selu,'actor','u'),Gate(AND,Atom(selv,'actor','v'),body))
        mode=r.choice(MODES)
        room=caps[b]-stocks[b]
        maxlegal= {'transfer':min(stocks[a],room),'consume':stocks[a],'produce':room}[mode]
        should_legal=r.random()<.72 and maxlegal>=1 and not all_invalid
        if should_legal:amount=r.randint(1,min(maxlegal,3 if not amount_ood else 7))
        else:amount=maxlegal+r.randint(1,3) if not amount_ood else maxlegal+r.randint(2,6)
        if amount_ood:amount=max(6,amount)
        law=Law(expr,mode,amount,f'{domain}:effect',bool(r.randrange(2)))
        proposals.append(Proposal(law,priorities[j],f'{domain}:rule:{j}'))
        plans.append((a,b,desired,should_legal))
    # Permute law order so proposal index cannot stand in for priority.
    order=list(range(laws));r.shuffle(order)
    proposals=tuple(proposals[q] for q in order)
    return MultiCase(actor,State(es,frozenset(facts),tuple(stocks),tuple(caps)),proposals,domain)
