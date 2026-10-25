"""Simultaneous multi-effect worlds; oracle exclusively for training/evaluation.

Teacher policy: consider individually valid proposals by descending priority, and
commit every effect compatible with previously accepted effects. Compatibility
requires aggregate resource stocks within capacities and no opposite writes to
the same relation. This is a deliberately limited greedily resolved law grammar;
not a universal conflict calculus. All reference functions are banned in inference.
"""
from dataclasses import dataclass,replace
from random import Random
from twm_v75.world import State,Atom,Gate,Law,AND,HELDOUT_TABLES,truth,numeric_legal,enumerate_bindings
from twm_v76.world import MultiCase,Proposal,single_cases,validate

SCENARIOS=('compatible','resource_conflict','relation_conflict','mixed','none','ordinary')

def delta(law,u,v):
    # Declared effect serialization, not semantic legality arbitration.
    d={}
    if law.mode in ('consume','transfer'):d[u]=d.get(u,0)-law.amount
    if law.mode in ('produce','transfer'):d[v]=d.get(v,0)+law.amount
    return d

def write_joint(m,selection):
    """Writes ALL neural-selected typed effects. Never checks or repairs legality.
    An opposing relation write is recorded as a conflict rather than resolved:
    serialization may have an arbitrary result, and conflict outputs count wrong.
    """
    s=m.state; stocks=list(s.stocks);facts=set(s.facts);ind={x:i for i,x in enumerate(s.entities)}
    writes={}; conflicting=set(); selected=[]
    cases=single_cases(m)
    for j,k in selection:
        c=cases[j];bound=list(enumerate_bindings(c))
        if k>=len(bound):raise ValueError('Bad binding')
        u,v,w=bound[k];law=c.law
        for node,change in delta(law,u,v).items():stocks[ind[node]]+=change
        fact=(law.effect_predicate,u,v)
        action=bool(law.relation_add)
        if fact in writes and writes[fact]!=action:conflicting.add(fact)
        writes[fact]=action
        if action:facts.add(fact)
        else:facts.discard(fact)
        selected.append((j,k))
    new=State(s.entities,frozenset(facts),tuple(stocks),s.capacities)
    invalid=any(x<0 or x>cap for x,cap in zip(stocks,s.capacities))
    return new,bool(selection),dict(invalid_numeric=invalid,opposed_relations=len(conflicting),edits=len(selection))

def reference_joint(m):
    """TEACHER ONLY: greedy priority with simultaneous aggregate feasibility."""
    validate(m)
    cases=single_cases(m); candidates=[]
    for j,c in enumerate(cases):
        for k,(u,v,w) in enumerate(enumerate_bindings(c)):
            if truth(c.law.expr,c,u,v,w) and numeric_legal(c,u,v):
                candidates.append((m.proposals[j].priority,j,k,u,v))
    candidates.sort(reverse=True)
    stock={e:val for e,val in zip(m.state.entities,m.state.stocks)}
    capacity=dict(zip(m.state.entities,m.state.capacities))
    writes={};chosen=[]; seen_law=set()
    for priority,j,k,u,v in candidates:
        # Research grammar: each rule triggers at most one unique selector binding.
        if j in seen_law:continue
        law=cases[j].law;ds=delta(law,u,v);key=(law.effect_predicate,u,v)
        if key in writes and writes[key]!=law.relation_add:continue
        if any(not (0<=stock[e]+dv<=capacity[e]) for e,dv in ds.items()):continue
        for e,dv in ds.items():stock[e]+=dv
        writes[key]=law.relation_add;chosen.append((j,k));seen_law.add(j)
    return tuple(sorted(chosen))

def reference_transition(m):
    selection=reference_joint(m)
    state,yes,diag=write_joint(m,selection)
    assert not diag['invalid_numeric'] and not diag['opposed_relations']
    return state,yes

def make_case(seed,scenario='ordinary',n=7,laws=4,heldout=False,amount_ood=False,domain='joint'):
    """Balanced synthetic cases. Not used in neural inference."""
    if scenario not in SCENARIOS:raise ValueError(scenario)
    if n<6 or laws<2 or laws>5:raise ValueError('2..5 laws, n>=6')
    r=Random(seed)
    actor=f'{domain}/actor/{seed}'; es=tuple(f'{domain}/obj/{seed}/{i}' for i in range(n))
    caps=[r.randint(9,16) for _ in es];stocks=[r.randint(1,c-1) for c in caps]
    facts=set();choices=[]
    base_u,base_v=es[0],es[1]
    pairs=[(es[0],es[1]),(es[2],es[3]),(es[4],es[5])]
    for j in range(laws):
        if scenario=='compatible':pair=pairs[j%3] if j<3 else (es[0],es[2])
        elif scenario in ('resource_conflict','relation_conflict') and j<2:
            pair=(base_u,base_v if scenario=='relation_conflict' else es[1+j])
        elif scenario=='mixed' and j in (0,1):pair=(base_u,es[j+1])
        elif scenario=='none':pair=(es[(2*j)%n],es[(2*j+1)%n])
        else:pair=r.choice(pairs)
        choices.append(pair)
    # Make numerical boundaries deliberately informative, including compound
    # conflicts where each individual effect is feasible.
    amount=r.randint(1,3) if not amount_ood else r.randint(5,8)
    if scenario=='resource_conflict':
        stocks[0]=min(caps[0],amount+1)
        for j in (1,2):stocks[j]=1;caps[j]=max(caps[j],amount+3)
    if scenario=='relation_conflict':stocks[0]=min(caps[0],amount*2+1)
    if scenario=='compatible':
        for u,v in pairs:stocks[es.index(u)]=caps[es.index(u)]-1;stocks[es.index(v)]=1
    laws_out=[]
    for j,(u,v) in enumerate(choices):
        su=f'{domain}/law/{j}/from';sv=f'{domain}/law/{j}/to';pred=f'{domain}/pre/{j}'
        facts.add((su,actor,u));facts.add((sv,actor,v))
        active=scenario!='none' or (j%2==0 and r.random()<.18)
        if scenario=='ordinary':active=r.random()<.7
        if scenario=='mixed':active=j<3 or r.random()<.4
        if active:facts.add((pred,u,v))
        expr=Gate(AND,Atom(su,'actor','u'),Gate(AND,Atom(sv,'actor','v'),Atom(pred,'u','v')))
        if heldout:
            # True-table variation is provided explicitly to compositional net.
            # In this task changing the binary operator may change feasibility.
            tab=r.choice(HELDOUT_TABLES)
            expr=Gate(AND,Atom(su,'actor','u'),Gate(AND,Atom(sv,'actor','v'),Gate(tuple(tab),Atom(pred,'u','v'),Atom(pred,'u','v'))))
        mode=r.choice(('transfer','consume','produce'))
        if scenario in ('resource_conflict','relation_conflict') and j<2:mode='consume'
        if scenario=='compatible' and j<3:mode='transfer'
        rel_add=bool(r.randrange(2))
        if scenario=='relation_conflict' and j<2:rel_add=bool(j==0)
        effect=f'{domain}/effect' if scenario=='relation_conflict' else f'{domain}/effect/{j%3}'
        law=Law(expr,mode,amount if j<2 else r.randint(1,3),effect,rel_add)
        laws_out.append(Proposal(law,priority=j+1,identity=f'{domain}/rule/{j}'))
    if scenario=='resource_conflict':
        # Both individually feasible, only the higher-priority consumption fits.
        pass
    if scenario=='relation_conflict':
        # Same pair, shared output predicate, opposite toggles.
        pass
    order=list(range(laws));r.shuffle(order)
    props=tuple(laws_out[i] for i in order)
    result=MultiCase(actor,State(es,frozenset(facts),tuple(stocks),tuple(caps)),props,domain)
    validate(result)
    return result
