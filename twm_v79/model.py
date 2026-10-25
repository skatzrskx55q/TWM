"""Neural resource-context energy for choosing a set of simultaneous effects.

The candidate encoder constructs *raw declared additive changes*, not a
legality decision: it never evaluates conditions, rejects a subset, clips stocks,
or resolves relation writes. The learned critic ranks these raw projections;
all state transitions are still determined by the neural subset decision.

This experiment is limited to an 8-proposal exhaustive selection grammar.
Typed declared-effect arithmetic remains infrastructure, not learned physics.
"""
import torch
from torch import nn
from torch.nn import functional as F
from twm_v75.world import enumerate_bindings
from twm_v76.world import single_cases
from twm_v78.model import binary_masks


def aggregate_context(world, locations):
    """Describe ALL subsets with *unfiltered* stock projections.

    No truth, priority, resource-bound or conflict test occurs here. Output is
    neural input, not a programmatic selector or a post-prediction correction.
    """
    n=len(locations);s=world.state
    m=binary_masks(n)
    entity_index={x:i for i,x in enumerate(s.entities)}
    rows=torch.zeros(n,len(s.entities),dtype=torch.float32)
    for i,(j,k) in enumerate(locations):
        case=single_cases(world)[j]
        u,v,w=list(enumerate_bindings(case))[k]
        effect=case.law
        if effect.mode in ('consume','transfer'):
            rows[i,entity_index[u]]-=float(effect.amount)
        if effect.mode in ('produce','transfer'):
            rows[i,entity_index[v]]+=float(effect.amount)
    # These quantities are *candidate properties*; nothing is tested against
    # a bound and no transition is accepted/rejected on the Python side.
    cap=torch.tensor(s.capacities,dtype=torch.float32)
    stock=torch.tensor(s.stocks,dtype=torch.float32)
    debits=m@rows.clamp(max=0.)
    credits=m@rows.clamp(min=0.)
    projected=stock[None,:]+m@rows
    stock_frac=(stock/cap).expand(m.shape[0],-1)
    available=((cap-stock)/cap).expand(m.shape[0],-1)
    proposal_count=(m.sum(1,keepdim=True)/max(1,n)).expand(-1,len(cap))
    context=torch.stack((stock_frac,available,debits/cap,credits/cap,
        projected/cap,cap.expand(m.shape[0],-1)/16.,proposal_count),dim=-1)
    return context


class AggregateResourceEnergy(nn.Module):
    def __init__(self,parent,width=64,use_aggregate=True):
        super().__init__()
        self.parent=parent
        for param in parent.parameters():param.requires_grad_(False)
        parent.eval()
        self.width=width
        self.use_aggregate=bool(use_aggregate)
        self.entity_net=nn.Sequential(nn.Linear(7,width),nn.GELU(),
            nn.Linear(width,width),nn.GELU())
        self.energy_net=nn.Sequential(nn.Linear(width*3,width),nn.GELU(),
            nn.Linear(width,1))
        # Residual starts nearly at zero; base subset energy remains intact.
        nn.init.normal_(self.energy_net[-1].weight,std=.012)
        nn.init.zeros_(self.energy_net[-1].bias)
        self.strength=nn.Parameter(torch.tensor(1.0))

    def train(self,mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    @torch.no_grad()
    def features(self,world):
        x,p,prior,loc=self.parent.features(world)
        context=aggregate_context(world,loc)
        return x,p,prior,context,loc

    def energies(self,x,p,prior,context):
        with torch.no_grad():base=self.parent.energies(x,p,prior)
        if not self.use_aggregate:
            # Same trainable critic architecture, but no object-specific
            # stocks or proposed aggregate deltas (ablation).
            context=torch.stack((context[...,6],)*7,dim=-1)
        h=self.entity_net(context)
        pooled=torch.cat((h.mean(dim=1),h.amax(dim=1),h.amin(dim=1)),dim=-1)
        extra=self.energy_net(pooled).flatten()
        return base+self.strength*extra

    def forward(self,world):
        x,p,prior,context,locations=self.features(world)
        return self.energies(x,p,prior,context),locations

    def selection(self,world):
        with torch.no_grad():
            scores,locations=self.forward(world)
            selection=binary_masks(len(locations),device=scores.device)[scores.argmax()]
            return tuple(locations[i] for i in range(len(locations)) if selection[i]>.5)
