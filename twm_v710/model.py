"""Joint neural energy for *raw* resource and relation write projections.

Features are descriptions of candidate effects, not a programmatic policy:
there is no legality check, no priority arbitration, no relation-conflict
elimination and no state repair. Exact relation keys are grouped to preserve
entity identity, but the learned scorer decides which subset to execute.
The fixed research grammar enumerates up to eight candidate effects.
"""
import torch
from torch import nn
from twm_v76.world import single_cases
from twm_v75.world import enumerate_bindings
from twm_v78.model import binary_masks
from twm_v79.model import aggregate_context


def relation_context(world,locations):
    """A [2^n, distinct facts, 12] tensor of raw writes in each subset.

    No teacher, truth-condition, resource bound or contradiction test is run.
    Multiplication by binary subset masks is only declarative aggregation.
    """
    n=len(locations)
    masks=binary_masks(n)
    cases=single_cases(world)
    keys=[]; props=[]
    for j,k in locations:
        c=cases[j];u,v,_=list(enumerate_bindings(c))[k]
        key=(c.law.effect_predicate,u,v)
        if key not in keys:keys.append(key)
        props.append((keys.index(key),float(c.law.relation_add),float(world.proposals[j].priority)/6.))
    num_keys=len(keys)
    raw=torch.zeros(n,num_keys,6)
    for i,(g,add,prio) in enumerate(props):
        raw[i,g,0]=add;raw[i,g,1]=1-add
        raw[i,g,2]=prio*add;raw[i,g,3]=prio*(1-add)
        raw[i,g,4]=float(prio);raw[i,g,5]=1.
    # Number of declared writes for selected and omitted candidate laws.
    count=masks@raw.reshape(n,-1)
    count=count.reshape(masks.shape[0],num_keys,6)
    omitted=raw.sum(dim=0)[None,:,:]-count
    # Max priority values computed as raw features, not used for decisions.
    priority_add=torch.zeros(masks.shape[0],num_keys)
    priority_del=torch.zeros_like(priority_add)
    for i in range(n):
        g,a,p=props[i]
        if a:priority_add[:,g]=torch.maximum(priority_add[:,g],masks[:,i]*p)
        else:priority_del[:,g]=torch.maximum(priority_del[:,g],masks[:,i]*p)
    present=torch.tensor([float(key in world.state.facts) for key in keys])[None,:].expand(len(masks),-1)
    # Note: simultaneous add/delete is NOT tested or blocked; the network may
    # learn from the independent selected counts and max priorities.
    features=torch.stack((count[:,:,0],count[:,:,1],count[:,:,2],count[:,:,3],
        priority_add,priority_del,omitted[:,:,0],omitted[:,:,1],present,
        count[:,:,4],omitted[:,:,4],count[:,:,5]),dim=-1)
    return features


class JointConsistencyEnergy(nn.Module):
    def __init__(self,parent,width=64,relation_features=True):
        super().__init__()
        self.parent=parent
        for p in parent.parameters():p.requires_grad_(False)
        parent.eval()
        self.width=width
        self.relation_features=bool(relation_features)
        self.resource_net=nn.Sequential(nn.Linear(7,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.relation_net=nn.Sequential(nn.Linear(12,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.cross_net=nn.Sequential(nn.Linear(width*6,width),nn.GELU(),
            nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        nn.init.normal_(self.cross_net[-1].weight,std=.006)
        nn.init.zeros_(self.cross_net[-1].bias)
        self.strength=nn.Parameter(torch.tensor(1.))

    def train(self,mode=True):
        super().train(mode);self.parent.eval();return self

    @torch.no_grad()
    def features(self,world):
        x,p,prior,resources,loc=self.parent.features(world)
        relation=relation_context(world,loc)
        return x,p,prior,resources,relation,loc

    def energies(self,x,p,prior,resources,relations):
        with torch.no_grad():
            base=self.parent.energies(x,p,prior,resources)
        nr=self.resource_net(resources)
        if self.relation_features:
            rel=self.relation_net(relations)
        else:
            # Controlled ablation: no information about actual relation keys,
            # directions or priorities, only number of proposal groups.
            rel=self.relation_net(torch.zeros_like(relations))
        pooled=torch.cat((nr.mean(1),nr.amax(1),nr.amin(1),
                          rel.mean(1),rel.amax(1),rel.amin(1)),dim=-1)
        return base+self.strength*self.cross_net(pooled).flatten()

    def forward(self,world):
        x,p,prior,res,rel,loc=self.features(world)
        return self.energies(x,p,prior,res,rel),loc

    def selection(self,world):
        with torch.no_grad():
            scores,loc=self(world)
            mask=binary_masks(len(loc),scores.device)[int(scores.argmax())]
            return tuple(loc[i] for i in range(len(loc)) if mask[i]>.5)
