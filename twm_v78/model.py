"""Learned joint-set energy for multi-law state transitions.

This neural model ranks EVERY subset of up to 8 retrieved proposals (v7.7
experiment uses 4 during training and 5 for OOD). Subset enumeration and
argmax are search infrastructure: there is NO conflict-checking, priority sort,
resource bound test or relational arbitration in the selector. All scores are
learned functions of the declarative candidate features and overlaps.

The frozen v7.7 proposer supplies neural candidate bindings and raw descriptors.
The model predicts a SET, not an independent threshold for every candidate.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .parent import load_parent


def binary_masks(n,device=None):
    if not 1<=n<=8: raise ValueError('The exhaustive research decoder supports 1..8 candidates')
    s=torch.arange(1<<n,device=device)
    return ((s[:,None]>>torch.arange(n,device=device))&1).float()


class JointSetEnergy(nn.Module):
    def __init__(self,parent=None,width=64,interactions=True):
        super().__init__()
        self.parent=parent if parent is not None else load_parent('checkpoints_v77/checkpoint_773.pt')
        for p in self.parent.parameters():p.requires_grad_(False)
        self.parent.eval()
        self.width=width;self.interactions=bool(interactions)
        self.emb=nn.Sequential(nn.Linear(15,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.unary=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.pair=nn.Sequential(nn.Linear(width*2+5,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.set_net=nn.Sequential(nn.Linear(width*2+3,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.parent_weight=nn.Parameter(torch.tensor(0.5))
        # Initial small adjustments retain the pretrained parent's broad competence.
        nn.init.normal_(self.unary[-1].weight,std=0.01);nn.init.zeros_(self.unary[-1].bias)
        nn.init.normal_(self.pair[-1].weight,std=0.01);nn.init.zeros_(self.pair[-1].bias)
        nn.init.normal_(self.set_net[-1].weight,std=0.01);nn.init.zeros_(self.set_net[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.parent.eval();return self

    @torch.no_grad()
    def features(self,world):
        x,p,loc=self.parent.features(world)
        prior=self.parent.logits(x,p).detach()
        return x,p,prior,loc

    def energies(self,x,p,prior):
        n=x.shape[0]
        masks=binary_masks(n,x.device)
        emb=self.emb(torch.cat((x,prior[:,None]),-1))
        unary=self.unary(emb).flatten()+self.parent_weight*prior
        score=masks@unary
        if self.interactions:
            a=emb.unsqueeze(1).expand(n,n,-1)
            b=emb.unsqueeze(0).expand(n,n,-1)
            inter=self.pair(torch.cat((a,b,p),-1)).squeeze(-1)
            inter=(inter+inter.T)*0.5
            inter=inter*(1-torch.eye(n,device=x.device))
            selected=masks.unsqueeze(2)*masks.unsqueeze(1)
            score=score+(selected*inter).sum((1,2))*0.5
        # Learned whole-set energy, conditioned on both selected and omitted rules.
        totals=masks@emb
        all_emb=emb.sum(0,keepdim=True)
        omitted=all_emb-totals
        count=masks.sum(1,keepdim=True)
        meta=torch.cat((count/n,count/8,torch.ones_like(count)),dim=1)
        pooled=torch.cat((totals/n,omitted/n,meta),-1)
        score=score+self.set_net(pooled).flatten()
        return score

    def forward(self,world):
        x,p,prior,loc=self.features(world)
        return self.energies(x,p,prior),loc

    def selection(self,world):
        with torch.no_grad():
            scores,loc=self(world)
            mask=binary_masks(len(loc),scores.device)[int(scores.argmax().item())]
            return tuple(loc[i] for i in range(len(loc)) if mask[i]>.5)
