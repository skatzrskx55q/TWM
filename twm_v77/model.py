"""Neural SET prediction over simultaneously proposed typed effects.

Frozen pretrained v7.5 estimates logical/numeric applicability. Candidate
binding retrieval chooses the highest neural score PER law (not a reference
lookup). Shared learned message-passing head decides an independently selected
subset, taking priorities, resource endpoints and shared output addresses as
INPUT FEATURES, not executing compatibility checks in Python.
"""
import torch
from torch import nn
from twm_v75.world import encode
from twm_v75.model import CompositionalRuleModel
from twm_v76.world import single_cases

class JointEffectsModel(nn.Module):
    def __init__(self,base=None,width=48,rounds=2):
        super().__init__()
        self.base=base or CompositionalRuleModel()
        for p in self.base.parameters():p.requires_grad_(False)
        self.base.eval();self.width=width;self.rounds=rounds
        self.embed=nn.Sequential(nn.Linear(14,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.pair=nn.Sequential(nn.Linear(width*2+5,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.update=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,width))
        self.output=nn.Sequential(nn.Linear(width+14,width),nn.GELU(),nn.Linear(width,1))

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    @torch.no_grad()
    def features(self,m):
        es=m.state.entities; cases=single_cases(m)
        rows=[];locations=[];metadata=[]
        for j,c in enumerate(cases):
            raw,num,bindings=encode(c)
            yes,logic,legal=self.base.forward_components(raw,num,c.law.expr)
            kk=int(yes.view(-1).argmax().item())
            u,v,w=bindings[kk]; law=c.law
            presence=float((law.effect_predicate,u,v) in m.state.facts)
            row=torch.cat((yes[kk],logic[kk],legal[kk],num[kk],
                           torch.tensor([m.proposals[j].priority/10.,float(law.relation_add),presence])))
            assert row.numel()==14
            rows.append(row);locations.append((j,kk))
            metadata.append((u,v,law.effect_predicate))
        pair=torch.zeros((len(rows),len(rows),5),dtype=torch.float32)
        for i,(u,v,p) in enumerate(metadata):
            for j,(uu,vv,pp) in enumerate(metadata):
                if i==j:continue
                pair[i,j]=torch.tensor((u==uu,v==vv,u==vv,v==uu,(p==pp and u==uu and v==vv)),dtype=torch.float32)
        return torch.stack(rows),pair,locations

    def logits(self,features,pairs):
        n=features.size(0);hidden=self.embed(features)
        for _ in range(self.rounds):
            a=hidden.unsqueeze(1).expand(n,n,-1)
            b=hidden.unsqueeze(0).expand(n,n,-1)
            info=self.pair(torch.cat((a,b,pairs),-1))
            mask=(1-torch.eye(n,device=features.device)).unsqueeze(-1)
            msg=(info*mask).sum(1)/max(1,n-1)
            hidden=hidden+self.update(torch.cat((hidden,msg),-1))*.5
        return self.output(torch.cat((hidden,features),-1)).flatten()

    def forward(self,m):
        x,p,loc=self.features(m)
        return self.logits(x,p),loc

class PairwiseJointEffectsModel(JointEffectsModel):
    """Joint head with LEARNED candidate-to-candidate inhibitory messages.

    Structural address overlaps are input. Whether overlapping proposals
    conflict, and how to arbitrate priorities, is supervised via neural weights.
    """
    def __init__(self,base=None,width=48,rounds=2):
        super().__init__(base,width,rounds)
        self.inhibition=nn.Sequential(nn.Linear(14*2+5,width),nn.GELU(),
                                      nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        nn.init.constant_(self.inhibition[-1].bias,-3.5)
        self.inhibition_scale=nn.Parameter(torch.tensor(4.0))

    def logits_and_pairs(self,features,pairs):
        main=super().logits(features,pairs)
        n=features.size(0)
        a=features.unsqueeze(1).expand(n,n,-1)
        b=features.unsqueeze(0).expand(n,n,-1)
        raw=self.inhibition(torch.cat((a,b,pairs),-1)).squeeze(-1)
        suppression=raw.sigmoid() * main.detach().sigmoid().unsqueeze(0)
        mask=1-torch.eye(n,device=features.device)
        suppression=suppression*mask
        logits=main-torch.clamp(self.inhibition_scale,min=0,max=12)*suppression.sum(1)
        return logits,raw

    def logits(self,features,pairs):
        return self.logits_and_pairs(features,pairs)[0]
