"""Differentiable neural truth-table attention with weight sharing over syntax trees.

No Boolean operator semantics are implemented with Python branches inside the
model. The *supplied* truth-table values are the law's definition. A neural
attention selector learns which value to read from that table.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .world import Atom,Gate

class CompositionalRuleModel(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        self.width=width
        self.literal=nn.Sequential(nn.Linear(2,width),nn.Tanh(),nn.Linear(width,1))
        self.gate_attention=nn.Sequential(nn.Linear(4,width),nn.Tanh(),nn.Linear(width,4))
        self.legal=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.decision=nn.Sequential(nn.Linear(2,width),nn.Tanh(),nn.Linear(width,1))

    def forward_components(self,raw,numbers,expr):
        # Raw graph lookups and syntax traversal are not logical execution.
        lit=torch.sigmoid(self.literal(raw))
        cursor=[0]
        def visit(node):
            if isinstance(node,Atom):
                k=cursor[0];cursor[0]+=1;return lit[:,k]
            a=visit(node.left);b=visit(node.right)
            inp=torch.cat((a,b,a*b,(1-a)*(1-b)),dim=-1)
            weights=F.softmax(self.gate_attention(inp),-1)
            table=torch.as_tensor(node.table,dtype=raw.dtype,device=raw.device)
            return (weights*table).sum(-1,keepdim=True)
        logic=visit(expr)
        legal=torch.sigmoid(self.legal(numbers))
        accepted=torch.sigmoid(self.decision(torch.cat((logic,legal),-1)))
        return accepted,logic,legal

    def forward(self,raw,numbers,expr):
        return self.forward_components(raw,numbers,expr)[0]
