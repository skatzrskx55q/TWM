"""Weights-only chain loader. Inference does NOT import teacher or training."""
import torch
from twm_v78.model import JointSetEnergy
from twm_v78.parent import load_parent as load_v77
from twm_v79.model import AggregateResourceEnergy

def load_parent(path):
    cfg=torch.load(path,map_location='cpu',weights_only=True)
    v78=torch.load(cfg['parent_path'],map_location='cpu',weights_only=True)
    parent=JointSetEnergy(load_v77(v78['parent_path']),v78['width'],v78['interactions'])
    parent.load_state_dict(v78['model']);parent.eval()
    m=AggregateResourceEnergy(parent,cfg['width'],cfg['use_aggregate'])
    m.load_state_dict(cfg['model']);m.eval()
    return m
