"""Inference-only checkpoint loading; no teacher imports."""
import torch
from .model import JointConsistencyEnergy
from .parent import load_parent

def load(path):
    cfg=torch.load(path,map_location='cpu',weights_only=True)
    model=JointConsistencyEnergy(load_parent(cfg['parent_path']),cfg['width'],cfg['relation_features'])
    model.load_state_dict(cfg['model']);model.eval();return model
