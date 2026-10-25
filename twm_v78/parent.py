"""Weights-only parent loader independent of any teacher/training module.

Inference imports the neural architecture and saved numeric tensors. It neither
imports nor invokes reference_joint or reference_transition functions.
"""
import torch
from twm_v75.model import CompositionalRuleModel
from twm_v77.model import JointEffectsModel,PairwiseJointEffectsModel

def load_parent(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    base_data=torch.load(data['base_checkpoint'],map_location='cpu',weights_only=True)
    base=CompositionalRuleModel(base_data['width'])
    base.load_state_dict(base_data['model'])
    cls=PairwiseJointEffectsModel if data['architecture']=='pairwise' else JointEffectsModel
    model=cls(base,data['width'],data['rounds'])
    model.load_state_dict(data['model']);model.eval()
    return model
