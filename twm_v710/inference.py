"""Neural inference. No teacher, validity checks or post-hoc correction."""
from twm_v77.world import write_joint

def select(model,world):
    return model.selection(world)

def predict(model,world):
    return write_joint(world,select(model,world))[:2]

def diagnose(model,world):
    return write_joint(world,select(model,world))
