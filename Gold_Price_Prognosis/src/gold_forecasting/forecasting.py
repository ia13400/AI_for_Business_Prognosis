"""Model construction for the naive/moving-average baselines."""
from .models import MODELS

def make_model(name, config, device, seed): return MODELS[name](config=config,device=device,seed=seed)
