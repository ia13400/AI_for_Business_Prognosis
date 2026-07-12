"""Configuration loading and deterministic seeding."""
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from .paths import CONFIGS

def load_yaml(name: str | Path) -> dict:
    path = Path(name)
    if not path.is_absolute(): path = CONFIGS / path
    with path.open(encoding="utf-8") as handle: return yaml.safe_load(handle) or {}

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def select_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
