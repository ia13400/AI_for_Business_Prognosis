"""Trailing moving-average baseline."""
import numpy as np
from .base import BaseForecaster

class MovingAverageForecaster(BaseForecaster):
    name = "moving_average"
    def __init__(self, config=None, **_): self.window = int((config or {}).get("window", 20))
    def fit(self, train, validation=None, checkpoint_path=None): self.level = float(np.mean(np.asarray(train, dtype=float)[-self.window:])); return self
    def predict(self, history, horizon, future_exogenous=None): return np.repeat(self.level, horizon)
