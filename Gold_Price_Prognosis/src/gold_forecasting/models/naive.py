"""Persistence baseline."""
import numpy as np
from .base import BaseForecaster

class NaiveForecaster(BaseForecaster):
    name = "naive"
    def __init__(self, **_): pass
    def fit(self, train, validation=None, checkpoint_path=None): self.last_value = float(np.asarray(train)[-1]); return self
    def predict(self, history, horizon, future_exogenous=None): return np.repeat(float(np.asarray(history)[-1]), horizon)
