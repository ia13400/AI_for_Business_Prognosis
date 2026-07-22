"""Trailing moving-average baseline."""
import numpy as np
from .base import BaseForecaster

class MovingAverageForecaster(BaseForecaster):
    name = "moving_average"
    def __init__(self, config=None, **_): self.window = int((config or {}).get("window", 20))
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        array = np.asarray(fit_data, dtype=float)
        target = array if array.ndim == 1 else array[:, 0]
        return np.repeat(float(np.mean(target[-self.window:])), horizon)
