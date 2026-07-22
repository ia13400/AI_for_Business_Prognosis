"""Persistence baseline."""
import numpy as np
from .base import BaseForecaster

class NaiveForecaster(BaseForecaster):
    name = "naive"
    def __init__(self, **_): pass
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        array = np.asarray(fit_data)
        last = array[-1] if array.ndim == 1 else array[-1, 0]
        return np.repeat(float(last), horizon)
