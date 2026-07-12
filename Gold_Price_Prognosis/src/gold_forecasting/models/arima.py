"""ARIMA classical baseline."""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from .base import BaseForecaster

class ArimaForecaster(BaseForecaster):
    name = "arima"
    def __init__(self, config, **_): self.order = tuple(config.get("order", [5,1,0])); self.result = None
    def fit(self, train, validation=None, checkpoint_path=None): self.result = ARIMA(np.asarray(train,float), order=self.order).fit(); return self
    def predict(self, history, horizon): return np.asarray(self.result.forecast(horizon), float)
