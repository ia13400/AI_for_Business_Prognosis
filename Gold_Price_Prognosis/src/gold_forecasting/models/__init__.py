"""Forecast model registry."""
from .naive import NaiveForecaster
from .arima import ArimaForecaster
from .lstm import LSTMForecaster
from .nbeats import NBeatsForecaster
from .patchtst import PatchTSTForecaster

MODELS = {"naive": NaiveForecaster, "arima": ArimaForecaster, "lstm": LSTMForecaster, "nbeats": NBeatsForecaster, "patchtst": PatchTSTForecaster}
