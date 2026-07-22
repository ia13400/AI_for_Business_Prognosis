"""Forecast model registry (used for the naive/moving-average baselines)."""
from .naive import NaiveForecaster
from .moving_average import MovingAverageForecaster
from .sarima import SarimaForecaster
from .sarimax import SarimaxForecaster
from .patchtst import PatchTSTForecaster
from .xgboost_model import XGBoostForecaster
from .tft import TFTForecaster
from .chronos_zero_shot import ChronosForecaster

MODELS = {
    "naive": NaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "sarima": SarimaForecaster,
    "sarimax": SarimaxForecaster,
    "patchtst": PatchTSTForecaster,
    "xgboost": XGBoostForecaster,
    "tft": TFTForecaster,
    "chronos": ChronosForecaster,
}
