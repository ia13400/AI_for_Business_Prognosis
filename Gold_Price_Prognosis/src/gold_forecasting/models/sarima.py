"""SARIMA with Optuna-driven order search scored over rolling-origin validation windows.

With `retrain_each_step` true (the default), every rolling window is a full
refit from scratch on all data available at that origin -- cheap enough for
statsmodels SARIMAX, and the simplest way to be obviously leakage-safe (no
persisted state between windows other than the frozen `order`). With
`retrain_each_step` false, only the *first* window's coefficients are
estimated; later windows cheaply extend the fitted state through newly
revealed real observations via `.append(refit=False)` (so the forecast
origin still advances correctly) without re-estimating the coefficients.
"""
import numpy as np
import optuna
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseForecaster
from ..hpo import run_study
from ..rolling import rolling_forecast
from ..paths import OPTUNA

def _fit_forecast(endog, order, seasonal_order, horizon):
    result = SARIMAX(np.asarray(endog, float), order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return result, np.asarray(result.forecast(horizon), float)

class SarimaForecaster(BaseForecaster):
    name = "sarima"
    def __init__(self, config, **_):
        self.config = config
        period = int(config.get("seasonal_period", 0))
        self.order = tuple(config["order"]); self.seasonal_order = (0, 0, 0, period) if period else (0, 0, 0, 0)
        self.retrain_each_step = bool(config.get("retrain_each_step", True))
        self.best_params = {"order": list(self.order), "seasonal_order": list(self.seasonal_order)}
        self._result = None; self._fitted_length = 0
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        array = np.asarray(fit_data, dtype=float)
        target = array if array.ndim == 1 else array[:, 0]
        if self.retrain_each_step or self._result is None:
            self._result, forecast = _fit_forecast(target, self.order, self.seasonal_order, horizon)
        else:
            new_observations = target[self._fitted_length:]
            if len(new_observations): self._result = self._result.append(new_observations, refit=False)
            forecast = np.asarray(self._result.forecast(horizon), float)
        self._fitted_length = len(target)
        return forecast

def run_sarima(train, validation, test, config, hpo, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    """Univariate SARIMA: Optuna order search scored via rolling-origin validation, evaluated via rolling-origin test."""
    from ..experiments import run_rolling_model
    space = config.get("search_space"); trials = int(hpo.get("n_trials", 0))
    retrain_each_step = bool(config.get("retrain_each_step", True))
    seasonal_period = int(config.get("seasonal_period", 0))
    study = None
    if trials and space:
        combined_tv = pd.concat([train, validation])
        validation_start, validation_end = len(train), len(train) + len(validation)
        def objective(trial):
            order = (trial.suggest_int("p", *space["p"]), trial.suggest_int("d", *space["d"]), trial.suggest_int("q", *space["q"]))
            model = SarimaForecaster(config={"order": order, "seasonal_period": seasonal_period, "retrain_each_step": retrain_each_step})
            try:
                result = rolling_forecast(model, combined_tv, validation_start, validation_end, horizon, step)
            except Exception:
                raise optuna.TrialPruned()
            return float(np.mean(np.abs(result["actual"] - result["predicted"])))
        study = run_study(f"sarima_{data_hash[:12]}", objective, trials, OPTUNA, seed=seed, timeout=hpo.get("timeout"))
        order = (study.best_params["p"], study.best_params["d"], study.best_params["q"])
    else:
        order = tuple(config.get("fallback", {}).get("order", [2, 1, 2]))
    model_config = {"order": list(order), "seasonal_period": seasonal_period, "retrain_each_step": retrain_each_step}
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints), "approach": "univariate"}
    return run_rolling_model(lambda checkpoint_path: SarimaForecaster(config=model_config), "sarima", train, validation, test, "univariate",
                              data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=study)
