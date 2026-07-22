"""SARIMAX with exogenous regressors, Optuna-driven order search scored over rolling-origin validation windows.

Exogenous regressors are scaled with a StandardScaler (coefficient
conditioning; the target is left unscaled). With `retrain_each_step` true
(the default), every rolling window refits from scratch -- including the
exogenous scaler -- on all data available at that origin. With
`retrain_each_step` false, the scaler and coefficients are frozen after the
first window; later windows cheaply extend the fitted state through newly
revealed real observations via `.append(refit=False)`.

Forecasting each window needs the *actual* realized exogenous values for
that window's dates -- a local, 20-day version of the "known-exogenous
backtest" assumption standard for exogenous regression models (much more
defensible than assuming exogenous values are known far into the future).
"""
import time
import warnings
import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseForecaster

# HPO's order search deliberately tries combinations that won't converge well
# (Optuna just scores them poorly); with hundreds of fits per rolling-origin
# evaluation, printing this every time floods notebook output for no benefit.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from ..hpo import run_study
from ..rolling import rolling_forecast
from ..paths import OPTUNA

def _fit(endog, exog, order, seasonal_order):
    return SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

class SarimaxForecaster(BaseForecaster):
    name = "sarimax"
    def __init__(self, config, **_):
        self.config = config
        period = int(config.get("seasonal_period", 0))
        self.order = tuple(config["order"]); self.seasonal_order = (0, 0, 0, period) if period else (0, 0, 0, 0)
        self.retrain_each_step = bool(config.get("retrain_each_step", True))
        self.best_params = {"order": list(self.order), "seasonal_order": list(self.seasonal_order)}
        self.exog_scaler = StandardScaler()
        self._result = None; self._fitted_length = 0
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        array = np.asarray(fit_data, dtype=float)
        target, exog = array[:, 0], array[:, 1:]
        if self.retrain_each_step or self._result is None:
            self.exog_scaler.fit(exog)
            self._result = _fit(target, self.exog_scaler.transform(exog), self.order, self.seasonal_order)
        else:
            new_target = target[self._fitted_length:]
            if len(new_target):
                new_exog = self.exog_scaler.transform(exog[self._fitted_length:])
                self._result = self._result.append(new_target, exog=new_exog, refit=False)
        self._fitted_length = len(target)
        future_scaled = self.exog_scaler.transform(np.asarray(future_exogenous, dtype=float))
        return np.asarray(self._result.forecast(horizon, exog=future_scaled), float)

def run_sarimax(train, validation, test, config, hpo, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    """Multivariate SARIMAX: train/validation/test are DataFrames with the target as column 0."""
    from ..experiments import run_rolling_model
    space = config.get("search_space"); trials = int(hpo.get("n_trials", 0))
    retrain_each_step = bool(config.get("retrain_each_step", True))
    seasonal_period = int(config.get("seasonal_period", 0))
    study = None
    hpo_started = time.perf_counter()
    if trials and space:
        combined_tv = pd.concat([train, validation])
        validation_start, validation_end = len(train), len(train) + len(validation)
        def objective(trial):
            order = (trial.suggest_int("p", *space["p"]), trial.suggest_int("d", *space["d"]), trial.suggest_int("q", *space["q"]))
            model = SarimaxForecaster(config={"order": order, "seasonal_period": seasonal_period, "retrain_each_step": retrain_each_step})
            try:
                result = rolling_forecast(model, combined_tv, validation_start, validation_end, horizon, step)
            except Exception:
                raise optuna.TrialPruned()
            return float(np.mean(np.abs(result["actual"] - result["predicted"])))
        study = run_study(f"sarimax_{data_hash[:12]}", objective, trials, OPTUNA, seed=seed, timeout=hpo.get("timeout"))
        order = (study.best_params["p"], study.best_params["d"], study.best_params["q"])
    else:
        order = tuple(config.get("fallback", {}).get("order", [2, 1, 2]))
    hpo_seconds = time.perf_counter() - hpo_started
    model_config = {"order": list(order), "seasonal_period": seasonal_period, "retrain_each_step": retrain_each_step}
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints),
            "approach": "multivariate-known-exogenous-backtest", "exogenous_columns": list(train.columns[1:])}
    return run_rolling_model(lambda checkpoint_path: SarimaxForecaster(config=model_config), "sarimax", train, validation, test, "multivariate",
                              data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=study, extra_seconds=hpo_seconds)
