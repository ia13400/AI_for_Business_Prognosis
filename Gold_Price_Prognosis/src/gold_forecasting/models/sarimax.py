"""SARIMAX with exogenous regressors and Optuna-driven order search.

Exogenous regressors are scaled with a StandardScaler fit on train only
(coefficient conditioning); the target is left unscaled so forecasts come
out directly in USD. Forecasting the test horizon requires exogenous values
for those future dates -- this uses the test period's *actual* realized
exogenous series (a standard "known-exogenous" backtest assumption for
exogenous regression models; a genuine future deployment forecast would
need its own forecasts of the exogenous series, which is out of scope here).
"""
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseForecaster
from ..hpo import run_study
from ..paths import OPTUNA

def _split_target_exog(frame):
    array = np.asarray(frame, dtype=float)
    return (array, None) if array.ndim == 1 else (array[:, 0], array[:, 1:])

def _fit(endog, exog, order, seasonal_order):
    return SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

class SarimaxForecaster(BaseForecaster):
    name = "sarimax"
    def __init__(self, config, **_):
        self.config = config
        period = int(config.get("seasonal_period", 0))
        self.seasonal_order = (0, 0, 0, period) if period else (0, 0, 0, 0)
        self.order = None; self.result = None; self.exog_scaler = StandardScaler(); self.best_params = {}
    def fit(self, train, validation=None, checkpoint_path=None):
        train_target, train_exog = _split_target_exog(train)
        self.exog_scaler.fit(train_exog); train_exog_scaled = self.exog_scaler.transform(train_exog)
        space = self.config.get("search_space"); hpo = self.config.get("hpo", {}); trials = int(hpo.get("n_trials", 0))
        validation_target, validation_exog = _split_target_exog(validation) if validation is not None and len(validation) else (None, None)
        if trials and space and validation_target is not None:
            validation_exog_scaled = self.exog_scaler.transform(validation_exog)
            def objective(trial):
                order = (trial.suggest_int("p", *space["p"]), trial.suggest_int("d", *space["d"]), trial.suggest_int("q", *space["q"]))
                try:
                    forecast = _fit(train_target, train_exog_scaled, order, self.seasonal_order).forecast(len(validation_target), exog=validation_exog_scaled)
                except Exception:
                    raise optuna.TrialPruned()
                return float(np.mean(np.abs(validation_target - np.asarray(forecast, float))))
            study_name = f"sarimax_{self.config.get('study_signature', 'default')}"
            study = run_study(study_name, objective, trials, OPTUNA, seed=int(self.config.get("seed", 42)), timeout=hpo.get("timeout"))
            best = study.best_params; self.order = (best["p"], best["d"], best["q"])
        else:
            self.order = tuple(self.config.get("fallback", {}).get("order", [2, 1, 2]))
        self.best_params = {"order": list(self.order), "seasonal_order": list(self.seasonal_order)}
        fitted = _fit(train_target, train_exog_scaled, self.order, self.seasonal_order)
        if validation_target is not None:
            fitted = fitted.append(validation_target, exog=self.exog_scaler.transform(validation_exog), refit=False)
        self.result = fitted
        return self
    def predict(self, history, horizon, future_exogenous=None):
        exog = self.exog_scaler.transform(np.asarray(future_exogenous, dtype=float)) if future_exogenous is not None else None
        return np.asarray(self.result.forecast(horizon, exog=exog), float)

def run_sarimax(train, validation, test, config, hpo, data_hash, seed, force_retrain=False, evaluation_horizons=(1, 7, 30)):
    """Multivariate SARIMAX: train/validation/test are DataFrames with the target as column 0."""
    from ..experiments import run_model
    model_config = {**config, "hpo": hpo, "study_signature": data_hash[:12], "seed": seed}
    model = SarimaxForecaster(config=model_config)
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "evaluation_horizons": list(evaluation_horizons),
            "approach": "multivariate-known-exogenous-backtest", "exogenous_columns": list(train.columns[1:])}
    return run_model(model, "sarimax", train, validation, test, "multivariate", data_hash, model_config, seed, meta, force_retrain)
