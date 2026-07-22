"""SARIMA with Optuna-driven order search (train/validation only, never test)."""
import numpy as np
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseForecaster
from ..hpo import run_study
from ..paths import OPTUNA

def _fit(endog, order, seasonal_order):
    return SARIMAX(np.asarray(endog, float), order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

class SarimaForecaster(BaseForecaster):
    name = "sarima"
    def __init__(self, config, **_):
        self.config = config
        period = int(config.get("seasonal_period", 0))
        self.seasonal_order = (0, 0, 0, period) if period else (0, 0, 0, 0)
        self.order = None; self.result = None; self.best_params = {}
    def fit(self, train, validation=None, checkpoint_path=None):
        space = self.config.get("search_space"); hpo = self.config.get("hpo", {})
        trials = int(hpo.get("n_trials", 0))
        if trials and space and validation is not None and len(validation):
            def objective(trial):
                order = (trial.suggest_int("p", *space["p"]), trial.suggest_int("d", *space["d"]), trial.suggest_int("q", *space["q"]))
                try:
                    forecast = _fit(train, order, self.seasonal_order).forecast(len(validation))
                except Exception:
                    raise optuna.TrialPruned()
                return float(np.mean(np.abs(np.asarray(validation, float) - np.asarray(forecast, float))))
            study_name = f"sarima_{self.config.get('study_signature', 'default')}"
            study = run_study(study_name, objective, trials, OPTUNA, seed=int(self.config.get("seed", 42)), timeout=hpo.get("timeout"))
            best = study.best_params; self.order = (best["p"], best["d"], best["q"])
        else:
            self.order = tuple(self.config.get("fallback", {}).get("order", [2, 1, 2]))
        self.best_params = {"order": list(self.order), "seasonal_order": list(self.seasonal_order)}
        fitted = _fit(train, self.order, self.seasonal_order)
        if validation is not None and len(validation):
            fitted = fitted.append(np.asarray(validation, float), refit=False)
        self.result = fitted
        return self
    def predict(self, history, horizon, future_exogenous=None):
        return np.asarray(self.result.forecast(horizon), float)

def run_sarima(train, validation, test, config, hpo, data_hash, seed, force_retrain=False, evaluation_horizons=(1, 7, 30)):
    """Univariate SARIMA: Optuna order search on train/validation, evaluated on test."""
    from ..experiments import run_model
    model_config = {**config, "hpo": hpo, "study_signature": data_hash[:12], "seed": seed}
    model = SarimaForecaster(config=model_config)
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "evaluation_horizons": list(evaluation_horizons), "approach": "univariate"}
    return run_model(model, "sarima", train, validation, test, "univariate", data_hash, model_config, seed, meta, force_retrain)
