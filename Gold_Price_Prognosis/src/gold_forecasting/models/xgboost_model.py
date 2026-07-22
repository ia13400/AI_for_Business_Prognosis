"""XGBoost with engineered lag/rolling/exogenous features, Optuna HPO scored over rolling-origin validation windows.

Named `xgboost_model.py` (not `xgboost.py`) to avoid shadowing the `xgboost`
package's own module name. Tree-based models are scale-invariant, so no
feature scaling is applied. With `retrain_each_step` true (the default),
every rolling window refits the regressor from scratch on `fit_data`'s
engineered features; with it false, the first window's model is cached and
reused for every later window. Each window's `horizon`-step forecast is
still generated recursively within that window: target lag/rolling features
are rebuilt from the model's own prior predictions inside the window (never
real future values), while exogenous features use that window's actual
realized values -- the same local "known-exogenous" assumption documented in
`sarimax.py`.
"""
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .base import BaseForecaster
from ..feature_engineering import create_features
from ..hpo import run_study
from ..rolling import rolling_forecast
from ..paths import OPTUNA

def _recursive_forecast(model, history_frame, horizon, future_exogenous, feature_config):
    target_col = history_frame.columns[0]; exog_cols = list(history_frame.columns[1:])
    window_size = max([*feature_config["lags"], *feature_config.get("rolling_windows", []), 1]) + 5
    working = history_frame.tail(window_size).copy()
    future_index = pd.bdate_range(history_frame.index.max() + pd.offsets.BDay(), periods=horizon)
    predictions = []
    for step in range(horizon):
        row_exog = future_exogenous[step] if future_exogenous is not None else (working[exog_cols].iloc[-1].values if exog_cols else [])
        row = [np.nan, *row_exog] if exog_cols else [np.nan]
        next_row = pd.DataFrame([row], columns=working.columns, index=[future_index[step]])
        working = pd.concat([working, next_row]).tail(window_size + 1)
        features = create_features(working[target_col], feature_config["lags"], feature_config.get("rolling_windows", []),
                                    feature_config.get("include_calendar", True),
                                    working[exog_cols] if exog_cols else None, feature_config.get("exogenous_lag", 1))
        row_features = features.drop(columns="target").iloc[[-1]]
        prediction = float(model.predict(row_features)[0])
        predictions.append(prediction)
        working.iloc[-1, working.columns.get_loc(target_col)] = prediction
    return np.asarray(predictions, dtype=float)

class XGBoostForecaster(BaseForecaster):
    name = "xgboost"
    def __init__(self, config, feature_config, **_):
        self.config = config; self.feature_config = feature_config
        self.retrain_each_step = bool(config.get("retrain_each_step", True))
        self.params = dict(config.get("params", {})); self.best_params = self.params
        self.seed = int(config.get("seed", 42)); self.model = None
        self.loss_history = {"train": [], "validation": []}
    def _fit(self, fit_data):
        target = fit_data.iloc[:, 0]; exogenous = fit_data.iloc[:, 1:] if fit_data.shape[1] > 1 else None
        features = create_features(target, self.feature_config["lags"], self.feature_config.get("rolling_windows", []),
                                    self.feature_config.get("include_calendar", True), exogenous, self.feature_config.get("exogenous_lag", 1)).dropna()
        X, y = features.drop(columns="target"), features["target"]
        # A trailing slice is held out purely to track a train/validation loss curve (boosting-round RMSE);
        # the model itself only ever sees real past data, same leakage discipline as everywhere else.
        validation_size = max(1, len(X) // 10)
        X_train, y_train = X.iloc[:-validation_size], y.iloc[:-validation_size]
        X_val, y_val = X.iloc[-validation_size:], y.iloc[-validation_size:]
        self.model = XGBRegressor(**self.params, random_state=self.seed, n_jobs=-1, eval_metric="rmse")
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        evals = self.model.evals_result()
        self.loss_history["train"].extend(evals["validation_0"]["rmse"])
        self.loss_history["validation"].extend(evals["validation_1"]["rmse"])
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        if self.retrain_each_step or self.model is None: self._fit(fit_data)
        return _recursive_forecast(self.model, fit_data, horizon, future_exogenous, self.feature_config)

def run_xgboost(train, validation, test, config, hpo, feature_config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    """Multivariate XGBoost: train/validation/test are DataFrames with the target as column 0."""
    from ..experiments import run_rolling_model
    space = config.get("search_space"); trials = int(hpo.get("n_trials", 0))
    retrain_each_step = bool(config.get("retrain_each_step", True))
    study = None
    hpo_started = time.perf_counter()
    if trials and space:
        combined_tv = pd.concat([train, validation])
        validation_start, validation_end = len(train), len(train) + len(validation)
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
                "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
                "subsample": trial.suggest_float("subsample", *space["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            }
            model = XGBoostForecaster(config={"params": params, "retrain_each_step": retrain_each_step, "seed": seed}, feature_config=feature_config)
            result = rolling_forecast(model, combined_tv, validation_start, validation_end, horizon, step)
            return float(np.mean(np.abs(result["actual"] - result["predicted"])))
        study = run_study(f"xgboost_{data_hash[:12]}", objective, trials, OPTUNA, seed=seed, timeout=hpo.get("timeout"))
        params = study.best_params
    else:
        params = dict(config.get("fallback", {}))
    hpo_seconds = time.perf_counter() - hpo_started
    model_config = {"params": params, "retrain_each_step": retrain_each_step, "seed": seed}
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints),
            "approach": "multivariate-known-exogenous-backtest", "exogenous_columns": list(train.columns[1:]), "feature_config": feature_config}
    return run_rolling_model(lambda checkpoint_path: XGBoostForecaster(config=model_config, feature_config=feature_config), "xgboost",
                              train, validation, test, "multivariate", data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=study, extra_seconds=hpo_seconds)
