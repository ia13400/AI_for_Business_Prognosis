"""XGBoost variant that predicts returns instead of raw price levels.

`xgboost_model.py`'s XGBoost predicts the absolute price level directly.
Investigating that model's unusually large multivariate test error showed why:
gold's price trended from ~$1,050-2,434 (train) to ~$3,293-5,318 (test) over
the dataset's span, and gradient-boosted trees output piecewise-constant leaf
averages learned from the training distribution -- they cannot linearly
extrapolate a trend the way SARIMAX's AR coefficients or a scaled neural net
can. Even once `retrain_each_step` has folded high test-period prices into
`fit_data`, those rows stay a small minority of the training set, so the
leaves covering that price range remain anchored near the much more populous
historical (lower-price) regime, and predictions systematically undershoot.

This model keeps the exact same lag/rolling/exogenous *features* as
`xgboost_model.py` (built via the same `feature_engineering.create_features`),
but predicts the next-step percentage return -- a stationary target whose
scale doesn't depend on the absolute price level -- and reconstructs each
step's absolute forecast by compounding the predicted return onto the last
known (real or previously forecasted) price. Both variants are kept and
compared side by side (not one replacing the other), since which
representation forecasts better is itself part of what's being evaluated.
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

def _recursive_forecast_diff(model, history_frame, horizon, future_exogenous, feature_config):
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
        predicted_return = float(model.predict(row_features)[0])
        last_price = working[target_col].iloc[-2]  # last known/forecasted price, before this step's (still-NaN) row
        price = last_price * (1.0 + predicted_return)
        predictions.append(price)
        working.iloc[-1, working.columns.get_loc(target_col)] = price
    return np.asarray(predictions, dtype=float)

class XGBoostDiffForecaster(BaseForecaster):
    name = "xgboost_diff"
    def __init__(self, config, feature_config, **_):
        self.config = config; self.feature_config = feature_config
        self.retrain_each_step = bool(config.get("retrain_each_step", True))
        self.params = dict(config.get("params", {})); self.best_params = self.params
        self.seed = int(config.get("seed", 42)); self.model = None
        self.loss_history = {"train": [], "validation": []}
    def _fit(self, fit_data):
        target = fit_data.iloc[:, 0]; exogenous = fit_data.iloc[:, 1:] if fit_data.shape[1] > 1 else None
        features = create_features(target, self.feature_config["lags"], self.feature_config.get("rolling_windows", []),
                                    self.feature_config.get("include_calendar", True), exogenous, self.feature_config.get("exogenous_lag", 1))
        # This-row return (t-1 -> t): the stationary target being predicted, not one of the (already-lagged) input features.
        returns = target.pct_change(fill_method=None)
        combined = features.drop(columns="target").assign(target_return=returns).dropna()
        X, y = combined.drop(columns="target_return"), combined["target_return"]
        # A trailing slice is held out purely to track a train/validation loss curve (boosting-round RMSE, on
        # the return scale); the model itself only ever sees real past data, same leakage discipline as everywhere else.
        validation_size = max(1, len(X) // 10)
        X_train, y_train = X.iloc[:-validation_size], y.iloc[:-validation_size]
        X_val, y_val = X.iloc[-validation_size:], y.iloc[-validation_size:]
        self.model = XGBRegressor(**self.params, random_state=self.seed, n_jobs=-1, eval_metric="rmse")
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        evals = self.model.evals_result()
        self.loss_history["train"].extend(evals["validation_0"]["rmse"])
        self.loss_history["validation"].extend(evals["validation_1"]["rmse"])
        self.feature_importances_ = dict(zip(X.columns, (float(v) for v in self.model.feature_importances_)))
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        if self.retrain_each_step or self.model is None: self._fit(fit_data)
        return _recursive_forecast_diff(self.model, fit_data, horizon, future_exogenous, self.feature_config)

def run_xgboost_diff(train, validation, test, config, hpo, feature_config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    """Multivariate XGBoost, return-based target: same call shape as `xgboost_model.run_xgboost`."""
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
            model = XGBoostDiffForecaster(config={"params": params, "retrain_each_step": retrain_each_step, "seed": seed}, feature_config=feature_config)
            result = rolling_forecast(model, combined_tv, validation_start, validation_end, horizon, step)
            return float(np.mean(np.abs(result["actual"] - result["predicted"])))
        study = run_study(f"xgboost_diff_{data_hash[:12]}", objective, trials, OPTUNA, seed=seed, timeout=hpo.get("timeout"))
        params = study.best_params
    else:
        params = dict(config.get("fallback", {}))
    hpo_seconds = time.perf_counter() - hpo_started
    model_config = {"params": params, "retrain_each_step": retrain_each_step, "seed": seed}
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints),
            "approach": "multivariate-known-exogenous-backtest-return-target", "exogenous_columns": list(train.columns[1:]), "feature_config": feature_config}
    return run_rolling_model(lambda checkpoint_path: XGBoostDiffForecaster(config=model_config, feature_config=feature_config), "xgboost_diff",
                              train, validation, test, "multivariate", data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=study, extra_seconds=hpo_seconds)
