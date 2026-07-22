"""XGBoost with engineered lag/rolling/exogenous features and Optuna HPO.

Named `xgboost_model.py` (not `xgboost.py`) to avoid shadowing the `xgboost`
package's own module name. Tree-based models are scale-invariant, so no
feature scaling is applied. The multi-step test forecast is generated
recursively: target lag/rolling features are rebuilt from the model's own
prior predictions (never real future target values), while exogenous
features use the test period's actual realized values -- the same
"known-exogenous" backtest assumption documented in `sarimax.py`.
"""
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
from .base import BaseForecaster
from ..feature_engineering import create_features
from ..hpo import run_study
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
        self.config = config; self.feature_config = feature_config; self.model = None; self.best_params = {}
    def _features(self, frame):
        target = frame.iloc[:, 0]; exogenous = frame.iloc[:, 1:] if frame.shape[1] > 1 else None
        built = create_features(target, self.feature_config["lags"], self.feature_config.get("rolling_windows", []),
                                 self.feature_config.get("include_calendar", True), exogenous, self.feature_config.get("exogenous_lag", 1))
        return built.dropna()
    def fit(self, train, validation=None, checkpoint_path=None):
        features = self._features(train); X, y = features.drop(columns="target"), features["target"]
        space = self.config.get("search_space"); hpo = self.config.get("hpo", {}); trials = int(hpo.get("n_trials", 0))
        if trials and space and validation is not None and len(validation):
            validation_target = np.asarray(validation.iloc[:, 0], float)
            validation_exog = validation.iloc[:, 1:].values if validation.shape[1] > 1 else None
            def objective(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
                    "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
                    "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
                    "subsample": trial.suggest_float("subsample", *space["subsample"]),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
                }
                trial_model = XGBRegressor(**params, random_state=int(self.config.get("seed", 42)), n_jobs=-1)
                trial_model.fit(X, y)
                forecast = _recursive_forecast(trial_model, train, len(validation), validation_exog, self.feature_config)
                return float(np.mean(np.abs(validation_target - forecast)))
            study_name = f"xgboost_{self.config.get('study_signature', 'default')}"
            study = run_study(study_name, objective, trials, OPTUNA, seed=int(self.config.get("seed", 42)), timeout=hpo.get("timeout"))
            params = study.best_params
        else:
            params = dict(self.config.get("fallback", {}))
        self.best_params = dict(params)
        combined = pd.concat([train, validation]) if validation is not None and len(validation) else train
        combined_features = self._features(combined)
        Xc, yc = combined_features.drop(columns="target"), combined_features["target"]
        self.model = XGBRegressor(**params, random_state=int(self.config.get("seed", 42)), n_jobs=-1)
        self.model.fit(Xc, yc)
        return self
    def predict(self, history, horizon, future_exogenous=None):
        return _recursive_forecast(self.model, history, horizon, future_exogenous, self.feature_config)

def run_xgboost(train, validation, test, config, hpo, feature_config, data_hash, seed, force_retrain=False, evaluation_horizons=(1, 7, 30)):
    """Multivariate XGBoost: train/validation/test are DataFrames with the target as column 0."""
    from ..experiments import run_model
    model_config = {**config, "hpo": hpo, "study_signature": data_hash[:12], "seed": seed}
    model = XGBoostForecaster(config=model_config, feature_config=feature_config)
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "evaluation_horizons": list(evaluation_horizons),
            "approach": "multivariate-known-exogenous-backtest", "exogenous_columns": list(train.columns[1:]), "feature_config": feature_config}
    return run_model(model, "xgboost", train, validation, test, "multivariate", data_hash, model_config, seed, meta, force_retrain)
