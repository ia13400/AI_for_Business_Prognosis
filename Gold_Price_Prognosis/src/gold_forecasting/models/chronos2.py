"""Zero-shot Chronos-2: pretrained, no training, no HPO -- unlike every other Chronos
variant in this project (`chronos_zero_shot.py`), Chronos-2 natively accepts past/future
covariates, so it is the one Chronos model that runs in the *multivariate* experiment,
competing zero-shot against SARIMAX/XGBoost/TFT on the same exogenous inputs. Each rolling
window's context (target and covariates alike) is the tail of `fit_data` truncated to
`context_length` real observations, same convention as the other Chronos wrappers.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the native hf_xet DLL is blocked in some locked-down Windows environments; plain HTTP download works fine
import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
from .base import BaseForecaster

class Chronos2Forecaster(BaseForecaster):
    name = "chronos2"
    def __init__(self, config, device=None, **_):
        self.model_id = config["model_id"]; self.context_length = int(config.get("context_length", 512))
        self.device = str(device or "cpu"); self.pipeline = None
        self.best_params = {"model_id": self.model_id}
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        if self.pipeline is None:
            self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device)
        frame = fit_data.to_frame() if isinstance(fit_data, pd.Series) else fit_data
        tail = frame.tail(self.context_length)
        target_col, exog_cols = tail.columns[0], list(tail.columns[1:])
        item = {"target": tail[target_col].to_numpy(dtype=np.float32)}
        if exog_cols:
            item["past_covariates"] = {c: tail[c].to_numpy(dtype=np.float32) for c in exog_cols}
            if future_exogenous is not None:
                future_array = np.asarray(future_exogenous, dtype=np.float32)
                item["future_covariates"] = {c: future_array[:, i] for i, c in enumerate(exog_cols)}
        _, mean = self.pipeline.predict_quantiles([item], prediction_length=horizon, quantile_levels=[0.5])
        return np.asarray(mean[0], dtype=float).reshape(-1)

def run_chronos2(train, validation, test, config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    """Multivariate, covariate-aware zero-shot Chronos-2: train/validation/test are DataFrames with the target as column 0."""
    from ..experiments import run_rolling_model
    from ..config import select_device
    model_config = {"model_id": config["model_id"], "context_length": config.get("context_length", 512)}
    device = select_device()
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints),
            "approach": "multivariate-zero-shot-covariates", "exogenous_columns": list(train.columns[1:])}
    return run_rolling_model(lambda checkpoint_path: Chronos2Forecaster(config=model_config, device=device), "chronos2",
                              train, validation, test, "multivariate", data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=None)
