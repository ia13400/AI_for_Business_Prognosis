"""Zero-shot Chronos / Chronos-Bolt: pretrained, no training, no HPO.

Chronos internally normalizes its own input context, so no external scaler
is needed or applied here.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the native hf_xet DLL is blocked in some locked-down Windows environments; plain HTTP download works fine
import numpy as np
import torch
from chronos import BaseChronosPipeline
from .base import BaseForecaster

class ChronosForecaster(BaseForecaster):
    name = "chronos"
    def __init__(self, config, device=None, **_):
        self.model_id = config["model_id"]; self.device = str(device or "cpu"); self.pipeline = None
        self.best_params = {"model_id": self.model_id}
    def fit(self, train, validation=None, checkpoint_path=None):
        self.pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device, torch_dtype=torch.float32)
        return self
    def predict(self, history, horizon, future_exogenous=None):
        context = torch.as_tensor(np.asarray(history, dtype=np.float32))
        forecast = self.pipeline.predict(context, prediction_length=horizon)
        array = np.asarray(forecast[0].detach().cpu() if torch.is_tensor(forecast[0]) else forecast[0], dtype=float)
        return array.mean(axis=0) if array.ndim > 1 else array

def run_chronos(train, validation, test, config, variant, data_hash, seed, force_retrain=False, evaluation_horizons=(1, 7, 30)):
    from ..experiments import run_model
    from ..config import select_device
    model_id = config["variants"][variant]
    model_config = {"model_id": model_id}
    model = ChronosForecaster(config=model_config, device=select_device())
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "evaluation_horizons": list(evaluation_horizons), "approach": "univariate-zero-shot"}
    return run_model(model, f"chronos_{variant}", train, validation, test, "univariate", data_hash, model_config, seed, meta, force_retrain)

def run_chronos_original(train, validation, test, config, data_hash, seed, force_retrain=False, evaluation_horizons=(1, 7, 30)):
    return run_chronos(train, validation, test, config, "original", data_hash, seed, force_retrain, evaluation_horizons)

def run_chronos_bolt(train, validation, test, config, data_hash, seed, force_retrain=False, evaluation_horizons=(1, 7, 30)):
    return run_chronos(train, validation, test, config, "bolt", data_hash, seed, force_retrain, evaluation_horizons)
