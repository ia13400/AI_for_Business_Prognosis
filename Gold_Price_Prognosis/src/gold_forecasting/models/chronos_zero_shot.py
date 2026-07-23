"""Zero-shot Chronos / Chronos-Bolt: pretrained, no training, no HPO.

Chronos internally normalizes its own input context, so no external scaler
is needed or applied here. Each rolling window's context is the tail of
`fit_data` truncated to `context_length` real observations -- both for
efficiency and because Chronos models have a fixed max supported context.
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
        self.model_id = config["model_id"]; self.context_length = int(config.get("context_length", 512))
        self.device = str(device or "cpu"); self.pipeline = None
        self.best_params = {"model_id": self.model_id}
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        if self.pipeline is None:
            self.pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device, torch_dtype=torch.float32)
        array = np.asarray(fit_data, dtype=np.float32)
        target = array if array.ndim == 1 else array[:, 0]
        context = torch.as_tensor(target[-self.context_length:])
        forecast = self.pipeline.predict(context, prediction_length=horizon)
        result = np.asarray(forecast[0].detach().cpu() if torch.is_tensor(forecast[0]) else forecast[0], dtype=float)
        return result.mean(axis=0) if result.ndim > 1 else result

def run_chronos(train, validation, test, config, variant, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    from ..experiments import run_rolling_model
    from ..config import select_device
    model_id = config["variants"][variant]
    model_config = {"model_id": model_id, "context_length": config.get("context_length", 512)}
    device = select_device()
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints), "approach": "univariate-zero-shot"}
    return run_rolling_model(lambda checkpoint_path: ChronosForecaster(config=model_config, device=device), f"chronos_{variant}",
                              train, validation, test, "univariate", data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=None)

def run_chronos_original(train, validation, test, config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    return run_chronos(train, validation, test, config, "original", data_hash, seed, horizon, step, force_retrain, lead_time_checkpoints)

def run_chronos_bolt(train, validation, test, config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    return run_chronos(train, validation, test, config, "bolt", data_hash, seed, horizon, step, force_retrain, lead_time_checkpoints)

def run_chronos_t5_base(train, validation, test, config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    return run_chronos(train, validation, test, config, "t5_base", data_hash, seed, horizon, step, force_retrain, lead_time_checkpoints)

def run_chronos_t5_large(train, validation, test, config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    return run_chronos(train, validation, test, config, "t5_large", data_hash, seed, horizon, step, force_retrain, lead_time_checkpoints)

def run_chronos_bolt_base(train, validation, test, config, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    return run_chronos(train, validation, test, config, "bolt_base", data_hash, seed, horizon, step, force_retrain, lead_time_checkpoints)
