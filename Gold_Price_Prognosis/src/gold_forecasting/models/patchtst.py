"""Native PyTorch PatchTST encoder for univariate forecasting.

Optuna HPO is scored over rolling-origin validation windows (see
`rolling.rolling_forecast`); the neural warm-start/retrain-each-step
machinery lives in `NeuralForecaster` (`base.py`) and applies identically
here without any extra code in this file.
"""
import numpy as np
import pandas as pd
from torch import nn
from .base import NeuralForecaster
from ..hpo import run_study
from ..rolling import rolling_forecast
from ..paths import OPTUNA

class _PatchTST(nn.Module):
    def __init__(self, context, patch, stride, d_model, nhead, layers):
        super().__init__(); self.patch,self.stride=patch,stride; self.embedding=nn.Linear(patch,d_model)
        encoder=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=d_model*2,batch_first=True,dropout=0.1)
        self.encoder=nn.TransformerEncoder(encoder,layers); self.head=nn.Linear(d_model,1)
    def forward(self,x):
        patches=x.squeeze(-1).unfold(1,self.patch,self.stride); encoded=self.encoder(self.embedding(patches)); return self.head(encoded.mean(1))

class PatchTSTForecaster(NeuralForecaster):
    name="patchtst"
    def build_model(self): return _PatchTST(self.context_length,int(self.config["patch_length"]),int(self.config["stride"]),int(self.config["d_model"]),int(self.config["nhead"]),int(self.config["layers"]))

def _round_d_model(d_model, nhead):
    """d_model must be divisible by nhead; round down within range, never below nhead itself."""
    return max(nhead, (d_model // nhead) * nhead)

def _sample_config(trial, context_length, space):
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    raw_d_model = trial.suggest_int("d_model", *space["d_model"])
    return {
        "context_length": context_length,
        "patch_length": trial.suggest_int("patch_length", *space["patch_length"]),
        "stride": trial.suggest_int("stride", *space["stride"]),
        "d_model": _round_d_model(raw_d_model, nhead),
        "nhead": nhead,
        "layers": trial.suggest_int("layers", *space["layers"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "batch_size": trial.suggest_int("batch_size", *space["batch_size"], step=32),
    }

def run_patchtst(train, validation, test, config, hpo, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    """Univariate PatchTST: Optuna search over architecture/optimization hyperparameters, scored via rolling-origin validation."""
    from ..experiments import run_rolling_model
    from ..config import select_device
    device = select_device(); context_length = int(config["context_length"]); space = config.get("search_space")
    trials = int(hpo.get("n_trials", 0))
    retrain_each_step = bool(config.get("retrain_each_step", True))
    update_epochs = int(hpo.get("update_epochs", max(1, int(hpo["epochs"]) // 10)))
    study = None
    if trials and space:
        combined_tv = pd.concat([train, validation])
        validation_start, validation_end = len(train), len(train) + len(validation)
        def objective(trial):
            trial_config = {**_sample_config(trial, context_length, space), "epochs": hpo["epochs"], "patience": hpo["patience"],
                             "retrain_each_step": retrain_each_step, "update_epochs": update_epochs}
            model = PatchTSTForecaster(config=trial_config, device=device, seed=seed)
            result = rolling_forecast(model, combined_tv, validation_start, validation_end, horizon, step)
            return float(np.mean(np.abs(result["actual"] - result["predicted"])))
        study = run_study(f"patchtst_{data_hash[:12]}", objective, trials, OPTUNA, seed=seed, timeout=hpo.get("timeout"))
        best = dict(study.best_params); best["d_model"] = _round_d_model(best["d_model"], best["nhead"])
        model_config = {"context_length": context_length, **best, "epochs": hpo["epochs"], "patience": hpo["patience"],
                         "retrain_each_step": retrain_each_step, "update_epochs": update_epochs}
    else:
        model_config = {"context_length": context_length, **config.get("fallback", {}), "epochs": hpo["epochs"], "patience": hpo["patience"],
                         "retrain_each_step": retrain_each_step, "update_epochs": update_epochs}
    def build(checkpoint_path):
        model = PatchTSTForecaster(config=model_config, device=device, seed=seed, checkpoint_path=checkpoint_path)
        model.best_params = model_config
        return model
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints), "approach": "univariate"}
    return run_rolling_model(build, "patchtst", train, validation, test, "univariate", data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=study)
