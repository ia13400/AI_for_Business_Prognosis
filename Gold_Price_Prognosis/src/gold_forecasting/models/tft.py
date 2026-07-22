"""Compact native PyTorch TFT-inspired forecaster for multivariate input.

Variable-selection gating -> LSTM encoder -> single multi-head self-attention
block -> gated residual output head. A lightweight approximation of the
Temporal Fusion Transformer, consistent with this project's native PyTorch
implementations (no pytorch-forecasting/lightning dependency). Optuna HPO is
scored over rolling-origin validation windows; the neural warm-start/
retrain-each-step machinery lives in `NeuralForecaster` (`base.py`) and
applies identically here. The recursive per-window forecast uses that
window's actual exogenous values, the same "known-exogenous" backtest
assumption documented in `sarimax.py`.
"""
import numpy as np
import pandas as pd
from torch import nn
from .base import NeuralForecaster
from ..hpo import run_study
from ..rolling import rolling_forecast
from ..paths import OPTUNA

class _VariableSelection(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__(); self.gate = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_features), nn.Softmax(dim=-1))
    def forward(self, x): return x * self.gate(x)

class _TFTLite(nn.Module):
    def __init__(self, n_features, hidden, heads, layers, dropout):
        super().__init__()
        self.selection = _VariableSelection(n_features, hidden)
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden); self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        encoded, _ = self.encoder(self.selection(x))
        attended, _ = self.attention(encoded, encoded, encoded)
        return self.head(self.norm(encoded + attended)[:, -1])

class TFTForecaster(NeuralForecaster):
    name = "tft"
    def build_model(self):
        return _TFTLite(self.n_features, int(self.config["hidden_size"]), int(self.config["attention_heads"]), int(self.config["lstm_layers"]), float(self.config["dropout"]))

def _round_hidden_size(hidden_size, heads):
    """hidden_size must be divisible by attention_heads; round down within range, never below heads itself."""
    return max(heads, (hidden_size // heads) * heads)

def _sample_config(trial, context_length, space):
    heads = trial.suggest_categorical("attention_heads", [1, 2, 4])
    raw_hidden_size = trial.suggest_int("hidden_size", *space["hidden_size"])
    return {
        "context_length": context_length,
        "hidden_size": _round_hidden_size(raw_hidden_size, heads),
        "attention_heads": heads,
        "lstm_layers": trial.suggest_int("lstm_layers", *space["lstm_layers"]),
        "dropout": trial.suggest_float("dropout", *space["dropout"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "batch_size": trial.suggest_int("batch_size", *space["batch_size"], step=32),
    }

def run_tft(train, validation, test, config, hpo, data_hash, seed, horizon, step, force_retrain=False, lead_time_checkpoints=(1, 10, 20)):
    """Multivariate TFT-lite: train/validation/test are DataFrames with the target as column 0."""
    from ..experiments import run_rolling_model
    from ..config import select_device
    device = select_device(); context_length = int(config["context_length"]); space = config.get("search_space")
    n_features = train.shape[1]; trials = int(hpo.get("n_trials", 0))
    retrain_each_step = bool(config.get("retrain_each_step", True))
    update_epochs = int(hpo.get("update_epochs", max(1, int(hpo["epochs"]) // 10)))
    study = None
    if trials and space:
        combined_tv = pd.concat([train, validation])
        validation_start, validation_end = len(train), len(train) + len(validation)
        def objective(trial):
            trial_config = {**_sample_config(trial, context_length, space), "epochs": hpo["epochs"], "patience": hpo["patience"],
                             "retrain_each_step": retrain_each_step, "update_epochs": update_epochs}
            model = TFTForecaster(config=trial_config, device=device, seed=seed, n_features=n_features)
            result = rolling_forecast(model, combined_tv, validation_start, validation_end, horizon, step)
            return float(np.mean(np.abs(result["actual"] - result["predicted"])))
        study = run_study(f"tft_{data_hash[:12]}", objective, trials, OPTUNA, seed=seed, timeout=hpo.get("timeout"))
        best = dict(study.best_params); best["hidden_size"] = _round_hidden_size(best["hidden_size"], best["attention_heads"])
        model_config = {"context_length": context_length, **best, "epochs": hpo["epochs"], "patience": hpo["patience"],
                         "retrain_each_step": retrain_each_step, "update_epochs": update_epochs}
    else:
        model_config = {"context_length": context_length, **config.get("fallback", {}), "epochs": hpo["epochs"], "patience": hpo["patience"],
                         "retrain_each_step": retrain_each_step, "update_epochs": update_epochs}
    def build(checkpoint_path):
        model = TFTForecaster(config=model_config, device=device, seed=seed, n_features=n_features, checkpoint_path=checkpoint_path)
        model.best_params = model_config
        return model
    meta = {"train_range": f"{train.index.min()}:{train.index.max()}", "validation_range": f"{validation.index.min()}:{validation.index.max()}",
            "test_range": f"{test.index.min()}:{test.index.max()}", "lead_time_checkpoints": list(lead_time_checkpoints),
            "approach": "multivariate-known-exogenous-backtest", "exogenous_columns": list(train.columns[1:])}
    return run_rolling_model(build, "tft", train, validation, test, "multivariate", data_hash, model_config, seed, meta, horizon, step, force_retrain, hpo_study=study)
