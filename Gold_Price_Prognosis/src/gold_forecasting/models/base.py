"""Common forecasting interface and neural training engine."""
from abc import ABC, abstractmethod
from pathlib import Path
import copy
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ..datasets import windows

# set_seed() opts into torch.use_deterministic_algorithms(True, warn_only=True),
# so this warning fires on virtually every non-deterministic GPU op (CuBLAS
# attention/matmul kernels) -- with many epochs x trials x rolling windows,
# that's thousands of near-identical lines flooding notebook output for a
# trade-off we already deliberately accepted.
warnings.filterwarnings("ignore", message=".*Deterministic behavior was enabled.*")

class BaseForecaster(ABC):
    """A model owns its complete fit/forecast lifecycle behind one call.

    `forecast_window` is invoked once per rolling-origin window (see
    `rolling.rolling_forecast`), always with `fit_data` = all real data
    strictly before that window. Each model decides internally -- based on
    its own state and its own `retrain_each_step` config flag -- whether this
    call performs a fresh fit, an incremental update, or reuses an existing
    fit; the caller never needs to know which.
    """
    name = "base"
    @abstractmethod
    def forecast_window(self, fit_data, horizon: int, future_exogenous: np.ndarray | None = None) -> np.ndarray: ...

class NeuralForecaster(BaseForecaster):
    """Windowed autoregressive PyTorch forecaster with rolling-origin warm-start.

    `fit_data`/`future_exogenous` accept either a 1-D univariate array/frame
    or an (n, n_features) array whose column 0 is the forecast target and
    whose remaining columns are exogenous regressors. On the first
    `forecast_window` call, trains from scratch for the full `epochs`/
    `patience` budget (early stopping against an internal trailing slice of
    `fit_data`), with epoch-level checkpoint/resume so an interrupted first
    fit doesn't restart from scratch. On every later call, if
    `retrain_each_step` is set, continues training the *existing* weights for
    `update_epochs` more epochs on the (larger) current `fit_data` -- a warm
    start, not a fresh fit -- with no per-window checkpointing (these updates
    are short; resuming mid-rolling-sequence is out of scope, see the plan).
    If `retrain_each_step` is false, the model is frozen after the first call.
    """
    def __init__(self, config: dict, device: torch.device, seed: int = 42, n_features: int = 1, checkpoint_path: Path | None = None):
        self.config, self.device, self.seed, self.n_features = config, device, seed, n_features
        self.context_length = int(config["context_length"])
        self.retrain_each_step = bool(config.get("retrain_each_step", True))
        self.update_epochs = int(config.get("update_epochs", max(1, int(config["epochs"]) // 10)))
        self.checkpoint_path = checkpoint_path
        self.scaler = StandardScaler()
        self.loss_history = {"train": [], "validation": []}
        self.model = self.build_model().to(device)
        self._initialized = False
    @abstractmethod
    def build_model(self) -> nn.Module: ...
    def _forward(self, x): return self.model(x)
    def _as_matrix(self, values):
        array = np.asarray(values, dtype=np.float32)
        return array[:, None] if array.ndim == 1 else array
    def _load_checkpoint(self):
        if not self.checkpoint_path or not self.checkpoint_path.exists(): return None
        saved = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        return None if saved.get("finished") else saved
    def _save_checkpoint(self, epoch, best_loss, best_state, remaining, finished):
        if not self.checkpoint_path: return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "scaler_mean": self.scaler.mean_, "scaler_scale": self.scaler.scale_,
                    "epoch": epoch, "best_loss": best_loss, "best_state": best_state, "remaining": remaining, "finished": finished}, self.checkpoint_path)
    def _initial_fit(self, values):
        resume = self._load_checkpoint()
        epochs, patience = int(self.config["epochs"]), int(self.config["patience"])
        if resume is None:
            self.scaler.fit(values); start_epoch, best_loss, remaining, best_state = 0, float("inf"), patience, None
        else:
            self.scaler.mean_, self.scaler.scale_ = resume["scaler_mean"], resume["scaler_scale"]
            self.scaler.var_, self.scaler.n_features_in_ = self.scaler.scale_ ** 2, len(self.scaler.mean_)
            self.model.load_state_dict(resume["state_dict"])
            start_epoch, best_loss, remaining, best_state = resume["epoch"] + 1, resume["best_loss"], resume["remaining"], resume["best_state"]
        best_state = self._train_loop(values, start_epoch, epochs, patience, best_loss, remaining, best_state, checkpoint=True)
        if best_state is not None: self.model.load_state_dict(best_state)
        self._save_checkpoint(epochs - 1, best_loss, best_state, 0, finished=True)
    def _update_fit(self, values):
        self._train_loop(values, 0, self.update_epochs, self.update_epochs, float("inf"), self.update_epochs, None, checkpoint=False)
    def _train_loop(self, values, start_epoch, epochs, patience, best_loss, remaining, best_state, checkpoint):
        scaled = self.scaler.transform(values); dataset = windows(scaled, self.context_length)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=int(self.config["batch_size"]), shuffle=True, generator=generator)
        fallback_validation = values[-max(self.context_length + 1, len(values)//10):]
        combined = np.concatenate([values[-self.context_length:], fallback_validation], axis=0)
        val_data = windows(self.scaler.transform(combined), self.context_length)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["learning_rate"]))
        criterion = nn.MSELoss()
        for epoch in range(start_epoch, epochs):
            self.model.train(); losses=[]
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True); prediction=self._forward(x.to(self.device)); loss=criterion(prediction, y.to(self.device)); loss.backward(); optimizer.step(); losses.append(loss.item())
            self.model.eval()
            with torch.no_grad(): val_loss=criterion(self._forward(val_data.tensors[0].to(self.device)), val_data.tensors[1].to(self.device)).item()
            self.loss_history["train"].append(float(np.mean(losses))); self.loss_history["validation"].append(val_loss)
            if val_loss < best_loss - 1e-8: best_loss, remaining, best_state = val_loss, patience, copy.deepcopy(self.model.state_dict())
            else: remaining -= 1
            if checkpoint: self._save_checkpoint(epoch, best_loss, best_state, remaining, finished=False)
            if remaining <= 0: break
        return best_state
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        values = self._as_matrix(fit_data)
        if not self._initialized:
            self._initial_fit(values); self._initialized = True
        elif self.retrain_each_step:
            self._update_fit(values)
        return self._recursive_forecast(values, horizon, future_exogenous)
    def _recursive_forecast(self, values, horizon, future_exogenous):
        context = list(self.scaler.transform(values)[-self.context_length:]); result=[]; self.model.eval()
        future = self._as_matrix(future_exogenous) if future_exogenous is not None else None
        with torch.no_grad():
            for step in range(horizon):
                window = np.stack(context[-self.context_length:])
                x=torch.tensor(window, dtype=torch.float32, device=self.device)[None]
                value=float(self._forward(x).cpu().item()); result.append(value)
                if self.n_features > 1:
                    exog_row = future[step] if future is not None else values[-1, 1:]
                    padded = np.concatenate([[0.0], exog_row]).astype(np.float32)
                    scaled_exog = self.scaler.transform(padded[None])[0, 1:]
                    context.append(np.concatenate([[value], scaled_exog]).astype(np.float32))
                else:
                    context.append(np.array([value], dtype=np.float32))
        result_values = np.asarray(result, dtype=np.float32).reshape(-1, 1)
        if self.n_features > 1:
            padded = np.concatenate([result_values, np.zeros((horizon, self.n_features - 1), dtype=np.float32)], axis=1)
            return self.scaler.inverse_transform(padded)[:, 0]
        return self.scaler.inverse_transform(result_values).ravel()
