"""Common forecasting interface and neural training engine."""
from abc import ABC, abstractmethod
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ..datasets import windows

class BaseForecaster(ABC):
    name = "base"
    @abstractmethod
    def fit(self, train, validation=None, checkpoint_path: Path | None = None): ...
    @abstractmethod
    def predict(self, history, horizon: int, future_exogenous: np.ndarray | None = None) -> np.ndarray: ...

class NeuralForecaster(BaseForecaster):
    """Windowed autoregressive PyTorch forecaster.

    `train`/`validation`/`history` accept either a 1-D univariate array or a
    (n, n_features) array whose column 0 is the forecast target and whose
    remaining columns are exogenous regressors. Training resumes from the
    last completed epoch if an unfinished checkpoint is found, so an
    interrupted run does not restart from scratch.
    """
    def __init__(self, config: dict, device: torch.device, seed: int = 42, n_features: int = 1):
        self.config, self.device, self.seed, self.n_features = config, device, seed, n_features
        self.context_length = int(config["context_length"]); self.scaler = StandardScaler(); self.loss_history = {"train": [], "validation": []}
        self.model = self.build_model().to(device)
    @abstractmethod
    def build_model(self) -> nn.Module: ...
    def _forward(self, x): return self.model(x)
    def _as_matrix(self, values):
        array = np.asarray(values, dtype=np.float32)
        return array[:, None] if array.ndim == 1 else array
    def _load_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not checkpoint_path.exists(): return None
        saved = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        return None if saved.get("finished") else saved
    def _save_checkpoint(self, checkpoint_path, epoch, best_loss, best_state, remaining, finished):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "optimizer_state": self._optimizer.state_dict(),
                    "scaler_mean": self.scaler.mean_, "scaler_scale": self.scaler.scale_, "config": self.config,
                    "loss_history": self.loss_history, "epoch": epoch, "best_loss": best_loss,
                    "best_state": best_state, "remaining": remaining, "finished": finished}, checkpoint_path)
    def fit(self, train, validation=None, checkpoint_path=None):
        values = self._as_matrix(train)
        resume = self._load_checkpoint(checkpoint_path)
        if resume is None:
            self.scaler.fit(values); start_epoch, best_loss, remaining, best_state = 0, float("inf"), int(self.config["patience"]), None
        else:
            self.scaler.mean_, self.scaler.scale_ = resume["scaler_mean"], resume["scaler_scale"]
            self.scaler.var_, self.scaler.n_features_in_ = self.scaler.scale_ ** 2, len(self.scaler.mean_)
            self.model.load_state_dict(resume["state_dict"]); self.loss_history = resume["loss_history"]
            start_epoch, best_loss, remaining, best_state = resume["epoch"] + 1, resume["best_loss"], resume["remaining"], resume["best_state"]
        scaled = self.scaler.transform(values); dataset = windows(scaled, self.context_length)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=int(self.config["batch_size"]), shuffle=True, generator=generator)
        fallback_validation = train[-max(self.context_length + 1, len(train)//10):]
        validation_values = self._as_matrix(validation if validation is not None and len(validation) else fallback_validation)
        combined = np.concatenate([values[-self.context_length:], validation_values], axis=0)
        val_data = windows(self.scaler.transform(combined), self.context_length)
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["learning_rate"]))
        if resume is not None: self._optimizer.load_state_dict(resume["optimizer_state"])
        criterion = nn.MSELoss()
        for epoch in range(start_epoch, int(self.config["epochs"])):
            self.model.train(); losses=[]
            for x, y in loader:
                self._optimizer.zero_grad(set_to_none=True); prediction=self._forward(x.to(self.device)); loss=criterion(prediction, y.to(self.device)); loss.backward(); self._optimizer.step(); losses.append(loss.item())
            self.model.eval()
            with torch.no_grad(): val_loss=criterion(self._forward(val_data.tensors[0].to(self.device)), val_data.tensors[1].to(self.device)).item()
            self.loss_history["train"].append(float(np.mean(losses))); self.loss_history["validation"].append(val_loss)
            if val_loss < best_loss - 1e-8: best_loss, remaining, best_state = val_loss, int(self.config["patience"]), copy.deepcopy(self.model.state_dict())
            else: remaining -= 1
            if checkpoint_path: self._save_checkpoint(checkpoint_path, epoch, best_loss, best_state, remaining, finished=False)
            if remaining <= 0: break
        if best_state is not None: self.model.load_state_dict(best_state)
        if checkpoint_path: self._save_checkpoint(checkpoint_path, int(self.config["epochs"]) - 1, best_loss, best_state, remaining, finished=True)
        return self
    def predict(self, history, horizon, future_exogenous=None):
        array = self._as_matrix(history)
        context = list(self.scaler.transform(array)[-self.context_length:]); result=[]; self.model.eval()
        future = self._as_matrix(future_exogenous) if future_exogenous is not None else None
        with torch.no_grad():
            for step in range(horizon):
                window = np.stack(context[-self.context_length:])
                x=torch.tensor(window, dtype=torch.float32, device=self.device)[None]
                value=float(self._forward(x).cpu().item()); result.append(value)
                if self.n_features > 1:
                    exog_row = future[step] if future is not None else array[-1, 1:]
                    padded = np.concatenate([[0.0], exog_row]).astype(np.float32)
                    scaled_exog = self.scaler.transform(padded[None])[0, 1:]
                    context.append(np.concatenate([[value], scaled_exog]).astype(np.float32))
                else:
                    context.append(np.array([value], dtype=np.float32))
        values = np.asarray(result, dtype=np.float32).reshape(-1, 1)
        if self.n_features > 1:
            padded = np.concatenate([values, np.zeros((horizon, self.n_features - 1), dtype=np.float32)], axis=1)
            return self.scaler.inverse_transform(padded)[:, 0]
        return self.scaler.inverse_transform(values).ravel()
