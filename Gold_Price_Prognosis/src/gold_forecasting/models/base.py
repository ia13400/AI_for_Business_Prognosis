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
    def predict(self, history, horizon: int) -> np.ndarray: ...

class NeuralForecaster(BaseForecaster):
    def __init__(self, config: dict, device: torch.device, seed: int = 42):
        self.config, self.device, self.seed = config, device, seed
        self.context_length = int(config["context_length"]); self.scaler = StandardScaler(); self.loss_history = {"train": [], "validation": []}
        self.model = self.build_model().to(device)
    @abstractmethod
    def build_model(self) -> nn.Module: ...
    def _forward(self, x): return self.model(x)
    def fit(self, train, validation=None, checkpoint_path=None):
        values = np.asarray(train, dtype=np.float32).reshape(-1, 1); self.scaler.fit(values)
        scaled = self.scaler.transform(values).ravel(); dataset = windows(scaled, self.context_length)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=int(self.config["batch_size"]), shuffle=True, generator=generator)
        validation_values = np.asarray(validation if validation is not None and len(validation) else train[-max(self.context_length + 1, len(train)//10):], dtype=np.float32).reshape(-1, 1)
        combined = np.concatenate([values[-self.context_length:], validation_values]); val_data = windows(self.scaler.transform(combined).ravel(), self.context_length)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["learning_rate"])); criterion = nn.MSELoss()
        best_loss, remaining, best_state = float("inf"), int(self.config["patience"]), None
        for _ in range(int(self.config["epochs"])):
            self.model.train(); losses=[]
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True); prediction=self._forward(x.to(self.device)); loss=criterion(prediction, y.to(self.device)); loss.backward(); optimizer.step(); losses.append(loss.item())
            self.model.eval()
            with torch.no_grad(): val_loss=criterion(self._forward(val_data.tensors[0].to(self.device)), val_data.tensors[1].to(self.device)).item()
            self.loss_history["train"].append(float(np.mean(losses))); self.loss_history["validation"].append(val_loss)
            if val_loss < best_loss - 1e-8: best_loss, remaining, best_state = val_loss, int(self.config["patience"]), copy.deepcopy(self.model.state_dict())
            else:
                remaining -= 1
                if remaining <= 0: break
        self.model.load_state_dict(best_state)
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "scaler_mean": self.scaler.mean_, "scaler_scale": self.scaler.scale_, "config": self.config, "loss_history": self.loss_history}, checkpoint_path)
        return self
    def predict(self, history, horizon):
        context = list(self.scaler.transform(np.asarray(history, dtype=np.float32).reshape(-1,1)).ravel()[-self.context_length:]); result=[]; self.model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                x=torch.tensor(context[-self.context_length:], dtype=torch.float32, device=self.device)[None,:,None]
                value=float(self._forward(x).cpu().item()); result.append(value); context.append(value)
        return self.scaler.inverse_transform(np.asarray(result).reshape(-1,1)).ravel()
