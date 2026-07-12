"""Native PyTorch LSTM forecaster."""
import torch
from torch import nn
from .base import NeuralForecaster

class _LSTM(nn.Module):
    def __init__(self, hidden, layers, dropout):
        super().__init__(); self.lstm=nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0); self.head=nn.Linear(hidden,1)
    def forward(self,x): return self.head(self.lstm(x)[0][:,-1])
class LSTMForecaster(NeuralForecaster):
    name="lstm"
    def build_model(self): return _LSTM(int(self.config["hidden_size"]), int(self.config["num_layers"]), float(self.config["dropout"]))
