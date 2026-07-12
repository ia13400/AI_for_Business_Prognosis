"""Compact native PyTorch N-BEATS-style residual stack."""
import torch
from torch import nn
from .base import NeuralForecaster

class _Block(nn.Module):
    def __init__(self, context, hidden): super().__init__(); self.net=nn.Sequential(nn.Linear(context,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU()); self.backcast=nn.Linear(hidden,context); self.forecast=nn.Linear(hidden,1)
    def forward(self,x): h=self.net(x); return self.backcast(h),self.forecast(h)
class _NBeats(nn.Module):
    def __init__(self,context,hidden,blocks): super().__init__(); self.blocks=nn.ModuleList([_Block(context,hidden) for _ in range(blocks)])
    # The explicit loop implements residual backcast subtraction.
    def forward(self,x):
        residual=x.squeeze(-1); forecast=torch.zeros((x.size(0),1),device=x.device)
        for block in self.blocks:
            backcast, contribution=block(residual); residual=residual-backcast; forecast=forecast+contribution
        return forecast
class NBeatsForecaster(NeuralForecaster):
    name="nbeats"
    def build_model(self): return _NBeats(self.context_length,int(self.config["hidden_size"]),int(self.config["blocks"]))
