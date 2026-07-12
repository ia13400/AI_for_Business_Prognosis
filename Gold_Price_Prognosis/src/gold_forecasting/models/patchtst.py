"""Native PyTorch PatchTST encoder for univariate forecasting."""
import torch
from torch import nn
from .base import NeuralForecaster

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
