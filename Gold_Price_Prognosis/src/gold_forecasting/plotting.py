"""Deterministic cached plotting functions."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .paths import FIGURES

def _save(fig, path: Path, force: bool) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    if force or not path.exists(): fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig); return path

def plot_history(series: pd.Series, signature: str, force=False):
    fig,ax=plt.subplots(figsize=(12,5)); series.plot(ax=ax); ax.set(title="Gold price history",ylabel="USD per troy ounce",xlabel="Date"); return _save(fig,FIGURES/f"history_{signature[:12]}.png",force)
def plot_split(series, cutoff, signature, force=False):
    fig,ax=plt.subplots(figsize=(12,5)); series.plot(ax=ax,label="Actual"); ax.axvline(pd.Timestamp(cutoff),color="red",ls="--",label="Cutoff"); ax.axvspan(pd.Timestamp(cutoff),series.index.max(),alpha=.12,color="red",label="Masked period"); ax.legend(); ax.set(ylabel="USD per troy ounce",title="Historical holdout split"); return _save(fig,FIGURES/f"holdout_split_{signature[:12]}.png",force)
def plot_chronological_split(series, split, signature, force=False):
    fig,ax=plt.subplots(figsize=(12,5)); series.plot(ax=ax,label="Actual",color="black",lw=1)
    ax.axvspan(split.train.index.min(),split.train.index.max(),alpha=.10,color="tab:blue",label="Train")
    ax.axvspan(split.validation.index.min(),split.validation.index.max(),alpha=.15,color="tab:orange",label="Validation")
    ax.axvspan(split.test.index.min(),split.test.index.max(),alpha=.15,color="tab:red",label="Test")
    ax.legend(); ax.set(ylabel="USD per troy ounce",title="Chronological train/validation/test split"); return _save(fig,FIGURES/f"chronological_split_{signature[:12]}.png",force)
def plot_predictions(frame, experiment, model, signature, cutoff=None, force=False):
    fig,ax=plt.subplots(figsize=(12,5)); frame[[c for c in ["actual","predicted"] if c in frame]].plot(ax=ax); 
    if cutoff: ax.axvline(pd.Timestamp(cutoff),color="red",ls="--")
    ax.set(title=f"{experiment}: {model}",ylabel="USD per troy ounce"); return _save(fig,FIGURES/f"{experiment}_{model}_{signature[:12]}.png",force)
def plot_combined(frame, experiment, signature, force=False):
    fig,ax=plt.subplots(figsize=(12,5)); frame.plot(ax=ax); ax.set(title=f"{experiment}: model comparison",ylabel="USD per troy ounce"); return _save(fig,FIGURES/f"{experiment}_combined_{signature[:12]}.png",force)
def plot_residuals(frame, model, signature, force=False):
    fig,ax=plt.subplots(figsize=(12,4)); (frame.actual-frame.predicted).plot(ax=ax); ax.axhline(0,color="black",lw=.8); ax.set(title=f"Residuals: {model}",ylabel="USD"); return _save(fig,FIGURES/f"holdout_{model}_residuals_{signature[:12]}.png",force)
def plot_losses(history, model, signature, force=False):
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(history["train"],label="Train"); ax.plot(history["validation"],label="Validation"); ax.legend(); ax.set(title=f"Loss: {model}",xlabel="Epoch",ylabel="MSE"); return _save(fig,FIGURES/f"training_{model}_{signature[:12]}.png",force)
