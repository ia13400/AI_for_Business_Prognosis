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
def plot_residuals(frame, model, signature, force=False):
    fig,ax=plt.subplots(figsize=(12,4)); (frame.actual-frame.predicted).plot(ax=ax); ax.axhline(0,color="black",lw=.8); ax.set(title=f"Residuals: {model}",ylabel="USD"); return _save(fig,FIGURES/f"holdout_{model}_residuals_{signature[:12]}.png",force)
def plot_trading_bot(bot, model, signature, force=False):
    """`bot`: a `trading.simulate_trading_bot`/`simulate_cheater_bot` result (date, actual, portfolio_value, position, action).

    Top panel: real gold price (left axis) vs. portfolio value (right axis) --
    separate axes since the two are both USD but on very different scales
    (price ~thousands/oz, portfolio value can range from a few thousand to
    several times the starting capital). Buy/sell decisions are marked
    directly on the price line. Bottom panel: the resulting cash/gold
    position over time as a step curve -- its rising/falling edges *are* the
    buy/sell decisions, its flat stretches are "hold".
    """
    fig,(ax_price,ax_position)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[3,1]})
    ax_price.plot(bot["date"],bot["actual"],color="black",lw=1,label="Gold price (USD/oz)")
    ax_portfolio=ax_price.twinx(); ax_portfolio.plot(bot["date"],bot["portfolio_value"],color="tab:blue",lw=1.5,label="Portfolio value (USD)")
    buys,sells=bot[bot["action"]=="buy"],bot[bot["action"]=="sell"]
    ax_price.scatter(buys["date"],buys["actual"],marker="^",color="green",s=70,zorder=5,label="Buy")
    ax_price.scatter(sells["date"],sells["actual"],marker="v",color="red",s=70,zorder=5,label="Sell")
    ax_price.set_ylabel("Gold price (USD/oz)"); ax_portfolio.set_ylabel("Portfolio value (USD)")
    lines1,labels1=ax_price.get_legend_handles_labels(); lines2,labels2=ax_portfolio.get_legend_handles_labels()
    ax_price.legend(lines1+lines2,labels1+labels2,loc="upper left",fontsize=8)
    ax_price.set_title(f"{model}: trading bot -- price, portfolio, position")
    position_numeric=(bot["position"]=="gold").astype(int)
    ax_position.step(bot["date"],position_numeric,where="post",color="tab:purple")
    ax_position.set_yticks([0,1]); ax_position.set_yticklabels(["Cash","Gold"]); ax_position.set_ylabel("Position")
    fig.tight_layout()
    return _save(fig,FIGURES/f"trading_{model}_{signature[:12]}.png",force)
