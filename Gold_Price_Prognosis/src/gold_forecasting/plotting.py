"""Deterministic cached static (matplotlib PNG) plotting functions.

Every chart applies scientific model display names (`display.py`), a light
grid, and a short methodology note below the plot -- consistent presentation
for an academic context, matching the same polish applied to the interactive
Plotly charts in `interactive_plots.py`. All user-facing text (titles, axis
labels, legends, notes) is German.
"""
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .paths import FIGURES
from .display import display_name, display_namespace

def _save(fig, path: Path, force: bool) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    if force or not path.exists(): fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig); return path

def _grid(ax):
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    return ax

# Opaque white box behind any annotation that sits on top of plotted data -- without it, a label
# can land directly on a curve/peak and become hard to read where the two overlap.
_LABEL_BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)

def _note(fig, text, gap=0.028):
    """A short, centered methodology caption placed directly under the axes -- the text is
    explicitly wrapped to a width matched to the figure's own size (not matplotlib's unreliable
    `wrap=True`, which can crop the final line under `bbox_inches="tight"`), so it never runs
    outside the saved image and never reserves more room than it actually needs. Returns the
    bottom fraction of figure height a caller should reserve (e.g. via
    `fig.tight_layout(rect=(0, reserved, 1, 1))`) to keep the axes' own x-label clear of the note --
    just `gap` itself: the note text lives entirely below y=0 regardless of how many lines it wraps
    to, and `bbox_inches="tight"` (in `_save`) already expands the saved canvas to fit it, so
    reserving extra rect space per line here would only add dead space between the two, not move
    the note any closer.
    """
    fig_width_in, _ = fig.get_size_inches()
    chars_per_line = max(40, int(fig_width_in * 13))
    lines = textwrap.wrap(text, width=chars_per_line) or [text]
    fig.text(0.5, -gap, "\n".join(lines), fontsize=8, color="#555", ha="center", va="top")
    return gap

def plot_history(series: pd.Series, signature: str, force=False):
    fig,ax=plt.subplots(figsize=(12,5)); series.plot(ax=ax); ax.set(title="Goldpreis-Verlauf",ylabel="USD je Feinunze",xlabel="Datum")
    _grid(ax); _note(fig, "Methode: taeglicher Schlusskurs (Yahoo Finance, GC=F), unveraendert, ohne Transformation.")
    return _save(fig,FIGURES/f"history_{signature[:12]}.png",force)
def plot_chronological_split(series, split, signature, force=False):
    fig,ax=plt.subplots(figsize=(12,5)); series.plot(ax=ax,label="Tatsächlich",color="black",lw=1)
    ax.axvspan(split.train.index.min(),split.train.index.max(),alpha=.10,color="tab:blue",label="Training")
    ax.axvspan(split.validation.index.min(),split.validation.index.max(),alpha=.15,color="tab:orange",label="Validierung")
    ax.axvspan(split.test.index.min(),split.test.index.max(),alpha=.15,color="tab:red",label="Test")
    ax.axvline(split.validation.index.min(),color="tab:orange",lw=1,ls=":")
    ax.axvline(split.test.index.min(),color="tab:red",lw=1,ls=":")
    ax.legend(loc="upper left")
    ymin,ymax=ax.get_ylim(); span=ymax-ymin
    # Two different vertical levels and opposite horizontal directions -- keeps the boxes apart
    # even when validation/test start dates sit close together on a multi-decade x-axis.
    ax.annotate(f"Validierung beginnt\n{split.validation.index.min().date()}",xy=(split.validation.index.min(),ymin+0.92*span),
                xytext=(-95,0),textcoords="offset points",fontsize=8,color="tab:orange",bbox=_LABEL_BOX,
                arrowprops=dict(arrowstyle="-",color="tab:orange",lw=0.7))
    ax.annotate(f"Test beginnt\n{split.test.index.min().date()}",xy=(split.test.index.min(),ymin+0.68*span),
                xytext=(10,0),textcoords="offset points",fontsize=8,color="tab:red",bbox=_LABEL_BOX,
                arrowprops=dict(arrowstyle="-",color="tab:red",lw=0.7))
    ax.set(ylabel="USD je Feinunze",xlabel="Datum",title="Chronologische Aufteilung: Training/Validierung/Test")
    _grid(ax); _note(fig, "Methode: harter chronologischer Schnitt, Validierung und Test je genau 1 Jahr, rueckwaerts vom "
                           "letzten verfuegbaren Datum -- alles Fruehere ist Trainingsdaten. Keine Durchmischung; strikte "
                           "zeitliche Reihenfolge.")
    return _save(fig,FIGURES/f"chronological_split_{signature[:12]}.png",force)
def plot_predictions(frame, experiment, model, signature, cutoff=None, force=False):
    fig,ax=plt.subplots(figsize=(12,5))
    columns=[c for c in ["actual","predicted"] if c in frame]
    frame[columns].rename(columns={"actual":"Tatsächlich","predicted":"Prognose"}).plot(ax=ax)
    if cutoff: ax.axvline(pd.Timestamp(cutoff),color="red",ls="--")
    ax.set(title=f"{display_namespace(experiment)}: {display_name(model)}",ylabel="USD je Feinunze",xlabel="Datum")
    _grid(ax); _note(fig, "Methode: aneinandergereihte rollierende (walk-forward) Prognose ueber den gesamten Testzeitraum "
                           "-- jeder Punkt nutzt nur Daten, die vor Beginn des jeweiligen 20-Tage-Fensters verfuegbar waren.")
    return _save(fig,FIGURES/f"{experiment}_{model}_{signature[:12]}.png",force)
def plot_residuals(frame, model, signature, force=False):
    fig,ax=plt.subplots(figsize=(12,4)); residual=frame.actual-frame.predicted; residual.plot(ax=ax); ax.axhline(0,color="black",lw=.8)
    ax.set(title=f"Residuen: {display_name(model)}",ylabel="USD",xlabel="Datum")
    _grid(ax); _note(fig, "Methode: Residuum = tatsaechlicher minus prognostizierter Preis, ueber den gesamten rollierenden "
                           "Testzeitraum. Anhaltend positive/negative Abschnitte deuten auf systematische Ueber-/Unterschaetzung hin.")
    return _save(fig,FIGURES/f"holdout_{model}_residuals_{signature[:12]}.png",force)
def plot_trading_bot(bot, model, signature, force=False):
    """`bot`: a `trading.simulate_trading_bot`/`simulate_cheater_bot` result (date, actual, portfolio_value, position, action).

    Top panel: real gold price and (for real models -- `bot` has a `predicted`
    column, unlike the cheater bot, which has none) that model's own predicted
    price, both on the left axis (same USD/oz unit, directly comparable), vs.
    portfolio value on the right axis -- a separate axis since portfolio value
    can range from a few thousand to several times the starting capital, a
    very different scale from the price itself. Buy/sell decisions are marked
    directly on the price line. Start/final portfolio value are called out as
    small ticks + labels pinned to the portfolio y-axis itself (outside the
    plotted curves entirely), so they can never overlap a data line. Bottom
    panel: the resulting cash/gold position over time as a step curve -- its
    rising/falling edges *are* the buy/sell decisions, its flat stretches are
    "hold".
    """
    fig,(ax_price,ax_position)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[3,1]})
    ax_price.plot(bot["date"],bot["actual"],color="black",lw=1,label="Goldpreis (USD/Feinunze)")
    if "predicted" in bot.columns:
        ax_price.plot(bot["date"],bot["predicted"],color="tab:orange",lw=1,ls="--",label="Prognostizierter Preis (USD/Feinunze)")
    ax_portfolio=ax_price.twinx(); ax_portfolio.plot(bot["date"],bot["portfolio_value"],color="tab:blue",lw=1.5,label="Portfoliowert (USD)")
    buys,sells=bot[bot["action"]=="buy"],bot[bot["action"]=="sell"]
    ax_price.scatter(buys["date"],buys["actual"],marker="^",color="green",s=70,zorder=5,label="Kauf")
    ax_price.scatter(sells["date"],sells["actual"],marker="v",color="red",s=70,zorder=5,label="Verkauf")

    ax_portfolio.tick_params(axis="y",which="both",labelright=False,right=False)  # replaced by the two explicit callouts below
    start_value,final_value=bot["portfolio_value"].iloc[0],bot["portfolio_value"].iloc[-1]
    y_lo,y_hi=ax_portfolio.get_ylim(); y_span=y_hi-y_lo
    close=abs(final_value-start_value)<0.05*y_span  # nudge the two labels apart if they'd otherwise sit on top of each other
    axis_transform=ax_portfolio.get_yaxis_transform()  # x: axes fraction, y: data -- lets text sit just outside the right spine
    for value,label,color,point_offset in ((start_value,"Startwert","dimgray",8 if close else 0),
                                            (final_value,"Endwert","tab:blue",-8 if close else 0)):
        ax_portfolio.plot([1.0,1.015],[value,value],transform=axis_transform,color=color,lw=1,clip_on=False)
        ax_portfolio.annotate(f"{label} = {value:,.0f} USD",xy=(1.02,value),xycoords=axis_transform,
                               xytext=(0,point_offset),textcoords="offset points",fontsize=8,color=color,
                               va="center",ha="left",annotation_clip=False,bbox=_LABEL_BOX)
    ax_price.set_ylabel("Goldpreis (USD/Feinunze)"); ax_portfolio.set_ylabel("Portfoliowert (USD)")
    lines1,labels1=ax_price.get_legend_handles_labels(); lines2,labels2=ax_portfolio.get_legend_handles_labels()
    ax_price.legend(lines1+lines2,labels1+labels2,loc="upper left",fontsize=8)
    ax_price.set_title(f"{display_name(model)}: Handelsbot -- Preis, Portfolio, Position")
    position_numeric=(bot["position"]=="gold").astype(int)
    ax_position.step(bot["date"],position_numeric,where="post",color="tab:purple")
    ax_position.set_yticks([0,1]); ax_position.set_yticklabels(["Bargeld","Gold"]); ax_position.set_ylabel("Position")
    _grid(ax_price); _grid(ax_position)
    if model=="cheater":
        note=("Methode: Referenzwert mit perfekter Voraussicht -- haelt Gold, wann immer der reale Preis von morgen ueber "
              "dem von heute liegt, taeglich neu entschieden anhand echter zukuenftiger Preise. Kein Modell, keine "
              "Schwelle, keine Gebuehren -- eine theoretische Obergrenze, keine in Echtzeit umsetzbare Strategie.")
    else:
        note=("Methode: entscheidet einmal pro rollierendem 20-Tage-Schritt (Kauf/Verkauf anhand einer +/-5-USD-Schwelle "
              "der prognostizierten Rendite am Beginn des Schritts), haelt die Position fuer den gesamten Schritt -- "
              "keine Gebuehren modelliert.")
    reserved=_note(fig, note, gap=0.03)
    fig.tight_layout(rect=(0,reserved,1,1))
    return _save(fig,FIGURES/f"trading_{model}_{signature[:12]}.png",force)
def plot_pnl_bar(summary, starting_capital, signature, force=False):
    """`summary`: DataFrame with 'model' and 'final_value' columns.

    Diverging horizontal bar rooted at `starting_capital` (not zero) -- each
    bar's own ending USD value is written directly next to it (left of the
    bar for a loss, right for a gain), so the exact number is readable
    without hovering. Named `pnl_summary_*` (not `trading_*`) so it doesn't
    collide with the per-bot `trading_<model>_*.png` glob used elsewhere.
    """
    ranked=summary.sort_values("final_value")
    labels=ranked["model"].map(display_name)
    colors=["#e34948" if v<starting_capital else "#2a78d6" for v in ranked["final_value"]]
    fig,ax=plt.subplots(figsize=(10,0.45*len(ranked)+1.5))
    ax.barh(labels,ranked["final_value"]-starting_capital,left=starting_capital,color=colors)
    ax.axvline(starting_capital,color="black",lw=1)
    lo,hi=min(starting_capital,ranked["final_value"].min()),max(starting_capital,ranked["final_value"].max())
    pad=(hi-lo)*0.15 if hi>lo else 1.0
    ax.set_xlim(lo-pad,hi+pad)
    for y,value in enumerate(ranked["final_value"]):
        ha="left" if value>=starting_capital else "right"
        offset=pad*0.15
        ax.text(value+(offset if ha=="left" else -offset),y,f"{value:,.0f} USD",va="center",ha=ha,fontsize=9)
    ax.set_xlabel("Endwert des Portfolios (USD)"); ax.set_title(f"Endwert des Portfolios je Bot (Startkapital {starting_capital:,.0f} USD)")
    _grid(ax)
    reserved=_note(fig, "Methode: derselbe rollierende Handelsbot-Backtest wie im Portfolio-Wert-Verlauf -- Balken "
                         "beginnen beim Startkapital, sodass die Achse direkt den End-USD-Betrag zeigt.", gap=0.02)
    fig.tight_layout(rect=(0,reserved,1,1))
    return _save(fig,FIGURES/f"pnl_summary_{signature[:12]}.png",force)
