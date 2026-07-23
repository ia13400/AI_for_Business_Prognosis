"""Interactive Plotly comparison charts for multiple models at once.

Unlike `plotting.py`'s static matplotlib PNGs (one model or one series),
these are built fresh at render time from the same underlying CSVs/frames
and shared verbatim between the notebook and the Streamlit dashboard --
clicking a legend entry toggles that model's trace on/off.

Every chart applies scientific model/series display names (`display.py`),
an explicit light grid where a shared reference grid aids reading values, and
a short methodology note pinned below the plot -- consistent presentation
for an academic context. Underlying data (dict keys, DataFrame columns)
stays snake_case throughout; display names are applied only at render time.
"""
from pathlib import Path
import plotly.graph_objects as go
from .display import display_name, display_series

_GRID = dict(showgrid=True, gridcolor="rgba(128,128,128,0.25)", zeroline=False)
# Opaque background + border for every annotation that sits on top of plotted data (as opposed to
# the `_note()` caption below the axes, which already lives in empty margin) -- without this, a
# label can land directly on a curve/marker and become unreadable where it overlaps.
_CALLOUT = dict(bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(0,0,0,0.25)", borderwidth=1, borderpad=4)

def _grid(fig):
    fig.update_xaxes(**_GRID); fig.update_yaxes(**_GRID)
    return fig

def _note(fig, text, bottom_margin=100):
    """A short methodology/interpretation caption pinned below the plot area, a fixed
    pixel distance down (`yshift`, not a paper-fraction offset, so this stays correctly
    placed regardless of the figure's own height) -- explains the method behind the chart
    and how to read it, matching an academic figure's caption."""
    fig.add_annotation(x=0, y=0, xref="paper", yref="paper", yshift=-(bottom_margin - 15),
                        xanchor="left", yanchor="top", showarrow=False, align="left",
                        font=dict(size=10.5, color="#555"), text=text)
    fig.update_layout(margin=dict(b=bottom_margin))
    return fig

def save_interactive_figure(fig: go.Figure, path: Path) -> Path:
    """Persist a Plotly figure as a standalone, still-interactive HTML file.

    Loads plotly.js from a CDN (`include_plotlyjs="cdn"`) rather than
    embedding it (~4.5MB per file) -- keeps these committed artifacts small,
    at the cost of needing internet access to render the saved file.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path

def combined_forecast_figure(combined, title: str) -> go.Figure:
    """`combined`: DataFrame with an 'actual' column plus one column per model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined.index, y=combined["actual"], name="Actual", line=dict(color="black", width=2)))
    for name in combined.columns:
        if name == "actual": continue
        fig.add_trace(go.Scatter(x=combined.index, y=combined[name], name=display_name(name), mode="lines"))
    last_date, last_actual = combined.index[-1], combined["actual"].iloc[-1]
    fig.add_annotation(x=last_date, y=last_actual, text=f"Last actual: {last_actual:,.0f} USD", showarrow=True, arrowhead=2, ax=40, ay=-40, **_CALLOUT)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="USD per troy ounce",
                       legend_title="Click to toggle", hovermode="x unified")
    _grid(fig)
    _note(fig, "Method: rolling-origin (walk-forward) forecasts concatenated across all test-period windows -- "
                "each point is that model's prediction for that date, made using only data available before the "
                "20-day window containing it began.")
    return fig

def error_by_lead_time_figure(metrics, title: str, metric: str = "mae", y_label: str | None = None) -> go.Figure:
    """`metrics`: a `rolling_metrics`-shaped table with rows per model per lead-time checkpoint plus a 'complete' row (dropped here).

    `metric`: which column to plot (default MAE); e.g. `directional_accuracy` for the accuracy-by-lead-time view.
    """
    part = metrics[metrics["horizon"] != "complete"].copy()
    part["horizon"] = part["horizon"].astype(int)
    fig = go.Figure()
    for name, group in part.groupby("model"):
        group = group.sort_values("horizon")
        fig.add_trace(go.Scatter(x=group["horizon"], y=group[metric], name=display_name(name), mode="lines+markers"))
    higher_is_better = metric == "directional_accuracy"
    best_idx = part[metric].idxmax() if higher_is_better else part[metric].idxmin()
    best = part.loc[best_idx]
    fig.add_annotation(x=best["horizon"], y=best[metric], showarrow=True, arrowhead=2, ay=-45, ax=30,
                        text=f"Best: {display_name(best['model'])} ({best[metric]:.2f})", **_CALLOUT)
    fig.update_layout(title=title, xaxis_title="Lead time (days)", yaxis_title=y_label or metric.upper(), legend_title="Click to toggle")
    _grid(fig)
    metric_definition=("Directional accuracy = the fraction of forecast points where the predicted price change "
                        "(predicted minus the last known actual) has the same sign as the actual price change "
                        "(actual minus the last known actual) -- 0.5 is coin-flip, 1.0 means every up/down move "
                        "was called correctly regardless of magnitude. ") if metric=="directional_accuracy" else ""
    _note(fig, f"Method: {metric_definition}{metric.replace('_', ' ').capitalize() if metric_definition else metric.replace('_', ' ')} "
                "computed across every rolling-origin test window, grouped by lead-time position within each 20-day "
                "window (day 1 = next-day forecast, day 20 = full-horizon forecast) -- always a slice of the same "
                f"forecast, never a different horizon. {'Higher is better.' if higher_is_better else 'Lower is better.'}")
    return fig

def leaderboard_figure(complete_metrics, title: str, metric: str = "mae") -> go.Figure:
    """`complete_metrics`: rows with a 'model' column and `metric` (typically the 'complete'-horizon rows of a `rolling_metrics` table).

    Ranked bar chart, best model at the top -- ascending (lower is better) for
    error metrics, descending for `directional_accuracy` (higher is better).
    """
    higher_is_better = metric == "directional_accuracy"
    ranked = complete_metrics[["model", metric]].sort_values(metric, ascending=not higher_is_better)
    labels = ranked["model"].map(display_name)
    fig = go.Figure(go.Bar(x=ranked[metric], y=labels, orientation="h", text=ranked[metric].map(lambda v: f"{v:.2f}"), textposition="outside"))
    best_label = labels.iloc[0]
    fig.update_layout(title=title, xaxis_title=metric.upper(), yaxis_title="Model", yaxis=dict(autorange="reversed"))
    fig.add_annotation(x=ranked[metric].iloc[0], y=best_label, text="Best", showarrow=True, arrowhead=2, ax=40, ay=0,
                        font=dict(color="green"), **_CALLOUT)
    _grid(fig)
    _note(fig, f"Method: each bar is the '{metric}' aggregated across the complete rolling-origin test period "
                f"(all windows, all lead times). {'Higher' if higher_is_better else 'Lower'} is better; "
                "the top bar is the best-performing model on this metric.")
    return fig

def feature_importance_figure(importances: dict, title: str, top_n: int = 15) -> go.Figure:
    """`importances`: {feature_name: importance}, e.g. `XGBRegressor.feature_importances_` zipped with column names."""
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in ranked][::-1]; values = [v for _, v in ranked][::-1]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", text=[f"{v:.3f}" for v in values], textposition="outside"))
    fig.update_layout(title=title, xaxis_title="Importance (fraction of total split gain)", yaxis_title="Feature")
    _grid(fig)
    _note(fig, "Method: gain-based feature importance from the final trained model (fit on train+validation, "
                "frozen hyperparameters) -- the fraction of total split-quality improvement attributable to each "
                "feature. Top bar is the single most-used feature.")
    return fig

def residual_histogram_figure(residuals: dict, title: str) -> go.Figure:
    """`residuals`: {model_name: array-like of (actual - predicted)} -- overlaid, legend-toggleable histograms."""
    fig = go.Figure()
    for name, values in residuals.items():
        fig.add_trace(go.Histogram(x=values, name=display_name(name), opacity=0.6))
    fig.add_vline(x=0, line=dict(color="black", width=1, dash="dot"))
    fig.update_layout(title=title, xaxis_title="Residual (actual - predicted, USD)", yaxis_title="Count",
                       barmode="overlay", legend_title="Click to toggle")
    _grid(fig)
    _note(fig, "Method: residual = actual minus predicted price, pooled across the complete rolling-origin test "
                "period. A distribution centered on the dashed zero line indicates no systematic bias; a shifted "
                "center indicates the model consistently over- or under-predicts.")
    return fig

def residual_boxplot_by_leadtime_figure(rolling_results: dict, title: str) -> go.Figure:
    """`rolling_results`: {model_name: rolling_result DataFrame} (needs 'lead_time', 'actual', 'predicted') -- one box per model per lead-time day."""
    fig = go.Figure()
    for name, result in rolling_results.items():
        fig.add_trace(go.Box(x=result["lead_time"], y=result["actual"] - result["predicted"], name=display_name(name)))
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"))
    fig.update_layout(title=title, xaxis_title="Lead time (days)", yaxis_title="Residual (actual - predicted, USD)",
                       boxmode="group", legend_title="Click to toggle")
    _grid(fig)
    _note(fig, "Method: one box per model per lead-time day, pooling every rolling-origin test window's residual "
                "at that day. Widening boxes at longer lead times show forecast uncertainty growing with horizon.")
    return fig

def hpo_convergence_figure(trials, title: str) -> go.Figure:
    """`trials`: an Optuna `study.trials_dataframe()`-shaped table (columns 'number', 'value', ...).

    Every HPO objective in this project minimizes validation MAE, so the
    "best so far" line is a running minimum -- flattening out signals the
    search has converged; still trending down at the last trial suggests
    more trials would likely have kept helping.
    """
    ordered = trials.sort_values("number")
    best_so_far = ordered["value"].cummin()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ordered["number"], y=ordered["value"], mode="markers", name="Trial"))
    fig.add_trace(go.Scatter(x=ordered["number"], y=best_so_far, mode="lines", name="Best so far", line=dict(color="red")))
    fig.add_annotation(x=ordered["number"].iloc[-1], y=best_so_far.iloc[-1], showarrow=True, arrowhead=2, ay=-40, ax=-40,
                        text=f"Final best: {best_so_far.iloc[-1]:.2f}", **_CALLOUT)
    fig.update_layout(title=title, xaxis_title="Trial", yaxis_title="Objective (validation MAE)", legend_title="Click to toggle")
    _grid(fig)
    _note(fig, "Method: each point is one Optuna trial's validation-region rolling-origin MAE; the red line is the "
                "running minimum. A flat tail means the search has converged; a still-falling tail suggests more "
                "trials would likely help further.")
    return fig

def hpo_param_relationship_figure(trials, title: str) -> go.Figure:
    """One subplot per tuned hyperparameter (`trials` columns prefixed `params_`) -- its sampled value vs. that trial's objective, across every trial."""
    from plotly.subplots import make_subplots
    param_columns = [c for c in trials.columns if c.startswith("params_")]
    best_idx = trials["value"].idxmin()
    fig = make_subplots(rows=1, cols=len(param_columns), subplot_titles=[c.removeprefix("params_") for c in param_columns])
    for i, column in enumerate(param_columns, start=1):
        fig.add_trace(go.Scatter(x=trials[column], y=trials["value"], mode="markers", showlegend=False,
                                  marker=dict(color="steelblue")), row=1, col=i)
        fig.add_trace(go.Scatter(x=[trials[column].loc[best_idx]], y=[trials["value"].loc[best_idx]], mode="markers",
                                  showlegend=(i == 1), name="Best trial", marker=dict(color="red", size=11, symbol="star")), row=1, col=i)
        fig.update_xaxes(title_text=column.removeprefix("params_"), row=1, col=i)
    fig.update_yaxes(title_text="Objective (validation MAE)", row=1, col=1)
    fig.update_layout(title=title, legend_title="Click to toggle")
    _grid(fig)
    _note(fig, "Method: each blue point is one trial's sampled value for that hyperparameter vs. its resulting "
                "validation MAE (red star = the overall best trial). A trend or U-shape suggests that parameter "
                "matters; best points clustered at one edge of the range suggests widening the search bound there.")
    return fig

def exogenous_overview_figure(data, columns, title: str) -> go.Figure:
    """One row per column, shared x-axis -- lets you visually compare each exogenous series' trend/co-movement against gold over time."""
    from plotly.subplots import make_subplots
    columns = list(columns)
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, subplot_titles=[display_series(c) for c in columns])
    for i, column in enumerate(columns, start=1):
        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode="lines", showlegend=False), row=i, col=1)
    fig.update_layout(title=title, height=220 * len(columns), margin=dict(b=110))
    _grid(fig)
    _note(fig, "Method: each panel is one exogenous series' full raw history. A single correlation coefficient "
                "(previous chart) averages over 24 years of very different regimes -- this view lets you check "
                "whether the relationship to gold looks stable over time or is regime-dependent.", bottom_margin=110)
    return fig

def correlation_heatmap_figure(correlation, title: str) -> go.Figure:
    """`correlation`: a square DataFrame (e.g. `frame.corr()`) -- cell values annotated, fixed -1..1 color scale."""
    columns = [display_series(c) for c in correlation.columns]
    values = correlation.values
    fig = go.Figure(go.Heatmap(z=values, x=columns, y=columns, zmin=-1, zmax=1, colorscale="RdBu_r",
                                text=correlation.round(2).values, texttemplate="%{text}", colorbar=dict(title="Correlation")))
    import numpy as np
    off_diag = values.copy(); np.fill_diagonal(off_diag, 0)
    i, j = np.unravel_index(np.argmax(np.abs(off_diag)), off_diag.shape)
    fig.add_annotation(x=columns[j], y=columns[i], text="Strongest pair", showarrow=True, arrowhead=2, ay=-40, **_CALLOUT)
    fig.update_layout(title=title)
    _note(fig, "Method: Pearson correlation coefficient over the complete available history (pre-split, purely "
                "descriptive -- not used by any model). A single coefficient over 24 years averages across very "
                "different market regimes; see the time-series overview below for the full picture.")
    return fig

def portfolio_value_figure(combined, starting_capital: float, title: str) -> go.Figure:
    """`combined`: DataFrame indexed by date, one column per bot (portfolio value, USD).

    A dashed gray horizontal line marks the starting capital -- the reference
    for "did this bot beat simply holding cash" -- instead of a second y-axis
    for the raw gold price (mixing "portfolio USD" and "USD/troy ounce" on one
    axis would be misleading despite the shared unit).
    """
    fig = go.Figure()
    for name in combined.columns:
        fig.add_trace(go.Scatter(x=combined.index, y=combined[name], name=display_name(name), mode="lines"))
    fig.add_hline(y=starting_capital, line=dict(color="gray", dash="dot"),
                  annotation_text=f"Starting capital ({starting_capital:,.0f} USD)", annotation_position="bottom left",
                  annotation_bgcolor=_CALLOUT["bgcolor"], annotation_bordercolor=_CALLOUT["bordercolor"], annotation_borderwidth=1)
    best_model = combined.iloc[-1].idxmax()
    fig.add_annotation(x=combined.index[-1], y=combined[best_model].iloc[-1], showarrow=True, arrowhead=2, ax=-50, ay=-40,
                        text=f"Best: {display_name(best_model)} ({combined[best_model].iloc[-1]:,.0f} USD)", **_CALLOUT)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Portfolio value (USD)",
                       legend_title="Click to toggle", hovermode="x unified")
    _grid(fig)
    _note(fig, "Method: each bot starts at the same capital and decides once per rolling 20-day step (buy/sell/hold "
                "vs. a +-5 USD predicted-return threshold), except 'Cheater', which has perfect foresight and can "
                "trade daily -- a theoretical upper bound, not a real model. No fees modeled.")
    return fig

def patchtst_sliding_window_figure(train, validation, test, context_length: int, horizon: int, step: int, title: str) -> go.Figure:
    """Illustrates the rolling-origin mechanism concretely for PatchTST: at every rolling window, the
    model sees a fixed `context_length`-day input window (its actual, bounded input -- see
    `NeuralForecaster._recursive_forecast` in `models/base.py`, which always feeds in just the trailing
    `context_length` real observations) immediately followed by the `horizon`-day forecast window,
    both regions the walk-forward engine steps forward by `step` days each time (`rolling.rolling_windows`).

    One row per rolling window, HPO/validation windows and final-test windows shown together (in
    different colors) so you can see both phases use the identical mechanism -- HPO just scores
    candidates against the validation region instead of test, per `rolling.rolling_forecast`.
    """
    from .rolling import rolling_windows
    combined = train.index.append(validation.index).append(test.index)
    validation_start, validation_end = len(train), len(train) + len(validation)
    test_start, test_end = validation_end, validation_end + len(test)
    fig = go.Figure()
    row = 0
    for phase, region_start, region_end, context_color, horizon_color in (
        ("HPO (Validierung)", validation_start, validation_end, "#a6cee3", "#ff7f0e"),
        ("Test", test_start, test_end, "#b2df8a", "#d62728"),
    ):
        seen_context, seen_horizon = False, False
        for window_start, window_end in rolling_windows(region_start, region_end, horizon, step):
            context_start = max(0, window_start - context_length)
            fig.add_trace(go.Scatter(x=[combined[context_start], combined[window_start - 1]], y=[row, row], mode="lines",
                                      line=dict(color=context_color, width=10), name=f"{phase}: Kontext ({context_length} Tage)",
                                      legendgroup=f"{phase}-context", showlegend=not seen_context))
            fig.add_trace(go.Scatter(x=[combined[window_start], combined[window_end - 1]], y=[row, row], mode="lines",
                                      line=dict(color=horizon_color, width=10), name=f"{phase}: Prognosehorizont ({horizon} Tage)",
                                      legendgroup=f"{phase}-horizon", showlegend=not seen_horizon))
            seen_context = seen_horizon = True
            row += 1
    fig.update_layout(title=title, xaxis_title="Datum", yaxis_title="Rollierendes Fenster (fortlaufender Index)",
                       legend_title="Klicken zum Ein-/Ausblenden")
    _grid(fig)
    _note(fig, "Methode: jede Zeile ist ein rollierendes Fenster -- der farbige Kontext-Balken zeigt PatchTSTs "
                "tatsaechliche, begrenzte Eingabe (die letzten context_length echten Tage), der zweite Balken den "
                "anschliessenden Prognosehorizont. Beide Bloecke wandern gemeinsam um step Tage nach rechts.")
    return fig

def patchtst_epoch_schedule_figure(loss_history: dict, n_windows: int, update_epochs: int, title: str) -> go.Figure:
    """Segments PatchTST's persisted loss history (concatenated across every rolling window's fit --
    see `NeuralForecaster`) back into per-window epoch blocks: the *first* window's full initial fit
    (early-stopped against `patience`, budget up to `epochs`), then every later window's short
    `update_epochs`-long warm-start continuation. Reconstructed purely from the loss history's total
    length, `n_windows`, and `update_epochs` -- every window after the first used exactly the same
    `update_epochs` budget, so the first window's (variable, early-stopped) length is just the remainder.
    """
    from plotly.subplots import make_subplots
    train_loss, validation_loss = loss_history["train"], loss_history["validation"]
    total_epochs = len(train_loss)
    first_window_epochs = total_epochs - (n_windows - 1) * update_epochs
    epochs_per_window = [first_window_epochs] + [update_epochs] * (n_windows - 1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Verlust je Epoche (fortlaufend, Fenstergrenzen gestrichelt)", "Epochen je rollierendem Fenster"))
    x = list(range(1, total_epochs + 1))
    fig.add_trace(go.Scatter(x=x, y=train_loss, mode="lines", name="Training"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=validation_loss, mode="lines", name="Validierung", line=dict(dash="dash")), row=1, col=1)
    cursor = 0
    for count in epochs_per_window[:-1]:
        cursor += count
        fig.add_vline(x=cursor + 0.5, line=dict(color="gray", dash="dot", width=1), row=1, col=1)

    colors = ["#d62728"] + ["#1f77b4"] * (n_windows - 1)
    fig.add_trace(go.Bar(x=list(range(1, n_windows + 1)), y=epochs_per_window, marker_color=colors,
                          name="Epochen", showlegend=False), row=1, col=2)
    fig.add_annotation(x=1, y=first_window_epochs, text="Erste vollstaendige Anpassung", showarrow=True, arrowhead=2, ay=-30, row=1, col=2, **_CALLOUT)
    if n_windows > 1:
        fig.add_annotation(x=n_windows, y=update_epochs, text="Warm-Start", showarrow=True, arrowhead=2, ay=-30, row=1, col=2, **_CALLOUT)
    fig.update_xaxes(title_text="Epoche (fortlaufend über alle Fenster)", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Rollierendes Fenster (Testzeitraum)", row=1, col=2)
    fig.update_yaxes(title_text="Anzahl Epochen", row=1, col=2)
    fig.update_layout(title=title, legend_title="Klicken zum Ein-/Ausblenden")
    _grid(fig)
    _note(fig, "Methode: aus der bereits gespeicherten Verlusthistorie rekonstruiert -- das erste Fenster durchlaeuft "
                "eine vollstaendige Anpassung (fruehes Stoppen via patience), jedes weitere nur einen kurzen "
                "Warm-Start (update_epochs). Kein zusaetzliches Training noetig.")
    return fig

def loss_curves_figure(loss_histories: dict, title: str) -> go.Figure:
    """`loss_histories`: {model_name: {"train": [...], "validation": [...]}}, only for models that expose one.

    Each model contributes two traces (train solid, validation dashed) sharing
    a legend group, so toggling a model hides both its curves together. The
    x-axis spans every epoch/boosting-round across the whole rolling
    evaluation (the initial fit plus every subsequent retrain/warm-start).
    """
    fig = go.Figure()
    for name, history in loss_histories.items():
        train, validation = history.get("train") or [], history.get("validation") or []
        if not train: continue
        label = display_name(name)
        fig.add_trace(go.Scatter(x=list(range(1, len(train) + 1)), y=train, name=f"{label} (train)",
                                  mode="lines", legendgroup=name))
        fig.add_trace(go.Scatter(x=list(range(1, len(validation) + 1)), y=validation, name=f"{label} (validation)",
                                  mode="lines", line=dict(dash="dash"), legendgroup=name))
    fig.update_layout(title=title, xaxis_title="Epoch / boosting round (across all rolling windows)",
                       yaxis_title="Loss", legend_title="Click to toggle")
    _grid(fig)
    _note(fig, "Method: loss concatenated across every rolling window's fit -- the initial full fit, then each "
                "later window's short warm-start continuation (neural models) or boosting rounds (XGBoost). "
                "Dashed = validation loss; a widening gap from the solid training loss signals overfitting.")
    return fig
