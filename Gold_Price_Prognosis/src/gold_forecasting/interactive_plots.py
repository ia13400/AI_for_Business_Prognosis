"""Interactive Plotly comparison charts for multiple models at once.

Unlike `plotting.py`'s static matplotlib PNGs (one model or one series),
these are built fresh at render time from the same underlying CSVs/frames
and shared verbatim between the notebook and the Streamlit dashboard --
clicking a legend entry toggles that model's trace on/off.
"""
from pathlib import Path
import plotly.graph_objects as go

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
        fig.add_trace(go.Scatter(x=combined.index, y=combined[name], name=name, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="USD per troy ounce",
                       legend_title="Click to toggle", hovermode="x unified")
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
        fig.add_trace(go.Scatter(x=group["horizon"], y=group[metric], name=name, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title="Lead time (days)", yaxis_title=y_label or metric.upper(), legend_title="Click to toggle")
    return fig

def leaderboard_figure(complete_metrics, title: str, metric: str = "mae") -> go.Figure:
    """`complete_metrics`: rows with a 'model' column and `metric` (typically the 'complete'-horizon rows of a `rolling_metrics` table).

    Ranked bar chart, best model at the top -- ascending (lower is better) for
    error metrics, descending for `directional_accuracy` (higher is better).
    """
    ranked = complete_metrics[["model", metric]].sort_values(metric, ascending=(metric != "directional_accuracy"))
    fig = go.Figure(go.Bar(x=ranked[metric], y=ranked["model"], orientation="h"))
    fig.update_layout(title=title, xaxis_title=metric.upper(), yaxis_title="Model", yaxis=dict(autorange="reversed"))
    return fig

def feature_importance_figure(importances: dict, title: str, top_n: int = 15) -> go.Figure:
    """`importances`: {feature_name: importance}, e.g. `XGBRegressor.feature_importances_` zipped with column names."""
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in ranked][::-1]; values = [v for _, v in ranked][::-1]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h"))
    fig.update_layout(title=title, xaxis_title="Importance", yaxis_title="Feature")
    return fig

def residual_histogram_figure(residuals: dict, title: str) -> go.Figure:
    """`residuals`: {model_name: array-like of (actual - predicted)} -- overlaid, legend-toggleable histograms."""
    fig = go.Figure()
    for name, values in residuals.items():
        fig.add_trace(go.Histogram(x=values, name=name, opacity=0.6))
    fig.update_layout(title=title, xaxis_title="Residual (actual - predicted, USD)", yaxis_title="Count",
                       barmode="overlay", legend_title="Click to toggle")
    return fig

def residual_boxplot_by_leadtime_figure(rolling_results: dict, title: str) -> go.Figure:
    """`rolling_results`: {model_name: rolling_result DataFrame} (needs 'lead_time', 'actual', 'predicted') -- one box per model per lead-time day."""
    fig = go.Figure()
    for name, result in rolling_results.items():
        fig.add_trace(go.Box(x=result["lead_time"], y=result["actual"] - result["predicted"], name=name))
    fig.update_layout(title=title, xaxis_title="Lead time (days)", yaxis_title="Residual (actual - predicted, USD)",
                       boxmode="group", legend_title="Click to toggle")
    return fig

def hpo_convergence_figure(trials, title: str) -> go.Figure:
    """`trials`: an Optuna `study.trials_dataframe()`-shaped table (columns 'number', 'value', ...).

    Every HPO objective in this project minimizes validation MAE, so the
    "best so far" line is a running minimum -- flattening out signals the
    search has converged; still trending down at the last trial suggests
    more trials would likely have kept helping.
    """
    ordered = trials.sort_values("number")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ordered["number"], y=ordered["value"], mode="markers", name="Trial"))
    fig.add_trace(go.Scatter(x=ordered["number"], y=ordered["value"].cummin(), mode="lines", name="Best so far", line=dict(color="red")))
    fig.update_layout(title=title, xaxis_title="Trial", yaxis_title="Objective (validation MAE)", legend_title="Click to toggle")
    return fig

def hpo_param_relationship_figure(trials, title: str) -> go.Figure:
    """One subplot per tuned hyperparameter (`trials` columns prefixed `params_`) -- its sampled value vs. that trial's objective, across every trial."""
    from plotly.subplots import make_subplots
    param_columns = [c for c in trials.columns if c.startswith("params_")]
    fig = make_subplots(rows=1, cols=len(param_columns), subplot_titles=[c.removeprefix("params_") for c in param_columns])
    for i, column in enumerate(param_columns, start=1):
        fig.add_trace(go.Scatter(x=trials[column], y=trials["value"], mode="markers", showlegend=False), row=1, col=i)
        fig.update_xaxes(title_text=column.removeprefix("params_"), row=1, col=i)
    fig.update_yaxes(title_text="Objective (validation MAE)", row=1, col=1)
    fig.update_layout(title=title)
    return fig

def exogenous_overview_figure(data, columns, title: str) -> go.Figure:
    """One row per column, shared x-axis -- lets you visually compare each exogenous series' trend/co-movement against gold over time."""
    from plotly.subplots import make_subplots
    columns = list(columns)
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, subplot_titles=columns)
    for i, column in enumerate(columns, start=1):
        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode="lines", showlegend=False), row=i, col=1)
    fig.update_layout(title=title, height=220 * len(columns))
    return fig

def correlation_heatmap_figure(correlation, title: str) -> go.Figure:
    """`correlation`: a square DataFrame (e.g. `frame.corr()`) -- cell values annotated, fixed -1..1 color scale."""
    columns = list(correlation.columns)
    fig = go.Figure(go.Heatmap(z=correlation.values, x=columns, y=columns, zmin=-1, zmax=1, colorscale="RdBu_r",
                                text=correlation.round(2).values, texttemplate="%{text}", colorbar=dict(title="Correlation")))
    fig.update_layout(title=title)
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
    fig.update_xaxes(title_text="Epoche (fortlaufend über alle Fenster)", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Rollierendes Fenster (Testzeitraum)", row=1, col=2)
    fig.update_yaxes(title_text="Anzahl Epochen", row=1, col=2)
    fig.update_layout(title=title, legend_title="Klicken zum Ein-/Ausblenden")
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
        fig.add_trace(go.Scatter(x=list(range(1, len(train) + 1)), y=train, name=f"{name} (train)",
                                  mode="lines", legendgroup=name))
        fig.add_trace(go.Scatter(x=list(range(1, len(validation) + 1)), y=validation, name=f"{name} (validation)",
                                  mode="lines", line=dict(dash="dash"), legendgroup=name))
    fig.update_layout(title=title, xaxis_title="Epoch / boosting round (across all rolling windows)",
                       yaxis_title="Loss", legend_title="Click to toggle")
    return fig
