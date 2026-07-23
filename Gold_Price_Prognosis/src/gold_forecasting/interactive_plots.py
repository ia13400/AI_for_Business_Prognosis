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
