"""Interactive Plotly comparison charts for multiple models at once.

Unlike `plotting.py`'s static matplotlib PNGs (one model or one series),
these are built fresh at render time from the same underlying CSVs/frames
and shared verbatim between the notebook and the Streamlit dashboard --
clicking a legend entry toggles that model's trace on/off.
"""
import plotly.graph_objects as go

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

def error_by_lead_time_figure(metrics, title: str) -> go.Figure:
    """`metrics`: a `rolling_metrics`-shaped table with rows per model per lead-time checkpoint plus a 'complete' row (dropped here)."""
    part = metrics[metrics["horizon"] != "complete"].copy()
    part["horizon"] = part["horizon"].astype(int)
    fig = go.Figure()
    for name, group in part.groupby("model"):
        group = group.sort_values("horizon")
        fig.add_trace(go.Scatter(x=group["horizon"], y=group["mae"], name=name, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title="Lead time (days)", yaxis_title="MAE (USD)", legend_title="Click to toggle")
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
