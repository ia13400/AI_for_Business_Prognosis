import pandas as pd
import plotly.graph_objects as go
from gold_forecasting.interactive_plots import combined_forecast_figure, error_by_lead_time_figure, loss_curves_figure, save_interactive_figure

def test_combined_forecast_figure_has_one_trace_per_series():
    index = pd.bdate_range("2024-01-01", periods=5)
    combined = pd.DataFrame({"actual": range(5), "modelA": range(1, 6), "modelB": range(2, 7)}, index=index)
    fig = combined_forecast_figure(combined, "test title")
    assert isinstance(fig, go.Figure)
    names = [trace.name for trace in fig.data]
    assert names == ["Actual", "modelA", "modelB"]

def test_error_by_lead_time_figure_excludes_complete_row():
    metrics = pd.DataFrame({
        "model": ["a", "a", "a", "b", "b", "b"],
        "horizon": [1, 10, "complete", 1, 10, "complete"],
        "mae": [1.0, 2.0, 1.5, 3.0, 4.0, 3.5],
    })
    fig = error_by_lead_time_figure(metrics, "test title")
    names = sorted(trace.name for trace in fig.data)
    assert names == ["a", "b"]
    for trace in fig.data:
        assert list(trace.x) == [1, 10]  # "complete" never becomes an x-value

def test_loss_curves_figure_pairs_train_and_validation_per_model():
    loss_histories = {
        "patchtst": {"train": [3.0, 2.0, 1.0], "validation": [3.5, 2.5, 1.5]},
        "naive_without_history": {"train": [], "validation": []},
    }
    fig = loss_curves_figure(loss_histories, "test title")
    names = [trace.name for trace in fig.data]
    assert names == ["patchtst (train)", "patchtst (validation)"]  # empty history is skipped entirely

def test_save_interactive_figure_writes_self_contained_html(tmp_path):
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    path = save_interactive_figure(fig, tmp_path / "nested" / "chart.html")
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
