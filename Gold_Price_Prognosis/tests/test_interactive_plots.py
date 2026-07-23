import pandas as pd
import plotly.graph_objects as go
from gold_forecasting.interactive_plots import (combined_forecast_figure, error_by_lead_time_figure, loss_curves_figure,
                                                  save_interactive_figure, correlation_heatmap_figure, leaderboard_figure,
                                                  feature_importance_figure, residual_histogram_figure,
                                                  residual_boxplot_by_leadtime_figure, hpo_convergence_figure,
                                                  hpo_param_relationship_figure, exogenous_overview_figure)

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

def test_error_by_lead_time_figure_supports_arbitrary_metric_column():
    metrics = pd.DataFrame({
        "model": ["a", "a"], "horizon": [1, 10], "directional_accuracy": [0.9, 0.6],
    })
    fig = error_by_lead_time_figure(metrics, "test title", metric="directional_accuracy", y_label="Directional accuracy")
    assert list(fig.data[0].y) == [0.9, 0.6]
    assert fig.layout.yaxis.title.text == "Directional accuracy"

def test_leaderboard_figure_ranks_best_model_first():
    metrics = pd.DataFrame({"model": ["slow", "best", "mid"], "mae": [10.0, 1.0, 5.0]})
    fig = leaderboard_figure(metrics, "test title")
    assert list(fig.data[0].y) == ["best", "mid", "slow"]  # ascending MAE -- best first

def test_leaderboard_figure_directional_accuracy_ranks_descending():
    metrics = pd.DataFrame({"model": ["low", "high"], "directional_accuracy": [0.4, 0.8]})
    fig = leaderboard_figure(metrics, "test title", metric="directional_accuracy")
    assert list(fig.data[0].y) == ["high", "low"]  # higher accuracy is better -- first

def test_feature_importance_figure_orders_highest_on_top_and_respects_top_n():
    importances = {f"feature_{i}": float(i) for i in range(20)}
    fig = feature_importance_figure(importances, "test title", top_n=5)
    assert len(fig.data[0].y) == 5
    assert fig.data[0].y[-1] == "feature_19"  # highest importance plotted last -> renders at the top

def test_residual_histogram_figure_one_trace_per_model():
    residuals = {"a": [1, -1, 2], "b": [0, 0.5]}
    fig = residual_histogram_figure(residuals, "test title")
    assert [trace.name for trace in fig.data] == ["a", "b"]
    assert fig.layout.barmode == "overlay"

def test_residual_boxplot_by_leadtime_figure_one_box_trace_per_model():
    results = {
        "a": pd.DataFrame({"lead_time": [1, 1, 2], "actual": [10, 11, 12], "predicted": [9, 10, 11]}),
        "b": pd.DataFrame({"lead_time": [1, 2, 2], "actual": [5, 6, 7], "predicted": [5, 5, 8]}),
    }
    fig = residual_boxplot_by_leadtime_figure(results, "test title")
    assert [trace.name for trace in fig.data] == ["a", "b"]

def test_hpo_convergence_figure_best_so_far_is_running_minimum():
    trials = pd.DataFrame({"number": [2, 0, 1], "value": [5.0, 10.0, 3.0]})
    fig = hpo_convergence_figure(trials, "test title")
    best_so_far = fig.data[1]
    assert list(best_so_far.x) == [0, 1, 2]
    assert list(best_so_far.y) == [10.0, 3.0, 3.0]  # cumulative minimum, ordered by trial number

def test_hpo_param_relationship_figure_one_subplot_per_param():
    trials = pd.DataFrame({"number": [0, 1], "value": [5.0, 3.0], "params_max_depth": [3, 5], "params_learning_rate": [0.1, 0.2]})
    fig = hpo_param_relationship_figure(trials, "test title")
    assert len(fig.data) == 2  # one scatter trace per tuned hyperparameter

def test_exogenous_overview_figure_one_row_per_column():
    index = pd.bdate_range("2024-01-01", periods=5)
    data = pd.DataFrame({"gold_usd": range(5), "silver": range(5, 10), "vix": range(10, 15)}, index=index)
    fig = exogenous_overview_figure(data, ["silver", "vix"], "test title")
    assert len(fig.data) == 2

def test_loss_curves_figure_pairs_train_and_validation_per_model():
    loss_histories = {
        "patchtst": {"train": [3.0, 2.0, 1.0], "validation": [3.5, 2.5, 1.5]},
        "naive_without_history": {"train": [], "validation": []},
    }
    fig = loss_curves_figure(loss_histories, "test title")
    names = [trace.name for trace in fig.data]
    assert names == ["patchtst (train)", "patchtst (validation)"]  # empty history is skipped entirely

def test_correlation_heatmap_figure_uses_fixed_scale_and_labels():
    frame = pd.DataFrame({"gold_usd": [1, 2, 3, 4], "dollar_index": [4, 3, 2, 1], "silver": [1, 2, 3, 4]})
    fig = correlation_heatmap_figure(frame.corr(), "test title")
    heatmap = fig.data[0]
    assert list(heatmap.x) == list(heatmap.y) == ["gold_usd", "dollar_index", "silver"]
    assert heatmap.zmin == -1 and heatmap.zmax == 1
    assert heatmap.z[0][0] == 1.0  # self-correlation

def test_save_interactive_figure_writes_self_contained_html(tmp_path):
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    path = save_interactive_figure(fig, tmp_path / "nested" / "chart.html")
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
