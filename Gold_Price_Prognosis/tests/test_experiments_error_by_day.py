import numpy as np, pandas as pd
from gold_forecasting.splitting import chronological_split
from gold_forecasting.experiments import (run_baselines, run_rolling_model, compare_error_by_day, compare_all_error_by_day,
                                           compare_leaderboard, compare_residual_diagnostics)
from gold_forecasting.models.xgboost_diff import XGBoostDiffForecaster

def _split(tmp_path, monkeypatch):
    # `experiments.py`/`plotting.py`/`mlflow_utils.py` each did `from .paths import X`, which
    # copies a reference into their own module namespace at import time -- patching
    # `gold_forecasting.paths.X` alone does NOT redirect those already-bound names, so every
    # module that actually writes a file must be patched directly, or `run_baselines` here would
    # write real signature-named artifacts into the tracked `artifacts/` tree instead of tmp_path.
    import gold_forecasting.experiments as experiments
    import gold_forecasting.plotting as plotting
    import gold_forecasting.mlflow_utils as mlflow_utils
    for name in ("PREDICTIONS", "METRICS", "CHECKPOINTS", "HPO_TRIALS", "LOSSES", "FEATURE_IMPORTANCE"):
        monkeypatch.setattr(experiments, name, tmp_path / name)
    monkeypatch.setattr(plotting, "FIGURES", tmp_path / "FIGURES")
    monkeypatch.setattr(mlflow_utils, "MLFLOW", tmp_path / "MLFLOW")
    frame = pd.DataFrame({"gold_usd": np.linspace(100, 150, 3000)}, index=pd.bdate_range("2015-01-01", periods=3000))
    return chronological_split(frame, validation_years=1, test_years=1)

def test_error_by_day_reuses_cached_rolling_result_no_retrain(tmp_path, monkeypatch):
    split = _split(tmp_path, monkeypatch)
    horizon = 20
    results = run_baselines(split.train.gold_usd, split.validation.gold_usd, split.test.gold_usd,
                             "univariate", "testhash", seed=42, meta={"lead_time_checkpoints": (1, 10, 20)},
                             horizon=horizon, step=horizon)
    metrics, fig = compare_error_by_day(results, "univariate", split.train.gold_usd, horizon)
    assert set(metrics["horizon"]) == set(range(1, horizon + 1))  # every day, not just checkpoints
    assert set(metrics["model"]) == set(results)
    assert set(["mae", "rmse", "mase", "smape", "directional_accuracy"]).issubset(metrics.columns)
    assert len(fig.data) == len(results)  # one legend-toggleable trace per model

def test_compare_all_error_by_day_combines_both_experiments(tmp_path, monkeypatch):
    split = _split(tmp_path, monkeypatch)
    horizon = 20
    uni = run_baselines(split.train.gold_usd, split.validation.gold_usd, split.test.gold_usd,
                         "univariate", "testhash", seed=42, meta={}, horizon=horizon, step=horizon)
    multi = run_baselines(split.train.gold_usd, split.validation.gold_usd, split.test.gold_usd,
                           "multivariate", "testhash", seed=42, meta={}, horizon=horizon, step=horizon)
    metrics, fig = compare_all_error_by_day(uni, multi, split.train.gold_usd, horizon)
    assert set(metrics["model"]) == set(uni) | set(multi)
    assert set(metrics["horizon"]) == set(range(1, horizon + 1))

def test_compare_leaderboard_ranks_by_complete_row(tmp_path, monkeypatch):
    split = _split(tmp_path, monkeypatch)
    horizon = 20
    results = run_baselines(split.train.gold_usd, split.validation.gold_usd, split.test.gold_usd,
                             "univariate", "testhash", seed=42, meta={}, horizon=horizon, step=horizon)
    fig = compare_leaderboard(results, "univariate")
    assert set(fig.data[0].y) == set(results)

def test_compare_residual_diagnostics_returns_histogram_and_boxplot(tmp_path, monkeypatch):
    split = _split(tmp_path, monkeypatch)
    horizon = 20
    results = run_baselines(split.train.gold_usd, split.validation.gold_usd, split.test.gold_usd,
                             "univariate", "testhash", seed=42, meta={}, horizon=horizon, step=horizon)
    histogram_fig, boxplot_fig = compare_residual_diagnostics(results, "univariate")
    assert {trace.name for trace in histogram_fig.data} == set(results)
    assert {trace.name for trace in boxplot_fig.data} == set(results)

def test_xgboost_diff_feature_importance_persists_and_reloads_on_cache_hit(tmp_path, monkeypatch):
    split = _split(tmp_path, monkeypatch)
    exogenous = pd.DataFrame({"e1": np.linspace(1, 2, 3000), "e2": np.linspace(2, 1, 3000)}, index=split.train.index.union(split.validation.index).union(split.test.index))
    multi = pd.concat([split.train, split.validation, split.test]).join(exogenous)
    train, validation, test = multi.iloc[:len(split.train)], multi.iloc[len(split.train):len(split.train) + len(split.validation)], multi.iloc[-len(split.test):]
    horizon = 20
    feature_config = {"lags": [1, 2, 5], "rolling_windows": [5], "include_calendar": True, "exogenous_lag": 1}
    config = {"params": {"max_depth": 3, "learning_rate": 0.2, "n_estimators": 20, "subsample": 1.0, "colsample_bytree": 1.0}, "retrain_each_step": True, "seed": 42}
    builder = lambda checkpoint_path: XGBoostDiffForecaster(config=config, feature_config=feature_config)

    first = run_rolling_model(builder, "xgboost_diff", train, validation, test, "multivariate", "testhash", config, 42, {}, horizon, horizon)
    assert first["cached"] is False
    assert first["feature_importance"]  # populated straight from the freshly-fit model

    second = run_rolling_model(builder, "xgboost_diff", train, validation, test, "multivariate", "testhash", config, 42, {}, horizon, horizon)
    assert second["cached"] is True
    assert second["feature_importance"] == first["feature_importance"]  # reloaded from the persisted CSV, not refit
