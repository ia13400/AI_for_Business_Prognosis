import numpy as np, pandas as pd
from gold_forecasting.splitting import chronological_split
from gold_forecasting.experiments import run_baselines, compare_error_by_day, compare_all_error_by_day

def _split(tmp_path, monkeypatch):
    # `experiments.py`/`plotting.py`/`mlflow_utils.py` each did `from .paths import X`, which
    # copies a reference into their own module namespace at import time -- patching
    # `gold_forecasting.paths.X` alone does NOT redirect those already-bound names, so every
    # module that actually writes a file must be patched directly, or `run_baselines` here would
    # write real signature-named artifacts into the tracked `artifacts/` tree instead of tmp_path.
    import gold_forecasting.experiments as experiments
    import gold_forecasting.plotting as plotting
    import gold_forecasting.mlflow_utils as mlflow_utils
    for name in ("PREDICTIONS", "METRICS", "CHECKPOINTS", "HPO_TRIALS", "LOSSES"):
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
