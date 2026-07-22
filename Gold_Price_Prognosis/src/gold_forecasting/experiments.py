"""Shared, model-agnostic orchestration: caching, MLflow logging, metrics, plots.

Each model's own file (`models/sarima.py`, `models/sarimax.py`, ...) owns its
fit/predict/HPO logic and calls `run_model` here to get caching, MLflow
tracking, metric computation and plotting for free. `train`/`validation`/
`test` may be a 1-D target Series (univariate models) or a DataFrame whose
first column is the target and remaining columns are exogenous regressors
(multivariate models) -- both flow through the same code path uniformly.
"""
import json
import time
import mlflow
import pandas as pd
from .hashing import stable_hash
from .paths import CHECKPOINTS, PREDICTIONS, METRICS, ensure_directories
from .plotting import plot_predictions, plot_residuals, plot_combined
from .metrics import horizon_metrics
from .mlflow_utils import tracked_run, log_dict_flat

def _target(frame): return frame.iloc[:, 0] if isinstance(frame, pd.DataFrame) else frame
def _future_exogenous(frame):
    return frame.iloc[:, 1:].values if isinstance(frame, pd.DataFrame) and frame.shape[1] > 1 else None

def run_model(model, name, train, validation, test, namespace, data_hash, config, seed, meta, force_retrain=False):
    ensure_directories()
    inputs = {"namespace": namespace, "data_hash": data_hash, "model": name, "hyperparameters": config, "seed": seed, "horizon": len(test), **meta}
    signature = stable_hash(inputs)
    prediction_path = PREDICTIONS/namespace/f"{name}_{signature[:12]}.csv"
    metric_path = METRICS/namespace/f"{name}_{signature[:12]}.csv"
    checkpoint = CHECKPOINTS/namespace/f"{name}_{signature[:12]}.pt"
    manifest = prediction_path.with_suffix(".manifest.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True); metric_path.parent.mkdir(parents=True, exist_ok=True)
    cached = prediction_path.exists() and manifest.exists() and not force_retrain
    test_target = _target(test)
    if cached:
        raw = pd.read_csv(prediction_path, index_col=0, parse_dates=True)
        result = pd.DataFrame({"actual": test_target.values, "predicted": raw["predicted"].values}, index=test_target.index)
    else:
        started = time.perf_counter()
        model.fit(train, validation, checkpoint)
        history = pd.concat([train, validation]) if isinstance(train, (pd.Series, pd.DataFrame)) else train
        predicted = model.predict(history, len(test_target), future_exogenous=_future_exogenous(test))
        result = pd.DataFrame({"actual": test_target.values, "predicted": predicted}, index=test_target.index)
        result.index.name = "date"; result.to_csv(prediction_path)
        tags = {"data_hash": data_hash, "cache_signature": signature, **{k: str(v) for k, v in meta.items()}}
        with tracked_run(name, namespace, tags) as run:
            log_dict_flat("model", {k: v for k, v in config.items() if not isinstance(v, dict)})
            mlflow.log_param("seed", seed); mlflow.log_metric("runtime_seconds", time.perf_counter() - started)
            mlflow.log_artifact(str(prediction_path))
            if checkpoint.exists(): mlflow.log_artifact(str(checkpoint))
            inputs["mlflow_run_id"] = run.info.run_id
        manifest.write_text(json.dumps({"signature": signature, "inputs": inputs}, indent=2, default=str), encoding="utf-8")
    metrics = horizon_metrics(result, _target(train), meta.get("evaluation_horizons", (1, 7, 30)))
    metrics.insert(0, "model", name)
    metrics.to_csv(metric_path, index=False)
    plot_predictions(result, namespace, name, signature)
    plot_residuals(result, name, signature)
    return {"predictions": result, "metrics": metrics, "signature": signature, "best_params": getattr(model, "best_params", {})}

def run_baselines(train, validation, test, namespace, data_hash, seed, meta, force_retrain=False, moving_average_window=20):
    from .forecasting import make_model
    from .config import select_device
    device = select_device()
    results = {}
    for name, config in (("naive", {}), ("moving_average", {"window": moving_average_window})):
        model = make_model(name, config, device, seed)
        results[name] = run_model(model, name, train, validation, test, namespace, data_hash, config, seed, meta, force_retrain)
    return results

def compare_results(results: dict, namespace: str, data_hash: str):
    metrics = pd.concat([r["metrics"] for r in results.values()], ignore_index=True)
    first = next(iter(results.values()))["predictions"]
    combined = pd.DataFrame({"actual": first["actual"]})
    for name, r in results.items(): combined[name] = r["predictions"]["predicted"]
    plot_path = plot_combined(combined, namespace, data_hash)
    return metrics.sort_values(["horizon", "mae"]), plot_path

def run_future(model, name, full_series, horizon, data_hash, seed, meta, force_retrain=False):
    """Refit on the entire available history and forecast `horizon` steps beyond it.

    No ground truth exists yet for genuine future dates, so this only caches
    predictions (no metrics). Restricted to univariate models -- forecasting
    exogenous variables themselves is out of scope, so multivariate models
    cannot produce a genuine unknown-future forecast.
    """
    ensure_directories()
    inputs = {"namespace": "future", "data_hash": data_hash, "model": name, "horizon": horizon, "seed": seed, **meta}
    signature = stable_hash(inputs)
    prediction_path = PREDICTIONS/"future"/f"{name}_{signature[:12]}.csv"
    manifest = prediction_path.with_suffix(".manifest.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    if not (prediction_path.exists() and manifest.exists() and not force_retrain):
        validation_size = max(1, len(full_series)//20)
        model.fit(full_series.iloc[:-validation_size], full_series.iloc[-validation_size:])
        dates = pd.bdate_range(full_series.index.max() + pd.offsets.BDay(), periods=horizon)
        predicted = model.predict(full_series, horizon)
        pd.DataFrame({"date": dates, "predicted": predicted}).to_csv(prediction_path, index=False)
        manifest.write_text(json.dumps({"signature": signature, "inputs": inputs}, indent=2, default=str), encoding="utf-8")
    return pd.read_csv(prediction_path, parse_dates=["date"], index_col="date")
