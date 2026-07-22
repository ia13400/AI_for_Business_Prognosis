"""Shared, model-agnostic rolling-origin orchestration: caching, MLflow
logging, metrics, plots, and HPO-trial documentation.

Each model's own file (`models/sarima.py`, `models/sarimax.py`, ...) owns its
fit/forecast/HPO logic -- scored via `rolling.rolling_forecast` over a
validation region -- and calls `run_rolling_model` here for the final
rolling-origin test evaluation, getting caching, MLflow tracking, metric
computation, plotting, and HPO-trial documentation for free. `train`/
`validation`/`test` may be a 1-D target Series (univariate models) or a
DataFrame whose first column is the target and remaining columns are
exogenous regressors (multivariate models).
"""
import json
import time
import mlflow
import pandas as pd
from .hashing import stable_hash
from .paths import CHECKPOINTS, PREDICTIONS, METRICS, HPO_TRIALS, LOSSES, ensure_directories
from .plotting import plot_predictions, plot_residuals
from .interactive_plots import combined_forecast_figure, error_by_lead_time_figure, loss_curves_figure
from .metrics import rolling_metrics
from .mlflow_utils import tracked_run, log_dict_flat
from .rolling import rolling_forecast

def _target(frame): return frame.iloc[:, 0] if isinstance(frame, pd.DataFrame) else frame

def run_rolling_model(model_builder, name, train, validation, test, namespace, data_hash, config, seed, meta, horizon, step, force_retrain=False, hpo_study=None):
    """`model_builder(checkpoint_path) -> model` is called once, only on a cache miss.

    Runs the rolling-origin forecast over the test region (train+validation
    is what the *first* test window's `fit_data` contains -- see
    `rolling.rolling_forecast` -- which is what makes "reuse validation as
    training data" hold automatically once HPO has frozen hyperparameters).
    """
    ensure_directories()
    inputs = {"namespace": namespace, "data_hash": data_hash, "model": name, "hyperparameters": config, "seed": seed,
              "horizon": horizon, "step": step, **meta}
    signature = stable_hash(inputs)
    prediction_path = PREDICTIONS/namespace/f"{name}_{signature[:12]}.csv"
    metric_path = METRICS/namespace/f"{name}_{signature[:12]}.csv"
    checkpoint = CHECKPOINTS/namespace/f"{name}_{signature[:12]}.pt"
    loss_path = LOSSES/namespace/f"{name}_{signature[:12]}.csv"
    manifest = prediction_path.with_suffix(".manifest.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True); metric_path.parent.mkdir(parents=True, exist_ok=True)
    cached = prediction_path.exists() and manifest.exists() and not force_retrain
    if cached:
        rolling_result = pd.read_csv(prediction_path, parse_dates=["date"])
        runtime_seconds = 0.0
        loss_history = pd.read_csv(loss_path).to_dict("list") if loss_path.exists() else None
    else:
        started = time.perf_counter()
        model = model_builder(checkpoint)
        combined = pd.concat([train, validation, test])
        test_start = len(train) + len(validation)
        rolling_result = rolling_forecast(model, combined, test_start, len(combined), horizon, step)
        rolling_result.to_csv(prediction_path, index=False)
        runtime_seconds = time.perf_counter() - started
        loss_history = getattr(model, "loss_history", None)
        if loss_history and loss_history.get("train"):
            loss_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"train": loss_history["train"], "validation": loss_history["validation"]}).to_csv(loss_path, index=False)
        else:
            loss_history = None
        tags = {"data_hash": data_hash, "cache_signature": signature, **{k: str(v) for k, v in meta.items()}}
        with tracked_run(name, namespace, tags) as run:
            log_dict_flat("model", {k: v for k, v in config.items() if not isinstance(v, dict)})
            mlflow.log_param("seed", seed); mlflow.log_metric("runtime_seconds", runtime_seconds)
            mlflow.log_artifact(str(prediction_path))
            if checkpoint.exists(): mlflow.log_artifact(str(checkpoint))
            inputs["mlflow_run_id"] = run.info.run_id
        if hpo_study is not None:
            trials_path = HPO_TRIALS/namespace/f"{name}_{signature[:12]}.csv"
            trials_path.parent.mkdir(parents=True, exist_ok=True)
            hpo_study.trials_dataframe().to_csv(trials_path, index=False)
        inputs["runtime_seconds"] = runtime_seconds
        manifest.write_text(json.dumps({"signature": signature, "inputs": inputs, "best_params": getattr(model, "best_params", {})}, indent=2, default=str), encoding="utf-8")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    metrics = rolling_metrics(rolling_result, _target(train), meta.get("lead_time_checkpoints", (1, 10, 20)))
    metrics.insert(0, "model", name)
    metrics.to_csv(metric_path, index=False)
    display_frame = rolling_result.set_index("date")[["actual", "predicted"]]
    plot_predictions(display_frame, namespace, name, signature)
    plot_residuals(display_frame, name, signature)
    return {"predictions": display_frame, "rolling_result": rolling_result, "metrics": metrics, "signature": signature,
            "best_params": manifest_data.get("best_params", {}), "runtime_seconds": manifest_data["inputs"].get("runtime_seconds", 0.0),
            "cached": cached, "loss_history": loss_history}

def run_baselines(train, validation, test, namespace, data_hash, seed, meta, horizon, step, force_retrain=False, moving_average_window=20, enabled=None):
    """`enabled`: optional {"naive": bool, "moving_average": bool}, defaulting to both enabled."""
    from .forecasting import make_model
    from .config import select_device
    enabled = enabled or {}
    device = select_device()
    results = {}
    for name, config in (("naive", {}), ("moving_average", {"window": moving_average_window})):
        if not enabled.get(name, True): continue
        builder = lambda checkpoint_path, n=name, c=config: make_model(n, c, device, seed)
        results[name] = run_rolling_model(builder, name, train, validation, test, namespace, data_hash, config, seed, meta, horizon, step, force_retrain)
    return results

def _require_results(results: dict, context: str):
    if not results:
        raise ValueError(f"No enabled models to compare for {context} -- check configs/models.yaml: at least one model's `enabled` must be true.")

def compare_results(results: dict, namespace: str, data_hash: str):
    """Returns (metrics, forecast_figure, lead_time_figure) -- both figures are interactive Plotly charts (click a legend entry to toggle a model)."""
    _require_results(results, namespace)
    metrics = pd.concat([r["metrics"] for r in results.values()], ignore_index=True)
    first = next(iter(results.values()))["predictions"]
    combined = pd.DataFrame({"actual": first["actual"]})
    for name, r in results.items(): combined[name] = r["predictions"]["predicted"]
    forecast_fig = combined_forecast_figure(combined, f"{namespace}: model comparison")
    lead_time_fig = error_by_lead_time_figure(metrics, f"{namespace}: MAE by lead time")
    return metrics.sort_values(["horizon", "mae"]), forecast_fig, lead_time_fig

def compare_all_results(univariate_results: dict, multivariate_results: dict, data_hash: str):
    """Combined actual-vs-forecast figure/table across both experiments (same target, same test dates)."""
    all_results = {**univariate_results, **multivariate_results}
    _require_results(all_results, "all models")
    metrics = pd.concat([r["metrics"] for r in all_results.values()], ignore_index=True)
    first = next(iter(all_results.values()))["predictions"]
    combined = pd.DataFrame({"actual": first["actual"]})
    for name, r in all_results.items(): combined[name] = r["predictions"]["predicted"]
    forecast_fig = combined_forecast_figure(combined, "all models: comparison")
    return metrics.sort_values(["horizon", "mae"]), forecast_fig

def compare_loss_curves(results: dict, namespace: str):
    """Interactive train/validation loss-curve figure for models that expose `loss_history` (PatchTST, TFT, XGBoost); None if none do."""
    loss_histories = {name: r["loss_history"] for name, r in results.items() if r.get("loss_history")}
    if not loss_histories: return None
    return loss_curves_figure(loss_histories, f"{namespace}: training/validation loss")

def run_future(model, name, full_series, horizon, data_hash, seed, meta, force_retrain=False):
    """One-shot genuine future forecast beyond the data's end (Kapitel 6, univariate models only).

    Not a rolling evaluation -- there is no real future data to roll forward
    against. `forecast_window` is called exactly once; for models with
    `retrain_each_step` semantics this is simply their first (and only) fit,
    using the entire available history.
    """
    ensure_directories()
    inputs = {"namespace": "future", "data_hash": data_hash, "model": name, "horizon": horizon, "seed": seed, **meta}
    signature = stable_hash(inputs)
    prediction_path = PREDICTIONS/"future"/f"{name}_{signature[:12]}.csv"
    manifest = prediction_path.with_suffix(".manifest.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    if not (prediction_path.exists() and manifest.exists() and not force_retrain):
        dates = pd.bdate_range(full_series.index.max() + pd.offsets.BDay(), periods=horizon)
        predicted = model.forecast_window(full_series, horizon)
        pd.DataFrame({"date": dates, "predicted": predicted}).to_csv(prediction_path, index=False)
        manifest.write_text(json.dumps({"signature": signature, "inputs": inputs}, indent=2, default=str), encoding="utf-8")
    return pd.read_csv(prediction_path, parse_dates=["date"], index_col="date")
