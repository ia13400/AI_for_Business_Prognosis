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
from .paths import CHECKPOINTS, PREDICTIONS, METRICS, HPO_TRIALS, LOSSES, FEATURE_IMPORTANCE, ensure_directories
from .plotting import plot_predictions, plot_residuals
from .interactive_plots import (combined_forecast_figure, error_by_lead_time_figure, loss_curves_figure,
                                 leaderboard_figure, feature_importance_figure, residual_histogram_figure,
                                 residual_boxplot_by_leadtime_figure)
from .metrics import rolling_metrics
from .mlflow_utils import tracked_run, log_dict_flat
from .rolling import rolling_forecast

def _target(frame): return frame.iloc[:, 0] if isinstance(frame, pd.DataFrame) else frame

def run_rolling_model(model_builder, name, train, validation, test, namespace, data_hash, config, seed, meta, horizon, step, force_retrain=False, hpo_study=None, extra_seconds=0.0):
    """`model_builder(checkpoint_path) -> model` is called once, only on a cache miss.

    Runs the rolling-origin forecast over the test region (train+validation
    is what the *first* test window's `fit_data` contains -- see
    `rolling.rolling_forecast` -- which is what makes "reuse validation as
    training data" hold automatically once HPO has frozen hyperparameters).

    `extra_seconds`: wall-clock time already spent *before* this call (e.g.
    the Optuna HPO search, which happens in the caller before `model_builder`
    is even constructed) -- added to this function's own measured time so
    `runtime_seconds` reflects the model's true total cost, not just the
    final evaluation pass.
    """
    ensure_directories()
    inputs = {"namespace": namespace, "data_hash": data_hash, "model": name, "hyperparameters": config, "seed": seed,
              "horizon": horizon, "step": step, **meta}
    signature = stable_hash(inputs)
    prediction_path = PREDICTIONS/namespace/f"{name}_{signature[:12]}.csv"
    metric_path = METRICS/namespace/f"{name}_{signature[:12]}.csv"
    checkpoint = CHECKPOINTS/namespace/f"{name}_{signature[:12]}.pt"
    loss_path = LOSSES/namespace/f"{name}_{signature[:12]}.csv"
    feature_importance_path = FEATURE_IMPORTANCE/namespace/f"{name}_{signature[:12]}.csv"
    manifest = prediction_path.with_suffix(".manifest.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True); metric_path.parent.mkdir(parents=True, exist_ok=True)
    cached = prediction_path.exists() and manifest.exists() and not force_retrain
    if cached:
        rolling_result = pd.read_csv(prediction_path, parse_dates=["date"])
        runtime_seconds = 0.0
        loss_history = pd.read_csv(loss_path).to_dict("list") if loss_path.exists() else None
        feature_importance = dict(pd.read_csv(feature_importance_path).values) if feature_importance_path.exists() else None
    else:
        started = time.perf_counter()
        model = model_builder(checkpoint)
        combined = pd.concat([train, validation, test])
        test_start = len(train) + len(validation)
        rolling_result = rolling_forecast(model, combined, test_start, len(combined), horizon, step)
        rolling_result.to_csv(prediction_path, index=False)
        loss_history = getattr(model, "loss_history", None)
        if loss_history and loss_history.get("train"):
            loss_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"train": loss_history["train"], "validation": loss_history["validation"]}).to_csv(loss_path, index=False)
        else:
            loss_history = None
        feature_importance = getattr(model, "feature_importances_", None)
        if feature_importance:
            feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"]).to_csv(feature_importance_path, index=False)
        # Logged into MLflow now (can't include the logging call's own overhead); the *complete* runtime_seconds
        # -- used for the manifest and the notebook's runtime report -- is measured after all housekeeping below.
        partial_runtime_seconds = extra_seconds + (time.perf_counter() - started)
        tags = {"data_hash": data_hash, "cache_signature": signature, **{k: str(v) for k, v in meta.items()}}
        with tracked_run(name, namespace, tags) as run:
            log_dict_flat("model", {k: v for k, v in config.items() if not isinstance(v, dict)})
            mlflow.log_param("seed", seed); mlflow.log_metric("runtime_seconds", partial_runtime_seconds)
            mlflow.log_artifact(str(prediction_path))
            if checkpoint.exists(): mlflow.log_artifact(str(checkpoint))
            inputs["mlflow_run_id"] = run.info.run_id
        if hpo_study is not None:
            trials_path = HPO_TRIALS/namespace/f"{name}_{signature[:12]}.csv"
            trials_path.parent.mkdir(parents=True, exist_ok=True)
            hpo_study.trials_dataframe().to_csv(trials_path, index=False)
        runtime_seconds = extra_seconds + (time.perf_counter() - started)
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
            "cached": cached, "loss_history": loss_history, "feature_importance": feature_importance}

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

def compare_error_by_day(results: dict, namespace: str, train, horizon: int):
    """Continuous per-day (1..horizon) error curve for every model in `results`.

    Purely a post-hoc recomputation over each model's already-cached
    `rolling_result` (which already records a `lead_time` value -- 1..horizon
    -- for every forecast point, see `rolling.rolling_forecast`) -- no model
    is refit and no cache signature is touched, unlike widening
    `experiments.yaml: lead_time_checkpoints` itself would (that list is
    folded into `run_rolling_model`'s cache signature, so changing it would
    invalidate every model's cached forecast for no methodological reason).
    Returns (metrics, figure); `metrics` has one row per model per day.
    """
    _require_results(results, namespace)
    frames = []
    for name, r in results.items():
        day_metrics = rolling_metrics(r["rolling_result"], _target(train), range(1, horizon + 1))
        day_metrics = day_metrics[day_metrics["horizon"] != "complete"]
        day_metrics.insert(0, "model", name)
        frames.append(day_metrics)
    metrics = pd.concat(frames, ignore_index=True)
    fig = error_by_lead_time_figure(metrics, f"{namespace}: MAE by day (1..{horizon})")
    return metrics.sort_values(["horizon", "mae"]), fig

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

def compare_all_error_by_day(univariate_results: dict, multivariate_results: dict, train, horizon: int):
    """Combined per-day (1..horizon) error curve across both experiments -- same reuse-not-retrain logic as `compare_error_by_day`."""
    all_results = {**univariate_results, **multivariate_results}
    return compare_error_by_day(all_results, "all models", train, horizon)

def compare_leaderboard(results: dict, namespace: str, metric: str = "mae"):
    """Ranked bar chart of the 'complete'-row `metric` value per model -- built from each model's already-computed `metrics`, no retraining."""
    _require_results(results, namespace)
    metrics = pd.concat([r["metrics"] for r in results.values()], ignore_index=True)
    complete = metrics[metrics["horizon"] == "complete"]
    return leaderboard_figure(complete, f"{namespace}: leaderboard ({metric})", metric)

def compare_residual_diagnostics(results: dict, namespace: str):
    """Residual histogram + boxplot-by-lead-time for every model in `results`, reusing each model's already-cached `rolling_result` -- no retraining."""
    _require_results(results, namespace)
    residuals = {name: r["rolling_result"]["actual"] - r["rolling_result"]["predicted"] for name, r in results.items()}
    histogram_fig = residual_histogram_figure(residuals, f"{namespace}: residual distribution")
    boxplot_fig = residual_boxplot_by_leadtime_figure({name: r["rolling_result"] for name, r in results.items()}, f"{namespace}: residual spread by lead time")
    return histogram_fig, boxplot_fig

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
