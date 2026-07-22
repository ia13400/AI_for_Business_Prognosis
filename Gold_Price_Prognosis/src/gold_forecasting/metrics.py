"""Forecast metrics in original price units."""
import numpy as np
import pandas as pd

def metric_set(actual, predicted, training=None) -> dict[str, float]:
    y = np.asarray(actual, float); p = np.asarray(predicted, float); error = y - p
    denominator = np.maximum(np.abs(y) + np.abs(p), np.finfo(float).eps)
    scale = np.mean(np.abs(np.diff(np.asarray(training, float)))) if training is not None and len(training) > 1 else np.nan
    direction = np.sign(np.diff(y)); predicted_direction = np.sign(p[1:] - y[:-1])
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))),
            "mase": float(np.mean(np.abs(error)) / scale) if scale and np.isfinite(scale) else np.nan,
            "smape": float(np.mean(200 * np.abs(error) / denominator)),
            "directional_accuracy": float(np.mean(direction == predicted_direction)) if len(y) > 1 else np.nan}

def rolling_metrics(rolling_result: pd.DataFrame, training, lead_time_checkpoints=(1, 10, 20)) -> pd.DataFrame:
    """Metrics for a concatenated rolling-origin forecast (see `rolling.rolling_forecast`).

    One row per lead-time checkpoint (e.g. day 1, day 10, day 20 of every
    20-day window -- always a subset of the same forecast, never a different
    horizon), plus one "complete" row over every forecast point.
    """
    rows = []
    for lead_time in lead_time_checkpoints:
        part = rolling_result[rolling_result["lead_time"] == lead_time].sort_values("date")
        if part.empty: continue
        rows.append({"horizon": lead_time, **metric_set(part["actual"], part["predicted"], training)})
    complete = rolling_result.sort_values("date")
    rows.append({"horizon": "complete", **metric_set(complete["actual"], complete["predicted"], training)})
    return pd.DataFrame(rows).drop_duplicates(subset="horizon")
