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

def horizon_metrics(predictions: pd.DataFrame, training, horizons=(1, 7, 30)) -> pd.DataFrame:
    rows = []
    for horizon in [*horizons, len(predictions)]:
        part = predictions.iloc[:min(horizon, len(predictions))]
        rows.append({"horizon": "complete" if horizon == len(predictions) else horizon, **metric_set(part.actual, part.predicted, training)})
    return pd.DataFrame(rows).drop_duplicates(subset="horizon")
