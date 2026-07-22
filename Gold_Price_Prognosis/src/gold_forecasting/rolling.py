"""Rolling-origin (walk-forward) evaluation engine.

Shared by every model's HPO objective (scored over the validation region)
and the final test evaluation -- the same mechanism, just handed a different
region. A window's forecast is only ever given `combined.iloc[:window_start]`
(all real data strictly before it), which is what makes every leakage rule
hold structurally rather than by convention.
"""
import pandas as pd

def rolling_windows(region_start: int, region_end: int, horizon: int, step: int) -> list[tuple[int, int]]:
    """Complete (start, end) position pairs of length `horizon`, stepping by `step`, fully inside [region_start, region_end). Incomplete trailing windows are dropped."""
    windows = []
    position = region_start
    while position + horizon <= region_end:
        windows.append((position, position + horizon))
        position += step
    return windows

def rolling_forecast(model, combined, region_start: int, region_end: int, horizon: int, step: int) -> pd.DataFrame:
    """Walk-forward rolling-origin forecast over `combined[region_start:region_end]`.

    `combined` is a continuous chronological Series (univariate target) or
    DataFrame (target column 0, exogenous columns after it); `region_start`/
    `region_end` are integer positions into it. Returns one row per forecast
    point: `step` (window index), `origin` (last date in the fitting data),
    `lead_time` (1-indexed position within its window), `date`, `actual`,
    `predicted`.
    """
    frame = combined.to_frame() if isinstance(combined, pd.Series) else combined
    records = []
    for step_number, (window_start, window_end) in enumerate(rolling_windows(region_start, region_end, horizon, step)):
        fit_data = combined.iloc[:window_start]
        window = frame.iloc[window_start:window_end]
        future_exogenous = window.iloc[:, 1:].values if frame.shape[1] > 1 else None
        predicted = model.forecast_window(fit_data, horizon, future_exogenous=future_exogenous)
        origin = fit_data.index[-1] if len(fit_data) else None
        for lead_time, (date, actual) in enumerate(zip(window.index, window.iloc[:, 0].values), start=1):
            records.append({"step": step_number, "origin": origin, "lead_time": lead_time, "date": date,
                             "actual": actual, "predicted": predicted[lead_time - 1]})
    return pd.DataFrame.from_records(records)
