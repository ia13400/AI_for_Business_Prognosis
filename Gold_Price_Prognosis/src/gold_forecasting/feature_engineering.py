"""Leakage-safe features based only on lagged observations."""
import pandas as pd

def create_features(target: pd.Series, lags: list[int], rolling_windows: list[int], include_calendar: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame({"target": target.astype(float)})
    past = frame["target"].shift(1)
    for lag in lags: frame[f"price_lag_{lag}"] = frame["target"].shift(lag)
    returns = frame["target"].pct_change(fill_method=None)
    frame["return_lag_1"] = returns.shift(1)
    for window in rolling_windows:
        frame[f"rolling_mean_{window}"] = past.rolling(window).mean()
        frame[f"rolling_std_{window}"] = past.rolling(window).std()
        frame[f"momentum_{window}"] = past / frame["target"].shift(window + 1) - 1
    if include_calendar:
        frame["day_of_week"] = frame.index.dayofweek
        frame["month"] = frame.index.month
    return frame
