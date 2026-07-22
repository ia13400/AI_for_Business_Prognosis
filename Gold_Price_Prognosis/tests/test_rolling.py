import numpy as np
import pandas as pd
from gold_forecasting.rolling import rolling_windows, rolling_forecast

def test_rolling_windows_drops_incomplete_trailing_window():
    windows = rolling_windows(region_start=0, region_end=45, horizon=20, step=20)
    assert windows == [(0, 20), (20, 40)]  # the trailing [40,45) is incomplete and dropped

def test_rolling_windows_step_can_differ_from_horizon():
    windows = rolling_windows(region_start=0, region_end=50, horizon=20, step=10)
    assert windows == [(0, 20), (10, 30), (20, 40), (30, 50)]

class _RecordingForecaster:
    """Records the max date seen in `fit_data` at every call -- used to assert no leakage."""
    def __init__(self):
        self.calls = []
    def forecast_window(self, fit_data, horizon, future_exogenous=None):
        self.calls.append(fit_data.index.max() if len(fit_data) else None)
        return np.repeat(float(fit_data.iloc[-1, 0]), horizon)

def test_rolling_forecast_never_sees_data_from_its_own_or_later_windows():
    index = pd.bdate_range("2023-01-01", periods=100)
    frame = pd.DataFrame({"target": np.arange(100, dtype=float)}, index=index)
    model = _RecordingForecaster()
    region_start, region_end = 40, 100
    result = rolling_forecast(model, frame, region_start, region_end, horizon=20, step=20)
    windows = rolling_windows(region_start, region_end, horizon=20, step=20)
    assert len(model.calls) == len(windows)
    for call_max_date, (window_start, _) in zip(model.calls, windows):
        assert call_max_date < frame.index[window_start]  # fit_data strictly precedes its own window
    assert set(result["step"]) == set(range(len(windows)))
    assert result["lead_time"].min() == 1 and result["lead_time"].max() == 20

def test_rolling_forecast_accepts_univariate_series():
    """Univariate models pass a plain Series (no exogenous columns); this must not error on `.shape[1]`."""
    index = pd.bdate_range("2023-01-01", periods=60)
    series = pd.Series(np.linspace(100, 160, 60), index=index)
    model = _RecordingForecaster.__new__(_RecordingForecaster)
    model.calls = []
    def forecast_window(fit_data, horizon, future_exogenous=None):
        model.calls.append(future_exogenous)
        return np.repeat(float(np.asarray(fit_data)[-1]), horizon)
    model.forecast_window = forecast_window
    result = rolling_forecast(model, series, region_start=20, region_end=60, horizon=20, step=20)
    assert len(result) == 40
    assert all(fe is None for fe in model.calls)  # no exogenous columns to pass through

def test_rolling_forecast_result_shape_and_dates():
    index = pd.bdate_range("2023-01-01", periods=60)
    frame = pd.DataFrame({"target": np.linspace(100, 160, 60)}, index=index)
    model = _RecordingForecaster()
    result = rolling_forecast(model, frame, region_start=20, region_end=60, horizon=20, step=20)
    assert len(result) == 40  # 2 windows x 20 days
    assert list(result.columns) == ["step", "origin", "lead_time", "date", "actual", "predicted"]
    assert sorted(result["date"]) == list(frame.index[20:60])
