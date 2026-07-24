import numpy as np
import pandas as pd
import pytest
from gold_forecasting.config import select_device, set_seed
from gold_forecasting.models.naive import NaiveForecaster
from gold_forecasting.models.moving_average import MovingAverageForecaster
from gold_forecasting.models.sarima import SarimaForecaster
from gold_forecasting.models.sarimax import SarimaxForecaster
from gold_forecasting.models.patchtst import PatchTSTForecaster
from gold_forecasting.models.tft import TFTForecaster
from gold_forecasting.models.xgboost_model import XGBoostForecaster
from gold_forecasting.models.xgboost_diff import XGBoostDiffForecaster

INDEX = pd.bdate_range("2023-01-01", periods=100)
SERIES = pd.Series(100 + np.sin(np.arange(100)/5) + np.arange(100)*.1, index=INDEX)
EXOG = pd.DataFrame({"e1": np.arange(100)*.2, "e2": np.cos(np.arange(100)/7)}, index=INDEX)
MULTI = pd.concat([SERIES.rename("target"), EXOG], axis=1)

@pytest.mark.parametrize("model,config", [
    (NaiveForecaster, {}),
    (MovingAverageForecaster, {"window": 5}),
])
def test_baseline_forecast_window(model, config):
    instance = model(config=config)
    prediction = instance.forecast_window(SERIES.iloc[:80], 3)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()

@pytest.mark.parametrize("retrain_each_step", [True, False])
def test_sarima_forecast_window_across_rolling_calls(retrain_each_step):
    set_seed(42)
    instance = SarimaForecaster(config={"order": [1, 1, 0], "seasonal_period": 0, "retrain_each_step": retrain_each_step})
    first = instance.forecast_window(SERIES.iloc[:80], 3)
    second = instance.forecast_window(SERIES.iloc[:90], 3)  # origin has advanced with real data
    assert first.shape == (3,) and second.shape == (3,)
    assert np.isfinite(first).all() and np.isfinite(second).all()

@pytest.mark.parametrize("retrain_each_step", [True, False])
def test_sarimax_forecast_window_across_rolling_calls(retrain_each_step):
    set_seed(42)
    instance = SarimaxForecaster(config={"order": [1, 1, 0], "seasonal_period": 0, "retrain_each_step": retrain_each_step})
    first = instance.forecast_window(MULTI.iloc[:80], 3, future_exogenous=EXOG.iloc[80:83].values)
    second = instance.forecast_window(MULTI.iloc[:90], 3, future_exogenous=EXOG.iloc[90:93].values)
    assert first.shape == (3,) and second.shape == (3,)
    assert np.isfinite(first).all() and np.isfinite(second).all()

@pytest.mark.parametrize("retrain_each_step", [True, False])
def test_xgboost_forecast_window_across_rolling_calls(retrain_each_step):
    set_seed(42)
    feature_config = {"lags": [1, 2, 5], "rolling_windows": [5], "include_calendar": True, "exogenous_lag": 1}
    instance = XGBoostForecaster(config={"params": {"max_depth": 3, "learning_rate": .2, "n_estimators": 30, "subsample": 1.0, "colsample_bytree": 1.0}, "retrain_each_step": retrain_each_step}, feature_config=feature_config)
    first = instance.forecast_window(MULTI.iloc[:80], 3, future_exogenous=EXOG.iloc[80:83].values)
    second = instance.forecast_window(MULTI.iloc[:90], 3, future_exogenous=EXOG.iloc[90:93].values)
    assert first.shape == (3,) and second.shape == (3,)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    # each forecast_window call refits (retrain_each_step) or reuses a cached fit (not); either way at least
    # one fit already ran, so 30 (n_estimators) train/validation loss values must have been recorded.
    expected_calls = 2 if retrain_each_step else 1
    assert len(instance.loss_history["train"]) == 30 * expected_calls
    assert len(instance.loss_history["validation"]) == len(instance.loss_history["train"])

@pytest.mark.parametrize("retrain_each_step", [True, False])
def test_xgboost_diff_forecast_window_across_rolling_calls(retrain_each_step):
    set_seed(42)
    feature_config = {"lags": [1, 2, 5], "rolling_windows": [5], "include_calendar": True, "exogenous_lag": 1}
    instance = XGBoostDiffForecaster(config={"params": {"max_depth": 3, "learning_rate": .2, "n_estimators": 30, "subsample": 1.0, "colsample_bytree": 1.0}, "retrain_each_step": retrain_each_step}, feature_config=feature_config)
    first = instance.forecast_window(MULTI.iloc[:80], 3, future_exogenous=EXOG.iloc[80:83].values)
    second = instance.forecast_window(MULTI.iloc[:90], 3, future_exogenous=EXOG.iloc[90:93].values)
    assert first.shape == (3,) and second.shape == (3,)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    # reconstructed absolute forecasts should stay in the same ballpark as the series, not diverge wildly
    assert np.all(np.abs(first - SERIES.iloc[80]) < 50) and np.all(np.abs(second - SERIES.iloc[90]) < 50)

def test_patchtst_forecast_window_warm_start():
    set_seed(42)
    config = {"context_length": 16, "patch_length": 4, "stride": 4, "d_model": 8, "nhead": 2, "layers": 1,
              "learning_rate": .01, "batch_size": 16, "epochs": 2, "patience": 1, "retrain_each_step": True, "update_epochs": 1}
    instance = PatchTSTForecaster(config=config, device=select_device(), seed=42)
    first = instance.forecast_window(SERIES.iloc[:80].values, 3)
    assert instance._initialized
    second = instance.forecast_window(SERIES.iloc[:90].values, 3)  # warm-start continuation, not a fresh fit
    assert first.shape == (3,) and second.shape == (3,)
    assert np.isfinite(first).all() and np.isfinite(second).all()

def test_tft_forecast_window_warm_start():
    set_seed(42)
    config = {"context_length": 16, "hidden_size": 8, "attention_heads": 2, "lstm_layers": 1, "dropout": 0.0,
              "learning_rate": .01, "batch_size": 16, "epochs": 2, "patience": 1, "retrain_each_step": True, "update_epochs": 1}
    instance = TFTForecaster(config=config, device=select_device(), seed=42, n_features=MULTI.shape[1])
    first = instance.forecast_window(MULTI.iloc[:80], 3, future_exogenous=EXOG.iloc[80:83].values)
    second = instance.forecast_window(MULTI.iloc[:90], 3, future_exogenous=EXOG.iloc[90:93].values)
    assert first.shape == (3,) and second.shape == (3,)
    assert np.isfinite(first).all() and np.isfinite(second).all()

def test_chronos_zero_shot_forecast_window():
    """Downloads amazon/chronos-bolt-small on first run; cached locally by huggingface_hub afterwards."""
    from gold_forecasting.models.chronos_zero_shot import ChronosForecaster
    instance = ChronosForecaster(config={"model_id": "amazon/chronos-bolt-small", "context_length": 64}, device="cpu")
    prediction = instance.forecast_window(SERIES.iloc[:90].values, 3)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()

def test_chronos2_forecast_window_with_covariates():
    """Downloads amazon/chronos-2 on first run; cached locally by huggingface_hub afterwards. Unlike the other
    Chronos variants above, Chronos-2 takes past/future covariates directly, so this exercises that path with MULTI/EXOG."""
    from gold_forecasting.models.chronos2 import Chronos2Forecaster
    instance = Chronos2Forecaster(config={"model_id": "amazon/chronos-2", "context_length": 64}, device="cpu")
    prediction = instance.forecast_window(MULTI.iloc[:90], 3, future_exogenous=EXOG.iloc[90:93].values)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()
