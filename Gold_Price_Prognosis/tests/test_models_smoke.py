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

INDEX = pd.bdate_range("2023-01-01", periods=100)
SERIES = pd.Series(100 + np.sin(np.arange(100)/5) + np.arange(100)*.1, index=INDEX)
EXOG = pd.DataFrame({"e1": np.arange(100)*.2, "e2": np.cos(np.arange(100)/7)}, index=INDEX)
MULTI = pd.concat([SERIES.rename("target"), EXOG], axis=1)

@pytest.mark.parametrize("model,config", [
    (NaiveForecaster, {}),
    (MovingAverageForecaster, {"window": 5}),
    (SarimaForecaster, {"seasonal_period": 0, "fallback": {"order": [1, 1, 0]}}),
])
def test_univariate_model_fit_predict(model, config):
    set_seed(42); instance = model(config=config, device=select_device(), seed=42)
    instance.fit(SERIES.iloc[:80], SERIES.iloc[80:90])
    prediction = instance.predict(SERIES.iloc[:90], 3)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()

def test_patchtst_fit_predict(tmp_path):
    set_seed(42)
    config = {"context_length": 16, "patch_length": 4, "stride": 4, "d_model": 8, "nhead": 2, "layers": 1, "learning_rate": .01, "batch_size": 16, "epochs": 1, "patience": 1}
    instance = PatchTSTForecaster(config=config, device=select_device(), seed=42)
    instance.fit(SERIES.iloc[:80].values, SERIES.iloc[80:90].values, tmp_path/"model.pt")
    prediction = instance.predict(SERIES.iloc[:90].values, 3)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()

def test_sarimax_fit_predict():
    set_seed(42)
    config = {"seasonal_period": 0, "fallback": {"order": [1, 1, 0]}}
    instance = SarimaxForecaster(config=config)
    instance.fit(MULTI.iloc[:80], MULTI.iloc[80:90])
    history = MULTI.iloc[:90]
    prediction = instance.predict(history, 3, future_exogenous=EXOG.iloc[90:93].values)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()

def test_xgboost_fit_predict():
    set_seed(42)
    feature_config = {"lags": [1, 2, 5], "rolling_windows": [5], "include_calendar": True, "exogenous_lag": 1}
    config = {"fallback": {"max_depth": 3, "learning_rate": .2, "n_estimators": 30, "subsample": 1.0, "colsample_bytree": 1.0}}
    instance = XGBoostForecaster(config=config, feature_config=feature_config)
    instance.fit(MULTI.iloc[:80], MULTI.iloc[80:90])
    history = MULTI.iloc[:90]
    prediction = instance.predict(history, 3, future_exogenous=EXOG.iloc[90:93].values)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()

def test_chronos_zero_shot_fit_predict():
    """Downloads amazon/chronos-bolt-small on first run; cached locally by huggingface_hub afterwards."""
    from gold_forecasting.models.chronos_zero_shot import ChronosForecaster
    instance = ChronosForecaster(config={"model_id": "amazon/chronos-bolt-small"}, device="cpu")
    instance.fit(SERIES.iloc[:80].values)
    prediction = instance.predict(SERIES.iloc[:90].values, 3)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()

def test_tft_fit_predict(tmp_path):
    set_seed(42)
    config = {"context_length": 16, "hidden_size": 8, "attention_heads": 2, "lstm_layers": 1, "dropout": 0.0, "learning_rate": .01, "batch_size": 16, "epochs": 1, "patience": 1}
    instance = TFTForecaster(config=config, device=select_device(), seed=42, n_features=MULTI.shape[1])
    instance.fit(MULTI.iloc[:80], MULTI.iloc[80:90], tmp_path/"tft.pt")
    history = MULTI.iloc[:90]
    prediction = instance.predict(history, 3, future_exogenous=EXOG.iloc[90:93].values)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()
