import numpy as np, pandas as pd
from gold_forecasting.splitting import chronological_split
from gold_forecasting.rolling import rolling_forecast
from gold_forecasting.models.naive import NaiveForecaster
from gold_forecasting.metrics import rolling_metrics

def test_quick_pipeline_without_network():
    frame=pd.DataFrame({"gold_usd":np.linspace(100,150,3000)},index=pd.bdate_range("2015-01-01",periods=3000))
    split=chronological_split(frame,validation_years=1,test_years=1)
    combined=pd.concat([split.train,split.validation,split.test])
    test_start=len(split.train)+len(split.validation)
    model=NaiveForecaster()
    result=rolling_forecast(model,combined,test_start,len(combined),horizon=20,step=20)
    metrics=rolling_metrics(result,split.train.gold_usd,lead_time_checkpoints=(1,10,20))
    assert set(["mae","rmse","mase","smape","directional_accuracy"]).issubset(metrics.columns)
    assert set(metrics["horizon"]) == {1,10,20,"complete"}
