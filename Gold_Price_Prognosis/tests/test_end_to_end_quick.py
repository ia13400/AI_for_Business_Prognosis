import numpy as np, pandas as pd
from gold_forecasting.splitting import chronological_split
from gold_forecasting.models.naive import NaiveForecaster
from gold_forecasting.metrics import horizon_metrics

def test_quick_pipeline_without_network():
    frame=pd.DataFrame({"gold_usd":np.linspace(100,150,3000)},index=pd.bdate_range("2015-01-01",periods=3000))
    split=chronological_split(frame,validation_years=1,test_years=1)
    model=NaiveForecaster().fit(split.train.gold_usd)
    history=pd.concat([split.train.gold_usd,split.validation.gold_usd])
    prediction=model.predict(history,len(split.test))
    result=pd.DataFrame({"actual":split.test.gold_usd,"predicted":prediction})
    assert set(["mae","rmse","mase","smape","directional_accuracy"]).issubset(horizon_metrics(result,split.train.gold_usd).columns)
