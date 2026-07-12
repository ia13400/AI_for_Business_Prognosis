import numpy as np,pandas as pd
from gold_forecasting.splitting import historical_split
from gold_forecasting.models.naive import NaiveForecaster
from gold_forecasting.metrics import horizon_metrics
def test_quick_pipeline_without_network():
    frame=pd.DataFrame({"gold_usd":np.linspace(100,150,300)},index=pd.bdate_range("2024-06-01",periods=300)); split=historical_split(frame,"2025-06-01"); model=NaiveForecaster().fit(split.train.gold_usd); prediction=model.predict(split.train.gold_usd,len(split.test)); result=pd.DataFrame({"actual":split.test.gold_usd,"predicted":prediction}); assert set(["mae","rmse","mase","smape","directional_accuracy"]).issubset(horizon_metrics(result,split.train.gold_usd).columns)
