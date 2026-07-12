import pandas as pd
from gold_forecasting.splitting import historical_split
def test_chronological_split():
    frame=pd.DataFrame({"gold_usd":range(100)},index=pd.bdate_range("2025-01-01",periods=100)); split=historical_split(frame,"2025-04-01",.2); assert split.train.index.max()<split.validation.index.min()<split.test.index.min()
