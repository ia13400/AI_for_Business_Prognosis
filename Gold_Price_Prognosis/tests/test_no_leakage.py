import pandas as pd
from gold_forecasting.feature_engineering import create_features
from gold_forecasting.splitting import historical_split
def test_features_do_not_see_current_or_future_target():
    index=pd.bdate_range("2024-01-01",periods=50); base=pd.Series(range(50),index=index,dtype=float); changed=base.copy(); changed.iloc[30:]=9999
    left=create_features(base,[1,2],[5]); right=create_features(changed,[1,2],[5]); pd.testing.assert_series_equal(left.iloc[30].drop("target"),right.iloc[30].drop("target"))
def test_holdout_target_is_strictly_hidden():
    frame=pd.DataFrame({"gold_usd":range(300)},index=pd.bdate_range("2024-06-01",periods=300)); split=historical_split(frame,"2025-06-01"); assert (split.train.index<"2025-06-01").all() and (split.validation.index<"2025-06-01").all() and (split.test.index>="2025-06-01").all()
