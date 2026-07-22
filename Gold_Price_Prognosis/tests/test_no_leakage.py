import pandas as pd
from gold_forecasting.feature_engineering import create_features
from gold_forecasting.splitting import chronological_split

def test_features_do_not_see_current_or_future_target():
    index=pd.bdate_range("2024-01-01",periods=50); base=pd.Series(range(50),index=index,dtype=float); changed=base.copy(); changed.iloc[30:]=9999
    left=create_features(base,[1,2],[5]); right=create_features(changed,[1,2],[5]); pd.testing.assert_series_equal(left.iloc[30].drop("target"),right.iloc[30].drop("target"))

def test_exogenous_features_do_not_see_future_exogenous():
    index=pd.bdate_range("2024-01-01",periods=50)
    target=pd.Series(range(50),index=index,dtype=float)
    exog=pd.DataFrame({"e":range(50)},index=index,dtype=float)
    changed_exog=exog.copy(); changed_exog.iloc[31:]=9999
    left=create_features(target,[1,2],[5],exogenous=exog,exogenous_lag=1)
    right=create_features(target,[1,2],[5],exogenous=changed_exog,exogenous_lag=1)
    pd.testing.assert_series_equal(left.iloc[30].drop("target"),right.iloc[30].drop("target"))

def test_split_test_partition_is_strictly_hidden_from_train_and_validation():
    frame=pd.DataFrame({"gold_usd":range(3000)},index=pd.bdate_range("2015-01-01",periods=3000))
    split=chronological_split(frame,validation_years=1,test_years=1)
    assert (split.train.index<split.test.index.min()).all()
    assert (split.validation.index<split.test.index.min()).all()
    assert (split.test.index>=split.test.index.min()).all()
