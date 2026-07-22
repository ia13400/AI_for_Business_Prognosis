import pandas as pd
from gold_forecasting.splitting import chronological_split

def test_chronological_split_orders_train_validation_test():
    frame = pd.DataFrame({"gold_usd": range(3000)}, index=pd.bdate_range("2015-01-01", periods=3000))
    split = chronological_split(frame, validation_years=1, test_years=1)
    assert split.train.index.max() < split.validation.index.min() <= split.validation.index.max() < split.test.index.min()

def test_chronological_split_lengths_track_years():
    frame = pd.DataFrame({"gold_usd": range(4000)}, index=pd.bdate_range("2010-01-01", periods=4000))
    split = chronological_split(frame, validation_years=2.5, test_years=2.5)
    approx_business_days_per_year = 260.9  # Mon-Fri calendar days, no holiday removal (matches pd.bdate_range)
    assert abs(len(split.validation) - 2.5 * approx_business_days_per_year) < 10
    assert abs(len(split.test) - 2.5 * approx_business_days_per_year) < 10

def test_chronological_split_rejects_empty_partitions():
    frame = pd.DataFrame({"gold_usd": range(50)}, index=pd.bdate_range("2024-01-01", periods=50))
    try:
        chronological_split(frame, validation_years=2.5, test_years=2.5)
        assert False, "expected ValueError"
    except ValueError:
        pass
