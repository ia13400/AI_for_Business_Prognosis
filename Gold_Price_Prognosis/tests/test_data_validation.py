import pandas as pd
import pytest

from gold_forecasting.data_validation import validate_market_data


def test_valid_exogenous_series_may_contain_negative_values():
    frame = pd.DataFrame(
        {"value": [18.0, -37.63, 10.0]},
        index=pd.date_range("2020-04-19", periods=3),
    )
    result = validate_market_data(frame, require_positive=False)
    assert result.loc["2020-04-20", "value"] == pytest.approx(-37.63)


def test_gold_target_requires_positive_values():
    frame = pd.DataFrame(
        {"value": [1800.0, 0.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    with pytest.raises(ValueError, match="target price"):
        validate_market_data(frame, require_positive=True)
