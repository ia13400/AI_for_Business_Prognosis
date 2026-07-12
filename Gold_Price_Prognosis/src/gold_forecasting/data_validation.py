"""Input data validation."""
import pandas as pd

def validate_market_data(
    frame: pd.DataFrame,
    value_column: str = "value",
    require_positive: bool = False,
) -> pd.DataFrame:
    """Validate a dated numeric market series.

    Positivity is optional because valid market series can contain zero or
    negative observations. WTI crude-oil futures, for example, traded below
    zero in April 2020. The gold target enables the stricter check explicitly.
    """
    if frame.empty: raise ValueError("Downloaded dataset is empty")
    result = frame.copy(); result.index = pd.to_datetime(result.index).tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if value_column not in result or not pd.api.types.is_numeric_dtype(result[value_column]): raise ValueError("Missing numeric value column")
    result = result.dropna(subset=[value_column])
    if require_positive and (result[value_column] <= 0).any():
        raise ValueError("The target price must be positive")
    return result
