"""Fixed-length chronological splits anchored to the series end."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

def _months(years: float) -> int:
    return round(years * 12)

def chronological_split(frame: pd.DataFrame, validation_years: float = 2.5, test_years: float = 2.5) -> TimeSplit:
    """Split strictly by time: train < validation < test.

    Validation and test are each exactly `*_years` long, measured backward
    from the series' last observation; everything earlier becomes train.
    """
    end = frame.index.max()
    test_start = end - pd.DateOffset(months=_months(test_years)) + pd.offsets.BDay(1)
    validation_start = test_start - pd.DateOffset(months=_months(validation_years))
    train = frame.loc[frame.index < validation_start]
    validation = frame.loc[(frame.index >= validation_start) & (frame.index < test_start)]
    test = frame.loc[frame.index >= test_start]
    if train.empty or validation.empty or test.empty: raise ValueError("Split configuration leaves an empty partition")
    if not (train.index.max() < validation.index.min() <= validation.index.max() < test.index.min()): raise AssertionError("Temporal split overlap")
    return TimeSplit(train, validation, test)
