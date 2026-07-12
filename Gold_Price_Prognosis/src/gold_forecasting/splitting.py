"""Time ordered, cutoff-safe splits."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

def historical_split(frame: pd.DataFrame, cutoff: str, validation_fraction: float = 0.15) -> TimeSplit:
    cutoff_ts = pd.Timestamp(cutoff); development = frame.loc[frame.index < cutoff_ts]; test = frame.loc[frame.index >= cutoff_ts]
    if development.empty or test.empty: raise ValueError("Cutoff must leave non-empty development and test periods")
    validation_size = max(1, int(len(development) * validation_fraction)); train = development.iloc[:-validation_size]; validation = development.iloc[-validation_size:]
    if train.index.max() >= validation.index.min() or validation.index.max() >= test.index.min(): raise AssertionError("Temporal split overlap")
    return TimeSplit(train, validation, test)
