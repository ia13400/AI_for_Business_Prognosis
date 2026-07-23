"""Scientific display names for models/namespaces, and small chart-polish
helpers shared by both matplotlib (`plotting.py`) and Plotly
(`interactive_plots.py`) figure builders, and by the dashboard -- so a
model's name never renders two different ways across the project.

Underlying data (dict keys, DataFrame "model" columns, filenames) stays the
technical snake_case identifier everywhere; `display_name` is applied only at
the point a label actually gets rendered (trace name, axis category, title).
"""
import re

_MODEL_NAMES = {
    "naive": "Naive",
    "moving_average": "Moving Average",
    "sarima": "SARIMA",
    "sarimax": "SARIMAX",
    "patchtst": "PatchTST",
    "xgboost": "XGBoost",
    "xgboost_diff": "XGBoost (Returns)",
    "tft": "TFT",
    "chronos_original": "Chronos (Original)",
    "chronos_bolt": "Chronos (Bolt)",
    "chronos_t5_base": "Chronos (T5, Base)",
    "chronos_t5_large": "Chronos (T5, Large)",
    "chronos_bolt_base": "Chronos (Bolt, Base)",
    "cheater": "Cheater (Perfect Foresight)",
}

_SERIES_NAMES = {
    "gold_usd": "Gold",
    "dollar_index": "Dollar Index",
    "silver": "Silver",
    "oil": "Oil",
    "sp500": "S&P 500",
    "vix": "VIX",
    "bitcoin": "Bitcoin",
}

def display_name(key: str) -> str:
    """A model's scientific display name, e.g. `"patchtst"` -> `"PatchTST"`. Unknown keys pass through unchanged."""
    return _MODEL_NAMES.get(key, key)

def display_series(key: str) -> str:
    """A price-series column's display name, e.g. `"sp500"` -> `"S&P 500"`. Unknown keys pass through unchanged."""
    return _SERIES_NAMES.get(key, key)

def display_namespace(namespace: str) -> str:
    """A namespace/experiment label, e.g. `"univariate"` -> `"Univariate"`, `"all models"` -> `"All Models"`."""
    return namespace.title()

def strip_signature(stem: str) -> str:
    """Strip a trailing `_<12-hex-char signature>` from an artifact filename stem, e.g.
    `"sarima_32611aca0aa3"` -> `"sarima"`. Shared by the dashboard so the content-addressed
    cache signature never leaks into anything user-facing next to a model's name."""
    return re.sub(r"_[0-9a-f]{12}$", "", stem)
