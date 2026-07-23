from gold_forecasting.display import display_name, display_namespace, display_series, strip_signature

def test_display_name_maps_known_models():
    assert display_name("patchtst") == "PatchTST"
    assert display_name("sarimax") == "SARIMAX"
    assert display_name("xgboost_diff") == "XGBoost (Returns)"
    assert display_name("chronos_t5_large") == "Chronos (T5, Large)"

def test_display_name_passes_through_unknown_keys():
    assert display_name("some_future_model") == "some_future_model"

def test_display_namespace_title_cases():
    assert display_namespace("univariate") == "Univariate"
    assert display_namespace("all models") == "All Models"

def test_display_series_maps_known_columns():
    assert display_series("sp500") == "S&P 500"
    assert display_series("gold_usd") == "Gold"
    assert display_series("unknown_col") == "unknown_col"

def test_strip_signature_removes_trailing_hash_only():
    assert strip_signature("sarima_32611aca0aa3") == "sarima"
    assert strip_signature("chronos_t5_large_10190b4d9395") == "chronos_t5_large"
    assert strip_signature("naive") == "naive"  # no signature to strip
