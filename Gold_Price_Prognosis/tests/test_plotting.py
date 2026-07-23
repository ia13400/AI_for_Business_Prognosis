import pandas as pd
import gold_forecasting.plotting as plotting
from gold_forecasting.plotting import plot_trading_bot

def test_plot_trading_bot_saves_png(tmp_path, monkeypatch):
    # `plotting.py` did `from .paths import FIGURES`, which copies a reference into its own
    # module namespace -- patch that name directly, not `gold_forecasting.paths.FIGURES`.
    monkeypatch.setattr(plotting, "FIGURES", tmp_path)
    bot = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=4),
        "actual": [100.0, 101.0, 99.0, 102.0],
        "portfolio_value": [10_000.0, 10_100.0, 10_100.0, 9_900.0],
        "position": ["cash", "gold", "gold", "cash"],
        "action": ["hold", "buy", "hold", "sell"],
    })
    path = plot_trading_bot(bot, "testmodel", "abcdef1234567890")
    assert path.exists()
    assert path.name == "trading_testmodel_abcdef123456.png"
