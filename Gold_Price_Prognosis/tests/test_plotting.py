import pandas as pd
import gold_forecasting.plotting as plotting
from gold_forecasting.plotting import plot_trading_bot, plot_pnl_bar

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

def test_plot_pnl_bar_saves_png_named_distinctly_from_per_bot_files(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "FIGURES", tmp_path)
    summary = pd.DataFrame({"model": ["winner", "loser", "flat"], "final_value": [15_000.0, 8_000.0, 10_000.0]})
    path = plot_pnl_bar(summary, starting_capital=10_000.0, signature="abcdef1234567890")
    assert path.exists()
    assert path.name == "pnl_summary_abcdef123456.png"
    assert not path.name.startswith("trading_")  # must not collide with the trading_<model>_*.png glob

def test_plot_trading_bot_with_predicted_column_saves_png(tmp_path, monkeypatch):
    # Real models' `simulate_trading_bot` output has a 'predicted' column; the cheater bot's
    # `simulate_cheater_bot` output does not -- both must render without error.
    monkeypatch.setattr(plotting, "FIGURES", tmp_path)
    bot = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=4),
        "actual": [100.0, 101.0, 99.0, 102.0],
        "predicted": [99.0, 103.0, 98.0, 104.0],
        "portfolio_value": [10_000.0, 10_100.0, 10_100.0, 9_900.0],
        "position": ["cash", "gold", "gold", "cash"],
        "action": ["hold", "buy", "hold", "sell"],
    })
    path = plot_trading_bot(bot, "testmodel", "abcdef1234567890")
    assert path.exists()
