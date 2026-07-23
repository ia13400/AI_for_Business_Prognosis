import pandas as pd
import pytest
from gold_forecasting.trading import simulate_trading_bot, simulate_cheater_bot

def _rolling_result(dates, steps, lead_times, actuals, end_of_step_predicted):
    """Build a synthetic rolling_result: `end_of_step_predicted` gives the predicted value
    for the (only) row per step where lead_time == max(lead_times for that step); every other
    row gets a throwaway predicted value (0), since only the end-of-step row drives decisions."""
    horizon = max(lead_times)
    predicted = [end_of_step_predicted[s] if lt == horizon else 0.0 for s, lt in zip(steps, lead_times)]
    return pd.DataFrame({"step": steps, "origin": dates, "lead_time": lead_times, "date": dates,
                          "actual": actuals, "predicted": predicted})

def test_simulate_trading_bot_decides_once_per_step_and_marks_to_market_daily():
    # 3 steps of 3 days each (horizon=3). Step 0: end-of-step forecast 108 vs prior_actual 100
    # -> predicted_return=8 > threshold(5) -> buy. Step 1: forecast 100 vs origin 103 -> -3,
    # inside the +-5 band -> hold (stays in gold). Step 2: forecast 95 vs origin 106 -> -11 <
    # -threshold -> sell. Only the lead_time==horizon row of each step should matter.
    dates = pd.bdate_range("2024-01-01", periods=9)
    steps = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    lead_times = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    actuals = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    end_of_step_predicted = {0: 108, 1: 100, 2: 95}
    rolling_result = _rolling_result(dates, steps, lead_times, actuals, end_of_step_predicted)

    bot = simulate_trading_bot(rolling_result, prior_actual=100.0, horizon=3, threshold=5.0, starting_capital=10_000.0)

    assert list(bot["action"]) == ["buy", "hold", "hold", "hold", "hold", "hold", "sell", "hold", "hold"]
    assert list(bot["position"]) == ["gold"] * 6 + ["cash"] * 3
    assert list(bot["portfolio_value"]) == pytest.approx([10100, 10200, 10300, 10400, 10500, 10600, 10600, 10600, 10600])

def test_simulate_trading_bot_ignores_non_end_of_step_forecasts():
    # A single 4-day step whose non-final lead_time rows carry huge, wildly wrong predicted
    # values -- if the bot used any of them, it would (wrongly) buy or sell mid-step. Only the
    # lead_time == horizon (4) row's predicted value (100.0) should drive the decision.
    dates = pd.bdate_range("2024-01-01", periods=4)
    frame = pd.DataFrame({"step": [0, 0, 0, 0], "origin": dates, "lead_time": [1, 2, 3, 4], "date": dates,
                           "actual": [101, 102, 103, 104], "predicted": [99999, -99999, 99999, 100.0]})
    bot = simulate_trading_bot(frame, prior_actual=100.0, horizon=4, threshold=5.0, starting_capital=10_000.0)
    # predicted_return = 100.0 - 100.0 = 0 -> within the +-5 band -> hold, stay cash throughout
    assert list(bot["action"]) == ["hold", "hold", "hold", "hold"]
    assert list(bot["position"]) == ["cash"] * 4
    assert list(bot["portfolio_value"]) == pytest.approx([10_000.0] * 4)

def test_simulate_trading_bot_threshold_is_strict_not_inclusive():
    dates = pd.bdate_range("2024-01-01", periods=2)
    # step 0: predicted_return = 108 - 100 = 8 > 5 -> buy (clean signal, gets us into gold).
    # step 1: predicted_return = 96 - 101 = -5 exactly -> must NOT sell (strict '<', not '<=').
    frame = pd.DataFrame({"step": [0, 1], "origin": dates, "lead_time": [1, 1], "date": dates,
                           "actual": [101, 102], "predicted": [108, 96]})
    bot = simulate_trading_bot(frame, prior_actual=100.0, horizon=1, threshold=5.0)
    assert list(bot["action"]) == ["buy", "hold"]
    assert list(bot["position"]) == ["gold", "gold"]  # exact -5 does not trigger a sell

def test_simulate_trading_bot_cash_exact_threshold_does_not_buy():
    dates = pd.bdate_range("2024-01-01", periods=1)
    frame = pd.DataFrame({"step": [0], "origin": dates, "lead_time": [1], "date": dates, "actual": [101], "predicted": [105]})
    bot = simulate_trading_bot(frame, prior_actual=100.0, horizon=1, threshold=5.0)  # predicted_return == 5 exactly
    assert list(bot["action"]) == ["hold"]
    assert list(bot["position"]) == ["cash"]
    assert bot["portfolio_value"].iloc[0] == pytest.approx(10_000.0)

def test_simulate_trading_bot_raises_when_a_step_is_missing_its_end_of_step_row():
    dates = pd.bdate_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"step": [0, 1], "origin": dates, "lead_time": [3, 2], "date": dates,  # step 1 never reaches lead_time==3
                           "actual": [101, 102], "predicted": [108, 100]})
    with pytest.raises(ValueError):
        simulate_trading_bot(frame, prior_actual=100.0, horizon=3)

def test_simulate_cheater_bot_buys_every_bottom_sells_every_top():
    actual = pd.Series([100, 110, 100, 125, 100], index=pd.bdate_range("2024-01-01", periods=5))
    bot = simulate_cheater_bot(actual, starting_capital=10_000.0)
    assert list(bot["action"]) == ["buy", "sell", "buy", "sell", "hold"]
    assert list(bot["position"]) == ["gold", "cash", "gold", "cash", "cash"]
    assert list(bot["portfolio_value"]) == pytest.approx([10_000, 11_000, 11_000, 13_750, 13_750])

def test_simulate_cheater_bot_portfolio_value_never_decreases():
    actual = pd.Series([100, 90, 95, 80, 130], index=pd.bdate_range("2024-01-01", periods=5))
    bot = simulate_cheater_bot(actual, starting_capital=10_000.0)
    assert (bot["portfolio_value"].diff().dropna() >= -1e-9).all()
