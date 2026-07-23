"""Backtest trading-bot simulation: pure functions, no caching/MLflow/plotting.

Both bots start 100% cash and are always either 100% cash or 100% gold (no
fees, no partial positions) -- see `simulate_trading_bot`'s decision table and
`simulate_cheater_bot`'s frictionless-foresight rule.
"""
import pandas as pd

def simulate_trading_bot(rolling_result: pd.DataFrame, prior_actual: float, horizon: int,
                          threshold: float = 5.0, starting_capital: float = 10_000.0) -> pd.DataFrame:
    """One row per test-period trading day: date, actual, predicted, step, origin_actual,
    end_of_step_forecast, predicted_return, position, action, portfolio_value.

    Decides once per rolling step -- at that step's origin (the last real day
    known before the step starts) -- using the model's own forecast for where
    this exact step will end (`predicted` at `lead_time == horizon`) against the
    real, already-known price at the origin. Never real future data. The
    resulting position is held for the entire step; days within a step are
    simply marked to market at the real `actual` price, with no new decision
    made until the next step's origin.

    Decision table (strict thresholds -- exactly +-threshold is a hold, not a trade):
      cash + predicted_return >  threshold -> buy all (at origin_actual)
      gold + predicted_return < -threshold -> sell all (at origin_actual)
      otherwise -> keep current position

    `prior_actual` seeds the first step's origin price (the last real
    validation-period price -- legitimately past data, not test-period leakage).
    Raises ValueError if any step is missing its `lead_time == horizon` row.
    """
    frame = rolling_result.sort_values(["step", "date"]).reset_index(drop=True).copy()
    end_of_step = frame.loc[frame["lead_time"] == horizon].set_index("step")["predicted"]
    missing = set(frame["step"].unique()) - set(end_of_step.index)
    if missing:
        raise ValueError(f"simulate_trading_bot requires every step to have a lead_time == horizon row; missing for steps {sorted(missing)}")
    frame["end_of_step_forecast"] = frame["step"].map(end_of_step)

    cash, shares, position = starting_capital, 0.0, "cash"
    last_actual = prior_actual
    current_step, origin_actual, predicted_return = None, None, None
    records = []
    for row in frame.itertuples(index=False):
        action = "hold"
        if row.step != current_step:
            current_step = row.step
            origin_actual = last_actual
            predicted_return = row.end_of_step_forecast - origin_actual
            if position == "cash" and predicted_return > threshold:
                shares, cash, position, action = cash / origin_actual, 0.0, "gold", "buy"
            elif position == "gold" and predicted_return < -threshold:
                cash, shares, position, action = shares * origin_actual, 0.0, "cash", "sell"
        records.append({"date": row.date, "actual": row.actual, "predicted": row.predicted, "step": row.step,
                         "origin_actual": origin_actual, "end_of_step_forecast": row.end_of_step_forecast,
                         "predicted_return": predicted_return, "position": position, "action": action,
                         "portfolio_value": cash + shares * row.actual})
        last_actual = row.actual
    return pd.DataFrame.from_records(records)

def simulate_cheater_bot(actual: pd.Series, starting_capital: float = 10_000.0) -> pd.DataFrame:
    """Perfect-foresight upper-bound bot: one row per date, decided fresh every
    day (unlike `simulate_trading_bot`, which only decides once per rolling
    step) since it has full access to real future prices.

    No transaction costs are modeled anywhere in this project's trading rules
    (the threshold in `simulate_trading_bot` *is* the stated cost-awareness
    mechanism), so "buy every bottom, sell every top" reduces to the classic
    frictionless-multiple-transactions optimum: hold gold on day d if
    tomorrow's real price is higher, else hold cash -- computed directly from
    the real price series, no model or threshold involved.
    """
    actual = actual.sort_index()
    values = actual.to_numpy()
    cash, shares, position = starting_capital, 0.0, "cash"
    records = []
    for i, (date, price) in enumerate(actual.items()):
        rising = i + 1 < len(values) and values[i + 1] > price
        action = "hold"
        if rising and position == "cash":
            shares, cash, position, action = cash / price, 0.0, "gold", "buy"
        elif not rising and position == "gold":
            cash, shares, position, action = shares * price, 0.0, "cash", "sell"
        records.append({"date": date, "actual": price, "position": position, "action": action,
                         "portfolio_value": cash + shares * price})
    return pd.DataFrame.from_records(records)
