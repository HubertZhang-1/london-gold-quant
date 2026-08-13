# -*- coding: utf-8 -*-
"""Self checks for the London gold toolkit (no network required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from london_gold.backtest import CostConfig, run_backtest
from london_gold.indicators import atr, donchian, rsi
from london_gold.intraday_strategy import open_range_breakout_signals
from london_gold.report import write_equity_svg
from london_gold.strategies import (
    donchian_breakout_signals,
    ema_cross_signals,
    rsi_reversal_signals,
)


def synthetic_frame(n=320, seed=7):
    t = np.arange(n)
    up = 1800 + 3.5 * t[: n // 2]
    peak = up[-1]
    down = peak - 2.0 * (t[n // 2 :] - n // 2)
    trend = np.concatenate([up, down])
    close = trend.copy()
    close[95:98] -= 150.0
    close[260:263] += 90.0
    high = close + 2.0
    low = close - 2.0
    open_ = np.concatenate([[close[0]], close[:-1]])
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"ok - {message}")


def main():
    df = synthetic_frame()
    atr14 = atr(df["high"], df["low"], df["close"], 14)
    check(atr14.dropna().gt(0).all(), "ATR is positive")

    r = rsi(df["close"], 14).dropna()
    check(r.between(0, 100).all(), "RSI stays in [0, 100]")

    upper, lower = donchian(df["high"], df["low"], 20)
    check(upper.dropna().ge(lower.dropna()).all(), "Donchian upper >= lower")

    don = donchian_breakout_signals(df, entry_n=20, exit_n=10, ma_filter=0, stop_mult=3.0)
    check(don["signal"].isin([-1, 0, 1]).all(), "Donchian signal values valid")
    check(don["signal"].abs().sum() > 0, "Donchian generates entries")

    ema = ema_cross_signals(df, fast_n=10, slow_n=40, stop_mult=3.0)
    check(set(ema["signal"].unique()) <= {-1, 0, 1} and ema["signal"].abs().sum() > 0, "EMA cross generates long/short signals")

    rsi_sig = rsi_reversal_signals(df, rsi_n=14, oversold=30, overbought=70, ma_filter=100, stop_mult=2.5)
    check(rsi_sig["signal"].abs().sum() > 0, "RSI reversal generates entries")

    flat = df.copy()
    flat["signal"] = 0
    flat["stop_dist"] = 0.0
    cost = CostConfig(capital=100000, position_oz=10, spread=0.35, slippage=0.1, commission_per_oz=0.1)
    flat_result = run_backtest(flat, cost=cost)
    check(flat_result["stats"]["trade_count"] == 0, "flat signal produces no trades")
    check(abs(flat_result["stats"]["final_equity"] - 100000) < 1, "flat signal keeps capital")

    trend = df.copy()
    mid = len(trend) // 2
    trend["signal"] = 0
    trend.loc[:mid, "signal"] = 1
    trend.loc[mid + 1 :, "signal"] = -1
    trend["stop_dist"] = 0.0
    with_cost = run_backtest(trend, cost=cost)
    no_cost_cfg = CostConfig(capital=100000, position_oz=10, spread=0, slippage=0, commission_per_oz=0)
    no_cost = run_backtest(trend, cost=no_cost_cfg)
    check(with_cost["stats"]["trade_count"] >= 2, "trend signal generates trades")
    check(no_cost["stats"]["total_return"] >= with_cost["stats"]["total_return"], "costs reduce backtest return")

    rising = df.iloc[:20].copy()
    rising["signal"] = 1
    rising["stop_dist"] = 0.0
    rising_result = run_backtest(rising, cost=no_cost_cfg)
    check(rising_result["stats"]["total_return"] > 0, "long position profits when price rises")
    check(rising_result["equity"][len(rising) // 2] > 100000, "long position marks equity above capital in a rally")

    falling = df.iloc[-20:].copy()
    falling["signal"] = 1
    falling["stop_dist"] = 0.0
    falling_result = run_backtest(falling, cost=no_cost_cfg)
    check(falling_result["stats"]["total_return"] < 0, "long position loses when price falls")

    short_rising = df.iloc[:20].copy()
    short_rising["signal"] = -1
    short_rising["stop_dist"] = 0.0
    short_rising_result = run_backtest(short_rising, cost=no_cost_cfg)
    check(short_rising_result["stats"]["total_return"] < 0, "short position loses when price rises")
    check(short_rising_result["equity"][len(short_rising) // 2] < 100000, "short position marks equity below capital in a rally")

    lev_cfg = CostConfig(capital=100000, position_oz=10, spread=0, slippage=0, commission_per_oz=0, leverage=50, margin_call_pct=0.5)
    leverage_test = df.iloc[-40:].copy()
    leverage_test["signal"] = 1
    leverage_test["stop_dist"] = 0.0
    leveraged = run_backtest(leverage_test, cost=lev_cfg)
    check(leveraged["trades"]["exit_reason"].eq("margin_call").any(), "50x long is margin-called on a fall")
    check(leveraged["stats"]["final_equity"] <= 50001, "margin call caps the loss near the configured threshold")

    hourly = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=720, freq="h"),
            "open": 4000 + 30 * np.sin(np.arange(720) / 24.0),
            "high": 4000 + 30 * np.sin(np.arange(720) / 24.0) + 5,
            "low": 4000 + 30 * np.sin(np.arange(720) / 24.0) - 5,
            "close": 4000 + 30 * np.sin(np.arange(720) / 24.0),
        }
    )
    orb = open_range_breakout_signals(hourly, range_hours=3, ma_filter=24, stop_mult=1.5)
    check(orb["signal"].abs().sum() > 0, "open-range breakout generates hourly signals")

    svg_path = Path("data") / "test_london_gold_equity.svg"
    svg_path.parent.mkdir(exist_ok=True)
    write_equity_svg([("test", trend["date"].tolist(), with_cost["equity"])], svg_path)
    check(svg_path.exists() and "<svg" in svg_path.read_text(encoding="utf-8"), "SVG equity chart written")

    print("\nall London gold self checks passed")


if __name__ == "__main__":
    main()
