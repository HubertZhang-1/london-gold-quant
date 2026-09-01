# -*- coding: utf-8 -*-
"""Tests for the 15-minute gold momentum-pullback backtest."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from london_gold.m15_pullback import M15StrategyConfig, momentum_pullback_signals
from london_gold.m15_execution import M15ExecutionConfig, run_m15_backtest
from scripts.london_gold_m15_pullback_backtest import prepare_period


def synthetic_m15(start: str, closes: list[float]) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(close), freq="15min", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.10,
            "low": np.minimum(open_, close) - 0.10,
            "close": close,
        }
    )


def relaxed_strategy_config() -> M15StrategyConfig:
    return M15StrategyConfig(
        ema_bars=3,
        fast_bars=1,
        slow_bars=4,
        breakout_bars=3,
        pullback_bars=3,
        atr_bars=3,
        min_body_ratio=0.20,
        max_breakout_atr=10.0,
        pullback_tolerance_atr=0.50,
        min_stop_atr=0.10,
        max_stop_atr=5.0,
    )


class M15SignalTests(unittest.TestCase):
    def test_overlap_handles_august_daylight_saving_time(self):
        bars = synthetic_m15("2026-08-03 11:45", [100.0] * 20)

        result = momentum_pullback_signals(bars, relaxed_strategy_config())

        at_noon = result.loc[result["date"] == pd.Timestamp("2026-08-03 12:00", tz="UTC")]
        at_close = result.loc[result["date"] == pd.Timestamp("2026-08-03 16:00", tz="UTC")]
        self.assertTrue(bool(at_noon["session_open"].item()))
        self.assertFalse(bool(at_close["session_open"].item()))

    def test_breakout_channel_excludes_current_bar(self):
        bars = synthetic_m15(
            "2026-08-03 10:30",
            [99.6, 99.8, 100.0, 100.1, 100.2, 100.3, 101.5, 101.3, 101.7],
        )

        result = momentum_pullback_signals(bars, relaxed_strategy_config())

        breakout_index = 6
        expected = bars["high"].iloc[3:6].max()
        self.assertAlmostEqual(float(result.loc[breakout_index, "breakout_high"]), float(expected))
        self.assertLess(float(result.loc[breakout_index, "breakout_high"]), float(bars.loc[breakout_index, "high"]))

    def test_pullback_confirmation_emits_close_signal(self):
        bars = synthetic_m15(
            "2026-08-03 10:30",
            [99.6, 99.8, 100.0, 100.1, 100.2, 100.3, 101.5, 100.45, 101.0, 101.2],
        )
        bars.loc[7, "low"] = 100.25

        result = momentum_pullback_signals(bars, relaxed_strategy_config())

        self.assertEqual(int(result.loc[6, "signal"]), 0)
        self.assertEqual(int(result.loc[8, "signal"]), 1)
        self.assertGreater(float(result.loc[8, "stop_dist"]), 0.0)
        self.assertEqual(result.loc[8, "setup_state"], "confirmed_long")

    def test_setup_expires_when_no_pullback_arrives(self):
        bars = synthetic_m15(
            "2026-08-03 10:30",
            [99.6, 99.8, 100.0, 100.1, 100.2, 100.3, 101.5, 101.7, 101.9, 102.1, 102.3],
        )

        result = momentum_pullback_signals(bars, relaxed_strategy_config())

        self.assertEqual(result.loc[10, "setup_state"], "idle")
        self.assertEqual(int(result["signal"].sum()), 0)


def execution_fixture() -> pd.DataFrame:
    bars = synthetic_m15(
        "2026-08-03 12:00",
        [100.0, 100.1, 100.2, 100.8, 101.3, 101.5, 101.4, 101.4],
    )
    bars["signal"] = 0
    bars.loc[1, "signal"] = 1
    bars["stop_dist"] = 1.0
    bars["atr"] = 0.5
    bars["session_open"] = [True, True, True, True, True, True, False, False]
    return bars


def loss_limit_fixture() -> pd.DataFrame:
    bars = synthetic_m15("2026-08-03 12:00", [100.0] * 10)
    bars["signal"] = 0
    bars.loc[[0, 2, 4, 6], "signal"] = 1
    bars["stop_dist"] = 1.0
    bars["atr"] = 0.5
    bars["session_open"] = True
    bars.loc[[2, 4, 6], "open"] = 98.0
    bars.loc[[2, 4, 6], "low"] = 98.0
    return bars


class M15ExecutionTests(unittest.TestCase):
    def test_signal_fills_on_next_bar_open(self):
        bars = execution_fixture()

        result = run_m15_backtest(bars, M15ExecutionConfig(max_hold_bars=3))

        self.assertEqual(result["trades"].iloc[0]["entry_time"], bars.iloc[2]["date"])

    def test_position_risks_no_more_than_half_percent(self):
        result = run_m15_backtest(execution_fixture(), M15ExecutionConfig(capital=100_000.0))

        trade = result["trades"].iloc[0]
        self.assertLessEqual(float(trade["initial_risk"]), 500.01)
        self.assertGreater(float(trade["position_oz"]), 0.0)

    def test_daily_loss_limit_blocks_fourth_entry(self):
        config = M15ExecutionConfig(
            capital=10_000.0,
            risk_per_trade=0.005,
            daily_loss_limit=0.015,
            max_daily_trades=10,
            spread=0.0,
            slippage=0.0,
            commission_per_oz=0.0,
        )

        result = run_m15_backtest(loss_limit_fixture(), config)

        self.assertEqual(len(result["trades"]), 2)
        self.assertTrue((result["skipped"]["reason"] == "daily_loss_limit").any())

    def test_position_closes_when_overlap_ends(self):
        result = run_m15_backtest(execution_fixture(), M15ExecutionConfig(max_hold_bars=12))

        trade = result["trades"].iloc[0]
        self.assertEqual(trade["exit_reason"], "session_end")
        self.assertEqual(trade["exit_time"], execution_fixture().iloc[6]["date"])


class M15CliTests(unittest.TestCase):
    def test_august_filter_is_inclusive_and_keeps_warmup(self):
        bars = synthetic_m15("2026-07-31 12:00", [100.0] * 100)

        prepared = prepare_period(bars, "2026-08-01", "2026-08-31", warmup_bars=16)

        evaluation = prepared["evaluation"]
        self.assertEqual(evaluation["date"].min(), pd.Timestamp("2026-08-01 00:00", tz="UTC"))
        self.assertLess(evaluation["date"].max(), pd.Timestamp("2026-09-01", tz="UTC"))
        self.assertGreater(len(prepared["input"]), len(evaluation))
        self.assertLess(prepared["input"]["date"].min(), evaluation["date"].min())


if __name__ == "__main__":
    unittest.main()
