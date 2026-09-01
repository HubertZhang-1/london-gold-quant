# -*- coding: utf-8 -*-
"""Tests for the bidirectional ATR-grid proxy backtester."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from london_gold.grid_replica_data import add_grid_step, normalize_ohlc, select_latest_days
from london_gold.grid_replica_engine import GridConfig, path_nodes, run_grid_backtest
from london_gold.grid_replica_report import compute_grid_stats, reconcile_result, write_grid_reports
from scripts.london_gold_grid_replica_backtest import assert_coverage, build_scenarios


def five_minute_bars(
    closes: list[float],
    start: str = "2026-06-01",
    freq: str = "5min",
) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(close), freq=freq, tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.20,
            "low": np.minimum(open_, close) - 0.20,
            "close": close,
        }
    )


def path_fixture(
    open_: float,
    high: float,
    low: float,
    close: float,
    grid_step: float = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-03 12:00", tz="UTC")],
            "open": [open_],
            "high": [high],
            "low": [low],
            "close": [close],
            "atr": [grid_step / 0.35],
            "grid_step": [grid_step],
        }
    )


def flat_fixture() -> pd.DataFrame:
    return path_fixture(100.0, 100.0, 100.0, 100.0)


def grid_config(**overrides) -> GridConfig:
    values = {
        "direction_target_fixed": 1_000_000.0,
        "global_target_fixed": 1_000_000.0,
        "hedge_loss_pct": 1.0,
        "daily_loss_limit": 1.0,
        "max_drawdown": 0.99,
    }
    values.update(overrides)
    return GridConfig(**values)


class GridDataTests(unittest.TestCase):
    def test_normalize_rejects_invalid_ohlc(self):
        bars = five_minute_bars([100.0, 101.0, 102.0])
        bars.loc[1, "high"] = 99.0

        with self.assertRaisesRegex(ValueError, "invalid OHLC"):
            normalize_ohlc(bars)

    def test_normalize_removes_duplicate_timestamp_and_reports_gap(self):
        bars = five_minute_bars([100.0, 101.0, 102.0, 103.0]).drop(index=2)
        bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)

        normalized, audit = normalize_ohlc(bars)

        self.assertEqual(audit.duplicate_rows, 1)
        self.assertEqual(audit.missing_intervals, 1)
        self.assertTrue(normalized["date"].is_monotonic_increasing)

    def test_grid_step_uses_only_previous_completed_bar(self):
        bars = five_minute_bars([100, 101, 102, 120, 121, 122] + [123] * 12)
        changed = bars.copy()
        changed.loc[3, ["high", "low", "close"]] = [200.0, 50.0, 125.0]

        result = add_grid_step(bars, atr_bars=3, atr_multiplier=0.35, min_step=0.60, max_step=2.50)
        changed_result = add_grid_step(changed, 3, 0.35, 0.60, 2.50)

        self.assertEqual(result.loc[3, "grid_step"], changed_result.loc[3, "grid_step"])
        self.assertNotEqual(result.loc[4, "grid_step"], changed_result.loc[4, "grid_step"])

    def test_latest_range_is_anchored_to_last_bar(self):
        bars = five_minute_bars([100.0] * (65 * 12), start="2026-06-01", freq="2h")

        prepared = select_latest_days(bars, days=60, warmup_bars=20)

        self.assertEqual(prepared.evaluation["date"].max(), bars["date"].max())
        self.assertGreaterEqual(prepared.input_bars.shape[0], prepared.evaluation.shape[0])
        self.assertLess(prepared.input_bars["date"].min(), prepared.evaluation["date"].min())


class GridCoreTests(unittest.TestCase):
    def test_path_nodes_preserve_selected_high_low_order(self):
        row = pd.Series({"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0})

        self.assertEqual(path_nodes(row, "OHLC"), (100.0, 105.0, 95.0, 102.0))
        self.assertEqual(path_nodes(row, "OLHC"), (100.0, 95.0, 105.0, 102.0))

    def test_cycle_opens_buy_and_sell_with_full_spread_marked(self):
        result = run_grid_backtest(flat_fixture(), grid_config(spread=0.40), "OHLC")

        opened = result.events.query("event == 'open' and layer == 1")
        self.assertEqual(opened["side"].tolist(), ["buy", "sell"])
        self.assertAlmostEqual(result.equity.iloc[0]["equity"], 100000.0 - 1.60, places=6)

    def test_falling_segment_opens_every_crossed_buy_level(self):
        bars = path_fixture(open_=100.0, high=100.0, low=95.0, close=95.0, grid_step=1.0)

        result = run_grid_backtest(bars, grid_config(), "OHLC")

        buys = result.events.query("event == 'open' and side == 'buy' and kind == 'grid'")
        self.assertEqual(buys["layer"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(buys["lots"].round(2).tolist(), [0.02, 0.03, 0.04, 0.05, 0.06, 0.07])

    def test_grid_never_exceeds_nine_layers_per_side(self):
        bars = path_fixture(open_=100.0, high=120.0, low=80.0, close=100.0, grid_step=1.0)

        result = run_grid_backtest(bars, grid_config(), "OHLC")

        opened = result.events.query("event == 'open' and kind == 'grid'")
        self.assertLessEqual(opened.groupby("side")["layer"].max().max(), 9)


class GridRiskTests(unittest.TestCase):
    def test_unlock_threshold_must_be_below_hedge_trigger(self):
        config = grid_config(hedge_loss_pct=0.01, hedge_unlock_loss_pct=0.02)

        with self.assertRaisesRegex(ValueError, "invalid grid configuration"):
            run_grid_backtest(flat_fixture(), config, "OHLC")

    def test_direction_basket_closes_only_profitable_grid_side(self):
        bars = path_fixture(open_=100.0, high=105.0, low=100.0, close=105.0, grid_step=100.0)
        config = grid_config(
            direction_target_fixed=5.0,
            direction_target_balance_pct=0.0,
            global_target_fixed=1_000_000.0,
        )

        result = run_grid_backtest(bars, config, "OHLC")

        closes = result.events.query("event == 'close' and reason == 'direction_target'")
        self.assertEqual(closes["side"].unique().tolist(), ["buy"])
        self.assertTrue((result.final_positions["side"] == "sell").any())

    def test_global_target_preempts_direction_target_at_same_crossing(self):
        bars = path_fixture(open_=100.0, high=106.0, low=99.0, close=106.0, grid_step=1.0)
        config = grid_config(
            global_target_fixed=3.55,
            global_target_balance_pct=0.0,
            direction_target_fixed=6.25,
            direction_target_balance_pct=0.0,
        )

        result = run_grid_backtest(bars, config, "OLHC")

        first_close = result.events.query("event == 'close'").iloc[0]
        self.assertEqual(first_close["reason"], "global_target")

    def test_hedge_requires_both_loss_and_net_exposure(self):
        bars = path_fixture(open_=100.0, high=100.0, low=94.0, close=94.0, grid_step=1.0)
        loss_only = run_grid_backtest(
            bars,
            grid_config(
                hedge_loss_pct=0.0001,
                hedge_unlock_loss_pct=0.00001,
                hedge_exposure_ratio=0.95,
            ),
            "OHLC",
        )
        both = run_grid_backtest(
            bars,
            grid_config(
                hedge_loss_pct=0.0001,
                hedge_unlock_loss_pct=0.00001,
                hedge_exposure_ratio=0.30,
            ),
            "OHLC",
        )

        self.assertFalse((loss_only.events["event"] == "hedge_open").any())
        self.assertTrue((both.events["event"] == "hedge_open").any())

    def test_hedge_floors_lots_and_pauses_exposed_grid_side(self):
        bars = path_fixture(open_=100.0, high=100.0, low=93.0, close=93.0, grid_step=1.0)
        config = grid_config(
            hedge_loss_pct=0.0001,
            hedge_unlock_loss_pct=0.00001,
            hedge_exposure_ratio=0.30,
        )

        result = run_grid_backtest(bars, config, "OHLC")

        hedge = result.events.query("event == 'hedge_open'").iloc[0]
        hedge_side = 1 if hedge["side"] == "buy" else -1
        pre_hedge_net = float(hedge["net_lots"]) - hedge_side * float(hedge["lots"])
        expected = np.floor(abs(pre_hedge_net) * config.hedge_fraction / config.lot_step) * config.lot_step
        self.assertAlmostEqual(float(hedge["lots"]), expected, places=8)
        later_buys = result.events.query(
            "sequence > @hedge.sequence and event == 'open' and kind == 'grid' and side == 'buy'"
        )
        self.assertTrue(later_buys.empty)

    def test_hedge_unlocks_after_floating_loss_recovers(self):
        bars = path_fixture(open_=100.0, high=105.0, low=94.0, close=105.0, grid_step=1.0)
        config = grid_config(
            hedge_loss_pct=0.0001,
            hedge_exposure_ratio=0.30,
            hedge_unlock_loss_pct=0.00005,
        )

        result = run_grid_backtest(bars, config, "OHLC")

        self.assertTrue((result.events["event"] == "hedge_open").any())
        self.assertTrue((result.events["event"] == "hedge_close").any())

    def test_direction_close_removes_orphaned_hedge(self):
        bars = path_fixture(open_=100.0, high=105.0, low=94.0, close=105.0, grid_step=1.0)
        config = grid_config(
            direction_target_fixed=10.0,
            direction_target_balance_pct=0.0,
            hedge_loss_pct=0.0001,
            hedge_unlock_loss_pct=0.000001,
            hedge_exposure_ratio=0.30,
        )

        result = run_grid_backtest(bars, config, "OLHC")

        cleanup = result.events.query("reason == 'orphan_hedge_cleanup'")
        self.assertFalse(cleanup.empty)
        self.assertTrue((cleanup["kind"] == "hedge").all())

    def test_daily_realized_loss_blocks_additions_until_next_utc_day(self):
        first = path_fixture(100.0, 105.0, 94.0, 105.0, 1.0)
        second = path_fixture(105.0, 105.0, 90.0, 90.0, 1.0)
        second["date"] = pd.Timestamp("2026-08-03 12:05", tz="UTC")
        third = path_fixture(90.0, 105.0, 90.0, 90.0, 1.0)
        third["date"] = pd.Timestamp("2026-08-04 12:00", tz="UTC")
        bars = pd.concat([first, second, third], ignore_index=True)
        config = grid_config(
            hedge_loss_pct=0.0001,
            hedge_unlock_loss_pct=0.00001,
            hedge_exposure_ratio=0.30,
            daily_loss_limit=0.00001,
        )

        result = run_grid_backtest(bars, config, "OLHC")

        blocked = result.events.query("event == 'addition_blocked' and reason == 'daily_loss_limit'")
        self.assertTrue((blocked["time"].dt.date == pd.Timestamp("2026-08-03").date()).any())
        next_day_adds = result.events.query("event == 'open' and reason == 'grid_add'")
        self.assertTrue((next_day_adds["time"].dt.date == pd.Timestamp("2026-08-04").date()).any())

    def test_global_close_respects_thirty_second_cooldown(self):
        bars = path_fixture(open_=100.0, high=106.0, low=99.0, close=106.0, grid_step=1.0)
        config = grid_config(
            global_target_fixed=2.0,
            global_target_balance_pct=0.0,
            cooldown_seconds=30,
        )

        result = run_grid_backtest(bars, config, "OHLC")

        close_time = result.events.query("reason == 'global_target'")["time"].min()
        restart_time = result.events.query("reason == 'cycle_restart' and time > @close_time")["time"].min()
        self.assertGreaterEqual(restart_time, close_time + pd.Timedelta(seconds=30))

    def test_drawdown_closes_all_and_permanently_stops(self):
        bars = path_fixture(open_=100.0, high=100.0, low=80.0, close=80.0, grid_step=1.0)
        config = grid_config(max_drawdown=0.001)

        result = run_grid_backtest(bars, config, "OHLC")

        breakers = result.events.query("event == 'circuit_breaker'")
        self.assertEqual(breakers["reason"].tolist(), ["max_drawdown"])
        breaker_sequence = int(breakers["sequence"].item())
        self.assertFalse((result.events.query("sequence > @breaker_sequence")["event"] == "open").any())
        self.assertTrue(result.final_positions.empty)


class GridReportTests(unittest.TestCase):
    def test_closed_and_open_pnl_reconcile_to_final_equity(self):
        result = run_grid_backtest(
            path_fixture(100.0, 102.0, 99.0, 101.0, 1.0),
            grid_config(),
            "OHLC",
        )

        errors = reconcile_result(result, initial_balance=100000.0)

        self.assertEqual(errors, [])
        closed = result.trades["pnl_usc"].sum()
        open_pnl = result.final_positions["unrealized_pnl_usc"].sum()
        self.assertAlmostEqual(
            100000.0 + closed + open_pnl,
            result.equity.iloc[-1]["equity"],
            places=6,
        )

    def test_wider_spread_does_not_improve_identical_flat_path(self):
        base = run_grid_backtest(flat_fixture(), grid_config(spread=0.35), "OHLC")
        wide = run_grid_backtest(flat_fixture(), grid_config(spread=0.70), "OHLC")

        self.assertLessEqual(wide.equity.iloc[-1]["equity"], base.equity.iloc[-1]["equity"] + 1e-9)

    def test_stats_include_grid_and_risk_metrics(self):
        result = run_grid_backtest(
            path_fixture(100.0, 103.0, 97.0, 101.0, 1.0),
            grid_config(),
            "OHLC",
        )

        stats = compute_grid_stats(result, initial_balance=100000.0)

        for name in (
            "final_balance",
            "final_equity",
            "max_drawdown_pct",
            "profit_factor",
            "max_gross_lots",
            "max_abs_net_lots",
            "hedge_count",
            "terminal_reason",
        ):
            self.assertIn(name, stats)

    def test_build_scenarios_returns_two_paths_times_three_spreads(self):
        config = {"scenarios": {"paths": ["OHLC", "OLHC"], "spreads": [0.35, 0.525, 0.70]}}

        scenarios = build_scenarios(config)

        self.assertEqual(len(scenarios), 6)
        self.assertEqual({scenario.path_mode for scenario in scenarios}, {"OHLC", "OLHC"})
        self.assertEqual({scenario.spread for scenario in scenarios}, {0.35, 0.525, 0.70})

    def test_coverage_rejects_less_than_fifty_nine_days(self):
        bars = five_minute_bars([100.0] * 41, start="2026-06-01", freq="1D")

        with self.assertRaisesRegex(ValueError, "insufficient 60-day coverage"):
            assert_coverage(bars, requested_days=60, tolerance_days=1)

    def test_report_writer_creates_summary_and_per_scenario_artifacts(self):
        result = run_grid_backtest(flat_fixture(), grid_config(), "OHLC")
        config = {"symbol": "GC=F", "proxy_for": "XAUUSD.c", "scenarios": {}}
        _, audit = normalize_ohlc(flat_fixture().drop(columns=["atr", "grid_step"]))
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir) / "grid_test"

            written = write_grid_reports([result], audit, config, prefix)

            names = {path.name for path in written}
            self.assertIn("grid_test.md", names)
            self.assertIn("grid_test_scenarios.csv", names)
            self.assertTrue(any(name.endswith("_events.csv") for name in names))
            report = (prefix.with_suffix(".md")).read_text(encoding="utf-8")
            self.assertIn("GC=F：双向 ATR 网格回测", report)
            json.dumps(config, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
