# -*- coding: utf-8 -*-
"""Entry point for the GC=F bidirectional ATR-grid proxy backtest."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.grid_replica_data import (
    add_grid_step,
    normalize_ohlc,
    select_latest_days,
)
from london_gold.grid_replica_engine import GridConfig, run_grid_backtest
from london_gold.grid_replica_report import reconcile_result, write_grid_reports


@dataclass(frozen=True)
class Scenario:
    path_mode: str
    spread: float


def assert_coverage(
    bars: pd.DataFrame,
    requested_days: int = 60,
    tolerance_days: int = 1,
) -> None:
    """Raise if the calendar-day time span of the bars is too short."""
    data = bars.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True)
    if data.empty:
        raise ValueError(f"insufficient {requested_days}-day coverage: no bars")
    coverage = int((data["date"].max() - data["date"].min()).days) + 1
    minimum = requested_days - tolerance_days
    if coverage < minimum:
        raise ValueError(
            f"insufficient {requested_days}-day coverage: got {coverage} days, need >= {minimum}"
        )


def build_scenarios(config: dict) -> list[Scenario]:
    """Expand the paths x spreads cartesian product into scenario objects."""
    scenarios_cfg = config.get("scenarios", {})
    paths = scenarios_cfg.get("paths", ["OHLC", "OLHC"])
    spreads = scenarios_cfg.get("spreads", [0.35, 0.525, 0.70])
    return [Scenario(path_mode=path, spread=spread) for path in paths for spread in spreads]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def run_all(
    data: pd.DataFrame,
    config: dict,
    days: int,
    warmup_bars: int,
    atr_bars: int,
    atr_multiplier: float,
    min_step: float,
    max_step: float,
    prefix: Path,
) -> list[Path]:
    normalized, audit = normalize_ohlc(data)
    stepped = add_grid_step(
        normalized,
        atr_bars=atr_bars,
        atr_multiplier=atr_multiplier,
        min_step=min_step,
        max_step=max_step,
    )
    prepared = select_latest_days(stepped, days=days, warmup_bars=warmup_bars)
    assert_coverage(prepared.evaluation, requested_days=days)

    results = []
    for scenario in build_scenarios(config):
        grid_config = GridConfig(spread=scenario.spread)
        result = run_grid_backtest(
            prepared.evaluation,
            grid_config,
            path_mode=scenario.path_mode,
        )
        errors = reconcile_result(result, initial_balance=grid_config.initial_balance_usc)
        if errors:
            raise RuntimeError(f"{result.scenario} failed reconciliation: {errors}")
        results.append(result)

    return write_grid_reports(results, audit, config, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="GC=F bidirectional ATR-grid proxy backtest")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "london_gold_grid_replica_config.json"))
    parser.add_argument("--csv", default=None, help="defaults to config data.cache")
    parser.add_argument("--symbol", default=None, help="overrides config symbol in report")
    parser.add_argument("--proxy", default=None, help="overrides config proxy_for in report")
    parser.add_argument("--interval", default=None, help="overrides config data.interval in report")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--warmup-bars", type=int, default=288)
    parser.add_argument("--prefix", default=None, help="defaults to reports/gc_h1_grid_replica")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_cfg = config.get("data", {})
    grid_cfg = config.get("grid", {})
    if args.symbol:
        config["symbol"] = args.symbol
    if args.proxy:
        config["proxy_for"] = args.proxy
    if args.interval:
        config.setdefault("data", {})["interval"] = args.interval
    csv_path = Path(args.csv) if args.csv else Path(str(PROJECT_ROOT / data_cfg.get("cache", "data/gc_h1.csv")))
    prefix = Path(args.prefix) if args.prefix else Path(str(PROJECT_ROOT / "reports" / "gc_h1_grid_replica"))

    written = run_all(
        data=load_data(csv_path),
        config=config,
        days=args.days,
        warmup_bars=data_cfg.get("warmup_bars", args.warmup_bars),
        atr_bars=grid_cfg.get("atr_bars", 14),
        atr_multiplier=grid_cfg.get("atr_multiplier", 0.35),
        min_step=grid_cfg.get("min_step", 0.60),
        max_step=grid_cfg.get("max_step", 2.50),
        prefix=prefix,
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
