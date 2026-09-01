# -*- coding: utf-8 -*-
"""Parameter sweep for the reverse-engineered martingale-grid EA."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "XAUUSD_5m_2026.csv"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    print(f"{'gridAtr':>8} {'tpAtr':>6} {'maxL':>4} {'trades':>7} {'win%':>6} {'PF':>6} "
          f"{'netPnL':>10} {'maxDD%':>8} {'hedge':>6} {'terminal':>14} {'ret%':>8}")
    print("-" * 96)
    for grid_mult in (0.8, 1.0, 1.2, 1.5):
        for tp_atr in (0.8, 1.0, 1.5):
            for max_layers in (4, 6):
                cfg = MartingaleConfig(
                    grid_atr_mult=grid_mult,
                    take_profit_atr=tp_atr,
                    max_layers=max_layers,
                )
                res = run_martingale_backtest(df, cfg)
                s = res.stats
                print(f"{grid_mult:8.2f} {tp_atr:6.2f} {max_layers:4d} {s['trades']:7d} "
                      f"{s['winrate']*100:6.1f} {s['profit_factor']:6.2f} "
                      f"{s['net_pnl']:10.2f} {s['max_drawdown_pct']:8.1f} "
                      f"{s['hedge_trades']:6d} {s['terminal_reason'] or 'none':>14} "
                      f"{s['total_return_pct']:8.2f}")


if __name__ == "__main__":
    main()
