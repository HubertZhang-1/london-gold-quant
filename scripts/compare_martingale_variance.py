# -*- coding: utf-8 -*-
"""Compare baseline martingale grid vs a low-risk variant, over full period and
rolling month-by-month, on real XAUUSD 5m data."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest  # noqa: E402

DF = pd.read_csv(PROJECT_ROOT / "data" / "XAUUSD_5m_2026.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
DF["ym"] = DF["date"].dt.strftime("%Y-%m")


def fmt(s: dict, label: str) -> str:
    return (f"{label:16s} trades={s['trades']:5d} win%={s['winrate']*100:5.1f} "
            f"PF={s['profit_factor']:5.2f} ret={s['total_return_pct']:+8.1f}% "
            f"maxDD={s['max_drawdown_pct']:5.1f}% term={s['terminal_reason'] or 'none'}")


def main() -> None:
    baseline = MartingaleConfig()
    low_risk = MartingaleConfig(
        grid_atr_mult=1.0, max_layers=3, take_profit_atr=1.0,
        stop_loss_atr=6.0, hedge_atr=2.5, lot_ladder=(0.3, 0.4, 0.5, 0.6),
        use_trend_filter=True,
    )
    equal_lot = MartingaleConfig(
        grid_atr_mult=1.0, max_layers=3, take_profit_atr=1.0,
        stop_loss_atr=6.0, lot_ladder=(0.3, 0.3, 0.3, 0.3),
        use_trend_filter=True,
    )

    print("=" * 100)
    print("FULL PERIOD  2026-01 .. 2026-08  (real XAUUSD 5m, with costs)")
    print("=" * 100)
    for cfg, label in [(baseline, "BASELINE(马丁)"), (low_risk, "LOW-RISK(温和)"), (equal_lot, "EQUAL-LOT")]:
        res = run_martingale_backtest(DF, cfg)
        print(fmt(res.stats, label))
        res.equity.to_csv(PROJECT_ROOT / "reports" / f"cmp_{label.lower()}_equity.csv",
                          index=False, encoding="utf-8-sig")

    print()
    print("=" * 100)
    print("ROLLING MONTHLY")
    print("=" * 100)
    months = sorted(DF["ym"].unique())
    header = f"{'month':>8} | {'B tr':>6} {'B ret%':>7} {'B DD%':>6} {'B term':>12} | {'LR tr':>6} {'LR ret%':>7} {'LR DD%':>6} {'LR term':>12}"
    print(header)
    print("-" * len(header))
    for ym in months:
        part = DF[DF["ym"] == ym]
        bl = run_martingale_backtest(part, baseline)
        lr = run_martingale_backtest(part, low_risk)
        print(f"{ym:>8} | {bl.stats['trades']:6d} {bl.stats['total_return_pct']:+7.2f} "
              f"{bl.stats['max_drawdown_pct']:6.1f} {bl.stats['terminal_reason'] or 'none':>12} | "
              f"{lr.stats['trades']:6d} {lr.stats['total_return_pct']:+7.2f} "
              f"{lr.stats['max_drawdown_pct']:6.1f} {lr.stats['terminal_reason'] or 'none':>12}")


if __name__ == "__main__":
    main()
