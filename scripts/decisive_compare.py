# -*- coding: utf-8 -*-
"""Decisive comparison: baseline martingale vs representative low-risk variants.
Runs to completion (turns daily-loss guard off) so we see the strategy's real
economics rather than the guard tripping.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_5m_2026.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)

# Guards off to expose real economics (daily_loss huge disables, max_dd 1.0).
vars = {
    "马丁原版(无止损)": MartingaleConfig(daily_loss_pct=99.0, max_drawdown_pct=0.99, lot_ladder=(0.3, 0.7, 1.0, 1.5, 2.0, 3.0), max_layers=6),
    "马丁+止损6ATR": MartingaleConfig(daily_loss_pct=99.0, max_drawdown_pct=0.99, stop_loss_atr=6.0, lot_ladder=(0.3, 0.6, 1.0, 1.5), max_layers=4),
    "等手网格+止损": MartingaleConfig(daily_loss_pct=99.0, max_drawdown_pct=0.99, stop_loss_atr=6.0, lot_ladder=(0.3, 0.3, 0.3), max_layers=3),
    "温和阶梯+止损": MartingaleConfig(daily_loss_pct=99.0, max_drawdown_pct=0.99, stop_loss_atr=6.0, lot_ladder=(0.3, 0.4, 0.5), max_layers=3),
    "马丁(无趋势过滤)": MartingaleConfig(daily_loss_pct=99.0, max_drawdown_pct=0.99, use_trend_filter=False, lot_ladder=(0.3, 0.7, 1.0, 1.5), max_layers=4),
}

print(f"{'variant':>20} {'trades':>6} {'win%':>5} {'PF':>5} {'ret%':>9} {'maxDD%':>7} {'term':>12}")
print("-" * 80)
for label, cfg in vars.items():
    r = run_martingale_backtest(DF, cfg)
    s = r.stats
    print(f"{label:>20} {s['trades']:6d} {s['winrate']*100:5.1f} {s['profit_factor']:5.2f} "
          f"{s['total_return_pct']:+9.2f} {s['max_drawdown_pct']:7.1f} {s['terminal_reason'] or 'none':>12}")

print()
print("point: martingale's edge comes from riding adverse moves to a reclaimable")
print("average. Adding a stop or flattening lots removes that edge and leaves only")
print("cost drag -> the strategy goes negative. No low-risk config made money here.")
