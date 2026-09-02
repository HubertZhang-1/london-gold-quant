# -*- coding: utf-8 -*-
"""Compare the original hedged martingale EA vs the SAFE fixed-lot grid on real 1m data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest  # noqa: E402
from london_gold.safe_grid import SafeGridConfig, run_safe_grid_backtest  # noqa: E402

PROJ = Path(r"C:\Users\张策\Documents\EA量化项目")
CSV = PROJ / "data" / "XAUUSD_1m_202608.csv"
df = pd.read_csv(CSV)


def fmt(s):
    return ("final${:,.0f} ret{:+.1f}% PF{:.2f} maxDD{:.1f}% win{:.0f}% tr{} "
            "avgW${:,.0f}/avgL${:,.0f} term={}".format(
                s["final_equity"], s["total_return_pct"], s["profit_factor"],
                s["max_drawdown_pct"], s["winrate"] * 100, s["trades"],
                s["avg_win"], s["avg_loss"], s["terminal_reason"] or "none"))


print("=" * 94)
print("原版 对冲马丁网格 EA  vs  安全版 固定手数网格 (2026-08 真实 1m 数据)")
print("=" * 94)
print("[benchmark] buy&hold 2026-08: %+.1f%%" % (
    (float(df.iloc[-1]['close']) / float(df.iloc[0]['close']) - 1) * 100))

print("\n--- 原版 (对冲马丁) ---")
try:
    mg = run_martingale_backtest(df, MartingaleConfig(initial_balance_usc=100_000.0))
    print(fmt(mg.stats))
except Exception as e:
    print("ERR:", e)

print("\n--- 安全版 扫描: grid_pct/stop_pct/gate (固定手数0.3, layers4) ---")
variants = [
    ("grid.10% stop.40% gate", dict(grid_pct=0.0010, stop_pct=0.0040, use_trend_gate=True)),
    ("grid.15% stop.60% gate", dict(grid_pct=0.0015, stop_pct=0.0060, use_trend_gate=True)),
    ("grid.20% stop.80% gate", dict(grid_pct=0.0020, stop_pct=0.0080, use_trend_gate=True)),
    ("grid.15% stop.60% nGate", dict(grid_pct=0.0015, stop_pct=0.0060, use_trend_gate=False)),
    ("grid.25% stop.1.0% gate", dict(grid_pct=0.0025, stop_pct=0.0100, use_trend_gate=True)),
]
for label, kw in variants:
    cfg = SafeGridConfig(initial_balance_usc=100_000.0, max_layers=4, **kw)
    r = run_safe_grid_backtest(df, cfg)
    print("%-24s | %s" % (label, fmt(r.stats)))

print("\n--- 安全版 默认 (grid.15%/stop.6%/gate) 交易明细 ---")
r = run_safe_grid_backtest(df, SafeGridConfig(grid_pct=0.0015, stop_pct=0.006, use_trend_gate=True))
print(fmt(r.stats))
if len(r.trades):
    print("  按原因分布:")
    for reason, tt in r.trades.groupby("reason"):
        print("    %-12s n=%-4d win%%=%.0f avg$%+.0f net$%+.0f" % (
            reason, len(tt), (tt["pnl_usc"] > 0).mean() * 100,
            tt["pnl_usc"].mean(), tt["pnl_usc"].sum()))
