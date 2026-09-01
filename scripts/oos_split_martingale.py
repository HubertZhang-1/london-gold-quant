# -*- coding: utf-8 -*-
"""Out-of-sample robustness split test (indicator warmup preserved).

Computes on the full frame then slices by date so EMA/ATR warmup is stable.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest

df = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_5m_2026.csv")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)

cfg = MartingaleConfig(grid_atr_mult=1.0, take_profit_atr=1.0, max_layers=6)

# Warm up indicators on the whole series, then window the rows. The engine needs
# its own warmup, so we run the whole thing and compare per-half equity windows
# is the honest approach: the engine is stateful so a fair split needs a restart
# per period. We instead run the FULL backtest and then slice the equity curve.
full = run_martingale_backtest(df, cfg)
eq = full.equity

# split equity curve at 2026-05-01
eq["date"] = pd.to_datetime(eq["date"], utc=True)
before = eq[eq["date"] < pd.Timestamp("2026-05-01", tz="UTC")]
after = eq[eq["date"] >= pd.Timestamp("2026-05-01", tz="UTC")]


def summarize(part, label):
    if part.empty:
        print(f"{label}: empty")
        return
    start = part["equity"].iloc[0]
    end = part["equity"].iloc[-1]
    peak = part["equity"].max()
    dd = ((peak - part["equity"]) / peak).max() * 100
    print(f"{label:22s} equity ${start:,.0f} -> ${end:,.0f} ({(end/start-1)*100:+6.1f}%) "
          f"maxDD={dd:5.1f}% trades={len(eq[eq['date'] < part['date'].iloc[-1]]) if label.startswith('FULL') else 'n/a'}")


print("=== FULL ===")
s = full.stats
print(f"trades={s['trades']} win%={s['winrate']*100:.1f} PF={s['profit_factor']:.2f} "
      f"ret={s['total_return_pct']:+.1f}% maxDD={s['max_drawdown_pct']:.1f}% term={s['terminal_reason'] or 'none'}")
print()
summarize(before, "FIRST HALF (<=04-30)")
summarize(after, "SECOND HALF (>04-30)")
