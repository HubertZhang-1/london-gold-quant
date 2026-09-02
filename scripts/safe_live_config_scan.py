# -*- coding: utf-8 -*-
"""Find a SAFE live config for the reproduced martingale-grid EA.

The user will run the reproduced strategy live but cannot access the original EA's
inputs (likely no built-in breaker). We reproduce it faithfully but with OUR OWN
safety layer: daily-loss breaker, max-drawdown breaker, and a tighter martingale
layer cap. We scan these knobs on BOTH a favourable month (Aug 2026, one-way up)
and a constructed one-way-down month, and pick settings that keep the downside
bounded (breaker triggers early) while the favourable month still profits.

Usage: py scripts/safe_live_config_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest  # noqa: E402

PROJ = Path(r"C:\Users\张策\Documents\EA量化项目")
df = pd.read_csv(PROJ / "data" / "XAUUSD_1m_202608.csv")


def adverse_month(d):
    """Construct a clear one-way-down month from the real Aug bars."""
    b = d.copy()
    start = float(b["close"].iloc[0])
    n = len(b)
    lin = start * (1 - 0.05 * np.arange(n) / n)  # ~-5% over the month
    b["close"] = lin
    b["open"] = b["close"].shift(1).fillna(start)
    b["high"] = b[["open", "close"]].max(axis=1) + 0.5
    b["low"] = b[["open", "close"]].min(axis=1) - 0.5
    b["date"] = pd.to_datetime(b["date"], utc=True)
    return b


def run(df, cfg):
    res = run_martingale_backtest(df, cfg)
    s = res.stats
    eq = res.equity
    dd = (eq["equity"].cummax() - eq["equity"]) / eq["equity"].cummax()
    return s, float(dd.max())


def fmt(s):
    return ("ret{:+.1f}% final${:,.0f} win{:.0f}% PF{:.2f} term={}".format(
        s["total_return_pct"], s["final_equity"], s["winrate"] * 100,
        s["profit_factor"], s["terminal_reason"] or "none"))


ABAD = adverse_month(df)

print("=" * 78)
print("实盘安全配置扫描: max_layers × 熔断线 (顺风8月 vs 构造单边下跌)")
print("=" * 78)
print("%-34s | %-28s | %-28s" % ("config", "favourable Aug", "adverse (down)"))
print("-" * 78)
for layers in [3, 4, 5, 6]:
    for dl, mdd in [(0.03, 0.30), (0.03, 0.20), (0.02, 0.20)]:
        cfg = MartingaleConfig(initial_balance_usc=100_000.0, stop_loss_atr=0.0,
                               use_trend_filter=True, max_layers=layers,
                               daily_loss_pct=dl, max_drawdown_pct=mdd)
        s1, dd1 = run(df, cfg)
        s2, dd2 = run(ABAD, cfg)
        name = "layers=%d dl=%.2f mdd=%.2f" % (layers, dl, mdd)
        print("%-34s | %-28s | %-28s" % (name, fmt(s1), fmt(s2)))

print("\n[基准] Aug 2026 黄金涨幅: %+.1f%%" % (
    (float(df.iloc[-1]["close"]) / float(df.iloc[0]["close"]) - 1) * 100))
