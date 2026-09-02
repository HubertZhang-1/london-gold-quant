# -*- coding: utf-8 -*-
"""Long-sample robustness check of the Asian->London session-follow strategy.

Uses XAUUSD_5m.csv (2004-06 -> 2025-09, ~21 years, ~5,383 trading days) to test
whether the session-continuation edge holds up over a LONG period, not just the
2 months of 2026 1m data. Reports per-year and aggregate stats.

Usage: py scripts/session_follow_long.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.session_follow import SessionFollowConfig, run_session_follow  # noqa: E402

PROJ = Path(r"C:\Users\张策\Documents\EA量化项目")
df = pd.read_csv(PROJ / "data" / "XAUUSD_5m.csv")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)
df["year"] = df["date"].dt.year


def fmt(s):
    ret_pct = s["net_pnl"] / 100000.0 * 100.0
    return ("{:+.1f}% PF{:.2f} win{:.0f}% tr{} net${:+,.0f} (L{} S{})".format(
        ret_pct, s["profit_factor"], s["winrate"] * 100, s["trades"], s["net_pnl"],
        s["long_trades"], s["short_trades"]))


# chosen best config from the 1m analysis
cfg = SessionFollowConfig(min_asia_ret_pct=0.15, min_vol=6.0, stop_mult=2.0, rr=2.0)

print("=== 长样本验证: 亚洲→欧盘延续策略 (2004-2025, XAUUSD 5m) ===")
print("配置: min_ret>=0.15% min_vol>=6 stop=2 rr=2   (与1m分析一致)")
print("-" * 74)

# whole-sample
res = run_session_follow(df, cfg)
print("全部 2004-2025 : %s" % fmt(res["stats"]))
print(f"  样本天数: {len(res['trades'])} 笔 / {df['year'].nunique()} 年")

print("\n分年:")
print("%6s | %-38s" % ("year", "result"))
print("-" * 74)
for y in sorted(df["year"].unique()):
    ydf = df[df["year"] == y]
    if len(ydf) < 500 or ydf["date"].max() - ydf["date"].min() < pd.Timedelta(days=100):
        continue
    r = run_session_follow(ydf.reset_index(drop=True), cfg)
    st = r["stats"]
    print("%6s | %s" % (y, fmt(st)))

print("\n[基准] 同期黄金累计涨幅: %.1f%%" % (
    (float(df.iloc[-1]["close"]) / float(df.iloc[0]["close"]) - 1) * 100))
