# -*- coding: utf-8 -*-
"""Formal 3x-leverage bull-only strategy with DOUBLE gate (bull regime + edge).

Recommended production config. Produces:
  - year-by-year / full bull-era / monthly report
  - equity + trades CSV
  - compares single-gate (bull only) vs double-gate (bull + edge)

Double gate:
  G1 bull regime : bull_score > bull_thr (clean uptrend)
  G2 strategy edge: recent micro score rolling mean > edge_thr (momentum edge
      intact) — suppresses chop/mean-reversion stretches and strategy-failure
      years (2022/2023 had high bull% but micro edge faded).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, trend_regime
from london_gold.leverage_backtest import run_leverage_backtest

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
close = DF["close"]
ema200 = ema(close, 4800)
ema_slope = ema200.diff(120)
up_trend = (close > ema(close, 1200)).astype(float)
trend = trend_regime(close, DF["high"], DF["low"], er_window=48, er_threshold=0.12,
                     adx_window=14, adx_threshold=20.0)
DF["bull"] = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)

MICRO_W = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
           "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}

WINDOWS = [("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31"),
           ("2020", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
           ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
           ("2026", "2026-01-01", "2026-08-28")]


def build_frame(part, thr=0.5, stop=2.5, rr=2.0, bull_thr=0.55,
                use_edge=False, edge_lookback=96, edge_thr=0.02):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    sig = np.where(part["bull"].to_numpy() > bull_thr, sig, 0)
    if use_edge:
        ms = pd.Series(micro.to_numpy(), index=range(len(part)))
        edge = ms.rolling(edge_lookback, min_periods=int(edge_lookback * 0.5)).mean().fillna(0.0).to_numpy()
        sig = np.where(edge > edge_thr, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    return pd.DataFrame({
        "date": part["date"], "open": part["open"], "high": part["high"],
        "low": part["low"], "close": part["close"], "signal": sig,
        "stop_dist": np.where(anim, a * stop, 0.0), "tp_dist": np.where(anim, a * stop * rr, 0.0)})


def run(part, lev=3.0, margin_call=0.30, **kw):
    f = build_frame(part, **kw)
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=lev, risk_per_trade_pct=0.20,
                      margin_call_pct=margin_call)
    return run_leverage_backtest(f, cost, "3x")


print("=== 3x LEVERAGE BULL STRATEGY: single-gate vs double-gate (year) ===")
print(f"{'year':>5} {'bull%':>5} | {'single ret':>10} {'sDD':>5} | {'double ret':>10} {'dDD':>5}")
print("-" * 60)
for wname, s, e in WINDOWS:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    sg = run(part, use_edge=False)["stats"]
    dg = run(part, use_edge=True)["stats"]
    print(f"{wname:>5} {part['bull'].mean()*100:5.1f} | {sg['total_return']:+10.1f} {sg['max_drawdown']:5.1f} | "
          f"{dg['total_return']:+10.1f} {dg['max_drawdown']:5.1f}")

print()
print("=== FULL BULL ERA 2024-2026 @ 3x (double gate) ===")
era = DF[(DF["date"] >= "2024-01-01") & (DF["date"] <= "2026-08-28")].reset_index(drop=True)
res = run(era, use_edge=True)
s = res["stats"]
print(f"trades={s['trade_count']} win%={s['win_rate']:.1f} ret={s['total_return']:+.1f}% "
      f"PF={s['profit_factor']:.2f} maxDD={s['max_drawdown']:.1f}% final=${s['final_equity']:,.0f}")

out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "bull_3x_double_gate"
pd.DataFrame({"date": res["dates"], "equity": res["equity"]}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
res["trades"].to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"saved {out}_equity.csv / _trades.csv")

# monthly
eq = pd.DataFrame({"date": pd.to_datetime(res["dates"], utc=True), "equity": res["equity"]})
eq["ym"] = eq["date"].dt.strftime("%Y-%m")
m = eq.groupby("ym").agg(first=("equity", "first"), last=("equity", "last"))
m["ret"] = (m["last"] / m["first"] - 1) * 100
print("\nmonthly returns:")
print(m["ret"].round(1).to_string())
