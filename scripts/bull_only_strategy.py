# -*- coding: utf-8 -*-
"""Final bull-only strategy: micro + bull gate + configurable risk budget.

Provides the recommended config (risk 1.5%) and full report on 2024-2026 + 2018-23
(bull vs bear comparison), saving equity/trades.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, trend_regime

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

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


def run(part, risk_pct, thr=0.20, bull_thr=0.55):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    sig = np.where(part["bull"].to_numpy() > bull_thr, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * 0.8, 0.0), "tp_dist": np.where(anim, a * 0.8 * 2.2, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, risk_per_trade_pct=risk_pct)
    return v3bt.backtest_v3(f, cost, "bull", {})


WINDOWS = [("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31"),
           ("2020", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
           ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
           ("2026", "2026-01-01", "2026-08-28")]

print("=== BULL-ONLY STRATEGY (bull gate + micro, risk 1.5%) by year ===")
print(f"{'year':>5} {'bull%':>5} label | {'ret%':>7} {'PF':>5} {'maxDD%':>6}")
print("-" * 46)
bull_tot = {"ret": 0.0}
for wname, s, e in WINDOWS:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    res = run(part, 0.015)
    st = res.stats
    bull_pct = part["bull"].mean() * 100
    lab = "BULL" if bull_pct > 55 else ("CHOP" if bull_pct > 35 else "BEAR")
    print(f"{wname:>5} {bull_pct:5.1f} {lab:>5} | {st['total_return']:+7.1f} {st['profit_factor']:5.2f} {st['max_drawdown']:6.1f}")

print()
print("=== BULL-ERA 2024-2026 (risk 1.5%) ===")
era = DF[(DF["date"] >= "2024-01-01") & (DF["date"] <= "2026-08-28")].reset_index(drop=True)
res = run(era, 0.015)
st = res.stats
print(f"trades={st['trade_count']} win%={st['win_rate']:.1f} ret={st['total_return']:+.1f}% "
      f"PF={st['profit_factor']:.2f} maxDD={st['max_drawdown']:.1f}% final=${st['final_equity']:,.0f}")
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "bull_only_strategy"
pd.DataFrame({"date": res.dates, "equity": res.equity}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(res.trades).to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved {out}_equity.csv / _trades.csv")
