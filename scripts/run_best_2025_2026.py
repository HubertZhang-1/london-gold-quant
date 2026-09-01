# -*- coding: utf-8 -*-
"""Final max-return strategy for 2025-01..2026-08: curated_strong factor ensemble.

Also produces the equity/trades artifacts and a monthly breakdown.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr as iatr

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
seg = DF[(DF["date"] >= "2025-01-01") & (DF["date"] <= "2026-08-28")].reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

curated_strong = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
                  "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}
THRESH = 0.30
STOP = 0.8
RR = 2.2

F = build_factors(seg)
score = aggregate_score(F, curated_strong)
atr14 = iatr(seg["high"], seg["low"], seg["close"], 14)
sig = np.where(score > THRESH, 1, np.where(score < -THRESH, -1, 0))
a = atr14.to_numpy(float); anim = ~np.isnan(a)
frame = pd.DataFrame({
    "date": seg["date"], "open": seg["open"], "high": seg["high"],
    "low": seg["low"], "close": seg["close"],
    "signal": sig, "stop_dist": np.where(anim & (sig != 0), a * STOP, 0.0),
    "tp_dist": np.where(anim & (sig != 0), a * STOP * RR, 0.0)})
res = v3bt.backtest_v3(frame, COST, "BEST", {})
s = res.stats
print("=" * 62)
print("BEST RETURN STRATEGY  | 2025-01 .. 2026-08 (XAUUSD 1h)")
print("=" * 62)
print(f"weight  : curated_strong (macd/aroon/trend_adx/ema_spread/bulls_bears/momentum/bb)")
print(f"params  : threshold={THRESH}  SL={STOP}xATR  TP={RR}xATR")
print("-" * 62)
print(f"final equity : ${s['final_equity']:,.0f}  (ret {s['total_return']:+.2f}%)")
print(f"trades       : {s['trade_count']}  (win {s['win_rate']:.1f}%)")
print(f"profit factor: {s['profit_factor']:.2f}")
print(f"max drawdown : {s['max_drawdown']:.1f}%")
print(f"initial      : $100,000")
print("=" * 62)

# save artifacts
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "xauusd_1h_2025_2026_best"
pd.DataFrame({"date": res.dates, "equity": res.equity}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(res.trades).to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved {out}_equity.csv / _trades.csv")

# monthly breakdown
eq = pd.DataFrame({"date": pd.to_datetime(res.dates, utc=True), "equity": res.equity})
eq["ym"] = eq["date"].dt.strftime("%Y-%m")
monthly = eq.groupby("ym").agg(first=("equity", "first"), last=("equity", "last"))
monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100
print("\n=== monthly returns ===")
print(monthly["ret"].round(2).to_string())
