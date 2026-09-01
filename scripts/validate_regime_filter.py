# -*- coding: utf-8 -*-
"""Validate the winning regime-filter config (ER=0.12, ADX=20) across full period."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.intraday_strategies_v3 import momentum_scalp_signals

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

BASE = dict(min_confidence=0.55, rr_target=1.8, min_adx=18, use_session_filter=False,
            regime_filter=False)  # explicit baseline without regime gate
best = {**BASE, "regime_filter": True, "er_threshold": 0.12, "adx_regime_threshold": 20.0}

# Full 2024-2026 continuous run
seg = DF[DF["date"] >= "2024-01-01"].reset_index(drop=True)
frame = momentum_scalp_signals(seg, **best)
res = v3bt.backtest_v3(frame, COST, "regime_filtered", {})
s = res.stats
print("=== FULL 2024-01 .. 2026-08 (regime-filtered) ===")
print(f"trades={s['trade_count']} win%={s['win_rate']:.1f} ret={s['total_return']:+.2f}% "
      f"PF={s['profit_factor']:.2f} maxDD={s['max_drawdown']:.1f}% final=${s['final_equity']:,.0f}")

# vs no-filter baseline
base_frame = momentum_scalp_signals(seg, **BASE)
bres = v3bt.backtest_v3(base_frame, COST, "no_filter", {})
bs = bres.stats
print(f"  no-filter baseline: ret={bs['total_return']:+.2f}% PF={bs['profit_factor']:.2f} maxDD={bs['max_drawdown']:.1f}%")

# save artifacts
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "xauusd_1h_regime_filtered"
pd.DataFrame({"date": res.dates, "equity": res.equity}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(res.trades).to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved {out}_equity.csv / _trades.csv")

# monthly breakdown
eq = pd.DataFrame({"date": res.dates, "equity": res.equity})
eq["date"] = pd.to_datetime(eq["date"], utc=True)
eq["ym"] = eq["date"].dt.strftime("%Y-%m")
monthly = eq.groupby("ym").agg(first=("equity", "first"), last=("equity", "last"))
monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100
print("\n=== monthly returns (regime-filtered) ===")
print(monthly["ret"].round(2).to_string())
