# -*- coding: utf-8 -*-
"""Full-period run of the robust best v3 config on 1h XAUUSD, saving report."""
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

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_2026.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

CONFIGS = {
    "B(0.55,RR2,ADX25)": dict(min_confidence=0.55, rr_target=2.0, min_adx=25, use_session_filter=False),
    "A(0.55,RR1.8,ADX18)": dict(min_confidence=0.55, rr_target=1.8, min_adx=18, use_session_filter=False),
    "C(0.60,RR2,ADX25)": dict(min_confidence=0.60, rr_target=2.0, min_adx=25, use_session_filter=False),
}

for label, kw in CONFIGS.items():
    frame = momentum_scalp_signals(DF, **kw)
    res = v3bt.backtest_v3(frame, COST, label, {})
    s = res.stats
    print(f"{label:22s} trades={s['trade_count']:4d} win%={s['win_rate']:5.1f} "
          f"ret={s['total_return']:+7.2f}% PF={s['profit_factor']:5.2f} "
          f"maxDD={s['max_drawdown']:5.1f}% final=${s['final_equity']:,.0f}")

# save best equity + trades
best = max(CONFIGS, key=lambda k: v3bt.backtest_v3(momentum_scalp_signals(DF, **CONFIGS[k]), COST, k, {}).stats["total_return"])
frame = momentum_scalp_signals(DF, **CONFIGS[best])
res = v3bt.backtest_v3(frame, COST, best, {})
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "xauusd_1h_v3_optimized"
pd.DataFrame({"date": res.dates, "equity": res.equity}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(res.trades).to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved best={best} -> {out}_equity.csv / _trades.csv")
