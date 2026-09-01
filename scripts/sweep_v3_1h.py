# -*- coding: utf-8 -*-
"""Parameter sweep for v3 intraday strategies on 1h XAUUSD data."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.backtest import CostConfig
from london_gold.intraday_strategies_v3 import mean_reversion_signals, momentum_scalp_signals

# load backtest_v3 from the v3 backtest script without running its main
spec = importlib.util.spec_from_file_location(
    "v3bt", str(PROJECT_ROOT / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(PROJECT_ROOT / "data" / "XAUUSD_1h_2026.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)


def run(frame, label):
    res = v3bt.backtest_v3(frame, COST, label, {})
    s = res.stats
    return s


print("=== MOMENTUM_SCALP sweep on 1h ===")
print(f"{'conf':>5} {'rr':>4} {'stop':>4} {'sess':>5} {'adx':>4} | {'trades':>6} {'win%':>5} "
      f"{'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 70)
best = None
for conf in (0.55, 0.60, 0.65, 0.70):
    for rr in (1.5, 2.0, 2.5):
        for stop in (1.0, 1.5):
            for sess in (True, False):
                frame = momentum_scalp_signals(
                    DF, min_confidence=conf, rr_target=rr, stop_mult=stop,
                    use_session_filter=sess,
                )
                s = run(frame, "MOM")
                row = (conf, rr, stop, sess, s["trade_count"], s["win_rate"],
                       s["total_return"], s["profit_factor"], s["max_drawdown"])
                print(f"{conf:5.2f} {rr:4.1f} {stop:4.1f} {str(sess):>5} {'25':>4} | "
                      f"{s['trade_count']:6d} {s['win_rate']:5.1f} {s['total_return']:+8.2f} "
                      f"{s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")
                if best is None or s["total_return"] > best["ret"]:
                    best = {"conf": conf, "rr": rr, "stop": stop, "sess": sess,
                            "ret": s["total_return"], "pf": s["profit_factor"],
                            "dd": s["max_drawdown"], "trades": s["trade_count"],
                            "win": s["win_rate"]}

print()
print("=== BEST ===")
print(best)
