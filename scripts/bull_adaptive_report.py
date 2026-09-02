# -*- coding: utf-8 -*-
"""Production report for the adaptive+circuit-breaker bull strategy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.bull_adaptive import AdaptiveConfig, prepare_daily, build_signals, run_adaptive
from london_gold.macro_factors import forward_fill_macro, macro_direction_score

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)
# Production config: confidence-scaled exposure (conf x2.5, floor 0.3) + macro
# risk-dampening (bearish macro -> lower leverage). See docs/bull_adaptive_strategy.md.
cfg = AdaptiveConfig(conf_mult=2.5, conf_power=1.0, conf_floor=0.3,
                     macro_lev_lo=0.5, macro_lev_hi=1.0)

# Macro direction score, forward-filled onto the daily gold bars for dampening.
MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)
MACRO_ON_BARS = forward_fill_macro(macro_direction_score(MACRO)["macro_score"],
                                   D["date"].to_numpy()).to_numpy()

WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]


def _macro_for(s, e):
    """Slice the full macro-on-bars array to the [s,e] date window (by position)."""
    idx = (D["date"] >= s) & (D["date"] <= e)
    return MACRO_ON_BARS[idx.to_numpy()]


print("=== ADAPTIVE + CIRCUIT BREAKER (production config) — per year ===")
print(f"{'year':>5} | {'ret%':>8} {'PF':>5} {'maxDD%':>6} {'trades':>6} {'avgLev':>6}")
print("-" * 50)
for wname, s, e in WINDOWS:
    part = D[(D["date"] >= s) & (D["date"] <= e)].reset_index(drop=True)
    res = run_adaptive(part, cfg, macro_series=_macro_for(s, e))
    st = res["stats"]
    avglev = part.assign(lev=prepare_daily(part, cfg)["lev"])["lev"].mean()
    print(f"{wname:>5} | {st['total_return']:+8.1f} {st['profit_factor']:5.2f} "
          f"{st['max_drawdown']:6.1f} {st['trade_count']:6d} {avglev:6.1f}")

print()
print("=== FULL BULL ERA 2024-2026 (production config) ===")
era_mask = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
era = D[era_mask].reset_index(drop=True)
res = run_adaptive(era, cfg, macro_series=MACRO_ON_BARS[era_mask.to_numpy()])
st = res["stats"]
print(f"trades={st['trade_count']} win%={st['win_rate']:.1f} ret={st['total_return']:+.1f}% "
      f"PF={st['profit_factor']:.2f} maxDD={st['max_drawdown']:.1f}% final=${st['final_equity']:,.0f}")

# save artifacts
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "bull_adaptive_circuit_breaker"
pd.DataFrame({"date": res["dates"], "equity": res["equity"]}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
res["trades"].to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
# monthly
eq = pd.DataFrame({"date": pd.to_datetime(res["dates"], utc=True), "equity": res["equity"]})
eq["ym"] = eq["date"].dt.strftime("%Y-%m")
m = eq.groupby("ym").agg(first=("equity", "first"), last=("equity", "last"))
m["ret"] = (m["last"] / m["first"] - 1) * 100
print("\nmonthly returns:")
print(m["ret"].round(1).to_string())
print(f"\nsaved {out}_equity.csv / _trades.csv")
