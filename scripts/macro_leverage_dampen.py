# -*- coding: utf-8 -*-
"""Macro-adjusted leverage: scale leverage DOWN when macro is bearish (risk-dampening).

Unlike a hard gate (which zeros leverage and forfeits return), this scales the
already-computed leverage by a function of the macro score: the more bearish the
macro, the lower the leverage (but still participating). This keeps bull-era upside
while trimming exposure exactly when macro conditions favour gold least.

Design:
  lev_gated = lev * macro_lev_mult(macro_score)
  macro_lev_mult in [lo, hi], mapped linearly from macro_score in [-1, 1]:
    macro>=+1 -> hi, macro<=-1 -> lo, linear in between.
  clamps lev_gated to the original tier set so we still respect the breaker logic.

Also supported: a macro-min floor (optionally keep at least lo even in extreme bear).

Usage: py scripts/macro_leverage_dampen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig  # noqa: E402
from london_gold.bull_adaptive import (  # noqa: E402
    AdaptiveConfig, build_signals, prepare_daily,
)
from london_gold.leverage_backtest import run_leverage_backtest  # noqa: E402
from london_gold.macro_factors import forward_fill_macro, macro_direction_score  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)

BASE = AdaptiveConfig(conf_mult=2.5, conf_power=1.0, conf_floor=0.3)
prepared = prepare_daily(D, BASE)
frame = build_signals(prepared, BASE)
frame["macro"] = forward_fill_macro(macro_direction_score(MACRO)["macro_score"],
                                    D["date"].to_numpy()).to_numpy()


def lev_mult(macro, lo, hi):
    """Linear map macro[-1,1] -> [lo,hi]. clip macro to [-1,1]."""
    m = np.clip(np.asarray(macro, dtype=float), -1.0, 1.0)
    return lo + (hi - lo) * (m + 1.0) / 2.0


def run(mask, lo, hi, lev_floor=0.0):
    sub = frame[mask].reset_index(drop=True).copy()
    base_lev = sub["lev"].to_numpy()
    mac = sub["macro"].to_numpy()
    mult = lev_mult(mac, lo, hi)
    lev = base_lev * mult
    # clamp to tier set so breaker logic stays meaningful: keep min(lev, base) and
    # a minimum participation floor (lev_floor) so we never fully zero a bull signal.
    lev = np.where(base_lev > 0, np.maximum(lev, lev_floor), base_lev)
    sub["lev"] = lev
    # risk stays as-is (confidence scaled), but scale it down a touch with lev too
    risk = sub["risk"].to_numpy()
    sub["risk"] = risk * np.where(base_lev > 0, np.clip(mult, 0.2, 1.0), 1.0)

    cost = CostConfig(capital=BASE.capital, position_oz=BASE.position_oz,
                      spread=BASE.spread, slippage=BASE.slippage,
                      commission_per_oz=BASE.commission_per_oz, leverage=3.0,
                      risk_per_trade_pct=BASE.risk_low, margin_call_pct=BASE.margin_call_pct)
    res = run_leverage_backtest(sub, cost, "dampen", leverage_series=sub["lev"].to_numpy(),
                                risk_series=sub["risk"].to_numpy())
    st = res["stats"]
    blew = st["max_drawdown"] >= BASE.margin_call_pct * 100 - 0.5
    return st, blew


def fmt(st, blew):
    return (f"ret{st['total_return']:+7.1f}% PF{st['profit_factor']:5.2f} "
            f"maxDD{st['max_drawdown']:5.1f}% win{st['win_rate']:3.0f}% tr{st['trade_count']:3d} "
            f"{'[BLOWUP]' if blew else 'safe'}")


def line(mask, lo, hi, floor, label):
    try:
        st, blew = run(mask, lo, hi, floor)
        return f"{label:<36} | {fmt(st, blew)}"
    except Exception as e:
        return f"{label:<36} | ERR {type(e).__name__}: {e}"


MASK_BULL = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
MASK_FULL = (D["date"] >= "2019-01-01") & (D["date"] <= "2026-08-28")

print("=== 宏观分 -> 杠杆降档 (不含硬门, 保留参与度) ===")
print("lev_mult: macro[+1,-1] -> [hi, lo]; 宏观越空杠杆越低")
print("-" * 92)
print("--- 无宏观调整 (基线) ---")
print(line(MASK_BULL, 1.0, 1.0, 0.0, "bull no-macro"))
print(line(MASK_FULL, 1.0, 1.0, 0.0, "full no-macro"))

print("\n--- 宏观降档 (bull-era 2024-26) [lo,hi] ---")
for lo, hi in [(0.2, 1.0), (0.3, 1.0), (0.4, 1.0), (0.5, 1.0), (0.3, 0.8), (0.5, 0.9)]:
    print(line(MASK_BULL, lo, hi, 0.0, f"bull lo{lo} hi{hi}"))

print("\n--- 宏观降档 (full 2019-26) [lo,hi] ---")
for lo, hi in [(0.2, 1.0), (0.3, 1.0), (0.4, 1.0), (0.5, 1.0), (0.3, 0.8), (0.5, 0.9)]:
    print(line(MASK_FULL, lo, hi, 0.0, f"full lo{lo} hi{hi}"))
