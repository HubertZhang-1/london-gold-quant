# -*- coding: utf-8 -*-
"""Macro-regime gate: stand aside (0 leverage) whenever macro is in sustained decline.

Drawdown attribution showed the strategy's worst drawdowns happen in CLEAN_TREND
@ 10x long while the macro score (USD/real-yield/VIX direction) is persistently
negative (-0.28). The macro layer says "bearish gold" yet the strategy still uses
max leverage long. This script adds a MACRO GATE: when macro_score < threshold the
strategy forces flat (0 leverage) indefinitely, with NO time limit — it stays out
for as long as the macro stays in decline.

macro_score (macro_direction_score) is a change-rate indicator over DXY/10Y/VIX, so
it holds its sign through a sustained move — exactly a "sustained decline" detector.
Coverage: macro data starts 2017-09, so the gate is active only from there; earlier
bars have no macro (treated as neutral/pass-through).

Usage: py scripts/macro_gate_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig  # noqa: E402
from london_gold.bull_adaptive import (  # noqa: E402
    MICRO_W, AdaptiveConfig, build_factors, build_signals, prepare_daily,
)
from london_gold.factor_library import aggregate_score  # noqa: E402
from london_gold.leverage_backtest import run_leverage_backtest  # noqa: E402
from london_gold.macro_factors import forward_fill_macro, macro_direction_score  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)

# baseline config = production confidence-scaled x2.5
BASE = AdaptiveConfig(conf_mult=2.5, conf_power=1.0, conf_floor=0.3)

prepared = prepare_daily(D, BASE)
frame = build_signals(prepared, BASE)
macro_score_series = macro_direction_score(MACRO)["macro_score"]
frame["macro"] = forward_fill_macro(macro_score_series, D["date"].to_numpy()).to_numpy()

# macro availability (for coverage reporting)
frame["macro_known"] = frame["macro"].to_numpy() != 0.0


def run_with_gate(mask, gate_thr, allow_unknown=True):
    """Run the adaptive backtest with a macro gate.

    When macro_score < gate_thr -> force lev=0 (flat). If macro data is unknown
    (pre-2017), allow_unknown=True lets it trade normally (neutral), else it is
    also flat. Returns stats + flat-fraction.
    """
    sub = frame[mask].reset_index(drop=True)
    sub = sub.copy()
    macro = sub["macro"].to_numpy()
    known = sub["macro_known"].to_numpy()

    lev = sub["lev"].to_numpy()
    # apply gate: force flat when macro in sustained decline
    gated = macro < gate_thr
    if not allow_unknown:
        gated = gated | (~known)
    lev = np.where(gated, 0.0, lev)
    sub["lev"] = lev
    # also zero the signal when flat, so no entry on flat bars
    sig = sub["signal"].to_numpy()
    sub["signal"] = np.where(gated, 0, sig)

    cost = CostConfig(capital=BASE.capital, position_oz=BASE.position_oz,
                      spread=BASE.spread, slippage=BASE.slippage,
                      commission_per_oz=BASE.commission_per_oz,
                      leverage=3.0, risk_per_trade_pct=BASE.risk_low,
                      margin_call_pct=BASE.margin_call_pct)
    res = run_leverage_backtest(sub, cost, "macrogate",
                                leverage_series=sub["lev"].to_numpy(),
                                risk_series=sub["risk"].to_numpy())
    st = res["stats"]
    blew = st["max_drawdown"] >= BASE.margin_call_pct * 100 - 0.5
    flat_frac = float(gated.mean() * 100)
    return st, blew, flat_frac


def fmt(st, blew, ff):
    return (f"ret{st['total_return']:+7.1f}% PF{st['profit_factor']:5.2f} "
            f"maxDD{st['max_drawdown']:5.1f}% win{st['win_rate']:3.0f}% tr{st['trade_count']:3d} "
            f"flat{ff:4.0f}% {('[BLOWUP]' if blew else 'safe')}")


def line(mask, gate_thr, label, allow_unknown=True):
    try:
        st, blew, ff = run_with_gate(mask, gate_thr, allow_unknown)
        return f"{label:<34} | {fmt(st, blew, ff)}"
    except Exception as e:
        return f"{label:<34} | ERR {type(e).__name__}: {e}"


MASK_BULL = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
MASK_FULL = (D["date"] >= "2019-01-01") & (D["date"] <= "2026-08-28")
MASK_TEST = (D["date"] >= "2017-09-01") & (D["date"] <= "2026-08-28")  # macro fully available

print("=== 宏观空仓门敏感度 (macro<阈值 -> 强制空仓不限时间) ===")
print("无门基线 compare; 不同阈值用于: bull-era 2024-26, macro-available 2017-26, full 2019-26")
print("-" * 92)
print("--- 无门基线 ---")
print(line(MASK_BULL, -1.01, "bull-era no-gate"))
print(line(MASK_TEST, -1.01, "2017-26 no-gate"))
print(line(MASK_FULL, -1.01, "full 2019-26 no-gate"))

print("\n--- 加宏观门 (bull-era 2024-26) ---")
for thr in [0.0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3]:
    print(line(MASK_BULL, thr, f"bull-era gate<{thr}"))

print("\n--- 加宏观门 (macro-available 2017-26) ---")
for thr in [0.0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3]:
    print(line(MASK_TEST, thr, f"2017-26 gate<{thr}"))

print("\n--- 加宏观门 (full 2019-26, allow unknown=pass-through) ---")
for thr in [0.0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3]:
    print(line(MASK_FULL, thr, f"full 2019-26 gate<{thr}", allow_unknown=True))
