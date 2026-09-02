# -*- coding: utf-8 -*-
"""Drawdown attribution: what did the signal/state/macro look like before each drawdown?

Goal: make the drawdown statistics clearer by attributing each drawdown event to
the regime / signal / macro conditions that preceded it, so we know WHICH kind of
condition produces the drawdowns the current adaptive+circuit-breaker strategy
suffers. This informs the "stand aside on sustained macro decline" gate.

Focus: the bull era 2024-2026 (the live production window), but run full history too.

Usage: py scripts/drawdown_attribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.bull_adaptive import (  # noqa: E402
    MICRO_W, AdaptiveConfig, build_factors, build_signals, prepare_daily,
)
from london_gold.factor_library import aggregate_score  # noqa: E402
from london_gold.macro_factors import forward_fill_macro, macro_direction_score  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)

cfg = AdaptiveConfig(conf_mult=2.5, conf_power=1.0, conf_floor=0.3)

# Build the full prepared+signals frame once
prepared = prepare_daily(D, cfg)
frame = build_signals(prepared, cfg)

# Attach macro score onto bars
macro_score_series = macro_direction_score(MACRO)["macro_score"]
frame["macro"] = forward_fill_macro(macro_score_series, D["date"].to_numpy()).to_numpy()

# micro composite
fac = build_factors(D)
micro = aggregate_score(fac, MICRO_W).fillna(0.0).to_numpy()
frame["micro"] = micro

# leverage-accurate equity curve + drawdown series
from london_gold.backtest import CostConfig  # noqa: E402
from london_gold.leverage_backtest import run_leverage_backtest  # noqa: E402
cost = CostConfig(capital=cfg.capital, position_oz=cfg.position_oz, spread=cfg.spread,
                  slippage=cfg.slippage, commission_per_oz=cfg.commission_per_oz,
                  leverage=3.0, risk_per_trade_pct=cfg.risk_low, margin_call_pct=cfg.margin_call_pct)
res = run_leverage_backtest(frame, cost, "attrib", leverage_series=frame["lev"].to_numpy(),
                            risk_series=frame["risk"].to_numpy())
eq_series = pd.Series(res["equity"]).ffill().to_numpy()


def pick_drawdowns(lo, hi, min_dd_pct=1.0):
    """Return dedup list of (peak_idx, trough_idx) drawdown episodes in [lo,hi).

    Intra-window drawdown: the running peak is reset at the window start so the
    % is measured against the equity level at the window's own peak, NOT the
    all-time peak. Events are merged when a small (sub-threshold) recovery is
    immediately followed by a deeper dip, so we don't report the same event many
    times with the same peak date.
    """
    eqs = eq_series[lo:hi]
    running_peak = eqs[0]
    events = []
    cur_start = None
    cur_trough = 0
    cur_peak = 0
    for k in range(len(eqs)):
        if eqs[k] > running_peak:
            running_peak = eqs[k]
        dd = (running_peak - eqs[k]) / running_peak
        if cur_start is None and dd >= min_dd_pct / 100.0:
            cur_start = k  # index into eqs where this drawdown began
            cur_peak = k
            cur_trough = k
        elif cur_start is not None:
            if eqs[k] > eqs[cur_peak]:
                cur_peak = k
            if eqs[k] < eqs[cur_trough]:
                cur_trough = k
            # if it fully recovers (back above the value at start of event), close it
            if eqs[k] >= running_peak * (1.0 - 1e-9):
                events.append((lo + cur_start, lo + cur_trough))
                cur_start = None
    if cur_start is not None:
        events.append((lo + cur_start, lo + cur_trough))
    return events


def state_of(i):
    b = prepared["bull"].iloc[i]
    er = prepared["er20"].iloc[i]
    vol = prepared["atr_pctl"].iloc[i]
    lev = frame["lev"].iloc[i]
    if b < cfg.bull_thr:
        st = "BEAR"
    elif vol > cfg.ext_vol_pctl:
        st = "EXTREME_VOL"
    elif vol > cfg.high_vol_pctl:
        st = "HIGH_VOL"
    elif er > cfg.er_clean:
        st = "CLEAN_TREND"
    elif er > cfg.er_bull:
        st = "BULL"
    else:
        st = "CHOP"
    return st


EDGES = [
    ("bull-era", "2024-01-01", "2026-08-28"),
    ("full", "2019-01-01", "2026-08-28"),
]

for label, s, e in EDGES:
    lo = D.index[D["date"] >= s][0]
    hi = D.index[D["date"] <= e][-1] + 1
    events = pick_drawdowns(lo, hi, min_dd_pct=0.5)
    print(f"\n============ {label} ({s} -> {e}) : {len(events)} 回撤事件 ============")
    print(f"{'peak_date':>11} {'trough':>11} {'dd%':>6} {'dd_days':>5} | "
          f"{'peak_state':>14} {'signal':>7} {'micro':>6} {'macro':>6} {'lev':>4}")
    print("-" * 96)
    enriched = []
    for pidx, tidx in events:
        p_date = str(D["date"].iloc[pidx].date())
        t_date = str(D["date"].iloc[tidx].date())
        # intra-window drawdown: peak (within window up to trough) vs trough equity
        wpeak = float(np.max(eq_series[lo:tidx + 1]))
        wtrough = float(eq_series[tidx])
        dd = (wpeak - wtrough) / wpeak * 100
        dd_days = int(tidx - pidx)
        st = state_of(pidx)
        sig = int(frame["signal"].iloc[pidx])
        mci = float(frame["micro"].iloc[pidx])
        mro = float(frame["macro"].iloc[pidx])
        lv = float(frame["lev"].iloc[pidx])
        enriched.append((dd, p_date, t_date, dd_days, st, sig, mci, mro, lv))
    for dd, p_date, t_date, dd_days, st, sig, mci, mro, lv in sorted(enriched, reverse=True):
        print(f"{p_date:>11} {t_date:>11} {dd:6.1f} {dd_days:5d} | "
              f"{st:>14} {sig:+7d} {mci:+6.2f} {mro:+6.2f} {lv:4.0f}")
