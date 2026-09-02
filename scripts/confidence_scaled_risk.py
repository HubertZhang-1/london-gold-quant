# -*- coding: utf-8 -*-
"""Confidence-scaled risk: give more exposure to strong signals, less to weak.

Baseline finding (risk_elasticity_sweep.py): the signal's edge is excellent on
the 2024-26 bull era (PF ~10.8) but the exposure is tiny, so total return is
only +12.4%. Uniformly scaling ALL risk xN does raise return linearly but is
dangerous — weak regimes (2020/2023/2026) have PF <1 and hit the 20% margin-call
halt first (2026 blows up at x6).

This script tests the better lever: instead of a uniform multiplier, scale each
trade's risk by the CONFIDENCE of its signal (|micro| composite score). Strong
signals get more risk; weak ones get little. That should lift return on good
years while still protecting the bad ones.

Usage: py scripts/confidence_scaled_risk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.bull_adaptive import (  # noqa: E402
    MICRO_W, AdaptiveConfig, build_signals, prepare_daily,
)
from london_gold.factor_library import aggregate_score, build_factors  # noqa: E402
from london_gold.leverage_backtest import run_leverage_backtest  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

YEARS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
         ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
         ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]

cfg = AdaptiveConfig()


def run_with_confidence_scaled(mask, base_mult=1.0, conf_power=1.0, conf_floor=0.3):
    """Run adaptive backtest where risk is scaled by |micro| confidence.

    risk = base_risk * base_mult * (conf_floor + (1-conf_floor) * |micro|^conf_power)
    where base_risk is the per-tier risk from the lever mapping (0.5%/1%/2%).
    """
    part = D[mask].reset_index(drop=True)
    prepared = prepare_daily(part, cfg)
    frame = build_signals(prepared, cfg)

    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W).fillna(0.0).abs().to_numpy()

    base_risk = frame["risk"].to_numpy()
    conf = np.clip(micro, 0.0, 1.0) ** conf_power
    scale = base_mult * (conf_floor + (1.0 - conf_floor) * conf)
    risk_series = base_risk * scale

    cost = CostConfig(capital=cfg.capital, position_oz=cfg.position_oz,
                      spread=cfg.spread, slippage=cfg.slippage,
                      commission_per_oz=cfg.commission_per_oz,
                      leverage=3.0, risk_per_trade_pct=cfg.risk_low,
                      margin_call_pct=cfg.margin_call_pct)
    res = run_leverage_backtest(frame, cost, "conf_scaled",
                                leverage_series=frame["lev"].to_numpy(),
                                risk_series=risk_series)
    st = res["stats"]
    blew = st["max_drawdown"] >= cfg.margin_call_pct * 100 - 0.5
    return st, blew


def fmt(st, blew):
    return (f"ret{st['total_return']:+7.1f}% PF{st['profit_factor']:5.2f} "
            f"maxDD{st['max_drawdown']:5.1f}% win{st['win_rate']:3.0f}% tr{st['trade_count']:3d} "
            f"{'[BLOWUP]' if blew else 'safe'}")


def line(label, base_mult, conf_power, conf_floor, mask):
    try:
        st, blew = run_with_confidence_scaled(mask, base_mult, conf_power, conf_floor)
        return f"{label:<34} | {fmt(st, blew)}"
    except Exception as e:
        return f"{label:<34} | ERR {type(e).__name__}: {e}"


MASK = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
FULL = pd.Series(True, index=D.index)

print("=== 信心缩放风险 (牛市区间 2024-2026) ===")
print("base_mult 抬高敞口; conf_floor=最低信心保底; conf_power=微分数映射强度")
print("-" * 84)
variants = [
    ("基线 x1", 1.0, 1.0, 1.0),
    ("conf x1.5 floor0.3", 1.5, 1.0, 0.3),
    ("conf x2.0 floor0.3", 2.0, 1.0, 0.3),
    ("conf x3.0 floor0.3", 3.0, 1.0, 0.3),
    ("conf x3.0 floor0.5", 3.0, 1.0, 0.5),
    ("conf x4.0 floor0.3", 4.0, 1.0, 0.3),
    ("conf x5.0 floor0.5", 5.0, 1.0, 0.5),
    ("conf x3.0 floor0.3 pw2", 3.0, 2.0, 0.3),
]
for label, bm, cp, cf in variants:
    print(line(label, bm, cp, cf, MASK))

print()
print("=== 逐年 (conf x3.0 floor0.3, 与统一x3对照) ===")
print("-" * 84)
for yname, s, e in YEARS:
    mask = (D["date"] >= s) & (D["date"] <= e)
    print(f"{yname:>6} | conf-scaled: " + line("", 3.0, 1.0, 0.3, mask))

print()
print("=== 完整区间 2004-2026 (conf-scaled) ===")
print("-" * 84)
for label, bm, cp, cf in variants:
    print(line(f"full ({label})", bm, cp, cf, FULL))
