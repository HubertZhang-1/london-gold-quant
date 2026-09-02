# -*- coding: utf-8 -*-
"""Test intermediate confidence-scaled multipliers and tighter circuit breaker.

confidence_scaled_risk.py showed conf x3 (floor 0.3) is the big win: bull-era
+28.4% / maxDD 2.2% and full-history +13.9% / maxDD 17.5% with NO blow-up,
while a uniform x3 blows up on the full history. But full-history maxDD 17.5%
is close to the 20% margin-call line. Here we probe x2.0-3.0 and a tighter
circuit breaker (earlier halt) to find the best stable point.

Usage: py scripts/confidence_sweep_refine.py
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


def run_conf(mask, base_mult, conf_power=1.0, conf_floor=0.3, mc=0.20):
    cfg = AdaptiveConfig(margin_call_pct=mc)
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
    res = run_leverage_backtest(frame, cost, "c",
                                leverage_series=frame["lev"].to_numpy(),
                                risk_series=risk_series)
    st = res["stats"]
    blew = st["max_drawdown"] >= mc * 100 - 0.5
    return st, blew


def fmt(st, blew):
    return (f"ret{st['total_return']:+7.1f}% PF{st['profit_factor']:5.2f} "
            f"maxDD{st['max_drawdown']:5.1f}% win{st['win_rate']:3.0f}% tr{st['trade_count']:3d} "
            f"{'[BLOWUP]' if blew else 'safe'}")


def line(bm, cf, cp, mask, mc, label):
    try:
        st, blew = run_conf(mask, bm, cp, cf, mc)
        return f"{label:<40} | {fmt(st, blew)}"
    except Exception as e:
        return f"{label:<40} | ERR {type(e).__name__}: {e}"


MASK = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
FULL = pd.Series(True, index=D.index)

print("=== 中间档位 + 熔断线 敏感 (牛市区间) ===")
print("-" * 86)
for bm in [1.5, 2.0, 2.5, 3.0]:
    for mc in [0.20, 0.15]:
        mclabel = "mc15%" if mc == 0.15 else "mc20%"
        print(line(bm, 0.3, 1.0, MASK, mc, "bull x%.1f %s" % (bm, mclabel)))

print()
print("=== 完整区间 (重点是 不爆仓 + 控制回撤) ===")
print("-" * 86)
for bm in [1.5, 2.0, 2.5, 3.0]:
    for mc in [0.20, 0.15]:
        mclabel = "mc15%" if mc == 0.15 else "mc20%"
        print(line(bm, 0.3, 1.0, FULL, mc, "full x%.1f %s" % (bm, mclabel)))

print()
print("=== 年度 (x2.5, mc 15%) ===")
print("-" * 86)
for yname, s, e in YEARS:
    mask = (D["date"] >= s) & (D["date"] <= e)
    print(f"{yname:>6} | " + line(2.5, 0.3, 1.0, mask, 0.15, "x2.5 mc15%"))
