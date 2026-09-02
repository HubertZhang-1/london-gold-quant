# -*- coding: utf-8 -*-
"""Lift return on the 2024-2026 window by increasing the risk exposure (conf_mult).

Diagnosis: the 2024-26 bull era has a superb signal (PF ~13, win ~74%) but maxDD is
only ~1.6% — the strategy barely uses its risk budget, so total return looks low.
Raising conf_mult (confidence-scaled exposure) scales the per-trade risk up and
should lift return substantially while maxDD stays far below the 20% breaker.

But exposure must stay SAFE across weak regimes too. We therefore validate each
conf_mult on BOTH the 2024-26 bull era AND the full 2019-2026 history (which includes
the losing 2022/2023 years) so we pick a level that lifts bull-era return WITHOUT
blowing up when the regime turns. Indicators are warmed on the full history.

Usage: py scripts/conf_scan_2024_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig  # noqa: E402
from london_gold.bull_adaptive import (  # noqa: E402
    AdaptiveConfig, apply_macro_leverage, build_signals, prepare_daily,
)
from london_gold.leverage_backtest import run_leverage_backtest  # noqa: E402
from london_gold.macro_factors import forward_fill_macro, macro_direction_score  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)
macro_on_bars = forward_fill_macro(macro_direction_score(MACRO)["macro_score"],
                                   D["date"].to_numpy()).to_numpy()

ERA = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
FULL = (D["date"] >= "2019-01-01") & (D["date"] <= "2026-08-28")


def run_for(conf_mult, mask):
    cfg = AdaptiveConfig(conf_mult=conf_mult, conf_power=1.0, conf_floor=0.3,
                         macro_lev_lo=0.5, macro_lev_hi=1.0)
    prepared = prepare_daily(D, cfg)
    frame = build_signals(prepared, cfg)
    frame["macro"] = macro_on_bars
    frame = apply_macro_leverage(frame, frame["macro"].to_numpy(), cfg)
    win = frame[mask].reset_index(drop=True).copy()
    cost = CostConfig(capital=cfg.capital, position_oz=cfg.position_oz,
                      spread=cfg.spread, slippage=cfg.slippage,
                      commission_per_oz=cfg.commission_per_oz,
                      leverage=3.0, risk_per_trade_pct=cfg.risk_low,
                      margin_call_pct=cfg.margin_call_pct)
    res = run_leverage_backtest(win, cost, "s", leverage_series=win["lev"].to_numpy(),
                                risk_series=win["risk"].to_numpy())
    st = res["stats"]
    blew = st["max_drawdown"] >= cfg.margin_call_pct * 100 - 0.5
    return st, blew


def fmt(st, blew):
    return (f"ret{st['total_return']:+7.1f}% PF{st['profit_factor']:5.2f} "
            f"maxDD{st['max_drawdown']:5.1f}% win{st['win_rate']:3.0f}% tr{st['trade_count']:3d} "
            f"{'[BLOWUP]' if blew else 'safe'}")


print("=== 2024-2026 窗口 · 提高 conf_mult (风险敞口) ===")
print("当前生产 conf_mult=2.5 → +17.2%。提高 conf_mult 会线性抬高收益。")
print("-" * 84)
print("--- 牛市区间 2024-2026 ---")
for cm in [2.5, 3.5, 5.0, 7.0, 10.0, 15.0]:
    try:
        st, blew = run_for(cm, ERA)
        print(f"  conf x{cm:<5.1f} | {fmt(st, blew)}")
    except Exception as e:
        print(f"  conf x{cm:<5.1f} | ERR {type(e).__name__}: {e}")

print("\n--- 完整区间 2019-2026 (含亏损年, 验证不爆仓) ---")
for cm in [2.5, 3.5, 5.0, 7.0, 10.0, 15.0]:
    try:
        st, blew = run_for(cm, FULL)
        print(f"  conf x{cm:<5.1f} | {fmt(st, blew)}")
    except Exception as e:
        print(f"  conf x{cm:<5.1f} | ERR {type(e).__name__}: {e}")
