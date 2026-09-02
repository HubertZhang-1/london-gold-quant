# -*- coding: utf-8 -*-
"""Backtest the adaptive+circuit-breaker strategy ONLY on 2024-2026.

Indicators (EMA200, ATR percentile, efficiency ratio, factor warmup) are computed
on the FULL daily history (2004+) so there is no warmup NaN inside the window; the
TRADING / performance window is strictly 2024-01-01 -> 2026-08-28 (the bull era),
which drops all pre-2024 data.

Production config (v3): confidence-scaled exposure (conf x2.5, floor 0.3) + macro
risk-dampening (macro_lev_lo=0.5/hi=1.0).

Usage: py scripts/bull_era_2024_2026.py
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

# Production config v3 (AGGRESSIVE): confidence-scaled exposure conf x15 + macro
# risk-dampening. NOTE: this blows up on the full history (non-bull years). Only use
# while gold is in a confirmed bull regime. See docs/conf_x15_aggressive.md.
cfg = AdaptiveConfig(conf_mult=15.0, conf_power=1.0, conf_floor=0.3,
                     macro_lev_lo=0.5, macro_lev_hi=1.0)

# ---- warm indicators on the FULL history, slice the signal window ----
prepared_full = prepare_daily(D, cfg)
frame_full = build_signals(prepared_full, cfg)

MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)
frame_full["macro"] = forward_fill_macro(macro_direction_score(MACRO)["macro_score"],
                                         D["date"].to_numpy()).to_numpy()
frame_full = apply_macro_leverage(frame_full, frame_full["macro"].to_numpy(), cfg)

# ---- restrict the trading window to 2024-01-01 .. 2026-08-28 ----
mask = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
win = frame_full[mask].reset_index(drop=True).copy()

cost = CostConfig(capital=cfg.capital, position_oz=cfg.position_oz,
                  spread=cfg.spread, slippage=cfg.slippage,
                  commission_per_oz=cfg.commission_per_oz,
                  leverage=3.0, risk_per_trade_pct=cfg.risk_low,
                  margin_call_pct=cfg.margin_call_pct)
res = run_leverage_backtest(win, cost, "bull-era",
                            leverage_series=win["lev"].to_numpy(),
                            risk_series=win["risk"].to_numpy())
st = res["stats"]

print("=" * 62)
print(f"自适应+熔断策略 · 2024-2026 牛市窗口回测 (v3 生产配置)")
print(f"窗口: 2024-01-02 → 2026-08-28   ({mask.sum()} 交易日)")
print("=" * 62)
print(f"最终权益   : ${st['final_equity']:,.0f}")
print(f"总收益     : {st['total_return']:+6.1f}%")
print(f"盈亏因子 PF: {st['profit_factor']:.2f}")
print(f"最大回撤   : {st['max_drawdown']:.1f}%")
print(f"胜率       : {st['win_rate']:.1f}%   ({st['trade_count']} 笔)")
print("=" * 62)

# per-year within the window
print("\n逐年:")
tr = res["trades"].copy()
if len(tr):
    tr["entry_date"] = pd.to_datetime(tr["entry_time"], utc=True)
    tr["year"] = tr["entry_date"].dt.year
    for y in sorted(tr["year"].unique()):
        tt = tr[tr["year"] == y]
        pnl = tt["pnl"].sum()
        wins = (tt["pnl"] > 0).sum()
        gp = tt.loc[tt["pnl"] > 0, "pnl"].sum()
        gl = abs(tt.loc[tt["pnl"] < 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0.0
        print(f"  {y}: 交易 {len(tt):2d}  胜率 {wins/len(tt)*100:4.0f}%  "
              f"净盈亏 ${pnl:+,.0f}  PF {pf:.2f}")

# save artifacts
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "bull_adaptive_2024_2026"
pd.DataFrame({"date": res["dates"], "equity": res["equity"]}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
res["trades"].to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"\n已保存 {out}_equity.csv / _trades.csv")
