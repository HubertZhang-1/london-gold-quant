# -*- coding: utf-8 -*-
"""Production report for the adaptive+circuit-breaker bull strategy (2024-2026 window).

The strategy indicators (EMA200, ATR percentile, efficiency ratio, factor warmup)
are computed on the FULL daily history (2004+) so there is no warmup NaN inside the
trading window. Performance is reported ONLY for the 2024-01-01 -> 2026-08-28 bull
era (all pre-2024 data is excluded from the analysis).

Production config (v3): confidence-scaled exposure (conf x2.5, floor 0.3) + macro
risk-dampening (macro_lev_lo=0.5/hi=1.0).

Usage: py scripts/bull_adaptive_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.bull_adaptive import (  # noqa: E402
    AdaptiveConfig, apply_macro_leverage, build_signals, prepare_daily,
)
from london_gold.leverage_backtest import run_leverage_backtest  # noqa: E402
from london_gold.macro_factors import forward_fill_macro, macro_direction_score  # noqa: E402
from london_gold.backtest import CostConfig  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

# Production config v3: confidence-scaled + macro risk-dampening.
# ATTENTION: conf_mult=15.0 is the AGGRESSIVE variant chosen for the 2024-26 bull
# window (+101%). It is ONLY safe while gold is in a confirmed bull regime — on the
# full 2019-2026 history (which includes the 2022/2023 losing years) it BLOWS UP
# (-100%). Never deploy this outside a verified bull era.
cfg = AdaptiveConfig(conf_mult=15.0, conf_power=1.0, conf_floor=0.3,
                     macro_lev_lo=0.5, macro_lev_hi=1.0)

# warm indicators on full history, then restrict the trading window
prepared = prepare_daily(D, cfg)
frame = build_signals(prepared, cfg)
MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)
frame["macro"] = forward_fill_macro(macro_direction_score(MACRO)["macro_score"],
                                    D["date"].to_numpy()).to_numpy()
frame = apply_macro_leverage(frame, frame["macro"].to_numpy(), cfg)

ERA = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
win = frame[ERA].reset_index(drop=True).copy()

cost = CostConfig(capital=cfg.capital, position_oz=cfg.position_oz,
                  spread=cfg.spread, slippage=cfg.slippage,
                  commission_per_oz=cfg.commission_per_oz,
                  leverage=3.0, risk_per_trade_pct=cfg.risk_low,
                  margin_call_pct=cfg.margin_call_pct)
res = run_leverage_backtest(win, cost, "adaptive",
                            leverage_series=win["lev"].to_numpy(),
                            risk_series=win["risk"].to_numpy())
st = res["stats"]

print("=" * 60)
print(f"伦敦金自适应+熔断策略 · 2024-2026 牛市窗口 (v3 生产配置)")
print(f"窗口: 2024-01-02 → 2026-08-28   ({int(ERA.sum())} 交易日)")
print("=" * 60)
print(f"总收益   : {st['total_return']:+7.1f}%")
print(f"盈亏因子 : {st['profit_factor']:.2f}")
print(f"最大回撤 : {st['max_drawdown']:.1f}%")
print(f"胜率     : {st['win_rate']:.1f}%   ({st['trade_count']} 笔)")
print(f"最终权益 : ${st['final_equity']:,.0f}")

# per-year breakdown within the window
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

# monthly
eq = pd.DataFrame({"date": pd.to_datetime(res["dates"], utc=True), "equity": res["equity"]})
eq["ym"] = eq["date"].dt.strftime("%Y-%m")
m = eq.groupby("ym").agg(first=("equity", "first"), last=("equity", "last"))
m["ret"] = (m["last"] / m["first"] - 1) * 100
print("\n月度收益:")
print(m["ret"].round(1).to_string())

# save artifacts
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "bull_adaptive_circuit_breaker"
pd.DataFrame({"date": res["dates"], "equity": res["equity"]}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
res["trades"].to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")

# --- safety check: full 2019-2026 history (includes losing years) ---
FULL = (D["date"] >= "2019-01-01") & (D["date"] <= "2026-08-28")
full_win = frame[FULL].reset_index(drop=True).copy()
res_full = run_leverage_backtest(full_win, cost, "full",
                                 leverage_series=full_win["lev"].to_numpy(),
                                 risk_series=full_win["risk"].to_numpy())
sf = res_full["stats"]
print("\n[安全校验] 完整区间 2019-2026 (含 2022/2023 亏损年):")
print(f"  ret {sf['total_return']:+7.1f}%  PF {sf['profit_factor']:.2f}  "
      f"maxDD {sf['max_drawdown']:.1f}%  tr {sf['trade_count']}")
if sf["max_drawdown"] >= cfg.margin_call_pct * 100 - 0.5:
    print("  ⚠️ 该配置在此区间爆仓 (触及22-25%熔断)。只适用于确认牛市前提。")

print(f"\nsaved {out}_equity.csv / _trades.csv")
