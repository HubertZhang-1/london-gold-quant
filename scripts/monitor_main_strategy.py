# -*- coding: utf-8 -*-
"""Live daily monitor + risk profile for the faithful main strategy (hedged martingale grid).

MAIN STRATEGY per the user's decision: two-sided hedged martingale grid, no bull/bear
filter (不分牛熊). Because it is high-frequency (tens of thousands of trades in a month),
a single end-of-data snapshot is NOT enough — the real risk lives in the PATH:
    - how deep the martingale layer stack goes (peak lots / layers, and how often)
    - how close equity came to the circuit breakers (3% daily loss / 30% max drawdown)
    - the daily P&L distribution (worst day, losing-day count)
    - how positions actually close (take-profit vs hedge-flat vs breaker)

This monitor reports both the current snapshot AND the full risk profile, with alerts
when exposure approaches a dangerous level.

Usage:
  py scripts/monitor_main_strategy.py                          # cached 1m Aug data
  py scripts/monitor_main_strategy.py --csv path.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest  # noqa: E402

# alert thresholds (configurable)
ALERT_MAIN_LOTS = 3.0    # warn when the martingale main stack exceeds this many lots
ALERT_DRAWDOWN = 0.20    # warn near-drawdown (as fraction) before the 30% breaker
ALERT_DAILY_LOSS = 0.02  # warn near the 3% daily-loss breaker


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily monitor + risk profile for the main martingale-grid EA")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "XAUUSD_1m_202608.csv"))
    parser.add_argument("--balance", type=float, default=100_000.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    cfg = MartingaleConfig(initial_balance_usc=args.balance, stop_loss_atr=0.0, use_trend_filter=True)

    res = run_martingale_backtest(df, cfg)
    s = res.stats
    ev = res.events
    tr = res.trades
    eq = res.equity

    last = df.iloc[-1]
    close = float(last["close"])
    ema60 = df["close"].ewm(span=cfg.trend_ema_bars, adjust=False).mean()
    trend = "UP ↑" if ema60.iloc[-1] > ema60.iloc[-2] else "DOWN ↓"

    print("=" * 64)
    print(f"主策略(双边对冲网格马丁) 监控+风险画像  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"数据: {last['date']}   收盘 ${close:,.1f}")
    print("=" * 64)

    # ---- market context ----
    print("[市场]")
    print(f"  金价收盘   : ${close:,.1f}")
    print(f"  EMA({cfg.trend_ema_bars}) 趋势 : {trend}")

    # ---- current snapshot (end of data) ----
    lastrow = eq.iloc[-1] if len(eq) else None
    main_lots = float(lastrow["main_lots"]) if lastrow is not None else 0.0
    hedge_lots = float(lastrow["hedge_lots"]) if lastrow is not None else 0.0
    equity_now = float(lastrow["equity"]) if lastrow is not None else args.balance
    balance = float(lastrow["balance"]) if lastrow is not None else args.balance

    print("\n[当前快照] (数据末端)")
    if main_lots > 0 or hedge_lots > 0:
        open_ev = ev[ev["event"] == "open"]
        last_open = open_ev.iloc[-1] if len(open_ev) else None
        mside = "做多" if (last_open is not None and pd.notna(last_open.get("main_side")) and last_open["main_side"] > 0) else "做空"
        layer_val = last_open.get("layer", 0) if last_open is not None else 0
        layer_n = int(layer_val) if pd.notna(layer_val) else 0
        print(f"  主仓方向/层数 : {mside} / {layer_n}层 (上限{cfg.max_layers})")
        print(f"  主仓手数   : {main_lots:.2f}   对冲手数 : {hedge_lots:.2f}")
    else:
        print("  当前空仓")
    print(f"  权益/余额  : ${equity_now:,.0f} / ${balance:,.0f}")

    # ---- full risk profile ----
    print("\n[全程风险画像]")
    open_ev = ev[ev["event"] == "open"]
    if len(open_ev):
        peak_lots = float(open_ev["main_lots"].max())
        peak_layer = float(open_ev["layer"].max())
        # how often deep
        deep = open_ev[open_ev["main_lots"] >= ALERT_MAIN_LOTS]
        print(f"  马丁峰值   : 主仓 {peak_lots:.2f} 手 / 层数 {peak_layer:.0f} (上限{cfg.max_layers})")
        print(f"  深仓次数   : 主仓≥{ALERT_MAIN_LOTS:.1f}手 出现 {len(deep)} 次 / 开仓 {len(open_ev)} 次")
        lots_dist = open_ev["main_lots"].value_counts().sort_index()
        print("  手数分布   : " + "  ".join(f"{k:.1f}v:{v}" for k, v in lots_dist.items()))
        if peak_lots >= ALERT_MAIN_LOTS:
            print(f"  ⚠️ 警告: 马丁主仓曾堆到 {peak_lots:.2f} 手(≥{ALERT_MAIN_LOTS:.1f})，单边反向会急速放大风险")

    dd_series = (eq["equity"].cummax() - eq["equity"]) / eq["equity"].cummax()
    max_dd = dd_series.max()
    print(f"  全程最大回撤: {max_dd*100:.1f}%  (熔断线 {cfg.max_drawdown_pct*100:.0f}%)")
    if max_dd >= ALERT_DRAWDOWN:
        print(f"  ⚠️ 警告: 已接近/超过 {ALERT_DRAWDOWN*100:.0f}% 回撤预警, 距 {cfg.max_drawdown_pct*100:.0f}% 熔断线不远")
    dd_tail = float(dd_series.iloc[-1])
    print(f"  当前回撤   : {dd_tail*100:.1f}%")

    # daily P&L distribution
    if len(tr):
        tr = tr.copy()
        tr["date"] = pd.to_datetime(tr["time"], utc=True).dt.date
        daily = tr.groupby("date")["pnl_usc"].sum()
        worst = float(daily.min())
        nday = len(daily)
        losing_days = int((daily < 0).sum())
        print(f"  当日盈亏分布: 最大单日亏 ${worst:,.0f}  亏钱天数 {losing_days}/{nday}")
        if worst <= -args.balance * ALERT_DAILY_LOSS:
            print(f"  ⚠️ 警告: 曾有单日亏损 ${worst:,.0f} ≥ {ALERT_DAILY_LOSS*100:.0f}% 本金, 接近日亏熔断")

    # exit reason breakdown
    if len(tr):
        reason_count = tr[tr["is_hedge"] == False]["reason"].value_counts()
        hedge_count = int(tr["is_hedge"].sum())
        print("  离场原因   : " + "  ".join(f"{k}:{v}" for k, v in reason_count.items()) + f"  对冲平仓:{hedge_count}")

    # breaker events
    cb = ev[ev["event"] == "circuit_breaker"]
    print(f"  熔断开停   : {len(cb)} 次")
    if len(cb):
        print(f"  ⚠️ 已触发熔断: {cb.iloc[-1]['reason']} @ {cb.iloc[-1]['time']}")

    print("\n[累计绩效]")
    print(f"  总收益     : {s['total_return_pct']:+.1f}%  → ${s['final_equity']:,.0f}")
    print(f"  交易数     : {s['trades']}   胜率 {s['winrate']*100:.1f}%   PF {s['profit_factor']:.2f}")
    print(f"  初始资金   : ${args.balance:,.0f}")


if __name__ == "__main__":
    main()
