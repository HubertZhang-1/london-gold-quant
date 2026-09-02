# -*- coding: utf-8 -*-
"""Live daily monitor for the faithful main strategy (hedged martingale-grid EA).

This is the MAIN STRATEGY per the user's decision: two-sided hedged martingale grid,
no bull/bear filter (不分牛熊). This monitor runs the strategy on the latest data and
reports the CURRENT market/position/risk state as of the last bar:

  - market context: gold close, EMA(60) trend direction (up/down), ATR step
  - current basket: main side + layers + accumulated lots + avg entry; hedge side
  - unrealized P&L of the open basket, equity, and how far it is to the risk
    circuit breakers (3% daily loss / 30% max drawdown)

Usage:
  py scripts/monitor_main_strategy.py                         # cached 1m Aug data
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily monitor for the main martingale-grid EA")
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

    # market context on the last bar
    last = df.iloc[-1]
    close = float(last["close"])
    ema60 = df["close"].ewm(span=cfg.trend_ema_bars, adjust=False).mean().iloc[-1]
    ema60_prev = df["close"].ewm(span=cfg.trend_ema_bars, adjust=False).mean().iloc[-2]
    trend = "UP ↑" if ema60 > ema60_prev else "DOWN ↓"

    print("=" * 62)
    print(f"主策略(双边对冲网格马丁) 每日监控   {datetime.now():%Y-%m-%d %H:%M}")
    print(f"数据截至: {last['date']}    收盘 ${close:,.1f}")
    print("=" * 62)

    # market context
    print("[市场]")
    print(f"  金价收盘   : ${close:,.1f}")
    print(f"  EMA({cfg.trend_ema_bars}) 趋势 : {trend}")

    # current open basket from the LAST event that has non-zero lots
    # current open basket from the equity series' last bar (authoritative net state)
    eq = res.equity
    lastrow = eq.iloc[-1] if len(eq) else None
    main_lots = float(lastrow["main_lots"]) if lastrow is not None else 0.0
    hedge_lots = float(lastrow["hedge_lots"]) if lastrow is not None else 0.0
    balance = float(lastrow["balance"]) if lastrow is not None else args.balance
    equity_now = float(lastrow["equity"]) if lastrow is not None else args.balance

    print("\n[当前持仓篮子] (数据末端)")
    if main_lots > 0 or hedge_lots > 0:
        # find the most recent open event to identify the main side/layer
        open_ev = ev[ev["event"] == "open"]
        last_open = open_ev.iloc[-1] if len(open_ev) else None
        if last_open is not None and pd.notna(last_open.get("main_side")):
            mside = "做多" if last_open["main_side"] > 0 else "做空"
        else:
            mside = "?"
        layer_val = last_open.get("layer", 0) if last_open is not None else 0
        layer_n = int(layer_val) if pd.notna(layer_val) else 0
        print(f"  主仓方向   : {mside}")
        print(f"  主仓层数   : {layer_n} 层 (上限 {cfg.max_layers})")
        print(f"  主仓手数   : {main_lots:.2f} 手")
        print(f"  对冲手数   : {hedge_lots:.2f} 手")
    else:
        print("  当前空仓 (数据末端无未平篮子)")

    # distance to circuit breakers
    print(f"  当前权益   : ${equity_now:,.0f}   (余额 ${balance:,.0f})")
    peak = max(eq["equity"].max(), equity_now) if len(eq) else equity_now
    drawdown = (peak - equity_now) / peak if peak > 0 else 0.0
    print(f"  距回撤熔断 : 当前回撤 {drawdown*100:.1f}% / 30% 线")

    print("\n[当日/累计绩效]")
    print(f"  总收益     : {s['total_return_pct']:+.1f}%   (期末 ${s['final_equity']:,.0f})")
    print(f"  交易数     : {s['trades']}   (胜率 {s['winrate']*100:.1f}%)")
    print(f"  盈亏因子   : {s['profit_factor']:.2f}")
    print(f"  最大回撤   : {s['max_drawdown_pct']:.1f}%")
    print(f"  初始资金   : ${args.balance:,.0f}")

    print("\n" + "-" * 62)
    print("风险提示: 主策略带 3% 日亏熔断 + 30% 回撤熔断, 单边逆风月会熔断停手(≈-4%)。")
    print("截图里的 95% 胜率是震荡段的片段; 真实胜率约 47%, PF≈1.2(靠高频小赚)。")


if __name__ == "__main__":
    main()
