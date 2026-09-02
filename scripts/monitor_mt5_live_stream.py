# -*- coding: utf-8 -*-
"""Live, continuously-refreshing MT5 monitor for the hedged martingale-grid strategy.

Runs an infinite loop, re-reading the MT5 terminal every ``--interval`` seconds, and
prints a compact live status: real-time XAUUSD quote, account equity/free margin,
the open grid basket (long/short lots, per-side unrealized PnL), and the distance to
the daily-loss / max-drawdown breakers. Risk thresholds are highlighted.

Requirements: MT5 terminal running locally + logged in + algorithmic trading enabled.

Usage:
  py scripts/monitor_mt5_live_stream.py                      # every 3s
  py scripts/monitor_mt5_live_stream.py --interval 5
  py scripts/monitor_mt5_live_stream.py --break-after 10     # run 10 cycles then stop
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import MetaTrader5 as mt5

SYMBOL = "XAUUSD"
DAILY_LOSS_PCT = 0.02    # 2% daily-loss breaker
MAX_DD_PCT = 0.20        # 20% max-drawdown breaker
ALERT_LOTS = 3.0         # warn when the martingale stack exceeds this many lots


def read_snapshot(sym: str = SYMBOL) -> dict:
    """Read one live snapshot from MT5; returns dict or raises if not connected."""
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("not logged in")
    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        raise RuntimeError("no quote (run mt5.symbol_select or add symbol)")
    pos = mt5.positions_get(symbol=sym) or []
    mt5.shutdown()
    buys = sum(p.volume for p in pos if p.type == mt5.POSITION_TYPE_BUY)
    sells = sum(p.volume for p in pos if p.type == mt5.POSITION_TYPE_SELL)
    buy_profit = sum(p.profit for p in pos if p.type == mt5.POSITION_TYPE_BUY)
    sell_profit = sum(p.profit for p in pos if p.type == mt5.POSITION_TYPE_SELL)
    peak = max(info.equity, info.balance)
    dd = (peak - info.equity) / peak if peak > 0 else 0.0
    return {
        "time": datetime.now(), "bid": tick.bid, "ask": tick.ask,
        "balance": info.balance, "equity": info.equity, "margin_free": info.margin_free,
        "n_pos": len(pos), "buys": buys, "sells": sells,
        "buy_profit": buy_profit, "sell_profit": sell_profit, "unreal": info.profit,
        "dd": dd, "total_lots": buys + sells,
    }


def fmt_snapshot(s: dict) -> str:
    risk = []
    if s["total_lots"] >= ALERT_LOTS:
        risk.append(f"⚠️手数{s['total_lots']:.2f}≥{ALERT_LOTS}")
    if s["dd"] >= MAX_DD_PCT * 0.8:
        risk.append(f"⚠️回撤{s['dd']*100:.1f}%接近{MAX_DD_PCT*100:.0f}%")
    risk_txt = "  " + "  ".join(risk) if risk else ""
    pos_txt = f"{s['n_pos']}单 多{s['buys']:.2f}/空{s['sells']:.2f}手 浮盈${s['unreal']:+,.2f}"
    return (f"[{s['time']:%H:%M:%S}] XAUUSD {s['bid']:.2f}/{s['ask']:.2f}  "
            f"权益${s['equity']:,.0f}/余额${s['balance']:,.0f}  "
            f"{pos_txt}  回撤{s['dd']*100:.2f}%{risk_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous live MT5 monitor")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between refreshes")
    parser.add_argument("--break-after", type=int, default=None, help="run N cycles then exit (else infinite)")
    args = parser.parse_args()
    sym = args.symbol

    print("=" * 66)
    print(f"主策略(双边网格马丁) 实时监控   [Ctrl+C 停止]   刷新间隔 {args.interval}s")
    print(f"风险线: 马丁手数≥{ALERT_LOTS}预警 / 回撤{MAX_DD_PCT*100:.0f}%熔断 / 日亏{DAILY_LOSS_PCT*100:.0f}%")
    print("=" * 66)

    n = 0
    try:
        while True:
            try:
                s = read_snapshot(sym)
                print(fmt_snapshot(s), flush=True)
                n += 1
                if args.break_after is not None and n >= args.break_after:
                    print("\n(达到 break-after 次数，停止)", flush=True)
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[{datetime.now():%H:%M:%S}] ⚠️ 读取失败: {e}   (等待下一轮)", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止监控。")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
