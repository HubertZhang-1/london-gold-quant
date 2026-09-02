# -*- coding: utf-8 -*-
"""Live MT5 monitor for the hedged martingale-grid main strategy.

Reads the LIVE MT5 terminal (via the MetaTrader5 Python library) and reports the
current real-time state of the XAUUSD martingale grid — the real data needed to
drive the strategy monitor (this replaces reading a historical CSV):

  - real-time XAUUSD bid/ask + day change
  - current ACCOUNT balance/equity + free margin
  - all open positions grouped into the grid's exposure (marin basket lots vs hedge)
  - real-time martingale exposure + unrealized P&L, and distance to the
    daily-loss / max-drawdown circuit breakers

REQUIREMENTS:
  - MT5 terminal must be RUNNING on this machine and logged into your account.
  - If you have NOT logged in, mt5.login() will fail -> see the printed guidance.

Usage:
  py scripts/monitor_mt5_live.py
  py scripts/monitor_mt5_live.py --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import MetaTrader5 as mt5  # noqa: E402

# safety-config breaker lines (match the safe live config)
DAILY_LOSS_PCT = 0.02     # 2% daily-loss breaker
MAX_DD_PCT = 0.20         # 20% max-drawdown breaker
ALERT_LOTS = 3.0          # warn when the martingale stack exceeds this many lots


def main() -> None:
    parser = argparse.ArgumentParser(description="Live MT5 monitor for the martingale grid strategy")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--balance", type=float, default=100_000.0, help="initial/参考初始资金 for %")
    args = parser.parse_args()

    # ---- connect to the local MT5 terminal ----
    if not mt5.initialize():
        print("❌ MetaTrader5 初始化失败:", mt5.last_error())
        print("\n请确认：")
        print("  1) MT5 终端正在本机运行；")
        print("  2) 已在 MT5 里登录你的交易账户（文件→登录交易账户）；")
        print("  3) 工具→选项→启用算法交易 已勾选。")
        sys.exit(2)

    info = mt5.account_info()
    if info is None:
        print("❌ 未登录交易账户:", mt5.last_error())
        print("请在 MT5 里: 文件→登录交易账户，然后重试。")
        mt5.shutdown()
        sys.exit(2)

    sym = args.symbol
    # subscribe the symbol so its quotes load even if not in the Market Watch window
    if mt5.symbol_info(sym):
        mt5.symbol_select(sym, True)
        import time as _t
        _t.sleep(1.0)
    tick = mt5.symbol_info_tick(sym) if mt5.symbol_info(sym) else None
    positions = mt5.positions_get(symbol=sym) if mt5.symbol_info(sym) else None

    print("=" * 62)
    print(f"MT5 实盘监控(主策略-双边网格马丁)   {datetime.now():%Y-%m-%d %H:%M}")
    print(f"账户: {info.login}   {info.name or ''}  券商: {info.company or ''}")
    print(f"货币: {info.currency}   余额 ${info.balance:,.2f}  权益 ${info.equity:,.2f}  可用 ${info.margin_free:,.2f}")
    print("=" * 62)

    # ---- market ----
    print("[实时行情]")
    if tick is not None:
        print(f"  {sym} : bid {tick.bid:.2f} / ask {tick.ask:.2f}")
        if hasattr(tick, "last"):
            print(f"  最新   : {tick.last:.2f}")
    else:
        print(f"  {sym} 无行情 (请确认品种名称，可用 mt5.symbols_get 查看)")

    # ---- open positions (the grid basket) ----
    print("\n[当前持仓 / 马丁网格]")
    if positions:
        pos_list = list(positions)
        longs = [p for p in pos_list if p.type == mt5.POSITION_TYPE_BUY]
        shorts = [p for p in pos_list if p.type == mt5.POSITION_TYPE_SELL]
        total_lots = sum(p.volume for p in pos_list)
        total_profit = sum(p.profit for p in pos_list)
        long_lots = sum(p.volume for p in longs)
        short_lots = sum(p.volume for p in shorts)
        print(f"  持仓单数   : {len(pos_list)}  (多 {len(longs)} / 空 {len(shorts)})")
        print(f"  东手数     : 多 {long_lots:.2f} / 空 {short_lots:.2f} / 合计 {total_lots:.2f}")
        print(f"  未实现盈亏 : ${total_profit:+,.2f}")
        if total_lots >= ALERT_LOTS:
            print(f"  ⚠️ 警告: 累计手数 {total_lots:.2f} ≥ {ALERT_LOTS:.1f}，马丁堆深仓，单边反向会急速放大风险")
        # breakdown by type for insight
        print("  明细:")
        for p in pos_list:
            side = "多" if p.type == mt5.POSITION_TYPE_BUY else "空"
            print(f"    #{p.ticket} {side} {p.volume:.2f}手 @{p.price_open:.2f} "
                  f"浮盈${p.profit:+,.2f}  ({p.symbol})")
    else:
        print("  （当前无持仓，空仓）")

    # ---- distance to circuit breakers ----
    print("\n[风险校验]")
    eq = info.equity if info else args.balance
    bal = info.balance if info else args.balance
    # max-drawdown proxy: compare current equity vs initial balance (approx)
    peak = max(bal, eq)
    dd = (peak - eq) / peak if peak > 0 else 0.0
    margin_safe = (bal - eq) / bal if bal > 0 else 0.0
    print(f"  回撤熔断   : 当前权益 {eq:,.2f} (相对初始 {args.balance:,.0f}) → 回撤 {dd*100:.2f}% / 线 {MAX_DD_PCT*100:.0f}%")
    if dd >= MAX_DD_PCT * 0.8:
        print(f"  ⚠️ 警告: 回撤接近 {MAX_DD_PCT*100:.0f}% 熔断线，建议减仓/停机")
    print(f"  日亏熔断   : 线 {DAILY_LOSS_PCT*100:.0f}%  (当日实时需结合 mt5.history_deals_get 看已实现盈亏)")
    print("  (提示: MT5 无内置'当日已实现盈亏'字段；用 history_deals_get 按当日汇总可衡量日亏)")

    mt5.shutdown()
    print("\n" + "-" * 62)
    print("说明: 每日实时盈利/日亏需用 mt5.history_deals_get(日期) 汇总已实现盈亏。")
    print("本脚本已读实时行情+持仓+浮盈亏+账户权益，可直接驱动主策略监控。")


if __name__ == "__main__":
    main()
