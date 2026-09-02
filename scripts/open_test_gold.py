# -*- coding: utf-8 -*-
"""Open a few gold positions on the (demo) account to test the live monitor.

SAFE: this only runs against a DEMO account (trade_mode == 0). It opens a small
two-sided hedged-grid set (a few buy + a few sell at the current market) to mimic the
martingale-grid basket, so the live monitor can read the open positions/unrealized PnL.

Usage:
  py scripts/open_test_gold.py            # open a small hedged set on XAUUSD (demo only)
  py scripts/open_test_gold.py --lots 0.1
  py scripts/open_test_gold.py --close-first   # close all XAUUSD before opening
"""
from __future__ import annotations

import argparse
import sys
import time

import MetaTrader5 as mt5

SYMBOL = "XAUUSD"


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a test gold basket on a DEMO account")
    parser.add_argument("--lots", type=float, default=0.1)
    parser.add_argument("--close-first", action="store_true")
    args = parser.parse_args()

    if not mt5.initialize():
        print("❌ MT5 init failed:", mt5.last_error())
        sys.exit(2)
    info = mt5.account_info()
    if info is None:
        print("❌ 未登录账户")
        sys.exit(2)
    if info.trade_mode != 0:
        print("⚠️ 非演示账户(trade_mode=%s)，为安全不自动下单。请用演示账户测试。" % info.trade_mode)
        mt5.shutdown()
        sys.exit(2)

    mt5.symbol_select(SYMBOL, True)
    time.sleep(1)
    tick = mt5.symbol_info_tick(SYMBOL)
    si = mt5.symbol_info(SYMBOL)
    if tick is None or si is None:
        print("❌ 无行情")
        sys.exit(2)

    # probe auto-trading is ON (retcode 10027 = AutoTrading disabled)
    probe = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
                            "volume": 0.01, "type": mt5.ORDER_TYPE_BUY,
                            "price": tick.ask, "deviation": 40, "magic": 0,
                            "comment": "autotrade_probe", "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": si.filling_mode or mt5.ORDER_FILLING_IOC})
    if probe.retcode == 10027:
        print("❌ 自动交易未开启 (retcode 10027)。")
        print("请在 MT5 里: 工具→选项→算法交易 → 勾选‘允许算法交易’，或点顶部‘算法交易’按钮开启。")
        mt5.shutdown()
        sys.exit(2)

    if args.close_first:
        poses = mt5.positions_get(symbol=SYMBOL) or []
        close_fill = mt5.ORDER_FILLING_IOC
        for p in poses:
            mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
                            "volume": p.volume, "type": mt5.ORDER_TYPE_BUY if p.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                            "position": p.ticket, "price": tick.bid, "deviation": 30, "magic": 0,
                            "comment": "close_test", "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": close_fill})
        print("已平掉现有持仓")

    lots = args.lots
    # resolve a usable filling mode: prefer IOC, then FOK, else the symbol's value
    si_fill = si.filling_mode
    for _fname, _fval in [("IOC", mt5.ORDER_FILLING_IOC), ("FOK", mt5.ORDER_FILLING_FOK)]:
        fill_probe = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
                                     "volume": 0.01, "type": mt5.ORDER_TYPE_BUY,
                                     "price": tick.ask, "deviation": 40, "magic": 0,
                                     "comment": "fill_probe", "type_time": mt5.ORDER_TIME_GTC,
                                     "type_filling": _fval})
        if fill_probe.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            # close the probe so the basket is clean
            probe_pos = mt5.positions_get(symbol=SYMBOL)
            if probe_pos:
                for pp in probe_pos:
                    if pp.comment == "fill_probe":
                        mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
                                        "volume": pp.volume,
                                        "type": mt5.ORDER_TYPE_SELL, "price": tick.bid,
                                        "deviation": 40, "magic": 0, "position": pp.ticket,
                                        "comment": "close_probe", "type_time": mt5.ORDER_TIME_GTC,
                                        "type_filling": _fval})
            FILL = _fval
            print(f"使用填充模式: {_fname}")
            break
    else:
        FILL = si_fill
        print(f"使用填充模式: symbol值({si_fill})")

    # a small hedged set: 2 buys + 2 sells at different lots, mimicking grid layers
    orders = [
        (mt5.ORDER_TYPE_BUY, lots),
        (mt5.ORDER_TYPE_BUY, round(lots * 2, 2)),
        (mt5.ORDER_TYPE_SELL, lots),
        (mt5.ORDER_TYPE_SELL, round(lots * 2, 2)),
    ]
    print(f"在 {SYMBOL} 开测试单 (演示账户, 手数基准 {lots}):")
    for typ, vol in orders:
        price = tick.ask if typ == mt5.ORDER_TYPE_BUY else tick.bid
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": vol,
               "type": typ, "price": price, "deviation": 40, "magic": 0,
               "comment": "grid_test", "type_time": mt5.ORDER_TIME_GTC,
               "type_filling": FILL}
        res = mt5.order_send(req)
        ok = res.retcode == mt5.TRADE_RETCODE_DONE
        print("  %s %.2f手 @%.2f -> %s %s" % (
            "BUY " if typ == mt5.ORDER_TYPE_BUY else "SELL", vol, price,
            "✅" if ok else "❌", res.comment))
        if not ok:
            print("     retcode:", res.retcode, res.comment)

    pos = mt5.positions_get(symbol=SYMBOL) or []
    total_lots = sum(p.volume for p in pos)
    total_profit = sum(p.profit for p in pos)
    print(f"\n当前持仓: {len(pos)} 单, 总手数 {total_lots:.2f}, 浮盈 ${total_profit:+,.2f}")
    print("接下来运行: py scripts/monitor_mt5_live.py  读取实时持仓/风险")
    mt5.shutdown()


if __name__ == "__main__":
    main()
