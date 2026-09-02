# -*- coding: utf-8 -*-
"""Close ALL open positions on the symbol (demo-safe). Use to reset before auto-trading."""
import argparse
import sys
import time

import MetaTrader5 as mt5


def main() -> None:
    parser = argparse.ArgumentParser(description="Close all open positions on a symbol (or all)")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--all", action="store_true", help="close positions on ALL symbols")
    args = parser.parse_args()

    if not mt5.initialize():
        print("❌ MT5 init failed:", mt5.last_error()); sys.exit(2)
    info = mt5.account_info()
    if info is None:
        print("❌ 未登录"); sys.exit(2)
    mt5.symbol_select(args.symbol, True)

    pos = mt5.positions_get() if args.all else mt5.positions_get(symbol=args.symbol)
    pos = pos or []
    print(f"待平仓持仓: {len(pos)} 单")
    for p in pos:
        tick = mt5.symbol_info_tick(p.symbol)
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
        res = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
                              "volume": p.volume, "type": close_type, "position": p.ticket,
                              "price": price, "deviation": 40, "magic": 0,
                              "comment": "reset_close", "type_time": mt5.ORDER_TIME_GTC,
                              "type_filling": mt5.ORDER_FILLING_IOC})
        print(f"  #{p.ticket} {p.symbol} {'BUY' if p.type==0 else 'SELL'} {p.volume:.2f}手 -> "
              f"{'✅' if res.retcode==mt5.TRADE_RETCODE_DONE else '❌'} {res.comment}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
