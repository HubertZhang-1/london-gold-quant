# -*- coding: utf-8 -*-
"""MT5 connectivity check: confirms the local terminal is reachable and logs in.

Use AFTER you have launched your MT5 terminal and logged into your trading account.

Usage:
  py scripts/mt5_connect_check.py                 # check current connection/account
  py scripts/mt5_connect_check.py --login 123 --password **** --server BrokerServer
                                                 # programmatic login
  py scripts/mt5_connect_check.py --list-symbols  # list gold-ish symbols available
"""
from __future__ import annotations

import argparse
import sys

import MetaTrader5 as mt5  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MetaTrader5 connectivity / login")
    parser.add_argument("--login", type=int, default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--server", default=None)
    parser.add_argument("--list-symbols", action="store_true")
    args = parser.parse_args()

    if args.login:
        ok = mt5.initialize(login=args.login, password=args.password, server=args.server)
        if not ok:
            print("❌ 登录失败:", mt5.last_error())
            print("  请确认账号/密码/服务器正确，且 MT5 终端已启动。")
            sys.exit(2)
        print("✅ 已登录")
    else:
        if not mt5.initialize():
            print("❌ MT5 未连接:", mt5.last_error())
            print("  请先启动 MT5 终端（本机），并在其中登录交易账户。")
            print("  或用程序化登录: --login *** --password *** --server ***")
            sys.exit(2)
        print("✅ MT5 终端已连接")

    info = mt5.account_info()
    if info:
        print(f"  账户 {info.login}  余额 ${info.balance:.2f}  权益 ${info.equity:.2f}  "
              f"货币 {info.currency}  杠杆 1:{info.leverage}")
    else:
        print("  未登录交易账户（请在 MT5 内登录，或程序化登录）")

    if args.list_symbols:
        print("\n黄金相关品种:")
        for s in mt5.symbols_get():
            if s.name.lower().startswith(("xau", "gold", "xauusd")):
                print(f"  {s.name}")
    print("\n提示: 之后运行 py scripts/monitor_mt5_live.py 读取实时持仓/行情")

    mt5.shutdown()


if __name__ == "__main__":
    main()
