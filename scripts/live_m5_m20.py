# -*- coding: utf-8 -*-
"""Live M5/M20 MA golden/dead-cross BOTH-direction executor (M5图上的金叉死叉双向).

User's strategy: on the M5 timeframe, trade the cross of the M5-MA vs M20-MA:
  - golden cross (M5-MA up through M20-MA) -> close short, go LONG
  - dead cross (M5-MA down through M20-MA) -> close long, go SHORT

Built off the verified backtest (M5/M20 on real data: +10.2%, PF 1.07, positive vs
cost $37/lot). Uses the current M5 EMA5/EMA20; re-evaluates each cycle. 5x leverage.
SAFETY: only auto-trades on a DEMO account (trade_mode==0); else dry-run.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "XAUUSD"
SPREAD = 0.37
USC = 100.0


def leverage_lots(equity, price, lev, oz=100.0):
    return max(0.01, round(lev * equity / (price * oz), 2))


def send_order(symbol, volume, order_type, price, magic, comment, filling=mt5.ORDER_FILLING_IOC):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": filling})


def main():
    p = argparse.ArgumentParser(description="Live M5/M20 cross both-direction executor")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=10.0)
    p.add_argument("--fast", type=int, default=5)
    p.add_argument("--slow", type=int, default=20)
    p.add_argument("--target-leverage", type=float, default=5.0)
    p.add_argument("--noise-amp", type=float, default=0.0, help="skip turns whose prior 6-bar swing < this")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_m5_m20_log.csv"))
    args = p.parse_args()

    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error()); sys.exit(2)
    mt5.symbol_select(args.symbol, True)
    info = mt5.account_info()
    if info is None:
        print("未登录"); sys.exit(2)
    demo = info.trade_mode == 0
    if not demo:
        print("非演示 -> DRY-RUN"); args.dry_run = True
    print("演示账户 %s | %s | M5/M20(快%d/慢%d) 双向 | 杠杆%.1fx | 噪声过滤%.1f点" % (
        info.login, "DRY-RUN" if args.dry_run else "自动下单",
        args.fast, args.slow, args.target_leverage, args.noise_amp))

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow([ts, ev, detail])
        print("  [%s] %s: %s" % (ts, ev, detail), flush=True)

    print("=" * 64)
    print("M5/M20 金叉死叉双向 运行中...")
    print("=" * 64)
    try:
        while True:
            try:
                rates = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M5, 0, 200)
                if rates is None or len(rates) < 30:
                    time.sleep(args.cycle_sec); continue
                d = pd.DataFrame(rates)
                close = d["close"].astype(float)
                high = d["high"].astype(float)
                low = d["low"].astype(float)
                f = close.ewm(span=args.fast, adjust=False).mean()
                s = close.ewm(span=args.slow, adjust=False).mean()
                spread_ma = (f - s).to_numpy()
                sig = 0
                if len(spread_ma) >= 2:
                    if spread_ma[-1] > 0 and spread_ma[-2] <= 0:
                        sig = 1
                    elif spread_ma[-1] < 0 and spread_ma[-2] >= 0:
                        sig = -1
                # noise filter
                if sig != 0 and args.noise_amp > 0:
                    amp = float(high.tail(6).max() - low.tail(6).min())
                    if amp < args.noise_amp:
                        sig = 0

                pos = mt5.positions_get(symbol=args.symbol) or []
                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                ai = mt5.account_info()
                equity = ai.equity if ai else 0
                px = float(close.iloc[-1])
                base_lot = leverage_lots(equity, px, args.target_leverage)

                ts = datetime.now().strftime("%H:%M:%S")
                log(ts, "tick", "close=%.2f 多%.2f/空%.2f lot=%.2f sig=%+d" % (
                    px, buy_lots, sell_lots, base_lot, sig))

                # exit current side when opposite signal
                cur = 1 if buy_lots > 0 else (-1 if sell_lots > 0 else 0)
                if sig != 0 and cur != 0 and sig != cur:
                    log(ts, "flip", "信号%+d 反向 - 平仓" % sig)
                    for x in pos:
                        ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                        if not args.dry_run:
                            send_order(args.symbol, x.volume, ct, cpx, 202605, "m520_flip")
                elif cur == 0 and sig != 0:
                    otype = mt5.POSITION_TYPE_BUY if sig > 0 else mt5.POSITION_TYPE_SELL
                    px0 = mt5.symbol_info_tick(args.symbol).ask if sig > 0 else mt5.symbol_info_tick(args.symbol).bid
                    log(ts, "open", "%s %.2f手 @%.2f (金叉/死叉%+d)" % (
                        "BUY" if sig > 0 else "SELL", base_lot, px0, sig))
                    if not args.dry_run:
                        send_order(args.symbol, base_lot, otype, px0, 202605, "m520_open")
            except Exception as e:  # noqa: BLE001
                print("[%s] 异常: %s" % (datetime.now().strftime("%H:%M:%S"), e), flush=True)
            time.sleep(args.cycle_sec)
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
