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
    p.add_argument("--lot-size", type=float, default=0.3, help="fixed lot per trade (0.3)")
    p.add_argument("--noise-amp", type=float, default=5.0, help="skip any cross whose recent 6-bar swing < this")
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
                ts = datetime.now().strftime("%H:%M:%S")
                sig = 0
                if len(spread_ma) >= 2:
                    if spread_ma[-1] > 0 and spread_ma[-2] <= 0:
                        sig = 1
                    elif spread_ma[-1] < 0 and spread_ma[-2] >= 0:
                        sig = -1
                # ---- K-line noise correction ----
                # a cross inside a tight MA-blend (small separation) or a small candle
                # body is likely chop -> treat as noise, skip it (只做明确的拐点).
                if sig != 0:
                    sep = abs(spread_ma[-1]) if len(spread_ma) else 0.0
                    body = abs(float(close.iloc[-1]) - float(d["open"].iloc[-1]))
                    ma_dist = abs(float(f.iloc[-1]) - float(s.iloc[-1]))
                    # noise: separation tiny / candle body tiny / recent swing small
                    swing = float(high.tail(6).max() - low.tail(6).min())
                    noisy = (ma_dist < 0.6) or (body < 0.8) or (swing < args.noise_amp)
                    if noisy:
                        log(ts, "noise_skip", "K线噪声: ma_dist=%.2f body=%.2f swing=%.2f 跳过" % (
                            ma_dist, body, swing))
                        sig = 0

                pos = mt5.positions_get(symbol=args.symbol) or []
                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                px = float(close.iloc[-1])
                base_lot = args.lot_size  # fixed 0.3 per trade (multi-position allowed)

                log(ts, "tick", "close=%.2f 多%.2f/空%.2f lot=%.2f sig=%+d" % (
                    px, buy_lots, sell_lots, base_lot, sig))

                # ---- multiple concurrent: on a fresh cross, open a new 0.3-lot; if an
                #      opposite signal appears, close the opposite-direction positions ----
                cur = 1 if buy_lots > 0 else (-1 if sell_lots > 0 else 0)
                if sig != 0:
                    if cur != 0 and sig != cur:
                        log(ts, "flip", "反向信号%+d - 平掉%+d方向仓位" % (sig, cur))
                        for x in pos:
                            opp = (x.type == mt5.POSITION_TYPE_BUY and sig < 0) or \
                                  (x.type == mt5.POSITION_TYPE_SELL and sig > 0)
                            if opp:
                                ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                                if not args.dry_run:
                                    send_order(args.symbol, x.volume, ct, cpx, 202605, "m520_flip")
                    # open a new 0.3-lot in signal direction (allows multiple concurrent)
                    otype = mt5.POSITION_TYPE_BUY if sig > 0 else mt5.POSITION_TYPE_SELL
                    px0 = mt5.symbol_info_tick(args.symbol).ask if sig > 0 else mt5.symbol_info_tick(args.symbol).bid
                    log(ts, "open", "%s %.2f手 @%.2f (信号%+d)" % (
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
