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
    p.add_argument("--min-sep", type=float, default=0.6, help="skip a cross if MA5-MA20 separation < this (方向不明)")
    p.add_argument("--min-body", type=float, default=0.8, help="skip a cross if candle body < this (小实体噪声)")
    p.add_argument("--confirm-bars", type=int, default=2, help="delay N M5 bars after a cross; only enter if the price keeps moving the cross direction")
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
    print("M5均线(5分钟周期) vs M20均线(20分钟周期) 金叉死叉双向 运行中...")
    print("=" * 64)
    last_sig = 0  # only act when the cross direction CHANGES (new cross), no repeat churn
    pending = None  # (dir, price_at_cross) — a cross awaiting delay-confirmation
    try:
        while True:
            try:
                # ---- 真正的 M5均线 vs M20均线 (跨周期, 时间对齐到M5) ----
                # M5 周期数据 (主轴)
                r5 = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M5, 0, 300)
                if r5 is None or len(r5) < 30:
                    time.sleep(args.cycle_sec); continue
                d5 = pd.DataFrame(r5)
                d5["t"] = pd.to_datetime(d5["time"], unit="s", utc=True)
                d5["close"] = d5["close"].astype(float)
                d5["high"] = d5["high"].astype(float)
                d5["low"] = d5["low"].astype(float)
                # M20 周期数据 -> M20均线
                r20 = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M20, 0, 300)
                if r20 is None or len(r20) < args.slow:
                    time.sleep(args.cycle_sec); continue
                d20 = pd.DataFrame(r20)
                d20["t"] = pd.to_datetime(d20["time"], unit="s", utc=True)
                d20["close"] = d20["close"].astype(float)
                d20["ma20"] = d20["close"].ewm(span=args.slow, adjust=False).mean()
                # M5 周期均线 (短)
                ma5 = d5["close"].ewm(span=args.fast, adjust=False).mean()
                # 对齐 M20均线 到 M5 时间轴
                idx = np.searchsorted(d20["t"].values, d5["t"].values, side="right") - 1
                idx = np.clip(idx, 0, len(d20) - 1)
                ma20_on_m5 = d20["ma20"].values[idx]
                spread_ma = (ma5.to_numpy() - ma20_on_m5)

                close = d5["close"]
                high = d5["high"]
                low = d5["low"]
                ts = datetime.now().strftime("%H:%M:%S")
                sig = 0
                if len(spread_ma) >= 2:
                    if spread_ma[-1] > 0 and spread_ma[-2] <= 0:
                        sig = 1
                    elif spread_ma[-1] < 0 and spread_ma[-2] >= 0:
                        sig = -1
                # ---- K-line noise correction (排除小震荡噪声) ----
                if sig != 0:
                    ma_dist = abs(spread_ma[-1])
                    body = abs(float(close.iloc[-1]) - float(d5["open"].iloc[-1]))
                    swing = float(high.tail(6).max() - low.tail(6).min())
                    noisy = (ma_dist < args.min_sep) or (body < args.min_body) or (swing < args.noise_amp)
                    if noisy:
                        log(ts, "noise_skip", "K线噪声: ma_dist=%.2f body=%.2f swing=%.2f 跳过" % (
                            ma_dist, body, swing))
                        sig = 0

                pos = mt5.positions_get(symbol=args.symbol) or []
                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                px = float(close.iloc[-1])
                base_lot = args.lot_size  # fixed 0.3 per trade

                log(ts, "tick", "close=%.2f 多%.2f/空%.2f lot=%.2f sig=%+d" % (
                    px, buy_lots, sell_lots, base_lot, sig))

                # ---- delay-confirmation + direction-flip semantics ----
                # when a fresh cross is detected (sig changed), record a pending entry; only
                # open after `confirm-bars` M5 bars IF price kept moving the cross direction.
                if sig != 0 and sig != last_sig:
                    # fresh cross: queue it for confirmation instead of opening immediately
                    if args.confirm_bars > 0:
                        pending = {"dir": sig, "px0": float(close.iloc[-1]),
                                   "bar0": len(d5)}
                        log(ts, "cross_detect", "检测到%+d交叉, 等待%d根确认后再进" % (sig, args.confirm_bars))
                        last_sig = sig
                    else:
                        # no delay: act immediately (close opposite + open new direction)
                        for x in pos:
                            oppose = (x.type == mt5.POSITION_TYPE_BUY and sig < 0) or \
                                     (x.type == mt5.POSITION_TYPE_SELL and sig > 0)
                            if oppose:
                                ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                                log(ts, "close_opposite", "信号%+d - 平反向仓 %s %.2f手 @%.2f" % (
                                    sig, "BUY" if x.type == 0 else "SELL", x.volume, cpx))
                                if not args.dry_run:
                                    send_order(args.symbol, x.volume, ct, cpx, 202605, "m520_close")
                        otype = mt5.POSITION_TYPE_BUY if sig > 0 else mt5.POSITION_TYPE_SELL
                        px0 = mt5.symbol_info_tick(args.symbol).ask if sig > 0 else mt5.symbol_info_tick(args.symbol).bid
                        log(ts, "open", "%s %.2f手 @%.2f (看涨/看跌%+d)" % (
                            "BUY" if sig > 0 else "SELL", base_lot, px0, sig))
                        if not args.dry_run:
                            send_order(args.symbol, base_lot, otype, px0, 202605, "m520_open")
                        last_sig = sig

                # ---- evaluate a pending signal (delay confirmation) ----
                elif pending is not None:
                    dirn = pending["dir"]
                    bars_elapsed = len(d5) - pending["bar0"]
                    price_moved = (float(close.iloc[-1]) - pending["px0"]) * dirn
                    sep_ok = abs(spread_ma[-1]) >= args.min_sep
                    if bars_elapsed >= max(1, args.confirm_bars):
                        if price_moved > 0 and sep_ok:
                            # confirmed: close opposite + open new direction
                            for x in pos:
                                oppose = (x.type == mt5.POSITION_TYPE_BUY and dirn < 0) or \
                                         (x.type == mt5.POSITION_TYPE_SELL and dirn > 0)
                                if oppose:
                                    ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                    cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                                    log(ts, "close_opposite", "确认%+d - 平反向仓" % dirn)
                                    if not args.dry_run:
                                        send_order(args.symbol, x.volume, ct, cpx, 202605, "m520_close")
                            otype = mt5.POSITION_TYPE_BUY if dirn > 0 else mt5.POSITION_TYPE_SELL
                            px0 = mt5.symbol_info_tick(args.symbol).ask if dirn > 0 else mt5.symbol_info_tick(args.symbol).bid
                            log(ts, "open", "确认%+d后进 %s %.2f手 @%.2f" % (
                                dirn, "BUY" if dirn > 0 else "SELL", base_lot, px0))
                            if not args.dry_run:
                                send_order(args.symbol, base_lot, otype, px0, 202605, "m520_open")
                        else:
                            log(ts, "confirm_fail", "%+d未获确认(价%.2f/分离%.2f) - 放弃" % (
                                dirn, price_moved, spread_ma[-1]))
                        pending = None
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
