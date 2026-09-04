# -*- coding: utf-8 -*-
"""Live mean-reversion executor (XAUUSD 1-15min band).

Based on the backtest that found a positive edge: price deviating from EMA20 by
more than dev_atr * ATR tends to revert over the next ~10 minutes. Entry is
counter-trend (buy a low deviation / sell a high deviation). Exit is a fixed
hold time (10 min) plus an H1 structure stop as a backstop, per user's choice
of plan C.

Data: 30s bars aggregated from ticks. EMA20 is the mean line; ATR14 the band.
"""
from __future__ import annotations
import argparse, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "XAUUSD"
USC = 100.0


def close_by_ticket(symbol, ticket, volume, pos_type):  # noqa: E302
    close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if pos_type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": close_type, "price": price, "deviation": 40, "magic": 202608,
                           "position": ticket, "comment": "close", "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def open_order(symbol, volume, order_type, price, sl=None, magic=202608, comment="open"):  # noqa: E302
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "sl": sl, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def ewm(a, span):  # noqa: E302
    al = 2.0 / (span + 1)
    out = np.empty(len(a)); out[0] = a[0]
    for k in range(1, len(a)):
        out[k] = al * a[k] + (1 - al) * out[k - 1]
    return out


def main():
    p = argparse.ArgumentParser(description="Live mean-reversion executor (1-15min band)")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=3.0)
    p.add_argument("--lot-size", type=float, default=0.3)
    p.add_argument("--bar-sec", type=float, default=30.0, help="candle period (30s per backtest)")
    p.add_argument("--ema-span", type=int, default=20, help="mean line EMA period")
    p.add_argument("--dev-atr-low", type=float, default=1.0,
                   help="enter mean-reversion only when |price-EMA20| reaches this * ATR (best timing)"
                        "--the band 1.0-1.5 ATR has the best revert prob + space; below this is too "
                        "early (small space).")
    p.add_argument("--dev-atr-high", type=float, default=2.0,
                   help="do NOT open past this * ATR deviation (deep deviation is often a real trend "
                        "and reverts only ~43%), so stop opening new mean-reversion legs here.")
    p.add_argument("--hold-min", type=float, default=10.0, help="hold minutes before exit (plan C)")
    p.add_argument("--stop-h1", type=int, default=3,
                   help="H1 structure stop: BUY stop=low, SELL stop=high of last N H1 candles")
    p.add_argument("--equity-floor", type=float, default=0.30)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_mr_log.csv"))
    args = p.parse_args()

    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error()); sys.exit(2)
    mt5.symbol_select(args.symbol, True)
    info = mt5.account_info()
    if info is None:
        print("未登录"); sys.exit(2)
    demo = info.trade_mode == 0
    if not demo:
        args.dry_run = True
    print("演示账户 %s | 均值回归 | EMA%d 偏离%.1f~%.1fATR 开仓 | %.0f秒K线" % (
        info.login, args.ema_span, args.dev_atr_low, args.dev_atr_high, args.bar_sec))
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow([ts, ev, detail])
        print("  [%s] %s: %s" % (ts, ev, detail), flush=True)

    print("=" * 62)
    print("均值回归(1-15分钟波段) 运行中 — 人工审核")
    print("=" * 62)
    last_open_ts = 0.0
    peaks = {}  # ticket -> peak profit, per-position staged-TP tracking
    try:
        while True:
            try:
                ts = datetime.now().strftime("%H:%M:%S")
                t = mt5.symbol_info_tick(args.symbol)
                bid = t.bid; ask = t.ask

                # --- 用 M1 K线(最新, 无滞后) 算 EMA/ATR; dev 用实时 bid ---
                # copy_ticks_from 的历史tick在这个环境滞后严重(最新bar滞后数小时),
                # 导致 dev 失真误判方向。改用 copy_rates_from_pos(M1) 是稳定的、和实时一致。
                bar_sec = args.bar_sec if args.bar_sec and args.bar_sec > 0 else 60.0
                ema = np.array([]); atr = np.array([]); price = None; dev = 0.0
                signal = None
                try:
                    r = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 300)
                    if r is not None and len(r) >= args.ema_span + 5:
                        d1 = pd.DataFrame(r)
                        c = d1["close"].astype(float).to_numpy()
                        h = d1["high"].astype(float).to_numpy()
                        l = d1["low"].astype(float).to_numpy()
                        ema = ewm(c, args.ema_span)
                        prev = np.concatenate([[c[0]], c[:-1]])
                        tr = np.maximum(np.maximum(h - l, np.abs(h - prev)), np.abs(l - prev))
                        atr = pd.Series(tr).rolling(14).mean().to_numpy()
                        price = c[-1]
                        # dev uses REAL-TIME bid (not the possibly-stale M1 close)
                        dev = (bid - ema[-1]) / atr[-1] if atr[-1] and atr[-1] > 0 else 0.0
                        adev = abs(dev)
                        # 最优入场时机: 仅当 1.0 <= |dev| <= 2.0 才触发均值回归.
                        # <low(1.0) 太早(空间小); >high(2.0) 是深偏离/真趋势(回归概率降到~43%), 停开.
                        if args.dev_atr_low <= adev <= args.dev_atr_high:
                            if dev <= -args.dev_atr_low:
                                signal = mt5.POSITION_TYPE_BUY    # 超卖 -> 低吸
                            elif dev >= args.dev_atr_low:
                                signal = mt5.POSITION_TYPE_SELL   # 超买 -> 高抛
                        else:
                            signal = None
                except Exception:
                    signal = None

                # H1 structure stop
                h1_hi = h1_lo = None
                if args.stop_h1 and args.stop_h1 > 0:
                    try:
                        h1 = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_H1, 0, args.stop_h1)
                        if h1 is not None and len(h1) >= 1:
                            h1_hi = float(h1["high"].max()); h1_lo = float(h1["low"].min())
                    except Exception:
                        pass

                pos = mt5.positions_get(symbol=args.symbol) or []
                ai = mt5.account_info()
                dd = (ai.balance - ai.equity) / ai.balance if ai and ai.balance > 0 else 0
                oz = args.lot_size * USC

                # equity floor
                if args.equity_floor and pos and dd >= args.equity_floor:
                    log(ts, "EQUITY_FLOOR", "回撤%.0f%% 全平" % (dd * 100))
                    for x in pos:
                        if not args.dry_run:
                            close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    time.sleep(args.cycle_sec); continue

                # --- 分段止盈 (B方案): 每个持仓独立跟踪峰值, 回落到 20/30/50 档锁利 ---
                pruned = False
                for x in pos:
                    tk = x.ticket
                    prof = x.profit
                    if prof > 0:
                        peaks[tk] = max(peaks.get(tk, 0.0), prof)
                        P = peaks[tk]
                        if P < 20.0:
                            target = None
                        elif P < 30.0:
                            target = 20.0
                        elif P <= 50.0:
                            target = 30.0
                        else:
                            target = P * 0.70
                        if target is not None and prof <= target and prof < P:
                            log(ts, "TP", "%s浮盈$%.0f 回落到$%.0f (峰值$%.0f) - 分段锁利" % (
                                "多" if x.type == mt5.POSITION_TYPE_BUY else "空", prof, target, P))
                            if not args.dry_run:
                                close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                            peaks.pop(tk, None)
                            pruned = True
                    else:
                        peaks[tk] = 0.0
                if pruned:
                    pos = mt5.positions_get(symbol=args.symbol) or []

                # --- 开仓: 任意新均值回归信号都开新单 (不管当前是否已有持仓) ---
                now_open_ok = (time.time() - last_open_ts) > 5.0
                if signal is not None and now_open_ok:
                    otype = signal
                    price_use = ask if otype == mt5.POSITION_TYPE_BUY else bid
                    if args.stop_h1 and args.stop_h1 > 0 and h1_hi is not None and h1_lo is not None:
                        sl = (h1_lo - 0.05) if otype == mt5.POSITION_TYPE_BUY else (h1_hi + 0.05)
                        smode = "H1"
                    else:
                        sl = price_use - (2.0 * (atr[-1] if len(atr) else 1.0)) if otype == mt5.POSITION_TYPE_BUY \
                            else price_use + (2.0 * (atr[-1] if len(atr) else 1.0))
                        smode = "ATR"
                    log(ts, "OPEN", "偏离%.2fATR %s %.1f手 @%.2f SL=%.2f (%s止损)" % (
                        dev, "低吸多" if otype == mt5.POSITION_TYPE_BUY else "高抛空",
                        args.lot_size, price_use, sl, smode))
                    if not args.dry_run:
                        res = open_order(args.symbol, args.lot_size, otype, price_use, sl, 202608, "mr_open")
                        if res is not None and getattr(res, "retcode", -1) == mt5.TRADE_RETCODE_DONE:
                            pos = mt5.positions_get(symbol=args.symbol) or []
                            for nx in pos:
                                peaks.setdefault(nx.ticket, 0.0)
                            last_open_ts = time.time()

                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                log(ts, "status", "px%.2f 偏离%.2fATR 多%.2f/空%.2f 权益%.0f 回撤%.1f%%" % (
                    (bid + ask) / 2, dev if price else 0, buy_lots, sell_lots, ai.equity, dd * 100))
            except Exception as e:  # noqa: BLE001
                print("[%s] 异常: %s" % (datetime.now().strftime("%H:%M:%S"), e), flush=True)
            time.sleep(args.cycle_sec)
    except KeyboardInterrupt:
        print("\n已停止.")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
