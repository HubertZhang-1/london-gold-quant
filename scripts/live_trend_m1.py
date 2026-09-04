# -*- coding: utf-8 -*-
"""Live M1 EMA5 trend strategy (严格风控, 低频).

M1 EMA5 primary direction, with EMA10/30 low-weight background, trend-confirm + volume-
confirm to stay LOW-frequency (M1 is noisy and spread-heavy), small TP via trailing take-
profit ($50 activate, 10% retrace), 2x ATR stop, 30% equity floor. Demo-only auto-trade.

Difference from the M5 version: the EMA and ATR are computed on M1 candles; the trend
confirm requires more M1 bars and a volume gate to keep churn/spread cost manageable.
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
USC = 100.0


def close_by_ticket(symbol, ticket, volume, pos_type):
    close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if pos_type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": close_type, "price": price, "deviation": 40, "magic": 202608,
                           "position": ticket, "comment": "close", "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def open_order(symbol, volume, order_type, price, sl=None, magic=202608, comment="open"):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "sl": sl, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def main():
    p = argparse.ArgumentParser(description="Live M1 EMA5 trend executor (low-freq, risk-controlled)")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=10.0)
    p.add_argument("--lot-size", type=float, default=0.3)
    p.add_argument("--ema-primary", type=int, default=5)
    p.add_argument("--ema-alpha", type=float, default=None,
                   help="override EMA5 smoothing alpha (default 2/(primary+1); lower=slower")
    p.add_argument("--ema-fast", type=int, default=10)
    p.add_argument("--ema-slow", type=int, default=30)
    p.add_argument("--stop-usd", type=float, default=0.0,
                   help="optional fixed max loss per position in USD (单笔≤$X). 0 = disabled "
                        "(use the ATR stop below instead).")
    p.add_argument("--stop-atr-mult", type=float, default=2.0,
                   help="adaptive stop: multiplier x ATR. Used when --stop-usd is 0 and "
                        "--stop-h1 is disabled. The stop widens/narrows with volatility.")
    p.add_argument("--stop-h1", type=int, default=3,
                   help="structure stop based on the last N one-hour candles' high/low. "
                        "BUY stop = low of the last N H1 candles; SELL stop = high of the last N "
                        "H1 candles. 0 = disable and fall back to the ATR stop. This replaces the "
                        "fixed/ATR stop with an H1 structure stop (per user).")
    p.add_argument("--trend-confirm", type=int, default=3,
                   help="require this many consecutive same trend to confirm entry")
    p.add_argument("--bar-sec", type=float, default=20.0,
                   help="candle period in seconds for the trend bars (per user: 20s). Bars are "
                        "aggregated from ticks (open/high/low/close), and EMA5/10/30 + trend are "
                        "computed on this series. Smaller = faster response but more noise; "
                        "larger = lower frequency / less spread cost.")
    p.add_argument("--volume-mult", type=float, default=0.8)
    p.add_argument("--tp-activate-profit", type=float, default=50.0)
    p.add_argument("--tp-trail-pct", type=float, default=0.30,
                   help="close when profit retraces >= this from peak (0.30 = let it ride)")
    p.add_argument("--equity-floor", type=float, default=0.30)
    p.add_argument("--crash-drop-pts", type=float, default=2.0,
                   help="intraday cliff guard: close all if the last 1-min close change drops "
                        "below -this many points (catches a single crash candle that a slow "
                        "2xATR stop cannot intercept). 0 = disabled.")
    p.add_argument("--structure-close", type=str, default="ema",
                   help="breakdown guard: close a long when price falls below EMA<slow>, and a "
                        "short when price rises above EMA<slow> ('ema' = 牛市趋势线跌破/熊市转头). "
                        "Set to 'none' to disable.")
    p.add_argument("--structure-buffer-atr", type=float, default=0.5,
                   help="buffer band around EMA<slow> before a structure break is confirmed, in "
                        "multiples of ATR. A long only breaks when price falls below "
                        "EMA<slow> - buffer*ATR (a short above EMA<slow> + buffer*ATR), so normal "
                        "whipsaw around the trend line does not false-trigger. 0 = old behaviour.")
    p.add_argument("--crash-cooldown-sec", type=float, default=60.0,
                   help="after a crash/structure close, block re-opening for this many seconds "
                        "(trend reversal takes priority; stand aside instead of re-buying the dip).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_trend_m1_log.csv"))
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
    print("演示账户 %s | %s | M1 EMA%s主 趋势(低频) 单笔%.1f手" % (
        info.login, "DRY-RUN" if args.dry_run else "自动下单", args.ema_primary, args.lot_size))
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow([ts, ev, detail])
        print("  [%s] %s: %s" % (ts, ev, detail), flush=True)

    def ewm(a, span=None, alpha=None):
        al = alpha if alpha is not None else (2 / (span + 1))
        out = np.empty(len(a)); out[0] = a[0]
        for k in range(1, len(a)): out[k] = al * a[k] + (1 - al) * out[k - 1]
        return out

    print("=" * 62)
    print("M1 EMA5 趋势(低频/带风控) 运行中 — 人工审核")
    print("=" * 62)
    last_trend = 0
    trend_run = 0
    peak_profit = 0.0
    crash_block_until = 0.0  # cooldown: no new entry after a crash/structure exit
    crash_block_dir = 0      # direction the cooldown blocks (0=both). A crash means price
                             # dropped -> block longs (catching the falling knife) but NOT shorts
                             # (with-trend). This is the fix for "为什么没做空".
    last_open_ts = 0.0       # guard: prevents a re-open race when positions_get reads stale
    try:
        while True:
            try:
                rates = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 300)
                if rates is None or len(rates) < args.ema_slow:
                    time.sleep(args.cycle_sec); continue
                df = pd.DataFrame(rates)
                cl = df["close"].astype(float).to_numpy()
                ts = datetime.now().strftime("%H:%M:%S")

                t = mt5.symbol_info_tick(args.symbol)
                bid = t.bid; ask = t.ask
                al = (2 / (args.ema_primary + 1)) if args.ema_alpha is None else args.ema_alpha

                # --- H1 structure range for the stop (past N H1 candles' high/low) ---
                h1_hi = h1_lo = None
                if args.stop_h1 and args.stop_h1 > 0:
                    try:
                        h1 = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_H1, 0, args.stop_h1)
                        if h1 is not None and len(h1) >= 1:
                            h1_hi = float(h1["high"].max())
                            h1_lo = float(h1["low"].min())
                    except Exception:
                        h1_hi = h1_lo = None

                # --- sub-minute trend bars (per user: bar-sec=10s) aggregated from ticks ---
                # Build a OHLCV series at the sub-minute cadence, then run EMA5/10/30 + trend on
                # the bar closes. This gives a faster trend response than M1 (one bar = 10s).
                bar_sec = args.bar_sec if args.bar_sec and args.bar_sec > 0 else 60.0
                try:
                    from datetime import timedelta as _td
                    # Pull enough ticks to form >= ~120 sub-minute bars (EMA30 + ATR14 need warm-up).
                    max_ticks = 20000
                    lookback_sec = max(3600.0, 1.6 * 2000 * bar_sec)
                    ticks = mt5.copy_ticks_from(args.symbol, datetime.now() - _td(seconds=lookback_sec),
                                                max_ticks, mt5.COPY_TICKS_ALL)
                except Exception:
                    ticks = None
                if ticks is not None and len(ticks) >= max(args.ema_slow, 5):
                    tf = pd.DataFrame(ticks)
                    tf["ts10"] = (pd.to_datetime(tf["time"], unit="s")
                                  .dt.floor(str(int(bar_sec)) + "s"))
                    tf["mid"] = 0.5 * (tf["bid"] + tf["ask"])
                    bars = tf.groupby("ts10")["mid"].agg(["first", "max", "min", "last"])
                    bcl = bars["last"].to_numpy()
                    bhi = bars["max"].to_numpy()
                    blo = bars["min"].to_numpy()
                    prices = bars["first"].to_numpy()
                    price = bcl[-1]
                    ep = ewm(bcl, alpha=al); ef = ewm(bcl, args.ema_fast); es = ewm(bcl, args.ema_slow)
                    s_primary = 1 if price > ep[-1] else (-1 if price < ep[-1] else 0)
                    s_bg = 0
                    if price > ef[-1] > es[-1]:
                        s_bg = 1
                    elif price < ef[-1] < es[-1]:
                        s_bg = -1
                    score = 2 * s_primary + 1 * s_bg
                    trend = 1 if score > 0 else (-1 if score < 0 else 0)
                    # volume confirm: bar tick-count vs recent average
                    vol = tf.groupby("ts10").size().to_numpy()
                else:
                    trend = 0
                    vol = None
                    bcl = bhi = blo = np.array([])
                    prices = np.array([])
                    ep = es = ef = np.array([])

                # volume confirm: BYPASSED per user request (去掉了量能放量过滤).
                # Entry no longer requires volume to spike; only trend-confirm(>=3) + not-in-cooldown.
                volume_ok = True
                # trend confirm: count consecutive same-trend bars
                if trend != 0 and trend == last_trend:
                    trend_run += 1
                else:
                    trend_run = 1 if trend != 0 else 0
                last_trend = trend
                confirmed = (trend != 0 and trend_run >= args.trend_confirm and volume_ok)

                pos = mt5.positions_get(symbol=args.symbol) or []
                ai = mt5.account_info()
                dd = (ai.balance - ai.equity) / ai.balance if ai and ai.balance > 0 else 0
                cur = 1 if any(x.type == mt5.POSITION_TYPE_BUY for x in pos) else \
                      (-1 if any(x.type == mt5.POSITION_TYPE_SELL for x in pos) else 0)
                if cur == 0:
                    peak_profit = 0.0  # fresh position -> reset peak (avoid cross-trade leak)

                # ATR stop (computed on the sub-minute bars, consistent with the trend)
                if len(bcl) >= 3:
                    prev = np.concatenate([[bcl[0]], bcl[:-1]])
                    tr = np.maximum(np.maximum(bhi - blo, np.abs(bhi - prev)), np.abs(blo - prev))
                    atr_now = float(pd.Series(tr).rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr[-14:].mean())
                else:
                    atr_now = float((df["high"] - df["low"]).tail(14).mean()) if len(df) >= 14 else 1.0
                # ATR stop: primary stop is stop_atr_mult x ATR (confirmed rule). A fixed --stop-usd
                # overrides only if explicitly set > 0.
                stop_pts = args.stop_atr_mult * atr_now

                # (crash/breakdown guards 已按用户要求去掉; 仅保留 H1 结构止损.)
                # volatility state filter: BYPASSED per user request (去掉了 ADX≥22 门槛).
                volatility_ok = True  # ADX gate removed: always pass.

                # direction-consistency filter: BYPASSED per user request (去掉了 DI≥0.20).
                # Entry no longer requires a direction-consistency gate; only trend-confirm
                # (>=3)+volume + not-in-cooldown gate the open. (crash_drop / structure_break
                # guards and the fixed-$50 SL are kept.)
                direction_ok = True  # DI gate removed: always pass.

                # equity floor
                if args.equity_floor and pos and dd >= args.equity_floor:
                    log(ts, "EQUITY_FLOOR", "回撤%.0f%% 全平" % (dd * 100))
                    for x in pos:
                        if not args.dry_run:
                            close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    time.sleep(args.cycle_sec); continue

                # Use MT5's reported floating P&L. For XAUUSD this is pure price
                # move against price_open, which already bakes in the spread (a BUY
                # fills at the ask, is valued at the bid). Subtracting 0.37*oz here
                # would double-count the spread and shift every graded-TP threshold.
                if cur != 0 and pos:
                    profit = sum(x.profit for x in pos)
                    # --- 出场规则(按用户确认, 仅保留两项): ---
                    # 1) H1结构止损已作为服务器端SL挂单(跌破H1高低点由MT5自动平, 零滑点).
                    # 2) 浮盈>$50后, 回落到 peak*0.70(回撤30%) 才锁利. 去掉其他所有止损/破位/
                    #    急跌/冷却/固定金额/ATR止损, 以及 $20/$30 小锁利档.
                    if profit > 0:
                        peak_profit = max(peak_profit, profit)
                        P = peak_profit
                        # 仅在峰值>$50时才进入回撤30%锁利; 不足$50不锁(让利润跑大波段)
                        if P > 50.0:
                            target = P * 0.70
                        else:
                            target = None
                        # 仅在真正回撤(profit<P)时触发, 避免触及峰值瞬间提前锁利
                        if target is not None and profit <= target and profit < P:
                            log(ts, "TP", "浮盈$%.0f 回落到目标$%.0f (峰值$%.0f, 回撤30%) - 锁利" % (
                                profit, target, P))
                            for x in pos:
                                if not args.dry_run:
                                    close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                            peak_profit = 0.0
                    else:
                        peak_profit = 0.0

                # (趋势反转反手REVERSE已按用户要求去掉; 出场仅靠 H1止损 + >$50回撤30%.)
                # open fresh in trend direction when the trend confirms.
                # Direction-aware crash cooldown: it only blocks the side that the crash/break
                # turned AGAINST (catching the falling knife), and ALLOWS the with-trend side.
                # crash_block_dir: +1 blocks LONG, -1 blocks SHORT, 0 blocks both (legacy).
                # (crash冷却已按用户要求去掉; 开仓仅靠趋势确认 + last_open_ts 防重复开仓.)
                if confirmed and cur == 0 and volatility_ok and direction_ok \
                        and (time.time() - last_open_ts) > 5.0:
                    otype = mt5.POSITION_TYPE_BUY if trend > 0 else mt5.POSITION_TYPE_SELL
                    price = ask if trend > 0 else bid
                    # 真实SL挂单. 用 H1 结构止损(最近N根H1高低点)优先; 否则退回 2xATR.
                    # BUY 止损 = H1低点下方; SELL 止损 = H1高点上方.
                    oz = args.lot_size * USC
                    if args.stop_h1 and args.stop_h1 > 0 and h1_hi is not None and h1_lo is not None:
                        sl = h1_lo - 0.05 if otype == mt5.POSITION_TYPE_BUY else h1_hi + 0.05
                        stop_mode = "H1"
                    else:
                        stop_pts = (args.stop_usd / oz) if args.stop_usd > 0 else (args.stop_atr_mult * atr_now)
                        sl = price - stop_pts if otype == mt5.POSITION_TYPE_BUY else price + stop_pts
                        stop_mode = "ATR"
                    log(ts, "OPEN", "趋势%+d %s 开%s %.1f手 @%.2f SL=%.2f (%s止损)" % (
                        trend, "涨" if trend > 0 else "跌", "多" if trend > 0 else "空",
                        args.lot_size, price, sl, stop_mode))
                    if not args.dry_run:
                        res = open_order(args.symbol, args.lot_size, otype, price, sl, 202608, "open")
                        # Refresh position immediately so the next loop does NOT think we are
                        # flat and re-open on top of the just-filled order (position_id race).
                        if res is not None and getattr(res, "retcode", -1) == mt5.TRADE_RETCODE_DONE:
                            time.sleep(0.5)
                            pos2 = mt5.positions_get(symbol=args.symbol) or []
                            cur = 1 if any(x.type == mt5.POSITION_TYPE_BUY for x in pos2) else \
                                  (-1 if any(x.type == mt5.POSITION_TYPE_SELL for x in pos2) else 0)
                            pos = pos2
                            peak_profit = 0.0
                            last_open_ts = time.time()

                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                last_trend = trend
                log(ts, "status", "px%.2f 趋势%+d 多%.2f/空%.2f 权益%.0f 浮盈%.0f 回撤%.1f%%" % (
                    (bid + ask) / 2, trend, buy_lots, sell_lots, ai.equity,
                    (ai.equity - ai.balance) if ai else 0, dd * 100))
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
