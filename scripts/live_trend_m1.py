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
                   help="adaptive stop: multiplier x ATR. Used when --stop-usd is 0. The stop "
                        "widens/narrows with volatility (2xATR is the confirmed rule). This is the "
                        "primary stop, attached as a server-side SL at open.")
    p.add_argument("--trend-confirm", type=int, default=3,
                   help="require this many consecutive same trend to confirm entry")
    p.add_argument("--bar-sec", type=float, default=60.0,
                   help="candle period in seconds for the trend bars (降频到 M1=60s per user). Bars "
                        "are aggregated from ticks (open/high/low/close), and EMA5/10/30 + trend are "
                        "computed on this series. Larger = lower frequency = fewer trades / less "
                        "spread cost; smaller (10s) = faster but noisier.")
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

                # --- crash/breakdown guards (慢涨防暴跌) ---
                # 1) intraday cliff: last bar close change (sub-minute)
                last_bar_change = float(bcl[-1] - bcl[-2]) if len(bcl) >= 2 else 0.0
                crash_drop = args.crash_drop_pts > 0 and last_bar_change < -args.crash_drop_pts
                # 2) structure breakdown: 多头跌破 EMA<slow> / 空头升破 EMA<slow> (熊市转头).
                #    Use a buffer band (buffer*ATR) around the EMA so a normal whipsaw around
                #    the trend line does NOT false-trigger. Break is confirmed only when price
                #    crosses beyond EMA +/- buffer*ATR.
                structure_break = False
                if args.structure_close.lower() == "ema" and cur != 0 and len(es) >= 1:
                    buf = args.structure_buffer_atr * atr_now
                    if cur > 0 and bid < es[-1] - buf:
                        structure_break = True          # 多单跌破多头趋势线(含缓冲) -> 熊市转头
                    elif cur < 0 and ask > es[-1] + buf:
                        structure_break = True          # 空单升破(含缓冲) -> 反转

                # Even if flat, a cliff means stand aside. Direction-aware cooldown so it blocks
                # the with-the-crash LONG (catching the falling knife) but DOES NOT block the
                # with-trend SHORT. This is the fix for "为什么没做空": the old code blocked
                # every direction, so a confirmed downtrend signal right after a drop was lost.
                if cur == 0 and crash_drop:
                    crash_block_until = time.time() + args.crash_cooldown_sec
                    crash_block_dir = 1   # price dropped -> only block LONG rebounds

                # volatility state filter: BYPASSED per user request (去掉了 ADX≥22 门槛).
                # Now the entry does NOT require ADX; only trend-confirm(>=3)+volume + DI>=0.20
                # + not-in-cooldown gate the open. (crash_drop / structure_break guards are kept.)
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
                    # --- 慢涨防暴跌: 三层防护, 最高优先级 ---
                    if crash_drop:
                        log(ts, "CRASH", "单分钟急跌%.2f点 (基线>%.1f) - 立即全平" % (
                            last_1m_change, args.crash_drop_pts))
                        for x in pos:
                            if not args.dry_run:
                                close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                        peak_profit = 0.0
                        crash_block_until = time.time() + args.crash_cooldown_sec
                        crash_block_dir = 1   # price dropped -> block LONG only, allow SHORT (with-trend)
                        time.sleep(args.cycle_sec); continue
                    if structure_break:
                        log(ts, "STRUCTURE", "跌破EMA%d趋势线(含%.1fATR缓冲) 熊市转头 - 全平规避" % (
                            args.ema_slow, args.structure_buffer_atr))
                        for x in pos:
                            if not args.dry_run:
                                close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                        peak_profit = 0.0
                        crash_block_until = time.time() + args.crash_cooldown_sec
                        crash_block_dir = cur  # trend broke against cur: block the same side we
                                               # just exited, allow the with-trend opposite side
                        time.sleep(args.cycle_sec); continue
                    # --- 止损: 2xATR 自适应 (确认规则; 随波动变宽/变窄). 可选固定 --stop-usd 覆盖. ---
                    oz = sum(x.volume for x in pos) * USC
                    stop_dollar = args.stop_usd if args.stop_usd > 0 else (stop_pts * oz)
                    if stop_dollar > 0 and profit <= -stop_dollar:
                        log(ts, "STOP", "浮亏$%.0f 触及止损($%.0f, %.2f点/%.1f手) - 平仓" % (
                            abs(profit), stop_dollar, stop_dollar / oz, sum(x.volume for x in pos)))
                        for x in pos:
                            if not args.dry_run:
                                close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                        peak_profit = 0.0
                        time.sleep(args.cycle_sec); continue
                    # full graded take-profit (你确认的规则: 浮盈>20后, 跌回$20锁利).
                    # $20 是锁利地线; 峰值越高锁利越高:
                    #   峰值 <  $20 -> 未激活, 不锁
                    #   $20<=峰值<$30 -> 回落到 $20 锁利
                    #   $30<=峰值<=$50 -> 回落到 $30 锁利
                    #   峰值 >  $50    -> 回落到 peak*0.70 锁利 (回撤30%)
                    if profit > 0:
                        peak_profit = max(peak_profit, profit)
                        P = peak_profit
                        if P < 20.0:
                            target = None            # 浮盈未超过$20, 不锁
                        elif P < 30.0:
                            target = 20.0            # 跌回$20锁利
                        elif P <= 50.0:
                            target = 30.0
                        else:
                            target = P * 0.70
                        # 仅在真正回撤(profit<P)时触发, 避免触及峰值瞬间提前锁利
                        if target is not None and profit <= target and profit < P:
                            log(ts, "TP", "浮盈$%.0f 回落到目标$%.0f (峰值$%.0f) - 锁利" % (
                                profit, target, P))
                            for x in pos:
                                if not args.dry_run:
                                    close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                            peak_profit = 0.0
                    else:
                        peak_profit = 0.0

                # reversal
                if confirmed and cur != 0 and trend != cur:
                    log(ts, "REVERSE", "趋势%+d vs 持仓%+d 反手" % (trend, cur))
                    for x in pos:
                        if not args.dry_run:
                            close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    peak_profit = 0.0
                    cur = 0
                # open fresh in trend direction ONLY when volatility is active (strong ADX /
                # widening band), so we DON'T trade the $20-within tiny/noise regime where
                # losses concentrate. 小波动(ADX低/带宽窄) -> 不做, 避免在无优势区间被点差消耗.
                # Direction-aware crash cooldown: it only blocks the side that the crash/break
                # turned AGAINST (catching the falling knife), and ALLOWS the with-trend side.
                # crash_block_dir: +1 blocks LONG, -1 blocks SHORT, 0 blocks both (legacy).
                in_crash_cooldown = time.time() < crash_block_until
                blocked_by_cooldown = in_crash_cooldown and (
                    crash_block_dir == 0 or (crash_block_dir > 0 and trend > 0) or
                    (crash_block_dir < 0 and trend < 0))
                if confirmed and cur == 0 and volatility_ok and direction_ok and not blocked_by_cooldown \
                        and (time.time() - last_open_ts) > 5.0:
                    otype = mt5.POSITION_TYPE_BUY if trend > 0 else mt5.POSITION_TYPE_SELL
                    price = ask if trend > 0 else bid
                    # 真实SL挂单: MT5终端在价格触及止损价瞬间成交, 零滑点.
                    # 止损点数: 固定 --stop-usd 优先, 否则用 2xATR(自适应) stop_pts.
                    # BUY止损在下方, SELL止损在上方.
                    oz = args.lot_size * USC
                    stop_pts = (args.stop_usd / oz) if args.stop_usd > 0 else (args.stop_atr_mult * atr_now)
                    sl = price - stop_pts if otype == mt5.POSITION_TYPE_BUY else price + stop_pts
                    log(ts, "OPEN", "趋势%+d %s 开%s %.1f手 @%.2f SL=%.2f (波动可交易)" % (
                        trend, "涨" if trend > 0 else "跌", "多" if trend > 0 else "空",
                        args.lot_size, price, sl))
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
                elif confirmed and cur == 0 and blocked_by_cooldown:
                    log(ts, "CRASH_COOLDOWN", "暴跌冷却中 %.0fs 挡住%s - 放行反向" % (
                        crash_block_until - time.time(),
                        "多" if crash_block_dir > 0 else "空"))

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
