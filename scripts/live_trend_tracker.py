# -*- coding: utf-8 -*-
"""Live TREND-TRACKING executor (看涨做多 / 看跌做空 / 反转反手 / 单笔不加仓 / 带止损).

User explicitly wants: follow the trend — long when trending up, short when trending down,
REVERSE (close + reopen) when the trend flips, NO martingale stacking, a hard stop to avoid
holding against a one-way move. Single position, fixed lot.

Trend: M5 EMA(fast) vs EMA(slow); +1 up, -1 down, 0 neutral. 
Reversal: close the opposite-side position by ticket, then open the trend direction.
Stop: hard ATR-based (or fixed points), so it never rides a losing position.
Safety: wide equity-floor stop too. Demo-only auto-trade.

User supervises manually.
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
                           "type": close_type, "price": price, "deviation": 40, "magic": 202607,
                           "position": ticket, "comment": "close", "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def open_order(symbol, volume, order_type, price, magic=202607, comment="open"):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def ema_trend(rates, fast, slow, primary=5, alpha=None):
    """Weighted trend: EMA(primary, smoothing alpha) drives; EMA(fast)/EMA(slow) background.
    alpha=None uses the standard span-derived coefficient 2/(primary+1); an explicit alpha
    overrides it (e.g. 0.25 = slower/more stable)."""
    import numpy as _np
    df = pd.DataFrame(rates)
    cl = df["close"].astype(float).to_numpy()

    def ewm(a, span=None, alpha=None):
        al = alpha if alpha is not None else (2 / (span + 1))
        out = _np.empty(len(a)); out[0] = a[0]
        for k in range(1, len(a)): out[k] = al * a[k] + (1 - al) * out[k - 1]
        return out

    ema5 = ewm(cl, span=primary, alpha=alpha)
    ef = ewm(cl, span=fast)
    es = ewm(cl, span=slow)
    # primary: price vs EMA5 (high weight)
    s_primary = 1 if cl[-1] > ema5[-1] else (-1 if cl[-1] < ema5[-1] else 0)
    # background (low weight): EMA_fast vs EMA_slow and close vs both
    s_bg = 0
    if cl[-1] > ef[-1] > es[-1]:
        s_bg = 1
    elif cl[-1] < ef[-1] < es[-1]:
        s_bg = -1
    # weighted: primary * 2 + background * 1 -> sign decides
    score = 2 * s_primary + 1 * s_bg
    return (1 if score > 0 else (-1 if score < 0 else 0))


def main():
    p = argparse.ArgumentParser(description="Live trend-tracking executor")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=10.0)
    p.add_argument("--lot-size", type=float, default=0.3)
    p.add_argument("--ema-fast", type=int, default=10)
    p.add_argument("--ema-slow", type=int, default=30)
    p.add_argument("--ema-primary", type=int, default=5,
                   help="primary EMA for the trend signal (high weight); EMA-fast/slow are a "
                        "low-weight big-picture background filter")
    p.add_argument("--ema-alpha", type=float, default=None,
                   help="override the primary EMA5 smoothing coefficient alpha (default "
                        "2/(primary+1)). e.g. 0.25 = slower/more stable; 0.4 = faster/more "
                        "sensitive (but more churn).")
    p.add_argument("--stop-atr-mult", type=float, default=2.0,
                   help="hard stop = this * ATR (to avoid riding a loss; 0 disables)")
    p.add_argument("--trend-confirm", type=int, default=2,
                   help="consecutive bars of the same trend before reversing (filters EMA jitter)")
    p.add_argument("--volume-mult", type=float, default=0.8,
                   help="require the last M5 tick_volume >= mean * this before trusting the "
                        "EMA trend (0.8 filters only significant volume-collapse/fake signals; "
                        "1.2 would be too strict as only ~25% of bars clear it; 0 disables)")
    p.add_argument("--tp-activate-profit", type=float, default=50.0,
                   help="trailing take-profit activates once unrealized profit >= this (USD)")
    p.add_argument("--tp-trail-pct", type=float, default=0.10,
                   help="after activation, close when profit retraces >= this fraction from "
                        "its peak (e.g. 0.10 = lock in after a 10% pullback / keep 90%)")
    p.add_argument("--equity-floor", type=float, default=0.30)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_trend_tracker_log.csv"))
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
    print("演示账户 %s | %s | 趋势跟踪 %s/%s 单笔%.1f手 止损%.0f ATR" % (
        info.login, "DRY-RUN" if args.dry_run else "自动下单",
        args.ema_fast, args.ema_slow, args.lot_size, args.stop_atr_mult))
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow([ts, ev, detail])
        print("  [%s] %s: %s" % (ts, ev, detail), flush=True)

    print("=" * 64)
    print("趋势跟踪(看涨做多/看跌做空/反转反手/不加仓/带止损) 运行中")
    print("=" * 64)
    last_trend = 0
    trend_run = 0   # consecutive bars with the same non-zero trend (confirmation guard)
    peak_profit = 0.0  # trailing take-profit peak (USD)
    try:
        while True:
            try:
                rates = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M5, 0, 60)
                if rates is None or len(rates) < args.ema_slow:
                    time.sleep(args.cycle_sec); continue
                trend = ema_trend(rates, args.ema_fast, args.ema_slow, args.ema_primary, args.ema_alpha)
                # volume-confirmation factor: only trust an EMA signal when the last M5 bar
                # traded on real turnover (tick_volume >= mean * volume_mult). A low-volume
                # EMA flip is likely a虚假/fake signal -> ignore it (wait for volume).
                volume_ok = True
                if args.volume_mult > 0:
                    try:
                        import pandas as _pd
                        _df = _pd.DataFrame(rates)
                        vol = _df["tick_volume"].astype(float).to_numpy()
                        last_vol = vol[-1]
                        mean_vol = float(vol[-20:].mean()) if len(vol) >= 20 else float(vol.mean())
                        volume_ok = last_vol >= mean_vol * args.volume_mult if mean_vol > 0 else True
                    except Exception:
                        volume_ok = True
                # trend-confirmation guard: count consecutive same non-zero trend bars, so a
                # single EMA-jitter bar doesn't trigger a false reverse. A NEUTRAL (+0) bar
                # pauses the count instead of resetting it — so an uptrend with a brief +0
                # still confirms, but a genuine reversal (+1->-1) resets it.
                if trend != 0 and trend == last_trend:
                    trend_run += 1
                elif trend != 0:
                    trend_run = 1
                # trend == 0 (neutral): keep trend_run unchanged (pause) — don't reset
                confirmed = trend != 0 and trend_run >= args.trend_confirm and volume_ok
                last_trend = trend  # CRITICAL: without this, the confirmation never advances
                pos = mt5.positions_get(symbol=args.symbol) or []
                ai = mt5.account_info()
                t = mt5.symbol_info_tick(args.symbol)
                bid = t.bid; ask = t.ask
                ts = datetime.now().strftime("%H:%M:%S")
                dd = (ai.balance - ai.equity) / ai.balance if ai and ai.balance > 0 else 0
                cur = 1 if any(x.type == mt5.POSITION_TYPE_BUY for x in pos) else \
                      (-1 if any(x.type == mt5.POSITION_TYPE_SELL for x in pos) else 0)

                # ATR-based stop
                df = pd.DataFrame(rates)
                tr = np.maximum(df["high"] - df["low"], np.maximum(
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()))
                atr_now = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 14 else 3.0
                stop_pts = args.stop_atr_mult * atr_now

                # equity floor
                if args.equity_floor and pos and dd >= args.equity_floor:
                    log(ts, "EQUITY_FLOOR", "回撤%.0f%% 强制全平" % (dd * 100))
                    for x in pos:
                        if not args.dry_run:
                            close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    time.sleep(args.cycle_sec); continue

                # hard stop for current position (favor/move with trend)
                if cur != 0 and args.stop_atr_mult > 0:
                    if cur > 0:
                        avg = np.mean([x.price_open for x in pos if x.type == mt5.POSITION_TYPE_BUY])
                        if bid <= avg - stop_pts:
                            log(ts, "STOP", "多单触发止损%.2f(ATR%.2f). 平" % (stop_pts, atr_now))
                            for x in pos:
                                if x.type == mt5.POSITION_TYPE_BUY and not args.dry_run:
                                    close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                            last_trend = 0
                    else:
                        avg = np.mean([x.price_open for x in pos if x.type == mt5.POSITION_TYPE_SELL])
                        if ask >= avg + stop_pts:
                            log(ts, "STOP", "空单触发止损. 平")
                            for x in pos:
                                if x.type == mt5.POSITION_TYPE_SELL and not args.dry_run:
                                    close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                            last_trend = 0

                # re-read positions after possible stop
                pos = mt5.positions_get(symbol=args.symbol) or []
                cur = 1 if any(x.type == mt5.POSITION_TYPE_BUY for x in pos) else \
                      (-1 if any(x.type == mt5.POSITION_TYPE_SELL for x in pos) else 0)

                # ---- trailing take-profit: activate once profit >= tp_activate; close when it
                #      retraces >= tp_trail_pct from the peak (lock in gains) ----
                if cur != 0 and pos:
                    px_avg = float(np.mean([x.price_open for x in pos]))
                    last_fx = bid if cur > 0 else ask
                    # unrealized profit in USD (price moved * lots * 100, minus spread)
                    gross = (last_fx - px_avg) * sum(x.volume for x in pos) * USC
                    profit = gross - 0.37 * sum(x.volume for x in pos) * USC
                    if profit >= args.tp_activate_profit:
                        peak_profit = max(peak_profit, profit)
                        if peak_profit > 0 and profit <= peak_profit * (1 - args.tp_trail_pct):
                            log(ts, "TRAIL_TP", "浮盈$%.0f 从峰值$%.0f回落%.0f%% - 锁利平仓" % (
                                profit, peak_profit, (1 - profit / peak_profit) * 100))
                            for x in pos:
                                if not args.dry_run:
                                    close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                            peak_profit = 0.0
                    elif profit < args.tp_activate_profit and peak_profit <= 0:
                        # Not yet activated: keep peak at 0 (no trailing until activated).
                        peak_profit = 0.0
                    # IMPORTANT: DON'T reset peak_profit when profit dips below activation
                    # mid-run — that would lower the trail trigger and lock in too late.

                # reversal: trend flipped vs current position -> close + reopen on trend side
                if confirmed and cur != 0 and trend != cur:
                    log(ts, "REVERSE", "趋势%+d 与持仓%+d相反 - 反手" % (trend, cur))
                    for x in pos:
                        if not args.dry_run:
                            close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    cur = 0
                # open fresh in trend direction
                if confirmed and cur == 0:
                    otype = mt5.POSITION_TYPE_BUY if trend > 0 else mt5.POSITION_TYPE_SELL
                    price = ask if trend > 0 else bid
                    log(ts, "OPEN", "趋势%+d %s 开%s %.1f手 @%.2f" % (
                        trend, "涨" if trend > 0 else "跌",
                        "多" if trend > 0 else "空", args.lot_size, price))
                    if not args.dry_run:
                        open_order(args.symbol, args.lot_size, otype, price, 202607, "open")

                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                unrl = (ai.equity - ai.balance) if ai else 0
                log(ts, "status", "px%.2f 趋势%+d 多%.2f/空%.2f 权益%.0f 浮盈%.0f 回撤%.1f%%" % (
                    (bid + ask) / 2, trend, buy_lots, sell_lots, ai.equity, unrl, dd * 100))
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
