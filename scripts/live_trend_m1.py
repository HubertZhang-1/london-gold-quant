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
from london_gold.indicators import adx  # noqa: E402

SYMBOL = "XAUUSD"
USC = 100.0


def close_by_ticket(symbol, ticket, volume, pos_type):
    close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if pos_type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": close_type, "price": price, "deviation": 40, "magic": 202608,
                           "position": ticket, "comment": "close", "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def open_order(symbol, volume, order_type, price, magic=202608, comment="open"):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
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
    p.add_argument("--stop-atr-mult", type=float, default=2.0)
    p.add_argument("--trend-confirm", type=int, default=3,
                   help="require this many consecutive same trend (M1 churn -> higher confirm)")
    p.add_argument("--volume-mult", type=float, default=0.8)
    p.add_argument("--adx-gate", type=float, default=22.0,
                   help="volatility filter: only open when ADX >= this (small-move/noise "
                        "regime with ADX below this is skipped, to avoid the $20-within losses)")
    p.add_argument("--tp-activate-profit", type=float, default=50.0)
    p.add_argument("--tp-trail-pct", type=float, default=0.30,
                   help="close when profit retraces >= this from peak (0.30 = let it ride)")
    p.add_argument("--equity-floor", type=float, default=0.30)
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
    try:
        while True:
            try:
                rates = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 300)
                if rates is None or len(rates) < args.ema_slow:
                    time.sleep(args.cycle_sec); continue
                df = pd.DataFrame(rates)
                cl = df["close"].astype(float).to_numpy()
                ts = datetime.now().strftime("%H:%M:%S")
                # trend (EMA5 primary + EMA10/30 background)
                al = (2 / (args.ema_primary + 1)) if args.ema_alpha is None else args.ema_alpha
                ep = ewm(cl, alpha=al); ef = ewm(cl, args.ema_fast); es = ewm(cl, args.ema_slow)
                s_primary = 1 if cl[-1] > ep[-1] else (-1 if cl[-1] < ep[-1] else 0)
                s_bg = 0
                if cl[-1] > ef[-1] > es[-1]:
                    s_bg = 1
                elif cl[-1] < ef[-1] < es[-1]:
                    s_bg = -1
                score = 2 * s_primary + 1 * s_bg
                trend = 1 if score > 0 else (-1 if score < 0 else 0)

                # volume confirm
                volume_ok = True
                if args.volume_mult > 0:
                    try:
                        vol = df["tick_volume"].astype(float).to_numpy()
                        volume_ok = vol[-1] >= vol[-40:].mean() * args.volume_mult
                    except Exception:
                        volume_ok = True
                # trend confirm (higher count for M1 churn)
                if trend != 0 and trend == last_trend:
                    trend_run += 1
                else:
                    trend_run = 1 if trend != 0 else 0
                confirmed = (trend != 0 and trend_run >= args.trend_confirm and volume_ok)

                pos = mt5.positions_get(symbol=args.symbol) or []
                ai = mt5.account_info()
                t = mt5.symbol_info_tick(args.symbol)
                bid = t.bid; ask = t.ask
                dd = (ai.balance - ai.equity) / ai.balance if ai and ai.balance > 0 else 0
                cur = 1 if any(x.type == mt5.POSITION_TYPE_BUY for x in pos) else \
                      (-1 if any(x.type == mt5.POSITION_TYPE_SELL for x in pos) else 0)
                if cur == 0:
                    peak_profit = 0.0  # fresh position -> reset peak (avoid cross-trade leak)

                # ATR stop
                tr = np.maximum(df["high"] - df["low"],
                                np.maximum((df["high"] - df["close"].shift()).abs().astype(float),
                                           (df["low"] - df["close"].shift()).abs().astype(float)))
                atr_now = float(pd.Series(tr).rolling(14).mean().iloc[-1]) if len(df) >= 14 else 1.0
                stop_pts = args.stop_atr_mult * atr_now

                # volatility state filter: small-move (ADX low / band narrow) -> don't trade
                # (the $20-within regime is where losses concentrate; avoid trading it).
                volatility_ok = True
                if args.adx_gate and args.adx_gate > 0:
                    try:
                        # use the verified library ADX (standard calculation)
                        adx_series = adx(df["high"].astype(float), df["low"].astype(float),
                                         df["close"].astype(float), 14)
                        adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0
                        volatility_ok = adx_val >= args.adx_gate
                    except Exception:
                        volatility_ok = True

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
                    # --- ATR stop-loss (2xATR, confirmed rule; was previously computed
                    #     as stop_pts but never enforced) ---
                    oz = sum(x.volume for x in pos) * USC
                    stop_dollar = stop_pts * oz
                    if stop_dollar > 0 and profit <= -stop_dollar:
                        log(ts, "STOP", "浮亏$%.0f 触及2xATR止损($%.0f) - 平仓" % (
                            abs(profit), stop_dollar))
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
                if confirmed and cur == 0 and volatility_ok:
                    otype = mt5.POSITION_TYPE_BUY if trend > 0 else mt5.POSITION_TYPE_SELL
                    price = ask if trend > 0 else bid
                    log(ts, "OPEN", "趋势%+d %s 开%s %.1f手 @%.2f (波动可交易)" % (
                        trend, "涨" if trend > 0 else "跌", "多" if trend > 0 else "空",
                        args.lot_size, price))
                    if not args.dry_run:
                        open_order(args.symbol, args.lot_size, otype, price, 202608, "open")
                elif confirmed and cur == 0 and not volatility_ok:
                    log(ts, "SKIP_LOWVOL", "小波动区(ADX低/带宽窄) - 不交易")

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
