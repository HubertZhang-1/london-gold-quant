# -*- coding: utf-8 -*-
"""Live TREND-FOLLOW engine for XAUUSD (5x leverage, hold the trend, bank gains).

This is a clean trend-following executor — the correct shape for the user's goal
("顺势做多、稳定持有、涨到就落袋"), unlike the martingale-grid executor which churns
(opens/closes/reverse-hedges and never lets profit reach the take-profit line).

Rules (only 3 exits, no churn):
  1. Signal: EMA-trend (close vs ema_fast vs ema_slow, with slope confirmation).
     - Cross to LONG (trend=+1): if flat, open long; if short, flip to long.
     - Cross to SHORT (trend=-1): if flat, open short; if long, flip to short.
  2. Hold: keep the direction open; do NOT add/close on small wiggles.
  3. Exits:
     - TAKE-PROFIT: trend-side unrealized profit >= tp_atr_mult * (M15 or chosen TF) ATR.
     - TREND-FLIP: EMA trend reverses vs the open side -> close (and flip).
     - RISK-STOP: unrealized loss >= max_unreal_loss_pct of equity -> close all.

Sizing: base lot = leverage * equity / (price * 100 oz) when target_leverage is set.

SAFETY: only auto-trades on a DEMO account (trade_mode == 0); else read-only/dry-run.

Usage:
  py scripts/live_trend_follow.py --target-leverage 5 --tp-atr-mult 3 --tp-atr-tf M15
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from london_gold.indicators import atr  # noqa: E402

SYMBOL = "XAUUSD"
_TF = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
       "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}


def ohlc(rates: np.ndarray) -> dict:
    d = pd.DataFrame(rates)
    return {"open": d["open"].astype(float), "high": d["high"].astype(float),
            "low": d["low"].astype(float), "close": d["close"].astype(float)}


def ema_trend(close: pd.Series, fast: int, slow: int) -> int:
    ef = float(close.ewm(span=fast, adjust=False).mean().iloc[-1])
    es = float(close.ewm(span=slow, adjust=False).mean().iloc[-1])
    es_prev = float(close.ewm(span=slow, adjust=False).mean().iloc[-2])
    price = float(close.iloc[-1])
    if price > ef > es and es >= es_prev:
        return 1
    if price < ef < es and es <= es_prev:
        return -1
    return 0


def leverage_lots(equity: float, price: float, lev: float, oz: float = 100.0) -> float:
    return max(0.01, round(lev * equity / (price * oz), 2))


def send_order(symbol, volume, order_type, price, magic, comment, filling):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": filling})


def main() -> None:
    p = argparse.ArgumentParser(description="Live trend-follow XAUUSD executor")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=60.0)
    p.add_argument("--ema-fast", type=int, default=10)
    p.add_argument("--ema-slow", type=int, default=30)
    p.add_argument("--target-leverage", type=float, default=5.0)
    p.add_argument("--tp-atr-mult", type=float, default=3.0)
    p.add_argument("--tp-atr-tf", default="M15")
    p.add_argument("--trail-activate-atr", type=float, default=1.0,
                   help="trailing stop activates once the trend-side profit reaches "
                        "this many (TP-TF) ATR")
    p.add_argument("--trail-back-atr", type=float, default=0.5,
                   help="after activation, close when profit retraces this many "
                        "(M1) ATR below its peak -> locks in gains, avoids give-back")
    p.add_argument("--max-unreal-loss-pct", type=float, default=0.05)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_trend_follow_log.csv"))
    args = p.parse_args()

    if not mt5.initialize():
        print("❌ MT5 init failed:", mt5.last_error()); sys.exit(2)
    mt5.symbol_select(args.symbol, True)
    info = mt5.account_info()
    if info is None:
        print("❌ 未登录"); sys.exit(2)
    demo = info.trade_mode == 0
    if not demo:
        print(f"🔒 非演示账户(trade_mode={info.trade_mode}) -> DRY-RUN"); args.dry_run = True
    print(f"✅ 演示账户 {info.login} | {'DRY-RUN' if args.dry_run else '自动下单'}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([ts, ev, detail])
        print(f"  [{ts}] {ev}: {detail}", flush=True)

    print("=" * 66)
    print(f"趋势跟踪引擎 {args.symbol}  EMA{args.ema_fast}/{args.ema_slow}  "
          f"杠杆{args.target_leverage}x  TP={args.tp_atr_mult}x{args.tp_atr_tf}-ATR  "
          f"风险止损{args.max_unreal_loss_pct*100:.0f}%")
    print("=" * 66)

    trail_peak = None  # peak trend-side profit (in price points) for the trailing stop

    try:
        while True:
            try:
                rates = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 200)
                if rates is None or len(rates) < 60:
                    time.sleep(args.cycle_sec); continue
                o = ohlc(rates)
                trend = ema_trend(pd.Series(o["close"]), args.ema_fast, args.ema_slow)
                pos = mt5.positions_get(symbol=args.symbol) or []
                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                ai = mt5.account_info()
                equity = ai.equity if ai else 0
                base_lot = leverage_lots(equity, float(o["close"].iloc[-1]), args.target_leverage)

                ts = datetime.now().strftime("%H:%M:%S")
                log(ts, "tick", f"close={o['close'].iloc[-1]:.2f} trend={trend:+d} "
                                f"多{buy_lots:.2f}/空{sell_lots:.2f} lot={base_lot:.2f}")

                # --- decide the target direction based on trend ---
                target_side = 0
                if trend == 1:
                    target_side = 1
                elif trend == -1:
                    target_side = -1

                # --- if flat and a clear trend, open ---
                if not pos and target_side != 0:
                    otype = mt5.POSITION_TYPE_BUY if target_side > 0 else mt5.POSITION_TYPE_SELL
                    px = mt5.symbol_info_tick(args.symbol).ask if target_side > 0 else mt5.symbol_info_tick(args.symbol).bid
                    log(ts, "open", f"{'BUY' if target_side>0 else 'SELL'} {base_lot}手 @{px:.2f}")
                    if not args.dry_run:
                        send_order(args.symbol, base_lot, otype, px, 202603, "tf_open", mt5.ORDER_FILLING_IOC)
                elif pos:
                    # --- TREND-FLIP: close the opposing side, then flip ---
                    have_long = buy_lots > 0
                    have_short = sell_lots > 0
                    if target_side != 0:
                        # close the side that OPPOSES the trend
                        opposing = (have_long and target_side < 0) or (have_short and target_side > 0)
                        if opposing:
                            log(ts, "trend_flip", f"趋势{target_side:+d}与持仓{('多' if have_long else '空')}相反 - 平该侧")
                            for x in pos:
                                close_side = (x.type == mt5.POSITION_TYPE_BUY and target_side < 0) or \
                                             (x.type == mt5.POSITION_TYPE_SELL and target_side > 0)
                                if close_side:
                                    ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                    cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                                    if not args.dry_run:
                                        send_order(args.symbol, x.volume, ct, cpx, 202603, "tf_flip", mt5.ORDER_FILLING_IOC)
        # ---- exits measured on the TREND side ----
                    # determine dominant side = long if we trend up
                    tside = 1 if trend >= 0 else -1
                    t_avg = np.mean([x.price_open for x in pos if (
                        (x.type == mt5.POSITION_TYPE_BUY and tside > 0) or
                        (x.type == mt5.POSITION_TYPE_SELL and tside < 0))]) if pos else 0
                    t_lots = sum(x.volume for x in pos if (
                        (x.type == mt5.POSITION_TYPE_BUY and tside > 0) or
                        (x.type == mt5.POSITION_TYPE_SELL and tside < 0)))
                    last = mt5.symbol_info_tick(args.symbol).bid if tside > 0 else mt5.symbol_info_tick(args.symbol).ask
                    if t_lots > 0:
                        profit_dist = (last - t_avg) if tside > 0 else (t_avg - last)
                        # TP ATR basis
                        tv = args.tp_atr_tf.upper()
                        if tv in _TF:
                            rt = mt5.copy_rates_from_pos(args.symbol, _TF[tv], 0, 40)
                            ot = ohlc(rt) if rt is not None else o
                            a_t = atr(pd.Series(ot["high"]), pd.Series(ot["low"]), pd.Series(ot["close"]), 14)
                            tp_atr = float(a_t.iloc[-1]) if not np.isnan(a_t.iloc[-1]) else 5.0
                        else:
                            tp_atr = 5.0
                        # M1 ATR for the trail-back distance
                        m1 = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 20)
                        o1 = ohlc(m1) if m1 is not None else o
                        a1 = atr(pd.Series(o1["high"]), pd.Series(o1["low"]), pd.Series(o1["close"]), 14)
                        m1_atr = float(a1.iloc[-1]) if not np.isnan(a1.iloc[-1]) else 2.0

                        # ---- TRAILING STOP: activate after trail-activate-atr * tp_atr,
                        #     then close if profit retraces trail-back-atr * m1_atr from peak ----
                        if args.trail_activate_atr and profit_dist >= tp_atr * args.trail_activate_atr:
                            if trail_peak is None or profit_dist > trail_peak:
                                trail_peak = profit_dist
                            trail_back = m1_atr * args.trail_back_atr
                            if profit_dist <= trail_peak - trail_back:
                                log(ts, "trail_stop", f"峰值{trail_peak:.2f}回撤至{profit_dist:.2f}(回吐{trail_peak-profit_dist:.2f}≥{trail_back:.2f}) - 落袋")
                                for x in pos:
                                    if (x.type == mt5.POSITION_TYPE_BUY and tside > 0) or \
                                       (x.type == mt5.POSITION_TYPE_SELL and tside < 0):
                                        ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                        cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                                        if not args.dry_run:
                                            send_order(args.symbol, x.volume, ct, cpx, 202603, "tf_trail", mt5.ORDER_FILLING_IOC)
                                        trail_peak = None
                        else:
                            trail_peak = None
                        # take-profit
                        if profit_dist >= tp_atr * args.tp_atr_mult:
                            log(ts, "take_profit", f"浮盈{profit_dist:.2f} ≥ {args.tp_atr_mult}x{tv}ATR({tp_atr*args.tp_atr_mult:.2f}) - 平顺势仓")
                            for x in pos:
                                if (x.type == mt5.POSITION_TYPE_BUY and tside > 0) or \
                                   (x.type == mt5.POSITION_TYPE_SELL and tside < 0):
                                    ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                    cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                                    if not args.dry_run:
                                        send_order(args.symbol, x.volume, ct, cpx, 202603, "tf_tp", mt5.ORDER_FILLING_IOC)
                        # risk-stop
                        if ai and ai.balance > 0:
                            unreal_frac = (ai.equity - ai.balance) / ai.balance
                            if unreal_frac <= -args.max_unreal_loss_pct:
                                log(ts, "risk_stop", f"浮亏{unreal_frac*100:.2f}% ≥ {args.max_unreal_loss_pct*100:.0f}% - 全平")
                                for x in pos:
                                    ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                    cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                                    if not args.dry_run:
                                        send_order(args.symbol, x.volume, ct, cpx, 202603, "tf_stop", mt5.ORDER_FILLING_IOC)
            except Exception as e:  # noqa: BLE001
                print(f"[{datetime.now():%H:%M:%S}] 异常: {e}", flush=True)
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
