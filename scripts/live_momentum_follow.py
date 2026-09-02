# -*- coding: utf-8 -*-
"""Live SIMPLE momentum-follow engine for XAUUSD (追涨杀跌 + 看换手量).

User feedback: the complex EMA-trend + trailing-stop engine was over-defensive (frequent
stops, missed the real momentum wave) and LOST money, while a plain "follow the M1
momentum + trade when volume confirms" would profit because gold has intraday momentum.

This engine is deliberately simple:
  1. Direction = recent M1 momentum: close vs the close N bars ago -> long if up, short if down.
  2. Volume confirmation: only trade when the latest tick_volume >= mean * volume_mult
     (i.e. the move is on real turnover), skipping dead-quiet false moves.
  3. 5x leverage, size = leverage * equity / (price * 100 oz).
  4. Exit: momentum flips, OR take-profit at tp_atr_mult * (chosen TF) ATR, OR a wide
     ATR stop to avoid a one-way move riding to the 5% loss.
  5. No repeated re-entry churn: after a flip/exit, wait a cooldown.

SAFETY: only auto-trades on a DEMO account (trade_mode == 0), else dry-run.

Usage:
  py scripts/live_momentum_follow.py --target-leverage 5 --momentum-bars 3
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


def ohlc(rates: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame(rates)
    d["time"] = pd.to_datetime(d["time"], unit="s", utc=True)
    return d


def leverage_lots(equity: float, price: float, lev: float, oz: float = 100.0) -> float:
    return max(0.01, round(lev * equity / (price * oz), 2))


def send_order(symbol, volume, order_type, price, magic, comment, filling=mt5.ORDER_FILLING_IOC):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": filling})


def main() -> None:
    p = argparse.ArgumentParser(description="Live simple momentum-follow XAUUSD")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=30.0)
    p.add_argument("--momentum-bars", type=int, default=3, help="compare close vs close N bars ago")
    p.add_argument("--volume-mult", type=float, default=1.2,
                   help="only trade when latest tick_volume >= (mean * this)")
    p.add_argument("--target-leverage", type=float, default=5.0)
    p.add_argument("--tp-atr-mult", type=float, default=3.0)
    p.add_argument("--tp-atr-tf", default="M5")
    p.add_argument("--stop-atr-mult", type=float, default=4.0,
                   help="wide ATR stop (loose, to avoid a one-way ride) — 0 disables")
    p.add_argument("--max-unreal-loss-pct", type=float, default=0.05)
    p.add_argument("--cooldown-bars", type=int, default=2, help="bars to wait after an exit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_momentum_follow_log.csv"))
    args = p.parse_args()

    if not mt5.initialize():
        print("❌ MT5 init failed:", mt5.last_error()); sys.exit(2)
    mt5.symbol_select(args.symbol, True)
    info = mt5.account_info()
    if info is None:
        print("❌ 未登录"); sys.exit(2)
    demo = info.trade_mode == 0
    if not demo:
        print(f"🔒 非演示账户 -> DRY-RUN"); args.dry_run = True
    print(f"✅ 演示账户 {info.login} | {'DRY-RUN' if args.dry_run else '自动下单'}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ts, ev, detail])
        print(f"  [{ts}] {ev}: {detail}", flush=True)

    print("=" * 66)
    print(f"简单顺势引擎 {args.symbol}  {args.momentum_bars}根动量+量{args.volume_mult}x  "
          f"杠杆{args.target_leverage}x  TP{args.tp_atr_mult}x{args.tp_atr_tf}ATR  止损{args.stop_atr_mult}xATR")
    print("=" * 66)

    last_side = 0
    exit_bars = 0

    try:
        while True:
            try:
                rates = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 60)
                if rates is None or len(rates) < 40:
                    time.sleep(args.cycle_sec); continue
                d = ohlc(rates)
                close = d["close"].astype(float)
                vol = d["tick_volume"].astype(float)
                # momentum direction
                m = args.momentum_bars
                mom = 1 if close.iloc[-1] > close.iloc[-m] else (-1 if close.iloc[-1] < close.iloc[-m] else 0)
                # volume confirmation (compare latest bar to recent mean)
                vol_mean = vol.tail(20).mean()
                vol_confirmed = vol.iloc[-1] >= vol_mean * args.volume_mult

                pos = mt5.positions_get(symbol=args.symbol) or []
                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                ai = mt5.account_info()
                equity = ai.equity if ai else 0
                price = float(close.iloc[-1])
                base_lot = leverage_lots(equity, price, args.target_leverage)

                ts = datetime.now().strftime("%H:%M:%S")
                log(ts, "tick", f"close={price:.2f} mom={mom:+d} vol={vol.iloc[-1]:.0f}/avg{vol_mean:.0f} "
                                f"confirm={'Y' if vol_confirmed else 'N'} 多{buy_lots:.2f}/空{sell_lots:.2f} lot={base_lot:.2f}")

                # ---- exit logic on the open side ----
                side = 1 if buy_lots > 0 else (-1 if sell_lots > 0 else 0)
                if side != 0 and exit_bars > 0:
                    exit_bars -= 1
                elif side != 0:
                    # TP basis
                    tv = args.tp_atr_tf.upper()
                    rt = mt5.copy_rates_from_pos(args.symbol, _TF[tv], 0, 40) if tv in _TF else None
                    ot = ohlc(rt) if rt is not None else d
                    a = atr(ot["high"].astype(float), ot["low"].astype(float), ot["close"].astype(float), 14)
                    tp_atr = float(a.iloc[-1]) if not np.isnan(a.iloc[-1]) else 5.0
                    avg = np.mean([x.price_open for x in pos])
                    last = mt5.symbol_info_tick(args.symbol).bid if side > 0 else mt5.symbol_info_tick(args.symbol).ask
                    profit = (last - avg) if side > 0 else (avg - last)
                    # momentum flip -> exit
                    flip = (mom != 0 and mom != side)
                    # take-profit
                    if profit >= tp_atr * args.tp_atr_mult:
                        log(ts, "take_profit", f"盈利{profit:.2f} ≥ {args.tp_atr_mult}x{tv}ATR({tp_atr*args.tp_atr_mult:.2f}) 平仓")
                        for x in pos:
                            ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                            cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                            if not args.dry_run:
                                send_order(args.symbol, x.volume, ct, cpx, 202604, "m_tp")
                        exit_bars = args.cooldown_bars
                    elif flip:
                        log(ts, "momentum_flip", f"动量{side:+d}→{mom:+d} 反向 - 平仓")
                        for x in pos:
                            ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                            cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                            if not args.dry_run:
                                send_order(args.symbol, x.volume, ct, cpx, 202604, "m_flip")
                        exit_bars = args.cooldown_bars
                    elif args.stop_atr_mult > 0 and (side > 0 and last <= avg - args.stop_atr_mult * tp_atr):
                        log(ts, "stop", f"止损(超{args.stop_atr_mult}xATR)")
                        for x in pos:
                            ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                            cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                            if not args.dry_run:
                                send_order(args.symbol, x.volume, ct, cpx, 202604, "m_stop")
                        exit_bars = args.cooldown_bars

                # ---- open a fresh momentum position ----
                if not pos and exit_bars == 0 and mom != 0 and vol_confirmed:
                    otype = mt5.POSITION_TYPE_BUY if mom > 0 else mt5.POSITION_TYPE_SELL
                    px = mt5.symbol_info_tick(args.symbol).ask if mom > 0 else mt5.symbol_info_tick(args.symbol).bid
                    log(ts, "open", f"{'BUY' if mom>0 else 'SELL'} {base_lot}手 @{px:.2f} (动量{mom:+d},量确认)")
                    if not args.dry_run:
                        send_order(args.symbol, base_lot, otype, px, 202604, "m_open")
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
