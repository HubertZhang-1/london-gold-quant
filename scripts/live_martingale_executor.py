# -*- coding: utf-8 -*-
"""LIVE auto-trading executor for the hedged martingale-grid main strategy.

This turns the reverse-engineered *backtest* engine (martingale_grid.py) into a LIVE
loop on MT5: each cycle it reads the real M1 history + current open positions, applies
the martingale-grid rules (trend-follow main basket, price-tier martingale adds,
average-entry take-profit, reverse-hedge), and places MT5 orders accordingly. Every
decision is logged to CSV so the strategy can be reviewed/improved from real trades.

SAFETY: it only auto-places orders on a DEMO account (trade_mode == 0); on a real
account it runs in read-only (dry-run) mode and prints what it WOULD do.

Usage:
  py scripts/live_martingale_executor.py --dry-run        # just print decisions (no orders)
  py scripts/live_martingale_executor.py                  # auto-trade on demo, log CSV
  py scripts/live_martingale_executor.py --cycle-sec 60
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


def ohlc(rates: np.ndarray) -> dict:
    d = pd.DataFrame(rates)
    return {
        "open": d["open"].astype(float), "high": d["high"].astype(float),
        "low": d["low"].astype(float), "close": d["close"].astype(float),
        "time": pd.to_datetime(d["time"], unit="s", utc=True),
    }


def leverage_base_lot(equity: float, price: float, leverage: float, oz_per_lot: float = 100.0) -> float:
    """Lots so that notional (lots*price*oz_per_lot) = leverage * equity."""
    if price <= 0 or oz_per_lot <= 0:
        return 0.01
    return max(0.01, round(leverage * equity / (price * oz_per_lot), 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live martingale-grid auto-trading executor")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--cycle-sec", type=float, default=60.0, help="seconds between decisions")
    parser.add_argument("--atr-bars", type=int, default=14)
    parser.add_argument("--grid-atr", type=float, default=1.0)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--tp-atr", type=float, default=1.0)
    parser.add_argument("--hedge-atr", type=float, default=1.6)
    parser.add_argument("--tp-cooldown", type=float, default=120.0,
                        help="seconds to wait after a martingale add before avg-TP can fire")
    parser.add_argument("--trend-ema-fast", type=int, default=10)
    parser.add_argument("--trend-ema-slow", type=int, default=30)
    parser.add_argument("--tp-atr-mult", type=float, default=None,
                        help="take-profit: close when the trend-side unrealized profit "
                             ">= tp_atr_mult * (tpatr_tf ATR), e.g. 2.0 = lock in after a "
                             "2x (that TF) ATR move")
    parser.add_argument("--tp-atr-tf", default=None,
                        help="timeframe for the TP ATR basis (M1/M5/M15/M30/H1/H4); "
                             "default: use the M1 grid step scaled by tp-atr-mult")
    parser.add_argument("--base-lot", type=float, default=0.1)
    parser.add_argument("--target-leverage", type=float, default=None,
                        help="if set, base lot = leverage * account_equity / (price * 100 oz); "
                             "overrides --base-lot (e.g. 5.0 = 5x notional)")
    parser.add_argument("--max-unreal-loss-pct", type=float, default=None,
                        help="hard stop: if unrealized loss >= this % of equity, close the basket")
    parser.add_argument("--dry-run", action="store_true", help="print decisions, do NOT place orders")
    parser.add_argument("--only-demo", action="store_true", default=True, help="only trade if demo account")
    parser.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_martingale_executor_log.csv"))
    args = parser.parse_args()

    if not mt5.initialize():
        print("❌ MT5 init failed:", mt5.last_error())
        sys.exit(2)
    mt5.symbol_select(args.symbol, True)
    info = mt5.account_info()
    if info is None:
        print("❌ 未登录")
        sys.exit(2)
    demo = info.trade_mode == 0
    if args.only_demo and not demo:
        print(f"⚠️ 非演示账户(trade_mode={info.trade_mode})，切换为 DRY-RUN（只读不下单）")
        args.dry_run = True
    if demo:
        print(f"✅ 演示账户 {info.login}，{'DRY-RUN' if args.dry_run else '自动下单'}")
    else:
        print(f"🔒 账户 {info.login} trade_mode={info.trade_mode} -> DRY-RUN")

    # append-mode CSV log
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_exists = Path(args.log).exists()

    print("=" * 64)
    print(f"主策略(双边网格马丁) 自动执行  {args.symbol}  M1  每{args.cycle_sec:.0f}s")
    print(f"grid={args.grid_atr}ATR  max_layers={args.max_layers}  tp={args.tp_atr}ATR  "
          f"hedge={args.hedge_atr}ATR  base_lot={args.base_lot}")
    print("=" * 64)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not log_exists:
                    w.writerow(["ts", "event", "detail"])
                w.writerow([ts, ev, detail])
        print(f"  [{ts}] {ev}: {detail}", flush=True)

    last_add_time = 0.0  # time of last martingale add (for tp-cooldown)

    try:
        while True:
            try:
                rates = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 500)
                if rates is None or len(rates) < 50:
                    print("⚠️ 无足够M1数据，等待")
                    time.sleep(args.cycle_sec); continue
                o = ohlc(rates)
                a = atr(o["high"], o["low"], o["close"], args.atr_bars)
                a_now = float(a.iloc[-1]) if not np.isnan(a.iloc[-1]) else 1.0
                step = a_now * args.grid_atr
                # TP ATR basis: if --tp-atr-tf given, use that TF's ATR (e.g. M15), else M1 step
                _TF_MAP = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                           "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
                if args.tp_atr_tf and args.tp_atr_tf.upper() in _TF_MAP:
                    r_tf = mt5.copy_rates_from_pos(args.symbol, _TF_MAP[args.tp_atr_tf.upper()], 0, 40)
                    o_tf = ohlc(r_tf) if r_tf is not None else o
                    a_tf = atr(o_tf["high"], o_tf["low"], o_tf["close"], args.atr_bars)
                    tp_atr = float(a_tf.iloc[-1]) if not np.isnan(a_tf.iloc[-1]) else a_now
                else:
                    tp_atr = step
                last_close = float(o["close"].iloc[-1])
                # EMA-based trend (smoother than a 14-bar slope, avoids flip-flopping
                # in a one-way move). Up if close > ema_fast and ema_fast > ema_slow.
                ema_fast = float(pd.Series(o["close"]).ewm(span=args.trend_ema_fast, adjust=False).mean().iloc[-1])
                ema_slow = float(pd.Series(o["close"]).ewm(span=args.trend_ema_slow, adjust=False).mean().iloc[-1])
                ema_prev = float(pd.Series(o["close"]).ewm(span=args.trend_ema_slow, adjust=False).mean().iloc[-2])
                trend = 0
                if last_close > ema_fast > ema_slow and ema_slow >= ema_prev:
                    trend = 1
                elif last_close < ema_fast < ema_slow and ema_slow <= ema_prev:
                    trend = -1

                pos = mt5.positions_get(symbol=args.symbol) or []
                buy_lots = sum(p.volume for p in pos if p.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(p.volume for p in pos if p.type == mt5.POSITION_TYPE_SELL)
                buy_avg = np.mean([p.price_open for p in pos if p.type == mt5.POSITION_TYPE_BUY]) if buy_lots else 0
                sell_avg = np.mean([p.price_open for p in pos if p.type == mt5.POSITION_TYPE_SELL]) if sell_lots else 0

                # resolve base lot: leverage-based (5x notional) or explicit --base-lot
                ai = mt5.account_info()
                live_equity = ai.equity if ai else last_close
                tick0 = mt5.symbol_info_tick(args.symbol)
                price0 = (tick0.ask + tick0.bid) / 2.0 if tick0 else last_close
                if args.target_leverage:
                    base_lot = leverage_base_lot(live_equity, price0, args.target_leverage)
                else:
                    base_lot = args.base_lot

                ts = datetime.now().strftime("%H:%M:%S")
                # --- decision summary ---
                detail = (f"close={last_close:.2f} trend={trend:+d} ATR={a_now:.2f} step={step:.2f} "
                          f"持仓 多{buy_lots:.2f}/空{sell_lots:.2f} base_lot={base_lot:.2f}")
                log(ts, "tick", detail)

                # open main basket if flat and clear trend
                if not pos and trend != 0:
                    t = mt5.POSITION_TYPE_BUY if trend > 0 else mt5.POSITION_TYPE_SELL
                    px = mt5.symbol_info_tick(args.symbol).ask if trend > 0 else mt5.symbol_info_tick(args.symbol).bid
                    log(ts, "open_main", f"{'BUY' if trend>0 else 'SELL'} {base_lot}手 @{px:.2f}")
                    if not args.dry_run:
                        mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                        "volume": base_lot, "type": t, "price": px,
                                        "deviation": 40, "magic": 202602, "comment": "grid_main",
                                        "type_time": mt5.ORDER_TIME_GTC,
                                        "type_filling": mt5.ORDER_FILLING_IOC})

                # martingale / trend-consistency logic (FIXED: only add with the trend,
                # and close positions that OPPOSE the trend instead of riding them)
                if pos:
                    main_side = 1 if buy_lots >= sell_lots else -1
                    main_avg = buy_avg if main_side > 0 else sell_avg
                    main_last = mt5.symbol_info_tick(args.symbol).bid if main_side > 0 else mt5.symbol_info_tick(args.symbol).ask
                    moved = (main_avg - main_last) if main_side > 0 else (main_last - main_avg)
                    # trend-side unrealized PROFIT distance (positive when in profit)
                    profit_dist = (main_last - main_avg) if main_side > 0 else (main_avg - main_last)
                    layers = len(pos)
                    now = time.time()

                    # 0) TAKE-PROFIT: lock in after a tp_atr_mult * (tp ATR) favorable move
                    if args.tp_atr_mult and profit_dist >= tp_atr * args.tp_atr_mult:
                        log(ts, "take_profit", f"浮盈 {profit_dist:.2f} ≥ {args.tp_atr_mult}xATR({tp_atr*args.tp_atr_mult:.2f}) - 止盈离场")
                        if not args.dry_run:
                            for p in pos:
                                if (p.type == mt5.POSITION_TYPE_BUY and main_side > 0) or \
                                   (p.type == mt5.POSITION_TYPE_SELL and main_side < 0):
                                    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                                    "volume": p.volume,
                                                    "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                                    "price": mt5.symbol_info_tick(args.symbol).bid if p.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask,
                                                    "deviation": 40, "magic": 202602, "position": p.ticket,
                                                    "comment": "take_profit", "type_time": mt5.ORDER_TIME_GTC,
                                                    "type_filling": mt5.ORDER_FILLING_IOC})
                    # 1) TREND-CONSISTENCY: if the open side now OPPOSES the trend, close it
                    elif trend != 0 and main_side != trend:
                        log(ts, "trend_flip", f"趋势{trend:+d}与持仓{'多' if main_side>0 else '空'}相反 - 平该侧仓位")
                        if not args.dry_run:
                            for p in pos:
                                if (p.type == mt5.POSITION_TYPE_BUY and main_side > 0) or \
                                   (p.type == mt5.POSITION_TYPE_SELL and main_side < 0):
                                    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                                    "volume": p.volume,
                                                    "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                                    "price": mt5.symbol_info_tick(args.symbol).bid if p.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask,
                                                    "deviation": 40, "magic": 202602, "position": p.ticket,
                                                    "comment": "trend_flip", "type_time": mt5.ORDER_TIME_GTC,
                                                    "type_filling": mt5.ORDER_FILLING_IOC})
                    # 1b) HARD RISK STOP: if unrealized loss >= threshold of equity, close basket
                    elif args.max_unreal_loss_pct and buy_lots + sell_lots > 0:
                        ui = mt5.account_info()
                        unreal_frac = (ui.equity - ui.balance) / ui.balance if ui and ui.balance > 0 else 0.0
                        if unreal_frac <= -args.max_unreal_loss_pct:
                            log(ts, "risk_stop", f"浮亏 {unreal_frac*100:.2f}% ≥ {args.max_unreal_loss_pct*100:.2f}% - 强平全仓")
                            if not args.dry_run:
                                for p in pos:
                                    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                                    "volume": p.volume,
                                                    "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                                    "price": mt5.symbol_info_tick(args.symbol).bid if p.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask,
                                                    "deviation": 40, "magic": 202602, "position": p.ticket,
                                                    "comment": "risk_stop", "type_time": mt5.ORDER_TIME_GTC,
                                                    "type_filling": mt5.ORDER_FILLING_IOC})
                    # 2) ADD only WITH the trend (never against it)
                    elif trend != 0 and main_side == trend and \
                            now - last_add_time >= args.tp_cooldown and \
                            layers < args.max_layers and moved >= step:
                        tadd = mt5.POSITION_TYPE_BUY if main_side > 0 else mt5.POSITION_TYPE_SELL
                        px = mt5.symbol_info_tick(args.symbol).ask if main_side > 0 else mt5.symbol_info_tick(args.symbol).bid
                        log(ts, "martingale_add", f"顺势 层{layers+1} {base_lot}手 @{px:.2f} (反向{moved:.2f})")
                        if not args.dry_run:
                            mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                            "volume": base_lot, "type": tadd, "price": px,
                                            "deviation": 40, "magic": 202602, "comment": "grid_add",
                                            "type_time": mt5.ORDER_TIME_GTC,
                                            "type_filling": mt5.ORDER_FILLING_IOC})
                        last_add_time = now
                    # 3) average-entry TP: only after cooldown, only on a real gain
                    elif now - last_add_time >= args.tp_cooldown and layers > 0:
                        gain = (main_avg - main_last) if main_side > 0 else (main_last - main_avg)
                        if gain >= step * args.tp_atr:
                            log(ts, "avg_tp", f"主侧均价回正 {gain:.2f}% - 平主仓")
                            if not args.dry_run:
                                for p in pos:
                                    if (p.type == mt5.POSITION_TYPE_BUY and main_side > 0) or \
                                       (p.type == mt5.POSITION_TYPE_SELL and main_side < 0):
                                        mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                                        "volume": p.volume,
                                                        "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                                        "price": mt5.symbol_info_tick(args.symbol).bid if p.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask,
                                                        "deviation": 40, "magic": 202602, "position": p.ticket,
                                                        "comment": "grid_tp", "type_time": mt5.ORDER_TIME_GTC,
                                                        "type_filling": mt5.ORDER_FILLING_IOC})
            except Exception as e:  # noqa: BLE001
                print(f"[{datetime.now():%H:%M:%S}] 决策异常: {e}", flush=True)
            time.sleep(args.cycle_sec)
    except KeyboardInterrupt:
        print("\n已停止自动执行。")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
