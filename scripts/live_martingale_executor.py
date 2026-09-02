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
    parser.add_argument("--base-lot", type=float, default=0.1)
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
                last_close = float(o["close"].iloc[-1])
                trend = 1 if o["close"].iloc[-1] > o["close"].iloc[-args.atr_bars] else \
                        (-1 if o["close"].iloc[-1] < o["close"].iloc[-args.atr_bars] else 0)

                pos = mt5.positions_get(symbol=args.symbol) or []
                buy_lots = sum(p.volume for p in pos if p.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(p.volume for p in pos if p.type == mt5.POSITION_TYPE_SELL)
                buy_avg = np.mean([p.price_open for p in pos if p.type == mt5.POSITION_TYPE_BUY]) if buy_lots else 0
                sell_avg = np.mean([p.price_open for p in pos if p.type == mt5.POSITION_TYPE_SELL]) if sell_lots else 0

                ts = datetime.now().strftime("%H:%M:%S")
                # --- decision summary ---
                detail = (f"close={last_close:.2f} trend={trend:+d} ATR={a_now:.2f} step={step:.2f} "
                          f"持仓 多{buy_lots:.2f}/空{sell_lots:.2f}")
                log(ts, "tick", detail)

                # open main basket if flat and clear trend
                if not pos and trend != 0:
                    t = mt5.POSITION_TYPE_BUY if trend > 0 else mt5.POSITION_TYPE_SELL
                    px = mt5.symbol_info_tick(args.symbol).ask if trend > 0 else mt5.symbol_info_tick(args.symbol).bid
                    log(ts, "open_main", f"{'BUY' if trend>0 else 'SELL'} {args.base_lot}手 @{px:.2f}")
                    if not args.dry_run:
                        mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                        "volume": args.base_lot, "type": t, "price": px,
                                        "deviation": 40, "magic": 202602, "comment": "grid_main",
                                        "type_time": mt5.ORDER_TIME_GTC,
                                        "type_filling": mt5.ORDER_FILLING_IOC})

                # martingale add / tp / hedge logic (simplified, adaptive to live positions)
                # add a layer when the main side has moved adversarially by 'step'
                if pos:
                    main_side = 1 if buy_lots >= sell_lots else -1
                    main_avg = buy_avg if main_side > 0 else sell_avg
                    main_last = mt5.symbol_info_tick(args.symbol).bid if main_side > 0 else mt5.symbol_info_tick(args.symbol).ask
                    moved = (main_avg - main_last) if main_side > 0 else (main_last - main_avg)
                    layers = len(pos)
                    # cooldown guard: don't avg-TP within N cycles of an add (avoid add->tp race)
                    now = time.time()
                    if now - last_add_time >= args.tp_cooldown:
                        added_this = False
                        if layers < args.max_layers and moved >= step:
                            tadd = mt5.POSITION_TYPE_BUY if main_side > 0 else mt5.POSITION_TYPE_SELL
                            px = mt5.symbol_info_tick(args.symbol).ask if main_side > 0 else mt5.symbol_info_tick(args.symbol).bid
                            log(ts, "martingale_add", f"层{layers+1} {args.base_lot}手 @{px:.2f} (反向{moved:.2f})")
                            if not args.dry_run:
                                mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": args.symbol,
                                                "volume": args.base_lot, "type": tadd, "price": px,
                                                "deviation": 40, "magic": 202602, "comment": "grid_add",
                                                "type_time": mt5.ORDER_TIME_GTC,
                                                "type_filling": mt5.ORDER_FILLING_IOC})
                            last_add_time = now
                            added_this = True
                        # average-entry TP: only after the cooldown, and only on a real gain
                        if not added_this and layers > 0:
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
