# -*- coding: utf-8 -*-
"""Live GRID/MARTINGALE executor (B + wide bottom-line stop). Replicates the friend's EA
style: open 0.3 base lot, add martingale layers on adverse moves, small per-cycle TP, and a
WIDE equity-floor stop (default 30% drawdown) as a safety net that only triggers in an
extreme one-way move — so the hi-win-rate accumulation is preserved while avoiding a total
blow-up.

User runs it and supervises manually. Demo-only auto-trade (real account = dry-run).

Parameters (validated by backtest):
  base_lot 0.3, martingale ladder [0.3,0.7,1.0,1.5,2.0,3.0,4.0,5.0],
  grid_pts ~1.0, take_profit ~0.8 pts, max_layers 6,
  equity_floor 0.30 (stop at 30% drawdown) -> normal months unaffected, crash capped at -30%.
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


def send_order(symbol, volume, order_type, price, magic, comment, filling=mt5.ORDER_FILLING_IOC):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": filling})


def main():
    p = argparse.ArgumentParser(description="Live grid/martingale executor (B + wide stop)")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=5.0)
    p.add_argument("--base-lot", type=float, default=0.3)
    p.add_argument("--grid-pts", type=float, default=1.0, help="grid spacing (USD/oz)")
    p.add_argument("--tp-pts", type=float, default=0.8, help="small per-cycle take profit (USD/oz)")
    p.add_argument("--max-layers", type=int, default=6)
    p.add_argument("--equity-floor", type=float, default=0.30,
                   help="force-close if drawdown >= this fraction (wide safety net)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_grid_ma_log.csv"))
    args = p.parse_args()

    ladder = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error()); sys.exit(2)
    mt5.symbol_select(args.symbol, True)
    info = mt5.account_info()
    if info is None:
        print("未登录"); sys.exit(2)
    demo = info.trade_mode == 0
    if not demo:
        print("非演示 -> DRY-RUN"); args.dry_run = True
    print("演示账户 %s | %s | 网格/马丁 0.3base 网格%.1f点 止盈%.1f点 底线止损%.0f%%" % (
        info.login, "DRY-RUN" if args.dry_run else "自动下单",
        args.grid_pts, args.tp_pts, args.equity_floor * 100))

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def log(ts, ev, detail):
        if not args.dry_run:
            with open(args.log, "a", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow([ts, ev, detail])
        print("  [%s] %s: %s" % (ts, ev, detail), flush=True)

    print("=" * 64)
    print("网格/马丁(B+底线止损) 运行中 — 人工审核, 注意单边行情风险")
    print("=" * 64)
    try:
        while True:
            try:
                pos = mt5.positions_get(symbol=args.symbol) or []
                ai = mt5.account_info()
                equity = ai.equity if ai else 0
                bal0 = ai.balance if ai else CAP
                t = mt5.symbol_info_tick(args.symbol)
                bid = t.bid; ask = t.ask
                px = (bid + ask) / 2
                ts = datetime.now().strftime("%H:%M:%S")

                # --- equity-floor safety net ---
                dd = (ai.balance - ai.equity) / ai.balance if ai and ai.balance > 0 else 0
                if args.equity_floor and pos and dd >= args.equity_floor:
                    log(ts, "EQUITY_FLOOR", "回撤%.0f%% >= %.0f%% - 强制离场(底线止损)" % (dd * 100, args.equity_floor * 100))
                    for x in pos:
                        ct = mt5.ORDER_TYPE_SELL if x.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        cpx = mt5.symbol_info_tick(args.symbol).bid if x.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(args.symbol).ask
                        if not args.dry_run:
                            send_order(args.symbol, x.volume, ct, cpx, 202606, "equity_floor")
                    time.sleep(args.cycle_sec); continue

                # --- take small profit (all lots = one TP level) ---
                if pos:
                    buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                    sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                    # assume a single direction basket (long here for the long bias demo)
                    if buy_lots > 0:
                        avg = np.mean([x.price_open for x in pos if x.type == mt5.POSITION_TYPE_BUY])
                        if bid >= avg + args.tp_pts:
                            log(ts, "TP", "多头均价%.2f 现价%.2f 达止盈%.2f - 平多仓" % (avg, bid, args.tp_pts))
                            for x in pos:
                                if x.type == mt5.POSITION_TYPE_BUY:
                                    if not args.dry_run:
                                        send_order(args.symbol, x.volume, mt5.ORDER_TYPE_SELL, bid, 202606, "tp")
                        # martingale add when price drops grid_pts * layers below avg
                        layers = sum(1 for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                        if layers < args.max_layers and bid <= avg - args.grid_pts * layers:
                            lot = ladder[min(layers, len(ladder) - 1)]
                            log(ts, "MART_ADD", "层%d 加%d手 @%.2f (均价%.2f)" % (layers + 1, lot, ask, avg))
                            if not args.dry_run:
                                send_order(args.symbol, lot, mt5.POSITION_TYPE_BUY, ask, 202606, "mart")
                    elif sell_lots > 0:
                        avg = np.mean([x.price_open for x in pos if x.type == mt5.POSITION_TYPE_SELL])
                        if ask <= avg - args.tp_pts:
                            log(ts, "TP", "空头均价%.2f 现价%.2f 达止盈" % (avg, ask))
                            for x in pos:
                                if x.type == mt5.POSITION_TYPE_SELL:
                                    if not args.dry_run:
                                        send_order(args.symbol, x.volume, mt5.ORDER_TYPE_BUY, ask, 202606, "tp")
                        layers = sum(1 for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                        if layers < args.max_layers and ask >= avg + args.grid_pts * layers:
                            lot = ladder[min(layers, len(ladder) - 1)]
                            log(ts, "MART_ADD", "层%d 加%d手 @%.2f" % (layers + 1, lot, bid))
                            if not args.dry_run:
                                send_order(args.symbol, lot, mt5.POSITION_TYPE_SELL, bid, 202606, "mart")
                else:
                    # open a fresh 0.3 lot long (grid bias; the friend EA is long-biased in demo)
                    log(ts, "OPEN", "开多 0.3手 @%.2f" % ask)
                    if not args.dry_run:
                        send_order(args.symbol, args.base_lot, mt5.POSITION_TYPE_BUY, ask, 202606, "open")

                # status log
                buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
                sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
                unrl = (ai.equity - ai.balance) if ai else 0
                log(ts, "status", "px%.2f 多%.2f/空%.2f 权益%.0f 浮盈%.0f 回撤%.1f%%" % (
                    px, buy_lots, sell_lots, equity, unrl, dd * 100))
            except Exception as e:  # noqa: BLE001
                print("[%s] 异常: %s" % (datetime.now().strftime("%H:%M:%S"), e), flush=True)
            time.sleep(args.cycle_sec)
    except KeyboardInterrupt:
        print("\n已停止(Martingale 人工审核在).")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
