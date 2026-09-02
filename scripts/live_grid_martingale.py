# -*- coding: utf-8 -*-
"""Live grid/martingale executor (CORRECT version: position-based close, both-direction).

The previous version had a critical bug: it "closed" a long by sending an opposite SELL
WITHOUT the `position` ticket, so MT5 opened a brand-new SHORT each time — shorts
accumulated and drew down. This version closes positions ONLY by ticket (position-based),
and manages BOTH long and short baskets correctly.

Rules (friend's grid/martingale style, B + wide equity-floor stop):
  - open a fresh 0.3 base lot (long bias as default);
  - per side: take small profit when price moves tp_pts in your favor; martingale-add
    (ladder) when price moves grid_pts*layers against; each side closes by ticket;
  - wide equity-floor stop at 30% drawdown (safety net, normal months unaffected).
Demo-only auto-trade (real account = dry-run). User supervises manually.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "XAUUSD"
USC = 100.0
LADDER = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def close_by_ticket(symbol, ticket, volume, pos_type):
    """Close a specific position by its ticket: if it's a BUY, SELL to close; if SELL, BUY."""
    close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if pos_type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": close_type, "price": price, "deviation": 40, "magic": 202606,
                           "position": ticket, "comment": "close_by_ticket",
                           "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC})


def open_order(symbol, volume, order_type, price, magic=202606, comment="open"):
    return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                           "type": order_type, "price": price, "deviation": 40, "magic": magic,
                           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                           "type_filling": mt5.ORDER_FILLING_IOC})


def main():
    p = argparse.ArgumentParser(description="Correct grid/martingale executor")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--cycle-sec", type=float, default=5.0)
    p.add_argument("--base-lot", type=float, default=0.3)
    p.add_argument("--grid-pts", type=float, default=1.0)
    p.add_argument("--tp-pts", type=float, default=0.8)
    p.add_argument("--max-layers", type=int, default=6)
    p.add_argument("--equity-floor", type=float, default=0.30)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log", default=str(PROJECT_ROOT / "reports" / "live_grid_ma_log.csv"))
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
    print("演示账户 %s | %s | 网格/马丁(修正版) 0.3base 网格%.1f 止盈%.1f 底线%.0f%%" % (
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
    print("网格/马丁(修正版, 按ticket平仓) 运行中 — 人工审核")
    print("=" * 64)
    try:
        while True:
            try:
                pos = mt5.positions_get(symbol=args.symbol) or []
                ai = mt5.account_info()
                t = mt5.symbol_info_tick(args.symbol)
                bid = t.bid; ask = t.ask
                ts = datetime.now().strftime("%H:%M:%S")
                dd = (ai.balance - ai.equity) / ai.balance if ai and ai.balance > 0 else 0

                # equity floor
                if args.equity_floor and pos and dd >= args.equity_floor:
                    log(ts, "EQUITY_FLOOR", "回撤%.0f%%>=%.0f%% 强制全部平仓(ticket)" % (dd * 100, args.equity_floor * 100))
                    for x in pos:
                        if not args.dry_run:
                            close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    time.sleep(args.cycle_sec); continue

                # manage buys and sells separately
                buys = [x for x in pos if x.type == mt5.POSITION_TYPE_BUY]
                sells = [x for x in pos if x.type == mt5.POSITION_TYPE_SELL]

                # ---- LONG basket ----
                if buys:
                    avg_b = np.mean([x.price_open for x in buys])
                    layers_b = len(buys)
                    if bid >= avg_b + args.tp_pts:
                        log(ts, "TP_LONG", "多均价%.2f 现价%.2f 达止盈%.2f - 逐笔按ticket平多" % (avg_b, bid, args.tp_pts))
                        for x in buys:
                            if not args.dry_run:
                                close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    elif layers_b < args.max_layers and bid <= avg_b - args.grid_pts * layers_b:
                        lot = LADDER[min(layers_b, len(LADDER) - 1)]
                        log(ts, "MART_LONG", "多仓层%d 加%.1f手 @%.2f" % (layers_b + 1, lot, ask))
                        if not args.dry_run:
                            open_order(args.symbol, lot, mt5.POSITION_TYPE_BUY, ask, 202606, "mart_long")

                # ---- SHORT basket ----
                if sells:
                    avg_s = np.mean([x.price_open for x in sells])
                    layers_s = len(sells)
                    if ask <= avg_s - args.tp_pts:
                        log(ts, "TP_SHORT", "空均价%.2f 现价%.2f 达止盈%.2f - 逐笔按ticket平空" % (avg_s, ask, args.tp_pts))
                        for x in sells:
                            if not args.dry_run:
                                close_by_ticket(args.symbol, x.ticket, x.volume, x.type)
                    elif layers_s < args.max_layers and ask >= avg_s + args.grid_pts * layers_s:
                        lot = LADDER[min(layers_s, len(LADDER) - 1)]
                        log(ts, "MART_SHORT", "空仓层%d 加%.1f手 @%.2f" % (layers_s + 1, lot, bid))
                        if not args.dry_run:
                            open_order(args.symbol, lot, mt5.POSITION_TYPE_SELL, bid, 202606, "mart_short")

                # ---- open fresh if completely flat ----
                if not pos:
                    log(ts, "OPEN", "开多 %s手 @%.2f (无仓)" % (args.base_lot, ask))
                    if not args.dry_run:
                        open_order(args.symbol, args.base_lot, mt5.POSITION_TYPE_BUY, ask, 202606, "open")

                buy_lots = sum(x.volume for x in buys)
                sell_lots = sum(x.volume for x in sells)
                unrl = (ai.equity - ai.balance) if ai else 0
                log(ts, "status", "px%.2f 多%.2f/空%.2f 权益%.0f 浮盈%.0f 回撤%.1f%%" % (
                    (bid + ask) / 2, buy_lots, sell_lots, ai.equity, unrl, dd * 100))
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
