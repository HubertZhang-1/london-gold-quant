# -*- coding: utf-8 -*-
"""Continuous monitor for the trend-tracking executor, pushing key events to Feishu.

Watches the XAUUSD account/positions and the M5 trend, and sends a Feishu alert on:
  - trend reversal (EMA fast/slow flips)  -> executor should close + short
  - the executor's position direction (long/short) changes
  - a hard stop (2x ATR) is likely hit (position drawdown growing)
  - drawdown alert thresholds (10% / 20% / 30%)
  - a sharp M5 drop (>5 USD) = crash warning
Runs forever; --interval controls the poll. Uses feishu_push.send_feishu for alerts.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.feishu_push import send_feishu, load_webhook  # noqa: E402

SYMBOL = "XAUUSD"
USC = 100.0


def ewm(a, span):
    al = 2 / (span + 1); out = np.empty(len(a)); out[0] = a[0]
    for k in range(1, len(a)): out[k] = al * a[k] + (1 - al) * out[k - 1]
    return out


def get_trend(rates, fast, slow):
    df = pd.DataFrame(rates)
    cl = df["close"].astype(float).to_numpy()
    ef = ewm(cl, fast); es = ewm(cl, slow)
    return (1 if cl[-1] > ef[-1] > es[-1] else (-1 if cl[-1] < ef[-1] < es[-1] else 0))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Monitor trend executor, push Feishu alerts")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--ema-fast", type=int, default=10)
    p.add_argument("--ema-slow", type=int, default=30)
    p.add_argument("--dd-alert", type=float, default=0.10, help="warn at this drawdown")
    p.add_argument("--crash-pts", type=float, default=5.0, help="M5 drop that triggers crash alert")
    args = p.parse_args()
    webhook = load_webhook()
    if not webhook:
        print("未配置 FEISHU_WEBHOOK; 监控继续但不会推送。")
    mt5.initialize()
    mt5.symbol_select(SYMBOL, True)

    last_trend = 0
    last_pos = 0
    last_crash_spam = 0.0
    last_dd_alert = 0.0
    print("监控启动: 趋势转跌/开空/止损/暴跌 -> 飞书推送")
    while True:
        try:
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 60)
            if rates is None or len(rates) < args.ema_slow:
                time.sleep(args.interval); continue
            trend = get_trend(rates, args.ema_fast, args.ema_slow)
            pos = mt5.positions_get(symbol=SYMBOL) or []
            ai = mt5.account_info()
            t = mt5.symbol_info_tick(SYMBOL)
            ts = datetime.now().strftime("%H:%M:%S")
            buy_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_BUY)
            sell_lots = sum(x.volume for x in pos if x.type == mt5.POSITION_TYPE_SELL)
            cur = 1 if buy_lots > 0 else (-1 if sell_lots > 0 else 0)
            dd = (ai.balance - ai.equity) / ai.balance if ai and ai.balance > 0 else 0
            # M5 recent drop
            cl = pd.Series(r["close"] for r in rates) if False else \
                pd.DataFrame(rates)["close"].astype(float)
            drop5 = float(cl.iloc[-1] - cl.iloc[-5])

            alerts = []
            # trend reversal (flip) — executor should close + short
            if last_trend != 0 and trend != 0 and trend != last_trend:
                alerts.append(f"🔁 趋势反转 {last_trend:+d}→{trend:+d} — 执行器将平旧仓+反手，请留意")
            # position direction change
            if cur != 0 and last_pos != 0 and cur != last_pos:
                alerts.append(f"↔️ 持仓方向变化: {last_pos:+d}→{cur:+d} (开{'多' if cur>0 else '空'})")
            # hard stop / big unrealized loss (position losing to the 2x ATR stop level)
            if cur != 0:
                px_open = [x.price_open for x in pos if
                           (x.type == mt5.POSITION_TYPE_BUY if cur > 0 else
                            x.type == mt5.POSITION_TYPE_SELL)]
                if px_open:
                    avg = float(np.mean(px_open))
                    last_bid = mt5.symbol_info_tick(SYMBOL).bid
                    last_ask = mt5.symbol_info_tick(SYMBOL).ask
                    # distance moved against the position
                    dist = (avg - last_bid) if cur > 0 else (last_ask - avg)
                    # ATR-based stop level
                    tr = np.maximum(pd.DataFrame(rates)["high"] - pd.DataFrame(rates)["low"],
                                    np.maximum((pd.DataFrame(rates)["high"] - pd.DataFrame(rates)["close"].shift()).abs(),
                                               (pd.DataFrame(rates)["low"] - pd.DataFrame(rates)["close"].shift()).abs()))
                    atr_now = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 5.0
                    stop_pts = 2.0 * atr_now
                    if dist >= stop_pts * 0.85:
                        alerts.append(f"🛑 止损临近: {'多' if cur>0 else '空'}仓均价{avg:.2f} 现价{last_bid:.2f} "
                                      f"逆势{dist:.2f} ≥ 2xATR({stop_pts:.2f}) — 即将触发止损")
            # drawdown
            if dd >= args.dd_alert and dd > last_dd_alert:
                alerts.append(f"⚠️ 回撤 {dd*100:.1f}% 达到预警线 {args.dd_alert*100:.0f}%")
                last_dd_alert = dd
            # crash
            if drop5 <= -args.crash_pts and time.time() - last_crash_spam > 180:
                alerts.append(f"🚨 暴跌预警: M5近5根 {drop5:+.2f} USD (触发阈值 -{args.crash_pts})")
                last_crash_spam = time.time()
            # position losing much (hard stop proximity)
            if cur != 0:
                avg = np.mean([x.price_open for x in pos if x.type == mt5.POSITION_TYPE_BUY if cur > 0] or
                              [x.price_open for x in pos if x.type == mt5.POSITION_TYPE_SELL if cur < 0] or [0])
                unrl = (ai.equity - ai.balance) if ai else 0

            for a in alerts:
                print(f"[{ts}] {a}", flush=True)
                if webhook:
                    try:
                        send_feishu(webhook, f"【黄金监控】{a}\n状态: 趋势{trend:+d} 多{buy_lots:.2f}/空{sell_lots:.2f} 权益${ai.equity:,.0f} 回撤{dd*100:.1f}%")
                    except Exception as e:
                        print(f"  飞书发送失败: {e}", flush=True)

            last_trend = trend
            last_pos = cur
        except Exception as e:  # noqa: BLE001
            print(f"[{datetime.now():%H:%M:%S}] 监控异常: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
