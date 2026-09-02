# -*- coding: utf-8 -*-
"""Live daily monitor for the adaptive+circuit-breaker bull strategy.

Reads the latest gold daily data, computes the current market state
(bull score / efficiency ratio / volatility percentile), and prints a
decision headline: suggested leverage, micro signal, stop/target, and a
one-line note. Saves a decision snapshot.

Usage:
  py scripts/monitor_bull_adaptive.py                # use cached daily CSV
  py scripts/monitor_bull_adaptive.py --csv path.csv # use a custom CSV
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.bull_adaptive import (  # noqa: E402
    MICRO_W,
    AdaptiveConfig,
    _lev_risk,
    build_signals,
    prepare_daily,
)
from london_gold.factor_library import aggregate_score, build_factors  # noqa: E402

STATE_NAMES = {
    "BEAR": "熊市/无趋势 — 空仓",
    "EXTREME_VOL": "极端波动 — 空仓避险",
    "HIGH_VOL": "高波动 — 1x 低杠杆",
    "CLEAN_TREND": "干净趋势 — 10x 高杠杆",
    "BULL": "普通牛市 — 5x 中杠杆",
    "CHOP": "震荡 — 2x 低杠杆",
}


def classify(row, cfg):
    b = row["bull"]
    er = row["er20"] if not np.isnan(row["er20"]) else 0.0
    vol = row["atr_pctl"] if not np.isnan(row["atr_pctl"]) else 0.5
    lev, risk = _lev_risk(row, cfg)
    if b < cfg.bull_thr:
        state = "BEAR"
    elif vol > cfg.ext_vol_pctl:
        state = "EXTREME_VOL"
    elif vol > cfg.high_vol_pctl:
        state = "HIGH_VOL"
    elif er > cfg.er_clean and vol < cfg.high_vol_pctl:
        state = "CLEAN_TREND"
    elif er > cfg.er_bull:
        state = "BULL"
    else:
        state = "CHOP"
    return state, lev, risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "XAUUSD_1d.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    cfg = AdaptiveConfig()

    prepared = prepare_daily(df, cfg)
    frame = build_signals(prepared, cfg)

    # compute micro score (factor composite) for a fuller readout
    fac = build_factors(df)
    micro_series = aggregate_score(fac, MICRO_W)
    micro = float(micro_series.iloc[-1])

    last = prepared.iloc[-1]
    state, lev, risk = classify(last, cfg)

    # Recommend the SIGNAL (micro score sign) as of last close; signal is 0/-1/+1
    sig = int(frame["signal"].iloc[-1])
    stop_dist = float(frame["stop_dist"].iloc[-1])
    tp_dist = float(frame["tp_dist"].iloc[-1])
    close = float(frame["close"].iloc[-1])

    action = {0: "观望/空仓", 1: "做多", -1: "做空"}[sig]

    print("=" * 60)
    print(f"伦敦金自适应策略 — 今日监控   {datetime.now():%Y-%m-%d %H:%M}")
    print(f"数据日期: {last['date'].date()}   收盘: ${close:,.1f}")
    print("=" * 60)
    print(f"市场状态 : {STATE_NAMES.get(state, state)}")
    print(f"建议杠杆 : {lev:.0f}x   (单笔风险 {risk*100:.1f}%)")
    print(f"信号方向 : {action}   (micro score={micro:+.3f})")
    if stop_dist > 0 and sig != 0:
        direction = 1 if sig > 0 else -1
        stop = close - direction * stop_dist
        target = close + direction * tp_dist
        print(f"建议止损 : ${stop:,.1f}   建议目标: ${target:,.1f}")
    print("=" * 60)
    note = {
        "BEAR": "趋势未确立，建议空仓等待，等 bull 分>0.55 再参与。",
        "EXTREME_VOL": "波动率冲高，规避黑天鹅，空仓。",
        "HIGH_VOL": "波动偏大，只做 1x 低杠杆，谨慎。",
        "CLEAN_TREND": "干净上升趋势，可用 10x 高杠杆放大。",
        "BULL": "牛市确认，可用 5x 中杠杆。",
        "CHOP": "震荡市，只做 2x 低杠杆或观望。",
    }[state]
    print(f"建议：{note}")

    # recommend signal (only if bull+clean enough)
    print(f"\n[信号] 当前建议{'做多' if sig > 0 else '做空' if sig < 0 else '不动'} — "
          f"持仓方向跟随信号，未确认趋势时以空仓为主。")

    # snapshot
    out = PROJECT_ROOT / "data"
    snap = {
        "date": str(last["date"].date()), "close": close, "state": state,
        "suggested_leverage": lev, "risk_pct": risk, "signal": sig, "micro": micro,
        "stop_dist": stop_dist, "tp_dist": tp_dist,
    }
    with open(out / "bull_adaptive_decision.json", "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=2)
    print(f"\n已保存决策快照 -> data/bull_adaptive_decision.json")


if __name__ == "__main__":
    main()
