# -*- coding: utf-8 -*-
"""Realtime signal scanner for London gold."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_SCAN_FILE, fetch_daily, fetch_realtime
from .strategies import donchian_breakout_signals, ema_cross_signals, rsi_reversal_signals

STRATEGY_FUNCS = {
    "donchian_breakout": donchian_breakout_signals,
    "ema_cross": ema_cross_signals,
    "rsi_reversal": rsi_reversal_signals,
}


def scan(config: dict, update_data: bool = False, save: bool = True) -> dict:
    """Build a JSON snapshot of current strategy signals plus the live quote."""
    data_cfg = config.get("data", {})
    df = fetch_daily(force=update_data, cache_path=data_cfg.get("cache", "data/london_gold_daily.csv"))
    if df is None or df.empty:
        raise RuntimeError("no London gold daily data available")

    quote = None
    try:
        quote = fetch_realtime()
    except Exception:
        quote = None

    snapshot = {
        "symbol": "XAU",
        "updated_at": pd.Timestamp.now().isoformat(),
        "last_close": float(df["close"].iloc[-1]),
        "last_date": str(df["date"].iloc[-1].date()),
        "quote": quote,
        "strategies": {},
    }

    strategies_cfg = config.get("strategies", {})
    for name, cfg in strategies_cfg.items():
        func = STRATEGY_FUNCS.get(name)
        if func is None:
            continue
        params = cfg.get("default", {})
        try:
            frame = func(df, **params)
        except Exception as exc:
            snapshot["strategies"][name] = {"error": str(exc)}
            continue
        last = frame.iloc[-1]
        signal = int(last["signal"])
        close = float(last["close"])
        stop_dist = float(last["stop_dist"]) if not np.isnan(last["stop_dist"]) else 0.0
        entry = {"signal": signal}
        if signal > 0:
            entry["action"] = "long"
            entry["stop"] = round(close - stop_dist, 2)
            entry["level"] = round(float(last.get("upper", np.nan)), 2) if name == "donchian_breakout" else None
        elif signal < 0:
            entry["action"] = "short"
            entry["stop"] = round(close + stop_dist, 2)
            entry["level"] = round(float(last.get("lower", np.nan)), 2) if name == "donchian_breakout" else None
        else:
            entry["action"] = "flat"
            entry["stop"] = None
            entry["level"] = None
        entry["atr"] = round(float(last["atr"]), 2)
        for col in ("upper", "lower", "ma", "fast", "slow", "rsi", "exit_upper", "exit_lower"):
            if col in frame.columns:
                value = last.get(col)
                entry[col] = None if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), 2)
        snapshot["strategies"][name] = entry

    if save:
        out = Path(data_cfg.get("scan", DEFAULT_SCAN_FILE))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return snapshot


def print_snapshot(snapshot: dict) -> None:
    quote = snapshot.get("quote") or {}
    print("=" * 64)
    print("London Gold Signal Scan")
    print(f"last close: {snapshot['last_close']:.2f} ({snapshot['last_date']})")
    if quote:
        print(f"live quote: {quote.get('last')}  bid={quote.get('bid')} ask={quote.get('ask')} "
              f"chg={quote.get('change_pct'):.2f}% time={quote.get('time')}")
    print("-" * 64)
    for name, entry in snapshot["strategies"].items():
        if "error" in entry:
            print(f"{name}: ERROR {entry['error']}")
            continue
        action = entry["action"]
        line = f"{name:<18} {action:<5} signal={entry['signal']:+d}"
        if entry.get("level") is not None:
            line += f" level={entry['level']}"
        if entry.get("stop") is not None:
            line += f" stop={entry['stop']}"
        if entry.get("rsi") is not None:
            line += f" rsi={entry['rsi']}"
        print(line)
    print("=" * 64)
