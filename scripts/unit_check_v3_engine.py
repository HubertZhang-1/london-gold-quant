# -*- coding: utf-8 -*-
"""Minimal unit sanity-check for the v3 backtest cash accounting."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig, _fill_price

# Build a tiny frame with a known long then short trade, verify PnL matches hand calc.
dates = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
frame = pd.DataFrame({
    "date": dates,
    "open": [100.0, 101.0, 102.0, 103.0, 102.0, 101.0],
    "high": [100.5, 101.5, 102.5, 103.5, 102.5, 101.5],
    "low": [99.5, 100.5, 101.5, 102.5, 101.5, 100.5],
    "close": [100.0, 101.0, 102.0, 103.0, 102.0, 101.0],
    "signal": [0, 1, 0, -1, 0, 0],   # go long @101 (bar1 signal -> fill @bar2 open 102)
    "stop_dist": [0, 1.0, 0, 1.0, 0, 0],
    "tp_dist": [0, 2.0, 0, 2.0, 0, 0],
})

cost = CostConfig(capital=100000, position_oz=10, spread=0.0, slippage=0.0,
                  commission_per_oz=0.0, risk_per_trade_pct=0.0)  # exact, no cost


def run(df):
    data = df.reset_index(drop=True)
    opens = data["open"].to_numpy(float); highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float); closes = data["close"].to_numpy(float)
    signals = data["signal"].to_numpy(int)
    stop = data["stop_dist"].to_numpy(float); tp = data["tp_dist"].to_numpy(float)
    cash = cost.capital; pos_oz = 0.0; entry_mid = 0.0; stop_dist = 0.0; tp_dist = 0.0
    entry_idx = None; cash_at_entry = 0.0; trades = []
    def close(i, mid, reason):
        nonlocal cash, pos_oz
        if pos_oz == 0: return
        closing_long = pos_oz > 0
        direction = -1 if closing_long else 1
        exit_fill = _fill_price(mid, direction, cost)
        if closing_long: cash += exit_fill * abs(pos_oz)
        else: cash -= exit_fill * abs(pos_oz)
        trades.append({"side": "long" if pos_oz > 0 else "short", "entry": entry_mid,
                       "exit": mid, "pnl": round(cash - cash_at_entry, 4), "reason": reason})
        pos_oz = 0.0
    for i in range(len(data)):
        if pos_oz != 0 and stop_dist > 0:
            sp = entry_mid - stop_dist if pos_oz > 0 else entry_mid + stop_dist
            if pos_oz > 0 and lows[i] <= sp: close(i, sp, "stop")
            elif pos_oz < 0 and highs[i] >= sp: close(i, sp, "stop")
        if pos_oz != 0 and tp_dist > 0:
            tpp = entry_mid + tp_dist if pos_oz > 0 else entry_mid - tp_dist
            if pos_oz > 0 and highs[i] >= tpp: close(i, tpp, "tp")
            elif pos_oz < 0 and lows[i] <= tpp: close(i, tpp, "tp")
        if i > 0:
            ps = signals[i-1]
            if pos_oz != 0 and (ps == 0 or (pos_oz > 0) != (ps > 0)):
                close(i, float(opens[i]), "signal")
            if pos_oz == 0 and ps != 0:
                entry_mid = float(opens[i])
                stop_dist = float(stop[i-1]); tp_dist = float(tp[i-1])
                oz = 10.0
                pos_oz = float(ps) * oz
                entry_idx = i; cash_at_entry = cash
                if ps > 0: cash -= opens[i] * oz
                else: cash += opens[i] * oz
    if pos_oz != 0: close(len(data)-1, float(closes[-1]), "end")
    return trades

trades = run(frame)
print("trades (no cost):")
for t in trades:
    print("  ", t)

# Expectation: buy 10oz @102 (bar2 open), close @103 (bar3) -> pnl +10.  [long fills @bar2 open=102]
# Then signal -1 at bar3 -> short fills @bar4 open=102, stops/tp ... 
print()
print("Manual: long 10oz @102 -> close 103 = +10; that is trade[0].pnl should be +10")
