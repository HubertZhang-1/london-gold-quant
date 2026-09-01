# -*- coding: utf-8 -*-
"""Margin-based 10x leverage backtest for the bull-only double-gate strategy.

Correct margin accounting for leverage:
  - account equity = cash + open-position unrealized PnL
  - nominal exposure = capital * leverage, but each position sized to risk only
    a fraction (risk_per_trade_pct) of capital given its stop distance. That
    avoids blowing up: worst case per trade = risk_pct of capital.
  - margin_call circuit breaker: if equity falls below (1 - margin_call_pct)
    of peak equity, force-flat and stop.

This does NOT reuse the full-cash backtest_v3 (whose cash -= price*lots is
wrong for leverage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from london_gold.backtest import CostConfig, _fill_price


def run_leverage_backtest(df: pd.DataFrame, cost: CostConfig, name: str = "") -> dict:
    """Backtest a signal/stop/tp frame with margin-account leverage."""
    data = df.reset_index(drop=True)
    if "signal" not in data:
        raise ValueError("signal column required")
    for c in ("stop_dist", "tp_dist"):
        if c not in data:
            data[c] = 0.0

    dates = data["date"].tolist()
    opens = data["open"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    signals = data["signal"].to_numpy(int)
    stop_dists = data["stop_dist"].to_numpy(float)
    tp_dists = data["tp_dist"].to_numpy(float)
    n = len(data)

    cash = float(cost.capital)          # margin account balance (unrealized excluded)
    pos_oz = 0.0                        # signed ounces
    entry_mid = 0.0
    stop_dist = 0.0
    tp_dist = 0.0
    entry_idx = None
    peak_equity = float(cost.capital)
    equity = np.full(n, np.nan)
    trades = []
    stopped = False

    def unrealized(mid: float) -> float:
        if pos_oz == 0:
            return 0.0
        exit_fill = _fill_price(mid, -np.sign(pos_oz), cost)
        return np.sign(pos_oz) * (exit_fill - entry_mid) * abs(pos_oz)

    def close_position(i, exit_mid, reason):
        nonlocal cash, pos_oz
        if pos_oz == 0:
            return
        closing_long = pos_oz > 0
        direction = -1 if closing_long else 1
        exit_fill = _fill_price(exit_mid, direction, cost)
        commission = cost.commission_per_oz * abs(pos_oz)
        pnl = np.sign(pos_oz) * (exit_fill - entry_mid) * abs(pos_oz) - commission
        cash += pnl
        trades.append({
            "side": "long" if pos_oz > 0 else "short",
            "entry_time": dates[entry_idx], "exit_time": dates[i],
            "entry_price": round(entry_mid, 2), "exit_price": round(exit_mid, 2),
            "pnl": round(pnl, 2), "bars_held": i - entry_idx, "exit_reason": reason,
        })
        pos_oz = 0.0

    for i in range(n):
        if pos_oz != 0 and stop_dist > 0:
            sp = entry_mid - stop_dist if pos_oz > 0 else entry_mid + stop_dist
            if pos_oz > 0 and lows[i] <= sp:
                close_position(i, sp, "stop")
            elif pos_oz < 0 and highs[i] >= sp:
                close_position(i, sp, "stop")
        if pos_oz != 0 and tp_dist > 0:
            tpp = entry_mid + tp_dist if pos_oz > 0 else entry_mid - tp_dist
            if pos_oz > 0 and highs[i] >= tpp:
                close_position(i, tpp, "take_profit")
            elif pos_oz < 0 and lows[i] <= tpp:
                close_position(i, tpp, "take_profit")

        if i > 0 and not stopped:
            ps = signals[i - 1]
            if pos_oz != 0 and (ps == 0 or (pos_oz > 0) != (ps > 0)):
                close_position(i, float(opens[i]), "signal")
            if pos_oz == 0 and ps != 0:
                entry_mid = float(opens[i])
                stop_dist = float(stop_dists[i - 1]) if not np.isnan(stop_dists[i - 1]) else 0.0
                tp_dist = float(tp_dists[i - 1]) if not np.isnan(tp_dists[i - 1]) else 0.0
                oz = cost.position_oz
                if cost.leverage > 0 and entry_mid > 0:
                    oz = cost.capital * cost.leverage / entry_mid
                if cost.risk_per_trade_pct > 0 and stop_dist > 0:
                    risk_oz = cost.capital * cost.risk_per_trade_pct / stop_dist
                    oz = min(oz, risk_oz)
                if cost.max_oz > 0:
                    oz = min(oz, cost.max_oz)
                oz = max(0.01, round(oz, 2))
                pos_oz = float(ps) * oz
                entry_idx = i

        equity[i] = cash + unrealized(float(closes[i]))
        peak_equity = max(peak_equity, equity[i])
        if cost.margin_call_pct > 0 and peak_equity > 0:
            dd = (peak_equity - equity[i]) / peak_equity
            if dd >= cost.margin_call_pct:
                if pos_oz != 0:
                    close_position(i, float(closes[i]), "margin_call")
                    equity[i] = cash
                stopped = True
                break

    if pos_oz != 0:
        close_position(n - 1, float(closes[-1]), "end")
        equity[-1] = cash

    equity = np.nan_to_num(equity)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    peak = np.maximum.accumulate(equity)
    dd = ((peak - equity) / peak).max() if len(equity) else 0.0
    total_return = (equity[-1] / cost.capital - 1.0) * 100 if len(equity) else 0.0
    stats = {
        "label": name, "final_equity": float(equity[-1]) if len(equity) else cost.capital,
        "total_return": total_return, "max_drawdown": dd * 100,
        "trade_count": len(trades), "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "profit_factor": gp / gl if gl > 0 else 0.0,
        "final_balance": float(cash),
    }
    return {"stats": stats, "equity": equity, "dates": dates, "trades": pd.DataFrame(trades)}
