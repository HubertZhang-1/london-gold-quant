# -*- coding: utf-8 -*-
"""Asian-session -> London-session continuation strategy (伦敦金).

Based on the session-trend analysis:
  - Asian (UTC 0-7) trend and London (UTC 7-13) trend are only WEAKLY correlated
    (same-sign 55.8%, r~0.25). But when they are the SAME sign, the London session
    has +0.83% avg; when they OPPOSE (46% of days), London avg is -0.16%.
  - Asian volatility is much lower than London (9.3 vs 14.8), and is often dead-low.

So the edge is: use the Asian session as a DIRECTION / VOLATILITY filter for a London
continuation trade:
    - only take a trade when the Asian session has a CLEAR direction and enough
      volatility (not dead water);
    - at the London open, enter in the Asian direction;
    - exit at London close (or on a stop/target sized from Asian volatility).
Only long/short the same direction; skip when direction is unclear or vol too low.

Contract: 100 oz/lot (XAUUSD CFD). Costs: spread/slippage/commission like the grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ASIA = (0, 7)      # UTC hours
LONDON = (7, 13)   # UTC hours


@dataclass
class SessionFollowConfig:
    initial_balance_usc: float = 100_000.0
    usc_per_price_lot: float = 100.0
    min_asia_ret_pct: float = 0.05   # min |Asian net move| (%) for a 'clear' direction
    min_vol: float = 5.0             # min Asian intraday std (usd) to avoid dead water
    stop_mult: float = 2.0           # stop = stop_mult * Asian vol
    rr: float = 2.0                  # TP = rr * stop
    spread: float = 0.35
    slippage: float = 0.10
    commission_per_lot: float = 0.10


def _fill(mid: float, side: int, spread: float, slippage: float = 0.0) -> float:
    return mid + side * (spread / 2.0 + slippage)


def _pnl(side: int, entry: float, exit: float, lots: float, usc: float, commission: float = 0.0) -> float:
    return side * (exit - entry) * lots * usc - commission * abs(lots)


def _ols_slope(close: np.ndarray) -> float:
    n = len(close)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    y = close.astype(float)
    denom = (x - x.mean()) @ (x - x.mean())
    return ((x - x.mean()) @ (y - y.mean())) / denom if denom > 0 else 0.0


def _session_sub(df, lo, hi):
    return df[(df["hour"] >= lo) & (df["hour"] < hi)]


def run_session_follow(df: pd.DataFrame, config: SessionFollowConfig | None = None) -> dict:
    config = config or SessionFollowConfig()
    data = df.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.date
    close = data["close"].to_numpy(float)

    balance = float(config.initial_balance_usc)
    equity_rows, trade_rows = [], []
    peak_equity = balance

    for day, g in data.groupby("day"):
        a = _session_sub(g, *ASIA)
        if len(a) < 5:
            continue
        a_open = float(a["open"].iloc[0])
        a_close = float(a["close"].iloc[-1])
        # Asian net move in % (clear direction + strength, right magnitude)
        a_ret_pct = (a_close - a_open) / a_open * 100.0
        a_vol = float(a["close"].std())
        a_end = a_close                    # Asian close price
        a_day = g["date"].iloc[0]

        # London open price = first London bar open
        ldn = _session_sub(g, *LONDON)
        if len(ldn) < 3:
            continue
        l_open = float(ldn["open"].iloc[0])
        l_close = float(ldn["close"].iloc[-1])
        l_hours = ldn["date"]

        # --- setup gate: clear Asian direction + enough volatility ---
        side = 0
        if abs(a_ret_pct) >= config.min_asia_ret_pct and a_vol >= config.min_vol:
            side = 1 if a_ret_pct > 0 else -1

        if side != 0:
            entry = _fill(l_open, side, config.spread, config.slippage)
            stop_dist = config.stop_mult * a_vol
            tp_dist = config.rr * stop_dist
            # risk-budgeted sizing: lose risk% of balance at the stop, capped lots
            risk_usc = balance * 0.01  # 1% risk per trade
            lots = risk_usc / max(stop_dist * config.usc_per_price_lot, 1e-9)
            lots = max(0.01, round(min(lots, 1.0), 2))  # cap lots to avoid over-sizing
            sp = entry - side * stop_dist
            tpp = entry + side * tp_dist
            # intrabar London path (direction-aware)
            ld_hi = float(ldn["high"].max())
            ld_lo = float(ldn["low"].min())
            hit_stop = (side > 0 and ld_lo <= sp) or (side < 0 and ld_hi >= sp)
            hit_tp = (side > 0 and ld_hi >= tpp) or (side < 0 and ld_lo <= tpp)
            if hit_stop:
                exit_px, reason = sp, "stop"
            elif hit_tp:
                exit_px, reason = tpp, "tp"
            else:
                exit_px, reason = l_close, "session_end"
            pnl = _pnl(side, entry, exit_px, lots, config.usc_per_price_lot, config.commission_per_lot)
            balance += pnl
            trade_rows.append({
                "date": a_day, "side": "long" if side > 0 else "short",
                "entry": round(entry, 2), "exit": round(exit_px, 2), "pnl": round(pnl, 2),
                "reason": reason, "asia_ret_pct": round(a_ret_pct, 3), "asia_vol": round(a_vol, 2),
            })

        # equity for the day's London close
        equity_rows.append({"date": a_day, "balance": balance, "equity": balance,
                            "drawdown_pct": 0.0})
        peak_equity = max(peak_equity, balance)

    trades = pd.DataFrame(trade_rows)
    if not len(trades):
        trades = pd.DataFrame(columns=["date", "side", "entry", "exit", "pnl", "reason",
                                       "asia_slope", "asia_vol"])
    wins = float(trades[trades["pnl"] > 0]["pnl"].sum()) if len(trades) else 0.0
    losses = abs(float(trades[trades["pnl"] < 0]["pnl"].sum())) if len(trades) else 0.0
    n = len(trades)
    eq = pd.DataFrame(equity_rows)
    stats = {
        "final_balance": balance,
        "final_equity": balance,
        "trades": n,
        "wins": int((trades["pnl"] > 0).sum()) if n else 0,
        "winrate": (trades["pnl"] > 0).mean() if n else 0.0,
        "net_pnl": float(trades["pnl"].sum()) if n else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "avg_win": float(wins / max(1, int((trades["pnl"] > 0).sum()))),
        "avg_loss": float(-losses / max(1, int((trades["pnl"] < 0).sum()))),
        "long_trades": int((trades["side"] == "long").sum()) if n else 0,
        "short_trades": int((trades["side"] == "short").sum()) if n else 0,
    }
    return {"stats": stats, "trades": trades, "equity": eq}
