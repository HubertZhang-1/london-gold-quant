# -*- coding: utf-8 -*-
"""Event-driven backtester for daily London gold strategies.

Fills happen on the open of the bar after a close signal. Costs are applied as
half spread plus slippage on every fill, plus a per-ounce commission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class CostConfig:
    capital: float = 100_000.0
    position_oz: float = 10.0
    spread: float = 0.35
    slippage: float = 0.10
    commission_per_oz: float = 0.10
    leverage: float = 0.0
    max_oz: float = 0.0
    risk_per_trade_pct: float = 0.0
    margin_call_pct: float = 0.0


def _fill_price(price: float, direction: int, cost: CostConfig) -> float:
    """Fill price for a buy (direction=1) or sell (direction=-1)."""
    half = cost.spread / 2.0
    if direction > 0:
        return price + half + cost.slippage
    return price - half - cost.slippage


def run_backtest(
    df: pd.DataFrame,
    cost: CostConfig | None = None,
    name: str = "",
    params: dict | None = None,
    reentry_after_stop: bool = True,
) -> dict:
    """Run one backtest over a frame containing ``signal`` and ``stop_dist``."""
    cost = cost or CostConfig()
    params = params or {}
    data = df.reset_index(drop=True)
    if "signal" not in data:
        raise ValueError("df must contain a 'signal' column")
    if "stop_dist" not in data:
        data["stop_dist"] = 0.0

    dates = data["date"].tolist()
    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    signals = data["signal"].to_numpy(dtype=int)
    stop_dists = data["stop_dist"].to_numpy(dtype=float)
    n = len(data)

    cash = cost.capital
    pos_oz = 0.0
    entry_mid = 0.0
    stop_dist = 0.0
    entry_idx = None
    cash_at_entry = 0.0
    equity = np.full(n, np.nan)
    trade_records = []
    stopped = False
    blocked_signal = 0

    def close_position(i, exit_mid, reason):
        nonlocal cash, pos_oz
        closing_long = pos_oz > 0
        direction = -1 if closing_long else 1
        exit_fill = _fill_price(exit_mid, direction, cost)
        commission = cost.commission_per_oz * abs(pos_oz)
        if closing_long:
            cash += exit_fill * abs(pos_oz) - commission
        else:
            cash -= exit_fill * abs(pos_oz) + commission
        pnl = cash - cash_at_entry
        trade_records.append(
            {
                "side": "long" if pos_oz > 0 else "short",
                "entry_time": dates[entry_idx],
                "exit_time": dates[i],
                "entry_price": round(entry_mid, 2),
                "exit_price": round(exit_mid, 2),
                "pnl": round(pnl, 2),
                "bars_held": i - entry_idx,
                "exit_reason": reason,
            }
        )
        pos_oz = 0.0

    for i in range(n):
        # Stop loss is evaluated on the intrabar range before signals.
        if pos_oz != 0 and stop_dist > 0:
            stop_price = entry_mid - stop_dist if pos_oz > 0 else entry_mid + stop_dist
            if pos_oz > 0 and lows[i] <= stop_price:
                close_position(i, stop_price, "stop")
                blocked_signal = signals[i - 1] if i > 0 else 0
            elif pos_oz < 0 and highs[i] >= stop_price:
                close_position(i, stop_price, "stop")
                blocked_signal = signals[i - 1] if i > 0 else 0

        # Signal exits and entries fill on the next bar open.
        if i > 0:
            prev_signal = signals[i - 1]
            if prev_signal == 0:
                blocked_signal = 0
            if pos_oz != 0 and (prev_signal == 0 or (pos_oz > 0) != (prev_signal > 0)):
                close_position(i, float(opens[i]), "signal")
            if (
                pos_oz == 0
                and prev_signal != 0
                and not stopped
                and (reentry_after_stop or prev_signal != blocked_signal)
            ):
                entry_mid = float(opens[i])
                entry_fill = _fill_price(entry_mid, prev_signal, cost)
                stop_dist = float(stop_dists[i - 1]) if not np.isnan(stop_dists[i - 1]) else 0.0
                oz = cost.position_oz
                if cost.leverage > 0:
                    oz = cost.capital * cost.leverage / entry_mid
                if cost.risk_per_trade_pct > 0 and stop_dist > 0:
                    risk_oz = cost.capital * cost.risk_per_trade_pct / stop_dist
                    oz = min(oz, risk_oz)
                if cost.max_oz > 0:
                    oz = min(oz, cost.max_oz)
                oz = max(0.01, round(oz, 2))
                pos_oz = float(prev_signal) * oz
                entry_idx = i
                cash_at_entry = cash
                commission = cost.commission_per_oz * abs(pos_oz)
                if prev_signal > 0:
                    cash -= entry_fill * abs(pos_oz) + commission
                else:
                    cash += entry_fill * abs(pos_oz) - commission

        if pos_oz != 0:
            equity[i] = cash + pos_oz * closes[i]
        else:
            equity[i] = cash

        if (
            cost.margin_call_pct > 0
            and pos_oz != 0
            and equity[i] <= cost.capital * (1.0 - cost.margin_call_pct)
        ):
            close_position(i, float(closes[i]), "margin_call")
            equity[i] = cash
            stopped = True

    if pos_oz != 0:
        close_position(n - 1, float(closes[-1]), "eod")
        equity[-1] = cash

    trades = pd.DataFrame(trade_records)
    stats = compute_stats(equity, trades, cost.capital)
    stats.update(
        {
            "name": name,
            "params": params,
            "costs": asdict(cost),
        }
    )
    return {
        "stats": stats,
        "equity": equity,
        "dates": dates,
        "trades": trades,
    }


def compute_stats(equity: np.ndarray, trades: pd.DataFrame, capital: float) -> dict:
    equity = pd.Series(equity).ffill()
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan)
    periods = max(len(equity) - 1, 1)
    total_return = float(equity.iloc[-1] / capital - 1.0) if len(equity) else 0.0
    annual_return = (1.0 + total_return) ** (TRADING_DAYS / periods) - 1.0 if total_return > -1 else -1.0

    vol = returns.std(ddof=0)
    sharpe = float(returns.mean() / vol * np.sqrt(TRADING_DAYS)) if vol and vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    wins = trades.loc[trades["pnl"] > 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    losses = trades.loc[trades["pnl"] < 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    win_rate = float(len(wins) / len(trades)) * 100 if len(trades) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)

    return {
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "trade_count": int(len(trades)),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_trade": round(float(trades["pnl"].mean()), 2) if len(trades) else 0.0,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "max_win": round(float(wins.max()), 2) if len(wins) else 0.0,
        "max_loss": round(float(losses.min()), 2) if len(losses) else 0.0,
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


def backtest_strategy(
    df: pd.DataFrame,
    signal_func: Callable,
    params: dict,
    cost: CostConfig,
    name: str = "",
) -> dict:
    """Convenience wrapper: generate signals, then backtest them."""
    signaled = signal_func(df, **params)
    return run_backtest(signaled, cost=cost, name=name, params=params)
