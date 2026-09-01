# -*- coding: utf-8 -*-
"""Execution and account-risk engine for 15-minute gold strategies."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class M15ExecutionConfig:
    capital: float = 100_000.0
    risk_per_trade: float = 0.005
    daily_loss_limit: float = 0.015
    max_daily_trades: int = 3
    spread: float = 0.35
    slippage: float = 0.10
    commission_per_oz: float = 0.10
    min_position_oz: float = 0.01
    max_hold_bars: int = 12
    trailing_atr: float = 1.50


TRADE_COLUMNS = [
    "side",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "position_oz",
    "initial_stop_dist",
    "initial_risk",
    "pnl",
    "r_multiple",
    "bars_held",
    "exit_reason",
    "entry_equity",
]
SKIPPED_COLUMNS = ["time", "direction", "reason"]


def _fill_price(mid: float, direction: int, config: M15ExecutionConfig) -> float:
    adverse_cost = config.spread / 2.0 + config.slippage
    return mid + adverse_cost if direction > 0 else mid - adverse_cost


def _round_trip_cost(config: M15ExecutionConfig) -> float:
    return config.spread + 2.0 * config.slippage + 2.0 * config.commission_per_oz


def _position_size(equity: float, stop_dist: float, config: M15ExecutionConfig) -> tuple[float, float]:
    risk_per_oz = stop_dist + _round_trip_cost(config)
    if equity <= 0 or stop_dist <= 0 or risk_per_oz <= 0 or config.min_position_oz <= 0:
        return 0.0, 0.0
    raw_ounces = equity * config.risk_per_trade / risk_per_oz
    units = floor(raw_ounces / config.min_position_oz)
    ounces = units * config.min_position_oz
    return ounces, ounces * risk_per_oz


def _stats(equity: np.ndarray, trades: pd.DataFrame, capital: float) -> dict:
    series = pd.Series(equity).ffill().fillna(capital)
    total_return = float(series.iloc[-1] / capital - 1.0) if len(series) and capital > 0 else 0.0
    drawdown = series / series.cummax() - 1.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    losses = trades.loc[trades["pnl"] < 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    return {
        "total_return": round(total_return * 100.0, 2),
        "max_drawdown": round(float(drawdown.min()) * 100.0, 2) if len(drawdown) else 0.0,
        "trade_count": int(len(trades)),
        "win_rate": round(len(wins) / len(trades) * 100.0, 1) if len(trades) else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_trade": round(float(trades["pnl"].mean()), 2) if len(trades) else 0.0,
        "expectancy_r": round(float(trades["r_multiple"].mean()), 3) if len(trades) else 0.0,
        "final_equity": round(float(series.iloc[-1]), 2) if len(series) else round(capital, 2),
    }


def run_m15_backtest(
    df: pd.DataFrame,
    config: M15ExecutionConfig | None = None,
) -> dict:
    """Execute completed-bar signals at the next open with daily controls."""
    config = config or M15ExecutionConfig()
    required = {"date", "open", "high", "low", "close", "signal", "stop_dist", "session_open"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = df.copy().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], utc=True)
    count = len(data)
    equity_curve = np.full(count, np.nan)
    realized_equity = float(config.capital)
    position: dict | None = None
    trades: list[dict] = []
    skipped: list[dict] = []
    current_day = None
    day_start_equity = realized_equity
    daily_realized = 0.0
    daily_entries = 0

    def close_position(index: int, exit_mid: float, reason: str) -> None:
        nonlocal position, realized_equity, daily_realized
        if position is None:
            return
        exit_direction = -position["direction"]
        exit_fill = _fill_price(float(exit_mid), exit_direction, config)
        price_pnl = position["direction"] * (exit_fill - position["entry_fill"]) * position["ounces"]
        commissions = 2.0 * config.commission_per_oz * position["ounces"]
        pnl = price_pnl - commissions
        realized_equity += pnl
        daily_realized += pnl
        initial_risk = position["initial_risk"]
        trades.append(
            {
                "side": "long" if position["direction"] > 0 else "short",
                "entry_time": position["entry_time"],
                "exit_time": data.loc[index, "date"],
                "entry_price": round(position["entry_fill"], 5),
                "exit_price": round(exit_fill, 5),
                "position_oz": round(position["ounces"], 4),
                "initial_stop_dist": round(position["initial_stop_dist"], 5),
                "initial_risk": round(initial_risk, 2),
                "pnl": round(pnl, 2),
                "r_multiple": round(pnl / initial_risk, 4) if initial_risk > 0 else 0.0,
                "bars_held": index - position["entry_index"] + 1,
                "exit_reason": reason,
                "entry_equity": round(position["entry_equity"], 2),
            }
        )
        position = None

    for index in range(count):
        timestamp = data.loc[index, "date"]
        trading_day = timestamp.date()
        if current_day != trading_day:
            current_day = trading_day
            day_start_equity = realized_equity
            daily_realized = 0.0
            daily_entries = 0

        open_price = float(data.loc[index, "open"])
        high_price = float(data.loc[index, "high"])
        low_price = float(data.loc[index, "low"])
        close_price = float(data.loc[index, "close"])
        in_session = bool(data.loc[index, "session_open"])
        stopped_this_bar = False

        if position is not None and not in_session:
            close_position(index, open_price, "session_end")

        if position is None and index > 0:
            previous_signal = int(data.loc[index - 1, "signal"])
            stop_distance = float(data.loc[index - 1, "stop_dist"])
            if previous_signal != 0:
                if not in_session:
                    skipped.append({"time": timestamp, "direction": previous_signal, "reason": "outside_session"})
                elif daily_realized <= -day_start_equity * config.daily_loss_limit:
                    skipped.append({"time": timestamp, "direction": previous_signal, "reason": "daily_loss_limit"})
                elif daily_entries >= config.max_daily_trades:
                    skipped.append({"time": timestamp, "direction": previous_signal, "reason": "daily_trade_limit"})
                else:
                    ounces, initial_risk = _position_size(realized_equity, stop_distance, config)
                    if ounces < config.min_position_oz or initial_risk <= 0:
                        skipped.append({"time": timestamp, "direction": previous_signal, "reason": "invalid_position_size"})
                    else:
                        entry_fill = _fill_price(open_price, previous_signal, config)
                        position = {
                            "direction": previous_signal,
                            "entry_time": timestamp,
                            "entry_index": index,
                            "entry_mid": open_price,
                            "entry_fill": entry_fill,
                            "entry_equity": realized_equity,
                            "ounces": ounces,
                            "initial_stop_dist": stop_distance,
                            "initial_risk": initial_risk,
                            "stop_mid": open_price - previous_signal * stop_distance,
                            "breakeven": False,
                        }
                        daily_entries += 1

        if position is not None:
            stop_hit = (
                position["direction"] > 0 and low_price <= position["stop_mid"]
            ) or (
                position["direction"] < 0 and high_price >= position["stop_mid"]
            )
            if stop_hit:
                if position["direction"] > 0:
                    stop_fill_mid = min(open_price, position["stop_mid"])
                else:
                    stop_fill_mid = max(open_price, position["stop_mid"])
                close_position(index, stop_fill_mid, "stop")
                stopped_this_bar = True

        if position is not None and not stopped_this_bar:
            direction = position["direction"]
            one_r_price = position["entry_mid"] + direction * position["initial_stop_dist"]
            reached_one_r = high_price >= one_r_price if direction > 0 else low_price <= one_r_price
            if reached_one_r:
                position["breakeven"] = True
            if position["breakeven"]:
                cost_cover = position["entry_mid"] + direction * _round_trip_cost(config)
                current_atr = float(data.loc[index, "atr"]) if "atr" in data and not pd.isna(data.loc[index, "atr"]) else 0.0
                trail = close_price - direction * current_atr * config.trailing_atr if current_atr > 0 else cost_cover
                candidate = max(cost_cover, trail) if direction > 0 else min(cost_cover, trail)
                if direction > 0:
                    position["stop_mid"] = max(position["stop_mid"], candidate)
                else:
                    position["stop_mid"] = min(position["stop_mid"], candidate)

            bars_held = index - position["entry_index"] + 1
            if bars_held >= config.max_hold_bars:
                close_position(index, close_price, "time_exit")

        if position is None:
            equity_curve[index] = realized_equity
        else:
            liquidation_fill = _fill_price(close_price, -position["direction"], config)
            unrealized = position["direction"] * (liquidation_fill - position["entry_fill"]) * position["ounces"]
            unrealized -= 2.0 * config.commission_per_oz * position["ounces"]
            equity_curve[index] = realized_equity + unrealized

    if position is not None and count:
        close_position(count - 1, float(data.loc[count - 1, "close"]), "end_of_data")
        equity_curve[-1] = realized_equity

    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    skipped_frame = pd.DataFrame(skipped, columns=SKIPPED_COLUMNS)
    return {
        "trades": trade_frame,
        "skipped": skipped_frame,
        "equity": equity_curve,
        "dates": data["date"].tolist(),
        "stats": _stats(equity_curve, trade_frame, config.capital),
    }
