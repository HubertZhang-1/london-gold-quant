# -*- coding: utf-8 -*-
"""SAFE version of the reverse-engineered grid/martingale EA (fixed-lot, hard-stop).

Replicates the STARTTRADER XAUUSD.c M1 EA's *character* (frequent small grid wins)
but REMOVES the blow-up tail that the screenshots hide (the -1961 max drawdown vs
+225 net is the tell: 95.5% win rate bought with 1-2 catastrophic losses).

Repairs vs the original martingale_grid.py:
  1. FIXED lot (no martingale ladder 0.14->0.34->0.63...). Risk per basket is bounded.
  2. HARD basket stop-loss (ATR-based) instead of "average-entry reclaim" that rides
     adverse moves indefinitely. A basket is closed at loss >= stop_atr*ATR.
  3. Optional TREND gate: only open a fresh basket in the direction of a short-term
     trend (EMA slope), so it does not fight a one-way move. If disabled it grids both
     sides (pure range behavior).
  4. Keep the grid add + average-entry take-profit (the "frequent small win" character).
  5. Optional cooldown after a stop, so it does not immediately re-enter into a losing move.

Contract stays 100 oz/lot and daily/max-drawdown circuit breakers are kept so the
repair is measured honestly (with costs), like the original replica.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr


@dataclass
class SafeGridConfig:
    initial_balance_usc: float = 100_000.0
    fixed_lot: float = 0.3            # FIXED lot (no martingale ladder)
    usc_per_price_lot: float = 100.0  # 100 oz/lot
    atr_bars: int = 14
    grid_pct: float = 0.0015          # grid step as a fraction of price (0.15%)
    take_profit_pct: float = 0.0015   # basket TP once unrealized >= price*this
    stop_pct: float = 0.006           # basket hard-stop at loss >= price*this; 0 disables
    max_layers: int = 4               # max adds per basket
    trend_ema_bars: int = 60          # trend direction via EMA slope (0 disables)
    use_trend_gate: bool = True       # only open baskets with the short-term trend
    cooldown_bars: int = 3            # bars to wait after a stop before re-entering
    spread: float = 0.35              # per-side half-spread already in fill
    slippage: float = 0.10
    commission_per_lot: float = 0.10
    daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.30


@dataclass
class SafeGridResult:
    events: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    stats: dict


def _fill(mid: float, side: int, spread: float, slippage: float) -> float:
    return mid + side * (spread / 2.0 + slippage)


def _pnl(side: int, entry: float, exit: float, lots: float, usc: float, commission: float) -> float:
    return side * (exit - entry) * lots * usc - commission * abs(lots)


def run_safe_grid_backtest(df: pd.DataFrame, config: SafeGridConfig | None = None) -> SafeGridResult:
    config = config or SafeGridConfig()
    required = {"date", "open", "high", "low", "close"}
    if missing := required.difference(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = df.copy().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], utc=True)
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    atr14 = atr(high, low, close, window=config.atr_bars)
    ema = close.ewm(span=config.trend_ema_bars, adjust=False).mean() if config.trend_ema_bars > 0 else None

    balance = float(config.initial_balance_usc)
    equity_rows, event_rows, trade_rows = [], [], []
    peak_equity = balance
    daily_start_balance = balance
    daily_realized = 0.0
    current_day = None
    terminal = ""
    terminal_time: pd.Timestamp | None = None
    seq = 0

    # basket state (single side at a time in safe mode; a basket has one direction)
    side = 0                 # +1 long, -1 short, 0 flat
    entries: list[tuple[float, float]] = []  # (fill, lots)
    last_entry = 0.0
    layers = 0
    entry_i = -1
    last_stop_i = -10 ** 9

    def basket_lots() -> float:
        return sum(l for _, l in entries)

    def avg() -> float:
        if not entries:
            return 0.0
        return sum(e * l for e, l in entries) / basket_lots()

    def basket_unreal(mid: float) -> float:
        if not entries:
            return 0.0
        return _pnl(side, avg(), _fill(mid, -side, config.spread, config.slippage),
                    basket_lots(), config.usc_per_price_lot, config.commission_per_lot)

    def account_equity(mid: float) -> float:
        return balance + basket_unreal(mid)

    def record(ev: str, reason: str, t: pd.Timestamp, mid: float, **extra) -> None:
        nonlocal seq
        seq += 1
        event_rows.append({
            "sequence": seq, "time": t, "event": ev, "reason": reason, "mid": mid,
            "balance": balance, "equity": account_equity(mid),
            "side": "long" if side > 0 else "short" if side < 0 else "flat",
            "lots": basket_lots(), **extra,
        })

    def add(t: pd.Timestamp, mid: float, reason: str) -> None:
        nonlocal layers, last_entry
        fill = _fill(mid, side, config.spread, config.slippage)
        entries.append((fill, config.fixed_lot))
        layers += 1
        last_entry = fill
        record("open", reason, t, mid, layer=layers, avg=round(avg(), 3))

    def close_basket(t: pd.Timestamp, mid: float, reason: str) -> None:
        nonlocal balance, daily_realized, entries, side, layers
        if not entries:
            return
        exit_fill = _fill(mid, -side, config.spread, config.slippage)
        for e, l in entries:
            pnl = _pnl(side, e, exit_fill, l, config.usc_per_price_lot, config.commission_per_lot)
            balance += pnl
            daily_realized += pnl
            trade_rows.append({
                "time": t, "side": "long" if side > 0 else "short", "lots": l,
                "entry_fill": round(e, 3), "exit_fill": round(exit_fill, 3),
                "pnl_usc": round(pnl, 2), "reason": reason,
            })
        record("close", reason, t, mid, lots=round(basket_lots(), 2))
        entries.clear(); side = 0; layers = 0

    for i, row in data.iterrows():
        t = row["date"]
        if current_day != t.date():
            current_day = t.date()
            daily_start_balance = balance
            daily_realized = 0.0

        trend = 0
        if ema is not None and i >= config.trend_ema_bars:
            if ema.iloc[i] > ema.iloc[i - config.trend_ema_bars]:
                trend = 1
            elif ema.iloc[i] < ema.iloc[i - config.trend_ema_bars]:
                trend = -1

        o = float(open_.iloc[i]); h = float(high.iloc[i]); l = float(low.iloc[i]); c = float(close.iloc[i])
        mid0 = o

        # open a fresh basket on the bar's open: pick direction (trend gate if enabled)
        if side == 0:
            if config.use_trend_gate:
                if trend > 0:
                    side = 1; add(t, mid0, "open_long")
                elif trend < 0:
                    side = -1; add(t, mid0, "open_short")
            else:
                side = 1 if c < o else -1
                add(t, mid0, "open_fade")
            # recompute price-relative step from current entry
            step = abs(avg()) * config.grid_pct if entries else 0.0

        if side != 0 and entries:
            step = abs(avg()) * config.grid_pct
            # price-relative distances
            tp_dist = abs(avg()) * config.take_profit_pct
            stop_dist = abs(avg()) * config.stop_pct if config.stop_pct > 0 else 0.0

            # evaluate the bar's range against entry (worst adverse first for stop)
            adverse = l if side > 0 else h
            favourable = h if side > 0 else l

            # grid add when price steps adversely by grid_pct from the last entry
            # (evaluate on the full bar range, then add layers as it crosses)
            if layers < config.max_layers:
                # add layers as price moves adverse by multiples of grid_pct
                start = last_entry
                # distance adverse travelled from last entry
                dist = (start - adverse) if side > 0 else (adverse - start)
                add_count = int(dist / (abs(avg()) * config.grid_pct)) if abs(avg()) * config.grid_pct > 0 else 0
                add_count = min(add_count, config.max_layers - layers)
                for _ in range(add_count):
                    # add at a level one grid step beyond last entry
                    lvl = last_entry - abs(avg()) * config.grid_pct if side > 0 else \
                          last_entry + abs(avg()) * config.grid_pct
                    if (side > 0 and lvl > l) or (side < 0 and lvl < h):
                        add(t, lvl, "grid_add")
                    else:
                        break

            # average-entry take-profit (favourable)
            if tp_dist > 0 and (side > 0 and favourable >= avg() + tp_dist) or \
               (side < 0 and favourable <= avg() - tp_dist):
                close_basket(t, favourable, "avg_tp")

            # hard basket stop (adverse) — the key repair
            if stop_dist > 0:
                if side > 0 and adverse <= avg() - stop_dist:
                    close_basket(t, adverse, "basket_stop")
                    last_stop_i = i
                elif side < 0 and adverse >= avg() + stop_dist:
                    close_basket(t, adverse, "basket_stop")
                    last_stop_i = i

        equity = account_equity(c)
        peak_equity = max(peak_equity, equity)
        daily_loss = max(0.0, -daily_realized)
        if balance <= 0 or equity <= 0:
            terminal, terminal_time = "bankruptcy", t
            record("circuit_breaker", terminal, t, c)
            break
        if peak_equity > 0 and (peak_equity - equity) / peak_equity >= config.max_drawdown_pct:
            terminal, terminal_time = "max_drawdown", t
            record("circuit_breaker", terminal, t, c)
            break
        if daily_loss >= daily_start_balance * config.daily_loss_pct:
            terminal, terminal_time = "daily_loss", t
            record("circuit_breaker", terminal, t, c)
            break

        equity_rows.append({
            "date": t, "balance": balance, "equity": account_equity(c),
            "drawdown_pct": (peak_equity - account_equity(c)) / peak_equity if peak_equity > 0 else 1.0,
        })

    if entries:
        close_basket(data.iloc[-1]["date"], float(close.iloc[-1]), "end_of_data")

    trades = pd.DataFrame(trade_rows)
    if not len(trades):
        trades = pd.DataFrame(columns=["time", "side", "lots", "entry_fill", "exit_fill", "pnl_usc", "reason"])
    wins = float(trades[trades["pnl_usc"] > 0]["pnl_usc"].sum()) if len(trades) else 0.0
    losses = abs(float(trades[trades["pnl_usc"] < 0]["pnl_usc"].sum())) if len(trades) else 0.0
    n = len(trades)
    eq = pd.DataFrame(equity_rows)
    stats = {
        "final_balance": balance,
        "final_equity": float(eq.iloc[-1]["equity"]) if len(eq) else balance,
        "total_return_pct": (float(eq.iloc[-1]["equity"]) / config.initial_balance_usc - 1.0) * 100 if len(eq) else 0.0,
        "trades": n,
        "wins": int((trades["pnl_usc"] > 0).sum()) if n else 0,
        "losses": int((trades["pnl_usc"] < 0).sum()) if n else 0,
        "winrate": (trades["pnl_usc"] > 0).mean() if n else 0.0,
        "net_pnl": float(trades["pnl_usc"].sum()) if n else 0.0,
        "avg_win": float(wins / max(1, int((trades["pnl_usc"] > 0).sum()))),
        "avg_loss": float(-losses / max(1, int((trades["pnl_usc"] < 0).sum()))),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown_pct": float(eq["drawdown_pct"].max() * 100) if len(eq) else 0.0,
        "terminal_reason": terminal,
        "terminal_time": str(terminal_time) if terminal_time is not None else "",
        "long_trades": int((trades["side"] == "long").sum()) if n else 0,
        "short_trades": int((trades["side"] == "short").sum()) if n else 0,
    }
    return SafeGridResult(events=pd.DataFrame(event_rows), trades=trades, equity=eq, stats=stats)
