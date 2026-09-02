# -*- coding: utf-8 -*-
"""Bull-only single-direction version of the screenshot EA (牛市单向版).

This converts the hedged martingale-grid EA from the MT5 XAUUSD.c screenshots into
a STRICT BULL-ONLY long strategy, aligned with the user's principle "只做牛市，
熊市不参与" (only trade bull markets, skip bear markets).

Key differences vs the original martingale_grid.py:
  1. LONG only. No shorts, no reverse hedge, no two-sided churn.
  2. Bull-regime gate: only take longs when the market state is a confirmed bull
     (bull score >= bull_thr). Bear / chop / extreme vol -> flat, indefinitely.
  3. Fixed lot (no martingale ladder). No averaging/basket-TP that rides adverse
     moves; risk is bounded per trade.
  4. Hard stop-loss per trade (ATR-based) instead of "average-entry reclaim".
  5. Take-profit at an ATR multiple (RR), so each trade has a defined payoff.

Entry rule: on a bullish bar (bull >= bull_thr) with a fresh long signal, buy at
the next bar's open. Exit: stop (entry - stop_mult*ATR) or TP (entry + rr*stop).

This keeps the EA's "frequent small wins" character but removes the martingale
blow-up tail and the bear-market participation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr, trend_regime
from .macro_factors import forward_fill_macro


@dataclass
class BullGridConfig:
    initial_balance_usc: float = 100_000.0
    base_lot: float = 0.3
    usc_per_price_lot: float = 100.0
    atr_bars: int = 14
    stop_mult: float = 2.0          # stop = entry - stop_mult * ATR
    rr: float = 2.0                 # TP = entry + rr * stop
    bull_ema_bars: int = 60          # trend for the bull score (EMA slope)
    er_window: int = 20              # efficiency ratio window
    bull_thr: float = 0.55           # bull score above this = allow longs
    er_threshold: float = 0.12
    adx_window: int = 14
    adx_threshold: float = 20.0
    max_bars_in_trade: int = 30      # time-stop: exit after N bars if neither hit
    spread: float = 0.35
    slippage: float = 0.10
    commission_per_lot: float = 0.10
    # optional macro risk-dampening: bearish macro lowers the lot multiplier
    macro_lev_lo: float = 1.0        # multiplier at macro=-1
    macro_lev_hi: float = 1.0        # multiplier at macro=+1 (1.0 disables)
    risk_per_trade_pct: float = 0.0  # if >0, size lots so a full stop loses this % of balance
                                     # (0.0 -> fixed base_lot)


def _fill(mid: float, side: int, spread: float, slippage: float = 0.0) -> float:
    return mid + side * (spread / 2.0 + slippage)


def _pnl(side: int, entry: float, exit: float, lots: float, usc: float, commission: float = 0.0) -> float:
    return side * (exit - entry) * lots * usc - commission * abs(lots)


def _bull_score(close: pd.Series, high: pd.Series, low: pd.Series, cfg: BullGridConfig) -> np.ndarray:
    """Reuse the adaptive strategy's bull composite (0..1) for the regime gate."""
    ema_s = close.ewm(span=cfg.bull_ema_bars, adjust=False).mean()
    ema_slope = ema_s.diff(cfg.bull_ema_bars // 2 or 1)
    up_trend = (close > ema_s).astype(float)
    trend = trend_regime(close, high, low, er_window=cfg.er_window,
                         er_threshold=cfg.er_threshold, adx_window=cfg.adx_window,
                         adx_threshold=cfg.adx_threshold).to_numpy()
    bull = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)
    return bull


def run_bull_grid_backtest(df: pd.DataFrame, config: BullGridConfig | None = None,
                           macro_series=None) -> dict:
    """Backtest the bull-only single-direction gold grid.

    Trades only LONG. Entry when bull >= bull_thr and a fresh long fires; exit on
    stop / TP / time-stop. Fixed lot (optionally scaled by a bearish macro score).
    Returns a dict with stats/equity/trades.
    """
    config = config or BullGridConfig()
    required = {"date", "open", "high", "low", "close"}
    if missing := required.difference(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = df.copy().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], utc=True)
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    atr14 = atr(high, low, close, window=config.atr_bars).to_numpy()

    bull = _bull_score(close, high, low, config)

    # macro: if supplied, map macro[-1,1] -> [lo, hi] lot multiplier
    macro = np.ones(len(data))
    if macro_series is not None:
        if isinstance(macro_series, pd.Series):
            macro_arr = pd.Series(macro_series).to_numpy()
        else:
            macro_arr = np.asarray(macro_series, dtype=float)
        if config.macro_lev_lo != config.macro_lev_hi:
            m = np.clip(np.nan_to_num(macro_arr, nan=0.0), -1.0, 1.0)
            macro = config.macro_lev_lo + (config.macro_lev_hi - config.macro_lev_lo) * (m + 1.0) / 2.0
            macro = np.where(np.isnan(macro_arr), 1.0, macro)

    balance = float(config.initial_balance_usc)
    equity_rows, trade_rows = [], []
    peak_equity = balance
    pos = 0.0
    entry = 0.0
    stop = 0.0
    tp = 0.0
    entry_i = -1
    pending_long = False  # signal from prior bar; fill this bar's open
    start_bar = 0

    def account_equity(mid: float) -> float:
        if pos == 0:
            return balance
        return balance + _pnl(1, entry, mid, pos, config.usc_per_price_lot, config.commission_per_lot)

    for i in range(len(data)):
        t = data["date"].iloc[i]
        o = float(open_.iloc[i]); h = float(high.iloc[i]); l = float(low.iloc[i]); c = float(close.iloc[i])
        a = atr14[i] if not np.isnan(atr14[i]) else 1.0

        # ---- manage open position intrabar ----
        if pos > 0:
            sp = entry - config.stop_mult * a
            tpp = entry + config.rr * config.stop_mult * a
            if l <= sp:
                balance += _pnl(1, entry, sp, pos, config.usc_per_price_lot, config.commission_per_lot)
                trade_rows.append({"time": t, "side": "long", "lots": pos, "entry": entry,
                                   "exit": sp, "pnl": _pnl(1, entry, sp, pos, config.usc_per_price_lot, config.commission_per_lot),
                                   "reason": "stop", "bars": i - entry_i})
                pos = 0.0
            elif h >= tpp:
                balance += _pnl(1, entry, tpp, pos, config.usc_per_price_lot, config.commission_per_lot)
                trade_rows.append({"time": t, "side": "long", "lots": pos, "entry": entry,
                                   "exit": tpp, "pnl": _pnl(1, entry, tpp, pos, config.usc_per_price_lot, config.commission_per_lot),
                                   "reason": "tp", "bars": i - entry_i})
                pos = 0.0
            elif (i - entry_i) >= config.max_bars_in_trade:
                balance += _pnl(1, entry, c, pos, config.usc_per_price_lot, config.commission_per_lot)
                trade_rows.append({"time": t, "side": "long", "lots": pos, "entry": entry,
                                   "exit": c, "pnl": _pnl(1, entry, c, pos, config.usc_per_price_lot, config.commission_per_lot),
                                   "reason": "time", "bars": i - entry_i})
                pos = 0.0

        # ---- entry on this bar's open if a bull-long signal fired ----
        if pos == 0 and pending_long:
            o_fill = _fill(o, 1, config.spread, config.slippage)
            stop_dist = config.stop_mult * a
            # risk-based sizing: lose risk% of balance at the stop distance
            if config.risk_per_trade_pct > 0 and stop_dist > 0:
                risk_usc = balance * config.risk_per_trade_pct
                lots = risk_usc / (stop_dist * config.usc_per_price_lot)
            else:
                lots = config.base_lot
            lots = lots * float(macro[i])
            lots = max(0.01, round(lots, 2))
            entry = o_fill
            pos = lots
            entry_i = i
            pending_long = False

        # ---- generate signal from THIS bar for next bar ----
        # only longs, only when bull-regime confirmed
        bull_long = bull[i] >= config.bull_thr and c > o
        pending_long = bool(bull_long)

        equity_rows.append({
            "date": t, "balance": balance, "equity": account_equity(c),
            "drawdown_pct": (peak_equity - account_equity(c)) / peak_equity if peak_equity > 0 else 1.0,
        })
        peak_equity = max(peak_equity, account_equity(c))

    trades = pd.DataFrame(trade_rows)
    if not len(trades):
        trades = pd.DataFrame(columns=["time", "side", "lots", "entry", "exit", "pnl", "reason", "bars"])
    wins = float(trades[trades["pnl"] > 0]["pnl"].sum()) if len(trades) else 0.0
    losses = abs(float(trades[trades["pnl"] < 0]["pnl"].sum())) if len(trades) else 0.0
    n = len(trades)
    eq = pd.DataFrame(equity_rows)
    stats = {
        "final_balance": balance,
        "final_equity": float(eq.iloc[-1]["equity"]) if len(eq) else balance,
        "total_return_pct": (float(eq.iloc[-1]["equity"]) / config.initial_balance_usc - 1.0) * 100 if len(eq) else 0.0,
        "trades": n,
        "wins": int((trades["pnl"] > 0).sum()) if n else 0,
        "losses": int((trades["pnl"] < 0).sum()) if n else 0,
        "winrate": (trades["pnl"] > 0).mean() if n else 0.0,
        "net_pnl": float(trades["pnl"].sum()) if n else 0.0,
        "avg_win": float(wins / max(1, int((trades["pnl"] > 0).sum()))),
        "avg_loss": float(-losses / max(1, int((trades["pnl"] < 0).sum()))),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown_pct": float(eq["drawdown_pct"].max() * 100) if len(eq) else 0.0,
    }
    return {"stats": stats, "equity": eq, "trades": trades}
