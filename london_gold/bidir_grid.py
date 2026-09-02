# -*- coding: utf-8 -*-
"""Bi-directional XAUUSD CFD strategy (牛市做多 / 熊市做空 / 震荡不参与).

The user's actual instrument is an XAUUSD CFD (spot/difference contract): it is
LINEAR, can go LONG or SHORT, and CAN blow up (margin call). So the correct usage
is to trade BOTH directions with the trend, NOT a bull-only long. This version:

  1. LONG when bull score >= bull_thr AND the market rhythm is a confirmed UP trend.
  2. SHORT when bull score < a bear threshold AND the rhythm is a confirmed DOWN trend.
  3. FLAT (no trade) when the rhythm says chop/range (震荡不参与).
  4. Risk management for a blow-up-prone linear contract:
       - risk-budget sizing (lose risk% of balance at the stop distance)
       - hard ATR stop (both sides) + TP (RR) + time-stop
       - optional macro dampening multiplier on lots.

Direction is chosen by the SIGNED signal of market_state (which is +1 in an up
trend, -1 in a down trend, 0 in chop). A one-sided bull gate addition: only take
LONG when bull>=bull_thr, and only take SHORT when bull<=bear_thr (else we could
short a bull regime, which fights the trend).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr, market_state, trend_regime
from .macro_factors import forward_fill_macro


@dataclass
class BidirGridConfig:
    initial_balance_usc: float = 100_000.0
    base_lot: float = 0.3
    usc_per_price_lot: float = 100.0
    atr_bars: int = 14
    stop_mult: float = 2.0          # stop distance (both sides) = stop_mult * ATR
    rr: float = 2.0                 # TP = rr * stop distance
    bull_ema_bars: int = 60          # trend EMA for regime
    er_window: int = 20
    bull_thr: float = 0.55           # bull >= this -> allow LONG
    bear_thr: float = 0.45           # bull <= this -> allow SHORT
    er_threshold: float = 0.12
    adx_window: int = 14
    adx_threshold: float = 20.0
    max_bars_in_trade: int = 30      # time-stop
    spread: float = 0.35
    slippage: float = 0.10
    commission_per_lot: float = 0.10
    macro_lev_lo: float = 1.0
    macro_lev_hi: float = 1.0
    risk_per_trade_pct: float = 0.02 # lose this % of balance at the stop
    rhythm_gate: bool = True
    rhythm_era_window: int = 20
    rhythm_er_thr: float = 0.10
    rhythm_adx_window: int = 14
    rhythm_adx_thr: float = 16.0
    rhythm_ema_window: int = 50
    chop_hi: float = 68.0
    # ---- strict short gate (fix the losing-short problem) ----
    short_off_high_min: float = 0.12  # require price at least 12% below the recent
                                      # N-bar high before going short (filter pullbacks)
    short_high_window: int = 60       # lookback for the recent high
    short_cooldown: int = 3           # bars to wait after a losing stop before re-shorting
    short_require_macro: bool = False # if True, only short when macro score is bearish
    short_macro_thr: float = -0.1     # macro below this count as bearish for shorts


def _fill(mid: float, side: int, spread: float, slippage: float = 0.0) -> float:
    return mid + side * (spread / 2.0 + slippage)


def _pnl(side: int, entry: float, exit: float, lots: float, usc: float, commission: float = 0.0) -> float:
    return side * (exit - entry) * lots * usc - commission * abs(lots)


def _bull_score(close: pd.Series, high: pd.Series, low: pd.Series, cfg: BidirGridConfig) -> np.ndarray:
    ema_s = close.ewm(span=cfg.bull_ema_bars, adjust=False).mean()
    ema_slope = ema_s.diff(cfg.bull_ema_bars // 2 or 1)
    up_trend = (close > ema_s).astype(float)
    trend = trend_regime(close, high, low, er_window=cfg.er_window,
                         er_threshold=cfg.er_threshold, adx_window=cfg.adx_window,
                         adx_threshold=cfg.adx_threshold).to_numpy()
    bull = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)
    return bull


def run_bidir_grid_backtest(df: pd.DataFrame, config: BidirGridConfig | None = None,
                            macro_series=None) -> dict:
    config = config or BidirGridConfig()
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

    # signed rhythm signal: +1 long (up trend), -1 short (down trend), 0 chop
    if config.rhythm_gate:
        rhythm = market_state(close, high, low,
                              er_window=config.rhythm_era_window,
                              er_thr=config.rhythm_er_thr,
                              adx_window=config.rhythm_adx_window,
                              adx_thr=config.rhythm_adx_thr,
                              chop_hi=config.chop_hi,
                              ema_window=config.rhythm_ema_window)
        rhythm_dir = rhythm["dir"].to_numpy()          # +1 / -1 / 0
        rhythm_state = rhythm["state"].to_numpy()      # trend / chop / neutral
    else:
        rhythm_dir = np.where(bull >= 0.5, 1, np.where(bull <= 0.5, -1, 0))
        rhythm_state = np.where(np.abs(rhythm_dir) > 0, "trend", "neutral")

    # macro lot multiplier
    macro = np.ones(len(data))
    macro_score = np.full(len(data), np.nan)  # raw macro score for the short gate
    if macro_series is not None:
        if isinstance(macro_series, pd.Series):
            macro_arr = pd.Series(macro_series).to_numpy()
        else:
            macro_arr = np.asarray(macro_series, dtype=float)
        macro_score = np.nan_to_num(macro_arr, nan=0.0)
        if config.macro_lev_lo != config.macro_lev_hi:
            m = np.clip(macro_score, -1.0, 1.0)
            macro = config.macro_lev_lo + (config.macro_lev_hi - config.macro_lev_lo) * (m + 1.0) / 2.0
            macro = np.where(np.isnan(macro_arr), 1.0, macro)

    balance = float(config.initial_balance_usc)
    equity_rows, trade_rows = [], []
    peak_equity = balance
    pos = 0.0       # signed lots (+ = long, - = short)
    entry = 0.0
    entry_i = -1
    pending_side = 0  # +1 / -1 / 0 from prior bar

    # strict-short state
    last_stop_bar = -10 ** 9
    # rolling recent high (past short_high_window bars, excluding current)
    recent_high = pd.Series(high).rolling(config.short_high_window, min_periods=config.short_high_window // 2).max().to_numpy()

    def account_equity(mid: float) -> float:
        if pos == 0:
            return balance
        side = 1 if pos > 0 else -1
        return balance + _pnl(side, entry, mid, abs(pos), config.usc_per_price_lot,
                              config.commission_per_lot)

    for i in range(len(data)):
        t = data["date"].iloc[i]
        o = float(open_.iloc[i]); h = float(high.iloc[i]); l = float(low.iloc[i]); c = float(close.iloc[i])
        a = atr14[i] if not np.isnan(atr14[i]) else 1.0

        # ---- manage open position intrabar (symmetric stop/TP) ----
        if pos != 0:
            side = 1 if pos > 0 else -1
            stop_dist = config.stop_mult * a
            sp = entry - side * stop_dist
            tpp = entry + side * config.rr * stop_dist
            hit_stop = (side > 0 and l <= sp) or (side < 0 and h >= sp)
            hit_tp = (side > 0 and h >= tpp) or (side < 0 and l <= tpp)
            timeout = (i - entry_i) >= config.max_bars_in_trade
            exit_px = None
            reason = None
            if hit_stop:
                exit_px = sp; reason = "stop"
            elif hit_tp:
                exit_px = tpp; reason = "tp"
            elif timeout:
                exit_px = c; reason = "time"
            if exit_px is not None:
                pnl = _pnl(side, entry, exit_px, abs(pos), config.usc_per_price_lot,
                           config.commission_per_lot)
                balance += pnl
                trade_rows.append({"time": t, "side": "long" if side > 0 else "short",
                                   "lots": abs(pos), "entry": entry, "exit": exit_px,
                                   "pnl": pnl, "reason": reason, "bars": i - entry_i})
                if reason == "stop":
                    last_stop_bar = i
                pos = 0.0

        # ---- entry on this bar's open if a signal fired ----
        if pos == 0 and pending_side != 0:
            side = pending_side
            o_fill = _fill(o, side, config.spread, config.slippage)
            stop_dist = config.stop_mult * a
            if config.risk_per_trade_pct > 0 and stop_dist > 0:
                risk_usc = balance * config.risk_per_trade_pct
                lots = risk_usc / (stop_dist * config.usc_per_price_lot)
            else:
                lots = config.base_lot
            lots = lots * float(macro[i])
            lots = max(0.01, round(lots, 2))
            entry = o_fill
            pos = side * lots
            entry_i = i
            pending_side = 0

        # ---- generate signal from THIS bar for next bar ----
        # only in a confirmed trend (rhythm_state == 'trend'), pick direction by
        # rhythm_dir, cross-checked against bull regime: long only if bull>=bull_thr,
        # short only if bull<=bear_thr AND the strict-short gate passes. Chop -> aside.
        side = 0
        if rhythm_state[i] == "trend" and rhythm_dir[i] != 0:
            if rhythm_dir[i] > 0 and bull[i] >= config.bull_thr:
                side = 1
            elif rhythm_dir[i] < 0 and bull[i] <= config.bear_thr:
                # strict short gate
                rh = recent_high[i]
                off_high = (close.iloc[i] - rh) / rh if rh and not np.isnan(rh) else 0.0
                cooled = (i - last_stop_bar) >= config.short_cooldown
                if off_high <= -config.short_off_high_min and cooled:
                    if config.short_require_macro:
                        if macro_score[i] <= config.short_macro_thr:
                            side = -1
                    else:
                        side = -1
        pending_side = side

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
        "long_trades": int((trades["side"] == "long").sum()) if n else 0,
        "short_trades": int((trades["side"] == "short").sum()) if n else 0,
    }
    return {"stats": stats, "equity": eq, "trades": trades}
