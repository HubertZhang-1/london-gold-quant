# -*- coding: utf-8 -*-
"""Hedged martingale-grid backtester replicating the reverse-engineered EA.

Faithful model of the MT5 XAUUSD.c screenshots ("对冲顺势加仓"):
  - ONE trend direction carries the martingale ladder (the "顺势/主仓" side).
  - adds at a fixed ATR-based grid step, lot ladder 0.3 -> 0.7 -> 1.0 -> ...
  - the whole basket closes when its average entry is reclaimable (small TP);
    unavoidable on the adverse side is a HEDGE opened at a loss threshold.
  - contract = 100 oz/lot => pnl = (exit-entry) * lots * 100.
  - risk cutoffs: daily loss, max drawdown, bankruptcy.

Intrabar path is direction-aware (long uses O-L-H-C, short uses O-H-L-C) so the
model cannot "eat both sides" inside one bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr


@dataclass
class MartingaleConfig:
    initial_balance_usc: float = 100_000.0
    base_lot: float = 0.3
    lot_ladder: tuple[float, ...] = (0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    usc_per_price_lot: float = 100.0
    atr_bars: int = 14
    grid_atr_mult: float = 1.0       # grid step = ATR * this (screenshot: ~5-6 USD)
    max_layers: int = 6              # max adds per side
    take_profit_atr: float = 1.0     # basket TP once unrealized >= ATR*this points
    hedge_atr: float = 2.0           # open hedge when adverse loss >= ATR*this points
    stop_loss_atr: float = 0.0       # basket hard-stop; 0 disables (martingale = no stop)
    trend_ema_bars: int = 60
    use_trend_filter: bool = True    # False opens both directions regardless of trend
    spread: float = 0.35             # per-side half-spread already in fill
    slippage: float = 0.10           # per side
    commission_per_lot: float = 0.10
    daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.30


@dataclass
class MartingaleResult:
    events: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    stats: dict


def _fill(mid: float, side: int, spread: float, slippage: float = 0.0) -> float:
    return mid + side * (spread / 2.0 + slippage)


def _pnl(side: int, entry: float, exit: float, lots: float, usc: float, commission: float = 0.0) -> float:
    return side * (exit - entry) * lots * usc - commission * abs(lots)


class Ladder:
    def __init__(self, side: int):
        self.side = side
        self.entries: list[tuple[float, float]] = []  # (entry_fill, lots)

    @property
    def lots(self) -> float:
        return sum(l for _, l in self.entries)

    @property
    def avg(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e * l for e, l in self.entries) / self.lots

    @property
    def last_entry(self) -> float | None:
        return self.entries[-1][0] if self.entries else None

    @property
    def layers(self) -> int:
        return len(self.entries)

    def unreal(self, mid: float, spread: float, usc: float, slippage: float = 0.0,
               commission: float = 0.0) -> float:
        if not self.entries:
            return 0.0
        return _pnl(self.side, self.avg, _fill(mid, -self.side, spread, slippage),
                    self.lots, usc, commission)


def run_martingale_backtest(
    df: pd.DataFrame,
    config: MartingaleConfig | None = None,
) -> MartingaleResult:
    config = config or MartingaleConfig()
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
    ema = close.ewm(span=config.trend_ema_bars, adjust=False).mean()

    balance = float(config.initial_balance_usc)
    equity_rows, event_rows, trade_rows = [], [], []
    peak_equity = balance
    daily_start_balance = balance
    daily_realized = 0.0
    current_day = None
    terminal = ""
    terminal_time: pd.Timestamp | None = None
    seq = 0

    main_ladder = Ladder(0)   # the trend/martingale side (side set at cycle open)
    hedge_ladder = Ladder(0)  # the opposite (hedge) side

    def ladders():
        return [l for l in (main_ladder, hedge_ladder) if l.entries]

    def account_equity(mid: float) -> float:
        return balance + sum(
            l.unreal(mid, config.spread, config.usc_per_price_lot, config.slippage, config.commission_per_lot)
            for l in ladders()
        )

    def record(ev: str, reason: str, t: pd.Timestamp, mid: float, **extra) -> None:
        nonlocal seq
        seq += 1
        event_rows.append({
            "sequence": seq, "time": t, "event": ev, "reason": reason, "mid": mid,
            "balance": balance, "equity": account_equity(mid),
            "main_lots": main_ladder.lots, "hedge_lots": hedge_ladder.lots,
            "main_side": main_ladder.side, **extra,
        })

    def add(lad: Ladder, t: pd.Timestamp, mid: float, reason: str) -> None:
        idx = lad.layers
        lots = config.lot_ladder[min(idx, len(config.lot_ladder) - 1)]
        fill = _fill(mid, lad.side, config.spread, config.slippage)
        lad.entries.append((fill, lots))
        record("open", reason, t, mid, side="buy" if lad.side > 0 else "sell",
               lots=round(lots, 2), layer=idx, avg=round(lad.avg, 3))

    def close_ladder(lad: Ladder, t: pd.Timestamp, mid: float, reason: str) -> None:
        nonlocal balance, daily_realized
        if not lad.entries:
            return
        exit_fill = _fill(mid, -lad.side, config.spread, config.slippage)
        for entry_fill, lots in lad.entries:
            pnl = _pnl(lad.side, entry_fill, exit_fill, lots, config.usc_per_price_lot,
                       config.commission_per_lot)
            balance += pnl
            daily_realized += pnl
            trade_rows.append({
                "side": "buy" if lad.side > 0 else "sell", "lots": lots,
                "entry_fill": round(entry_fill, 3), "exit_fill": round(exit_fill, 3),
                "pnl_usc": round(pnl, 2), "time": t, "reason": reason,
                "is_hedge": lad is hedge_ladder,
            })
        record("close", reason, t, mid, lots=round(lad.lots, 2))
        lad.entries.clear()

    for i, row in data.iterrows():
        t = row["date"]
        if current_day != t.date():
            current_day = t.date()
            daily_start_balance = balance
            daily_realized = 0.0
        step = float(atr14.iloc[i]) * config.grid_atr_mult if not np.isnan(atr14.iloc[i]) else 5.0 * config.grid_atr_mult

        # trend direction from EMA slope
        trend = 0
        if i >= config.trend_ema_bars:
            if ema.iloc[i] > ema.iloc[i - config.trend_ema_bars]:
                trend = 1
            elif ema.iloc[i] < ema.iloc[i - config.trend_ema_bars]:
                trend = -1

        # direction-aware intrabar path (cannot eat both sides)
        if trend > 0:
            path = (open_.iloc[i], low.iloc[i], high.iloc[i], close.iloc[i])
        elif trend < 0:
            path = (open_.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i])
        else:
            path = (open_.iloc[i], close.iloc[i])

        # open a fresh cycle when the main basket is flat and a trend is present
        if not main_ladder.entries and trend != 0:
            main_ladder.side = trend
            add(main_ladder, t, open_.iloc[i], "cycle_open")

        for mid in path:
            if terminal:
                break
            if not main_ladder.entries:
                continue

            # martingale add on the MAIN side when price steps adversely
            if main_ladder.layers < config.max_layers and main_ladder.last_entry is not None:
                adverse = main_ladder.last_entry - step if main_ladder.side > 0 else main_ladder.last_entry + step
                crossed = mid <= adverse if main_ladder.side > 0 else mid >= adverse
                if crossed:
                    add(main_ladder, t, mid, "martingale_add")

            # hedge: if the main basket is deeply underwater, open the hedge side
            loss = -main_ladder.unreal(mid, config.spread, config.usc_per_price_lot, config.slippage, config.commission_per_lot)
            if loss >= step * config.hedge_atr and not hedge_ladder.entries:
                hedge_ladder.side = -main_ladder.side
                add(hedge_ladder, t, mid, "hedge_open")

            # main basket TP once average reclaimable -> close main AND hedge together
            tp_target = step * config.take_profit_atr
            if main_ladder.unreal(mid, config.spread, config.usc_per_price_lot, config.slippage, config.commission_per_lot) >= tp_target:
                close_ladder(hedge_ladder, t, mid, "hedge_close_with_main")
                close_ladder(main_ladder, t, mid, "main_tp")

            # hard stop-loss per basket (low-risk mode): cut instead of riding
            if config.stop_loss_atr > 0 and main_ladder.entries:
                cur = main_ladder.unreal(mid, config.spread, config.usc_per_price_lot, config.slippage, config.commission_per_lot)
                if cur <= -step * config.stop_loss_atr:
                    close_ladder(hedge_ladder, t, mid, "hedge_close_with_stop")
                    close_ladder(main_ladder, t, mid, "main_stop")

            # risk cutoffs
            equity = account_equity(mid)
            peak_equity = max(peak_equity, equity)
            daily_loss = max(0.0, -daily_realized)
            if balance <= 0 or equity <= 0:
                terminal, terminal_time = "bankruptcy", t
                record("circuit_breaker", terminal, t, mid)
                break
            if peak_equity > 0 and (peak_equity - equity) / peak_equity >= config.max_drawdown_pct:
                terminal, terminal_time = "max_drawdown", t
                record("circuit_breaker", terminal, t, mid)
                break
            if daily_loss >= daily_start_balance * config.daily_loss_pct:
                terminal, terminal_time = "daily_loss", t
                record("circuit_breaker", terminal, t, mid)
                break

        if terminal:
            for lad in ladders():
                close_ladder(lad, terminal_time, close.iloc[i], "terminal_close")
            break

        equity = account_equity(close.iloc[i])
        peak_equity = max(peak_equity, equity)
        equity_rows.append({
            "date": t, "balance": balance, "equity": equity,
            "main_lots": main_ladder.lots, "hedge_lots": hedge_ladder.lots,
            "drawdown_pct": (peak_equity - equity) / peak_equity if peak_equity > 0 else 1.0,
        })

    final_mid = float(close.iloc[-1])
    for lad in ladders():
        close_ladder(lad, data.iloc[-1]["date"], final_mid, "end_of_data")

    equity_frame = pd.DataFrame(equity_rows)
    trade_frame = pd.DataFrame(trade_rows, columns=[
        "side", "lots", "entry_fill", "exit_fill", "pnl_usc", "time", "reason", "is_hedge"])
    trades = trade_frame
    wins = float(trades[trades["pnl_usc"] > 0]["pnl_usc"].sum())
    losses = abs(float(trades[trades["pnl_usc"] < 0]["pnl_usc"].sum()))
    n = len(trades)
    stats = {
        "final_balance": balance,
        "final_equity": float(equity_frame.iloc[-1]["equity"]) if len(equity_frame) else balance,
        "total_return_pct": (float(equity_frame.iloc[-1]["equity"]) / config.initial_balance_usc - 1.0) * 100 if len(equity_frame) else 0.0,
        "trades": n,
        "wins": int((trades["pnl_usc"] > 0).sum()),
        "losses": int((trades["pnl_usc"] < 0).sum()),
        "winrate": (trades["pnl_usc"] > 0).mean() if n else 0.0,
        "net_pnl": float(trades["pnl_usc"].sum()),
        "avg_win": float(wins / max(1, int((trades["pnl_usc"] > 0).sum()))),
        "avg_loss": float(-losses / max(1, int((trades["pnl_usc"] < 0).sum()))),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown_pct": float(equity_frame["drawdown_pct"].max() * 100) if len(equity_frame) else 0.0,
        "terminal_reason": terminal,
        "terminal_time": str(terminal_time) if terminal_time is not None else "",
        "hedge_trades": int(trades["is_hedge"].sum()) if n else 0,
    }
    return MartingaleResult(events=pd.DataFrame(event_rows), trades=trades,
                            equity=equity_frame, stats=stats)
