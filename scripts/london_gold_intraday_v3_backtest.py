# -*- coding: utf-8 -*-
"""Third-gen intraday strategy backtest on real 5m XAUUSD data.

Supports ATR risk:reward (SL/TP) exits using the local backtest engine's event
loop extended with a take-profit path. Compares the new confidence-scored
strategies against the existing v2 momentum / reversion baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.backtest import CostConfig, _fill_price
from london_gold.intraday_strategies_v2 import momentum_trend_signals, zscore_reversion_signals
from london_gold.intraday_strategies_v3 import (
    mean_reversion_signals,
    momentum_scalp_signals,
)
from london_gold.report import format_table

DEFAULT_CSV = str(PROJECT_ROOT / "data" / "XAUUSD_5m_2026.csv")


@dataclass
class V3Result:
    name: str
    dates: list
    equity: np.ndarray
    trades: list
    stats: dict


def backtest_v3(df: pd.DataFrame, cost: CostConfig, name: str, params: dict) -> V3Result:
    """Event backtest with both stop_dist and tp_dist exits.

    ``df`` is a strategy-signal frame with signal/stop_dist/tp_dist columns.
    Uses the same signed-cash accounting as the production run_backtest, and
    carries a signed pos_oz (positive = long, negative = short).
    """
    data = df.reset_index(drop=True)
    if "signal" not in data or "stop_dist" not in data:
        raise ValueError("v3 frame needs at least signal/stop_dist")
    if "tp_dist" not in data:
        data["tp_dist"] = 0.0
    opens = data["open"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    signals = data["signal"].to_numpy(int)
    stop_dists = data["stop_dist"].to_numpy(float)
    tp_dists = data["tp_dist"].to_numpy(float)
    dates = data["date"].tolist()
    n = len(data)

    cash = cost.capital
    pos_oz = 0.0
    entry_mid = 0.0
    stop_dist = 0.0
    tp_dist = 0.0
    entry_idx = None
    cash_at_entry = 0.0
    equity = np.full(n, np.nan)
    trades = []
    blocked_signal = 0

    def close_position(i, exit_mid, reason):
        nonlocal cash, pos_oz
        if pos_oz == 0:
            return
        closing_long = pos_oz > 0
        direction = -1 if closing_long else 1
        exit_fill = _fill_price(exit_mid, direction, cost)
        commission = cost.commission_per_oz * abs(pos_oz)
        if closing_long:
            cash += exit_fill * abs(pos_oz) - commission
        else:
            cash -= exit_fill * abs(pos_oz) + commission
        trades.append({
            "side": "long" if pos_oz > 0 else "short",
            "entry_time": dates[entry_idx],
            "exit_time": dates[i],
            "entry_price": round(entry_mid, 2),
            "exit_price": round(exit_mid, 2),
            "pnl": round(cash - cash_at_entry, 2),
            "bars_held": i - entry_idx,
            "exit_reason": reason,
        })
        pos_oz = 0.0

    for i in range(n):
        # intrabar stop/take eval on range before signal fills
        if pos_oz != 0 and stop_dist > 0:
            stop_price = entry_mid - stop_dist if pos_oz > 0 else entry_mid + stop_dist
            if pos_oz > 0 and lows[i] <= stop_price:
                close_position(i, stop_price, "stop")
                blocked_signal = signals[i - 1] if i > 0 else 0
            elif pos_oz < 0 and highs[i] >= stop_price:
                close_position(i, stop_price, "stop")
                blocked_signal = signals[i - 1] if i > 0 else 0
        if pos_oz != 0 and tp_dist > 0:
            tp_price = entry_mid + tp_dist if pos_oz > 0 else entry_mid - tp_dist
            if pos_oz > 0 and highs[i] >= tp_price:
                close_position(i, tp_price, "take_profit")
            elif pos_oz < 0 and lows[i] <= tp_price:
                close_position(i, tp_price, "take_profit")

        # signal-driven exits/entries on next open
        if i > 0:
            prev_signal = signals[i - 1]
            if prev_signal == 0:
                blocked_signal = 0
            if pos_oz != 0 and (prev_signal == 0 or (pos_oz > 0) != (prev_signal > 0)):
                close_position(i, float(opens[i]), "signal")
            if pos_oz == 0 and prev_signal != 0 and prev_signal != blocked_signal:
                entry_mid = float(opens[i])
                entry_fill = _fill_price(entry_mid, prev_signal, cost)
                stop_dist = float(stop_dists[i - 1]) if not np.isnan(stop_dists[i - 1]) else 0.0
                tp_dist = float(tp_dists[i - 1]) if not np.isnan(tp_dists[i - 1]) else 0.0
                oz = cost.position_oz
                if cost.risk_per_trade_pct > 0 and stop_dist > 0:
                    oz = cost.capital * cost.risk_per_trade_pct / stop_dist
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

        equity[i] = cash + pos_oz * closes[i] if pos_oz != 0 else cash

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
        "label": name,
        "name": name,
        "params": params,
        "total_return": total_return,
        "annual_return": total_return,
        "sharpe": float(np.nanmean(np.diff(equity)) / (np.nanstd(np.diff(equity)) + 1e-12)),
        "max_drawdown": dd * 100,
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "profit_factor": gp / gl if gl > 0 else 0.0,
        "final_equity": float(equity[-1]),
        "final_balance": float(cash),
    }
    return V3Result(name=name, dates=dates, equity=equity, trades=trades, stats=stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 intraday gold strategies backtest (5m XAUUSD)")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--position-oz", type=float, default=10.0)
    parser.add_argument("--risk-pct", type=float, default=0.02)
    parser.add_argument("--out", default=f"xauusd_v3_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-28")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    mask = (df["date"] >= pd.Timestamp(args.start, tz="UTC")) & (df["date"] <= pd.Timestamp(args.end, tz="UTC"))
    df = df[mask].reset_index(drop=True)

    cost = CostConfig(
        capital=args.capital,
        position_oz=args.position_oz,
        spread=0.35,
        slippage=0.10,
        commission_per_oz=0.10,
        risk_per_trade_pct=args.risk_pct,
    )
    variants = {
        "V3_MOM_SCALP": lambda: momentum_scalp_signals(
            df, min_confidence=0.65, rr_target=2.0, stop_mult=1.0,
        ),
        "V3_MOM_SCALP_STRONG": lambda: momentum_scalp_signals(
            df, min_confidence=0.75, rr_target=2.0, stop_mult=1.0,
        ),
        "V3_MEANREV": lambda: mean_reversion_signals(
            df, entry_z=2.0, rr_target=2.0, stop_mult=1.5, max_adx_meanrev=32.0,
        ),
        "V2_MOM_BASE": lambda: momentum_trend_signals(df),
        "V2_ZSCORE_BASE": lambda: zscore_reversion_signals(df),
    }

    results = []
    # V3 strategies (with ATR SL/TP) use the v3 engine; use module defaults (optimized)
    for label, fn in [("V3_MOM_SCALP", lambda: momentum_scalp_signals(df)),
        ("V3_MOM_SCALP_STRONG", lambda: momentum_scalp_signals(df, min_confidence=0.65, rr_target=2.0)),
        ("V3_MEANREV", lambda: mean_reversion_signals(df))]:
        frame = fn()
        res = backtest_v3(frame, cost, label, {})
        results.append(res)

    # V2 baselines use the production run_backtest (verified accounting)
    from london_gold.backtest import run_backtest
    for label, fn in [("V2_MOM_BASE", lambda: momentum_trend_signals(df)),
                      ("V2_ZSCORE_BASE", lambda: zscore_reversion_signals(df))]:
        frame = fn()
        r = run_backtest(frame, cost=cost, name=label, params={})
        s = r["stats"]
        results.append(type("R", (), {
            "name": label,
            "dates": r["dates"],
            "equity": np.array(r["equity"]),
            "trades": r["trades"],
            "stats": {
                "label": label, "name": label, "params": {}, "trade_count": s["trade_count"],
                "win_rate": s["win_rate"], "profit_factor": s["profit_factor"],
                "total_return": s["total_return"], "max_drawdown": s["max_drawdown"],
                "sharpe": s["sharpe"], "final_equity": s["final_equity"],
                "final_balance": s["final_equity"],
            },
        })())

    for r in results:
        st = r.stats
        print(f"{st['label']:20s} trades={st['trade_count']:4d} "
              f"win%={st['win_rate']:5.1f} ret={st['total_return']:+7.2f}% "
              f"PF={st['profit_factor']:5.2f} maxDD={st['max_drawdown']:5.1f}%")

    # summary table
    s = [r.stats for r in results]
    rows = [[r["label"], f"{r['total_return']:.1f}", f"{r['sharpe']:.2f}",
             f"{r['max_drawdown']:.1f}", r["trade_count"], f"{r['win_rate']:.1f}",
             f"{r['profit_factor']:.1f}"] for r in s]
    print()
    print(format_table(["策略", "总收益%", "夏普", "最大回撤%", "交易数", "胜率%", "盈亏比"], rows))

    # save best equity
    best = max(results, key=lambda r: r.stats["total_return"])
    out = PROJECT_ROOT / "reports" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": best.dates, "equity": best.equity}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best.trades).to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\nbest={best.name}  wrote {out}_equity.csv / _trades.csv")


if __name__ == "__main__":
    main()
