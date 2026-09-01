# -*- coding: utf-8 -*-
"""Data preparation for the GC=F bidirectional grid proxy backtest."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr


@dataclass(frozen=True)
class DataAudit:
    rows_in: int
    rows_out: int
    duplicate_rows: int
    missing_intervals: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp


@dataclass(frozen=True)
class PreparedRange:
    input_bars: pd.DataFrame
    evaluation: pd.DataFrame
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


def normalize_ohlc(df: pd.DataFrame) -> tuple[pd.DataFrame, DataAudit]:
    """Normalize UTC OHLC data and return an audit of duplicates and gaps."""
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = df[["date", "open", "high", "low", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    for name in ("open", "high", "low", "close"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    if data.empty:
        raise ValueError("empty OHLC data")
    prices = data[["open", "high", "low", "close"]]
    if data["date"].isna().any() or prices.isna().any().any() or not np.isfinite(prices).all().all():
        raise ValueError("non-finite OHLC data")

    duplicate_rows = int(data.duplicated("date", keep="last").sum())
    data = data.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    invalid = (data["high"] < data[["open", "close", "low"]].max(axis=1)) | (
        data["low"] > data[["open", "close", "high"]].min(axis=1)
    )
    if invalid.any():
        raise ValueError(f"invalid OHLC rows: {invalid[invalid].index.tolist()}")

    deltas = data["date"].diff().dropna()
    intervals = (deltas / pd.Timedelta(minutes=5)).round() - 1
    missing_intervals = int(intervals.clip(lower=0).sum())
    audit = DataAudit(
        rows_in=len(df),
        rows_out=len(data),
        duplicate_rows=duplicate_rows,
        missing_intervals=missing_intervals,
        first_timestamp=data.iloc[0]["date"],
        last_timestamp=data.iloc[-1]["date"],
    )
    return data, audit


def add_grid_step(
    df: pd.DataFrame,
    atr_bars: int = 14,
    atr_multiplier: float = 0.35,
    min_step: float = 0.60,
    max_step: float = 2.50,
) -> pd.DataFrame:
    """Add previous-completed-bar Wilder ATR and the clamped grid step."""
    if atr_bars <= 0 or atr_multiplier <= 0 or min_step <= 0 or max_step < min_step:
        raise ValueError("invalid ATR grid parameters")
    data = df.copy()
    current_atr = atr(data["high"], data["low"], data["close"], window=atr_bars)
    data["atr"] = current_atr.shift(1)
    data["grid_step"] = (data["atr"] * atr_multiplier).clip(lower=min_step, upper=max_step)
    return data


def select_latest_days(
    df: pd.DataFrame,
    days: int = 60,
    warmup_bars: int = 20,
) -> PreparedRange:
    """Select a latest-calendar-day evaluation slice plus earlier warmup rows."""
    if df.empty:
        raise ValueError("empty OHLC data")
    if days <= 0 or warmup_bars < 0:
        raise ValueError("days must be positive and warmup_bars non-negative")
    data = df.copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    end = data["date"].max()
    start = end - pd.Timedelta(days=days)
    evaluation_mask = (data["date"] >= start) & (data["date"] <= end)
    positions = data.index[evaluation_mask]
    if len(positions) == 0:
        raise ValueError("no bars in latest-day evaluation range")
    first = int(positions[0])
    evaluation = data.loc[evaluation_mask].reset_index(drop=True)
    input_bars = data.iloc[max(0, first - warmup_bars) :].reset_index(drop=True)
    return PreparedRange(
        input_bars=input_bars,
        evaluation=evaluation,
        evaluation_start=evaluation.iloc[0]["date"],
        evaluation_end=evaluation.iloc[-1]["date"],
    )
