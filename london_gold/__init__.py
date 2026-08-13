# -*- coding: utf-8 -*-
"""London gold (XAU) quant toolkit."""
from .data import fetch_daily, fetch_realtime, load_daily
from .backtest import CostConfig, run_backtest
from .strategies import (
    donchian_breakout_signals,
    ema_cross_signals,
    rsi_reversal_signals,
)

__all__ = [
    "fetch_daily",
    "fetch_realtime",
    "load_daily",
    "CostConfig",
    "run_backtest",
    "donchian_breakout_signals",
    "ema_cross_signals",
    "rsi_reversal_signals",
]
