# -*- coding: utf-8 -*-
"""London gold (XAU) quant toolkit."""
from .data import fetch_daily, fetch_realtime, load_daily
from .backtest import CostConfig, run_backtest
from .intraday_data import fetch_intraday, load_intraday
from .intraday_strategy import open_range_breakout_signals
from .strategies import (
    donchian_breakout_signals,
    ema_cross_signals,
    rsi_reversal_signals,
)

__all__ = [
    "fetch_daily",
    "fetch_realtime",
    "load_daily",
    "fetch_intraday",
    "load_intraday",
    "open_range_breakout_signals",
    "CostConfig",
    "run_backtest",
    "donchian_breakout_signals",
    "ema_cross_signals",
    "rsi_reversal_signals",
]
