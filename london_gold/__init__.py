# -*- coding: utf-8 -*-
"""London gold (XAU) quant toolkit."""
from .data import fetch_daily, fetch_realtime, load_daily
from .backtest import CostConfig, run_backtest
from .intraday_data import fetch_intraday, load_intraday
from .intraday_strategy import open_range_breakout_signals
from .intraday_strategies_v2 import (
    combine_ensemble,
    momentum_trend_signals,
    session_breakout_signals,
    zscore_reversion_signals,
)
from .strategies import (
    donchian_breakout_signals,
    ema_cross_signals,
    rsi_reversal_signals,
)
from .intraday_strategies_v3 import (
    mean_reversion_signals,
    momentum_scalp_signals,
)
from .factor_library import (
    FACTOR_BUILDERS,
    aggregate_score,
    build_factors,
)
from .macro_factors import (
    forward_fill_macro,
    macro_direction_score,
)
from .cot_factors import (
    compute_cot_factors,
    cot_timing_score,
)
from .gold_system import (
    MICRO_WEIGHTS,
    SystemConfig,
    build_three_line_frame,
)

__all__ = [
    "fetch_daily",
    "fetch_realtime",
    "load_daily",
    "fetch_intraday",
    "load_intraday",
    "open_range_breakout_signals",
    "session_breakout_signals",
    "momentum_trend_signals",
    "zscore_reversion_signals",
    "combine_ensemble",
    "momentum_scalp_signals",
    "mean_reversion_signals",
    "build_factors",
    "aggregate_score",
    "FACTOR_BUILDERS",
    "macro_direction_score",
    "forward_fill_macro",
    "compute_cot_factors",
    "cot_timing_score",
    "SystemConfig",
    "build_three_line_frame",
    "MICRO_WEIGHTS",
    "CostConfig",
    "run_backtest",
    "donchian_breakout_signals",
    "ema_cross_signals",
    "rsi_reversal_signals",
]
