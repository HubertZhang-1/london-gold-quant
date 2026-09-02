# -*- coding: utf-8 -*-
"""MACRO layer for the three-line gold system: derive a [-1,1] macro direction score.

Based on research findings (Siebert regression: R^2~0.32, real-yield -6%/pt,
DXY -0.85%/1%, VIX conditional):
  - actual/nominal yield rising -> bearish gold (negative driver)
  - USD strengthening -> bearish gold (negative driver)
  - VIX spiking (crisis) -> short-term bullish gold, but only conditionally
The macro score feeds the long-period direction (wind vane), not short timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def macro_direction_score(macro: pd.DataFrame, tnx_window: int = 60,
                          dxy_window: int = 60, vix_window: int = 20) -> pd.DataFrame:
    """Compute a [-1,1] macro direction score on the daily macro frame.

    Columns expected: date, dxy, vix, tnx (may be partial).
    Returns the same frame plus:
      macro_score  : composite direction (-1 bearish..+1 bullish gold)
      yield_chg    : normalized change in 10y nominal yield (negative to gold)
      dxy_chg      : normalized DXY change (negative to gold)
      vix_gate     : crisis-level VIX (positive to gold, conditional)
    """
    out = macro.copy().sort_values("date").reset_index(drop=True)

    dxy = out["dxy"]
    tnx = out["tnx"]
    vix = out["vix"]
    # Changes over a rolling window, normalized by rolling std
    dxy_chg = dxy.diff(dxy_window) / (dxy.rolling(dxy_window).std() + 1e-9)
    tnx_chg = tnx.diff(tnx_window) / (tnx.rolling(tnx_window).std() + 1e-9)
    vix_chg = vix.diff(vix_window) / (vix.rolling(vix_window).std() + 1e-9)

    # Driver contributions: gold is NEGATIVE to yield rises and USD rises.
    # Scale to ~[-1,1] via tanh.
    yield_contrib = -np.tanh(tnx_chg / 2.0)      # rising yield -> -gold
    dxy_contrib = -np.tanh(dxy_chg / 2.0)        # rising DXY  -> -gold
    # crisis gate: only when VIX level is high does a jump help gold
    vix_level = vix / 100.0
    crisis_ok = (vix > 25.0).astype(float)       # above 25 = elevated stress
    vix_contrib = np.tanh(vix_chg / 2.0) * crisis_ok  # conditional

    # Composite: weight research coefficients (yield -6pts, dxy -0.85) but
    # normalized. Give yield a bit more weight.
    score = 0.5 * yield_contrib + 0.35 * dxy_contrib + 0.15 * vix_contrib
    score = np.clip(score, -1.0, 1.0)

    out["macro_score"] = score
    out["yield_chg"] = yield_contrib
    out["dxy_chg"] = dxy_contrib
    out["vix_gate"] = vix_contrib
    # Align the frame to tz-aware date index so forward_fill_macro (which keys on
    # the Series index) can map the daily macro score onto gold-bar dates. Without
    # this the macro_score Series had a RangeIndex and ffilled to a constant.
    out.index = out["date"]
    return out


def forward_fill_macro(macro_score_series: pd.Series, gold_dates) -> pd.Series:
    """Map the daily macro score onto (higher-frequency) gold bar dates via ffill.

    ``gold_dates`` may be a DatetimeIndex, an ndarray of datetime64, or a Series.
    We coerce both the macro index and the target index to tz-aware UTC so the
    reindex works regardless of tz-naive vs tz-aware inputs.
    """
    import pandas as _pd
    m = macro_score_series.dropna()
    # ensure tz-aware UTC index, period start
    macro_idx = _pd.DatetimeIndex(m.index)
    if macro_idx.tz is None:
        macro_idx = macro_idx.tz_localize("UTC")
    else:
        macro_idx = macro_idx.tz_convert("UTC")
    s = _pd.Series(m.values, index=macro_idx)

    # normalize target dates to a tz-aware UTC datetime index
    if isinstance(gold_dates, _pd.Series):
        target = _pd.DatetimeIndex(gold_dates)
    else:
        target = _pd.DatetimeIndex(gold_dates)
    if target.tz is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")

    return s.reindex(target, method="ffill").fillna(0.0)
