# -*- coding: utf-8 -*-
"""Scan low-risk variants to see if any config stays profitable & drawdown-bounded."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_5m_2026.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)

print(f"{'step':>5} {'tp':>4} {'stop':>4} {'layers':>6} {'ladder':>16} {'tr':>4} "
      f"{'trades':>6} {'win%':>5} {'PF':>5} {'ret%':>8} {'maxDD%':>7} {'term':>12}")
print("-" * 92)

ladders = {
    "flat.3": (0.3, 0.3, 0.3, 0.3),
    "mild": (0.3, 0.4, 0.5, 0.6),
    "mod": (0.3, 0.5, 0.7, 0.9),
    "mart": (0.3, 0.6, 1.0, 1.5),
}
for step in (1.0, 1.2, 1.5):
    for tp in (1.0, 1.5, 2.0):
        for stop in (3.0, 4.0, 6.0, 8.0):
            for layers in (3, 4):
                for lname, ladder in ladders.items():
                    cfg = MartingaleConfig(
                        grid_atr_mult=step, take_profit_atr=tp, stop_loss_atr=stop,
                        hedge_atr=2.5, max_layers=layers, lot_ladder=ladder,
                        use_trend_filter=True,
                    )
                    r = run_martingale_backtest(DF, cfg)
                    s = r.stats
                    # keep only non-terminal-disaster, positive or near-flat
                    if s["trades"] < 50:
                        continue
                    print(f"{step:5.1f} {tp:4.1f} {stop:4.1f} {layers:6d} {lname:>16} "
                          f"{s['trades']:6d} {s['winrate']*100:5.1f} {s['profit_factor']:5.2f} "
                          f"{s['total_return_pct']:+8.2f} {s['max_drawdown_pct']:7.1f} {s['terminal_reason'] or 'none':>12}")
