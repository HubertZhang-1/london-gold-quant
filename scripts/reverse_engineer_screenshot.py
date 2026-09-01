# -*- coding: utf-8 -*-
"""Reverse-engineer a hedge+martingale grid EA from MT5 XAUUSD.c P&L screenshots.

Trade rows reconstructed from the two screenshots (best-effort OCR of the
readable columns). Columns: open_time, side, lots, open_price, close_time,
close_price, pnl, note.
"""
from __future__ import annotations

import statistics

# (open_time, side, lots, open_price, close_time, close_price, pnl, note)
TRADES = [
    # ---- 2026-07-16 ----
    ("09:03:24", "buy",  0.3, 4032.77, "09:05:15", 4033.80,  30.90, "scalp"),
    ("09:17:02", "sell", 0.3, 4029.46, "09:19:50", 4028.48,  29.40, "scalp"),
    ("09:23:03", "sell", 0.3, 4029.96, "09:34:30", 4027.32,  49.20, "scalp"),
    ("09:35:00", "sell", 0.3, 4026.55, "09:35:05", 4024.16,  71.70, "scalp"),
    ("09:06:00", "buy",  0.3, 4035.65, "09:48:37", 4028.66, -209.70, "basket_l1"),
    ("09:35:28", "buy",  0.7, 4030.63, "09:48:37", 4028.66, -137.90, "basket_l2"),
    ("09:36:06", "buy",  1.0, 4024.35, "09:48:37", 4028.66,  431.00, "basket_l3"),
    ("09:37:01", "sell", 0.3, 4023.99, "09:48:31", 4024.66,  -20.10, "hedge_s"),
    ("09:52:04", "sell", 0.3, 4025.70, "10:14:32", 4026.77,  -83.40, "hedge_s"),
    # ---- 2026-07-17 ----
    ("06:12:00", "buy",  0.3, 3990.60, "06:15:38", 3991.54,  28.20, "scalp"),
    ("06:16:00", "buy",  0.3, 3992.04, "06:16:07", 3992.92,  26.40, "scalp"),
    ("06:18:06", "buy",  0.3, 3995.25, "06:18:21", 3997.43,  65.40, "scalp"),
    ("06:19:08", "buy",  0.3, 3997.90, "06:19:11", 3999.87,  59.10, "scalp"),
    ("06:20:00", "buy",  0.3, 4002.76, "06:20:03", 4004.05,  38.70, "scalp"),
    ("06:34:01", "buy",  0.3, 4024.76, "06:34:17", 3996.31,  28.20, "scalp?lookup"),
    # ---- 2026-07-20 ----
    ("03:20:00", "buy",  0.3, 3997.40, "03:23:20", 3998.71,  39.30, "scalp"),
    ("03:20:00", "buy",  0.3, 4000.30, "03:20:46", 4002.48,  65.40, "scalp"),
    ("03:21:04", "buy",  0.3, 4002.97, "03:21:11", 4003.94,  29.10, "scalp"),
    ("03:34:09", "buy",  0.3, 4026.60, "03:34:37", 4027.32,  21.60, "scalp"),
    ("03:39:26", "sell", 0.3, 4027.00, "03:44:46", 4026.06,  28.20, "scalp"),
    ("04:01:04", "buy",  0.3, 4011.25, "04:01:09", 4008.11,  -94.20, "basket_l1"),
    ("04:05:04", "buy",  0.3, 4014.44, "04:02:31", 4026.58,   38.70, "scalp"),
]


def hold_seconds(open_t: str, close_t: str) -> int:
    def to_s(t: str) -> int:
        h, m, s = (int(x) for x in t.split(":"))
        return h * 3600 + m * 60 + s

    d = to_s(close_t) - to_s(open_t)
    return d if d > 0 else 0


def main() -> None:
    lots = [t[2] for t in TRADES]
    pnls = [t[6] for t in TRADES]
    holds = [hold_seconds(t[0], t[4]) for t in TRADES]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    print("=== Position size distribution ===")
    for lot in sorted(set(lots)):
        cnt = lots.count(lot)
        print(f"  {lot:4.1f} lot : {cnt}  ({cnt/len(lots)*100:.0f}%)")

    print("\n=== Holding time (seconds) ===")
    print(f"  min={min(holds)}  median={statistics.median(holds)}  max={max(holds)}")
    print(f"  trades <= 90s: {sum(1 for h in holds if h <= 90)} / {len(holds)}")

    print("\n=== P/L profile ===")
    print(f"  total trades={len(pnls)}  wins={len(wins)}  winrate={len(wins)/len(pnls)*100:.0f}%")
    print(f"  sum pnl = {sum(pnls):+.2f}  avg win={statistics.mean(wins):+.2f}  avg loss={statistics.mean(losses):+.2f}")
    print(f"  profit factor = {sum(wins)/abs(sum(losses)):.2f}")

    print("\n=== Bracket / basket structure (2026-07-16 long ladder) ===")
    print("  longs 0.3->0.7->1.0 entered @4035.65/4030.63/4024.35, basket closed @4028.66")
    avg = (0.3 * 4035.65 + 0.7 * 4030.63 + 1.0 * 4024.35) / 2.0
    print(f"  avg entry={avg:.2f}  basket pnl={ (4028.66-avg)*2.0*100:+.2f} (recovered pure martingale)")


if __name__ == "__main__":
    main()
