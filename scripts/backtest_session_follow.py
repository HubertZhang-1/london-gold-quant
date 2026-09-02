# -*- coding: utf-8 -*-
"""Backtest + parameter scan for the Asian->London session-continuation strategy."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.session_follow import SessionFollowConfig, run_session_follow  # noqa: E402

PROJ = Path(r"C:\Users\张策\Documents\EA量化项目")
d7 = pd.read_csv(PROJ / "data" / "XAUUSD_1m_202607.csv")
d8 = pd.read_csv(PROJ / "data" / "XAUUSD_1m_202608.csv")
df = pd.concat([d7, d8]).sort_values("date").reset_index(drop=True)


def fmt(s):
    # total % on 100k initial; net_pnl is in USD
    ret_pct = s["net_pnl"] / 100000.0 * 100.0
    return ("ret{:+.1f}% PF{:.2f} win{:.0f}% tr{} net${:+,.0f} avgW${:+,.0f}/avgL${:+,.0f} (L{} S{})".format(
        ret_pct, s["profit_factor"], s["winrate"] * 100, s["trades"], s["net_pnl"],
        s["avg_win"], s["avg_loss"], s["long_trades"], s["short_trades"]))


print("=== 亚洲盘→欧盘 延续策略 回测 (2026 07+08, 43交易日) ===")
cfg = SessionFollowConfig()
res = run_session_follow(df, cfg)
print("默认 (min_ret=%.2f%% min_vol=%.1f stop=%.1fx rr=%.1f) : %s" % (
    cfg.min_asia_ret_pct, cfg.min_vol, cfg.stop_mult, cfg.rr, fmt(res["stats"])))

print("\n--- 参数扫描: min_asia_ret_pct / min_vol ---")
for ms, mv in [(0.05, 4.0), (0.1, 5.0), (0.15, 6.0), (0.2, 8.0), (0.1, 3.0), (0.3, 8.0)]:
    c = SessionFollowConfig(min_asia_ret_pct=ms, min_vol=mv)
    r = run_session_follow(df, c)
    print("min_ret>=%.2f%% min_vol=%.1f : %s" % (ms, mv, fmt(r["stats"])))

print("\n--- 参数扫描: stop_mult / rr (min_ret=0.15 min_vol=6) ---")
for sm, rr in [(1.5, 1.5), (2.0, 2.0), (2.5, 2.0), (2.0, 3.0), (3.0, 3.0)]:
    c = SessionFollowConfig(min_asia_ret_pct=0.15, min_vol=6.0, stop_mult=sm, rr=rr)
    r = run_session_follow(df, c)
    print("stop=%s rr=%s : %s" % (sm, rr, fmt(r["stats"])))

# per-side breakdown for the best config
c = SessionFollowConfig(min_asia_ret_pct=0.15, min_vol=6.0, stop_mult=2.0, rr=2.0)
r = run_session_follow(df, c)
tr = r["trades"]
print("\n--- 最佳配置 (min_ret=0.15 min_vol=6 stop=2 rr=2) 明细 ---")
print("总: %s" % fmt(r["stats"]))
if len(tr):
    print("  按方向:")
    for side in ["long", "short"]:
        tt = tr[tr["side"] == side]
        if len(tt):
            print("    %-5s n=%-3d win%%=%.0f net$%+.0f" % (
                side, len(tt), (tt["pnl"] > 0).mean() * 100, tt["pnl"].sum()))
    print("  按离场原因:")
    for reason, tt in tr.groupby("reason"):
        print("    %-11s n=%-3d win%%=%.0f net$%+.0f" % (
            reason, len(tt), (tt["pnl"] > 0).mean() * 100, tt["pnl"].sum()))
