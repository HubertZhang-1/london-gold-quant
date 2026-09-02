# -*- coding: utf-8 -*-
"""Analyse whether the Asian session and European (London) session trends agree.

Uses the project's UTC session definition (intraday_strategies_v3.SESSIONS_UTC):
    asia   = hour in [0,7)
    london = hour in [7,13)
For each UTC trading day, it fits a linear regression (OLS slope) on the 1m close
over the Asian window and over the London window, then measures:
  - how often the two slopes agree in SIGN (direction) — the 'same-trend' rate
  - the correlation of the two session returns / slopes
  - Asian-session volatility (which the user wants to use in the strategy)

Data: XAUUSD_1m_202607.csv + XAUUSD_1m_202608.csv (58,737 bars).
Usage: py scripts/session_trend_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ASIA = (0, 7)      # UTC hours
LONDON = (7, 13)   # UTC hours


def load_months():
    parts = []
    for f in ("XAUUSD_1m_202607.csv", "XAUUSD_1m_202608.csv"):
        p = PROJECT_ROOT / "data" / f
        if p.is_file():
            d = pd.read_csv(p)
            d["date"] = pd.to_datetime(d["date"], utc=True)
            parts.append(d)
    df = pd.concat(parts).sort_values("date").reset_index(drop=True)
    return df


def window_slope(df, lo, hi, cols, price="close"):
    """OLS slope of price vs time index over rows in [lo,hi) hour window, per day."""
    sub = df[(df["hour"] >= lo) & (df["hour"] < hi)]
    out = {}
    for col in cols:
        x = np.arange(len(sub), dtype=float)
        y = sub[price].to_numpy(dtype=float)
        if len(x) >= 3:
            denom = (x - x.mean()) @ (x - x.mean())
            slope = ((x - x.mean()) @ (y - y.mean())) / denom if denom > 0 else 0.0
        else:
            slope = np.nan
        out[col] = slope
    out["prev_close"] = sub["prev_close"].iloc[0] if len(sub) else np.nan
    out["ret"] = sub["price_ret"].sum() if len(sub) else np.nan
    out["vol"] = sub["close"].std() if len(sub) else np.nan
    out["_date"] = sub["date"].iloc[0].date() if len(sub) else None
    return out


def main():
    df = load_months()
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.date
    df["prev_close"] = df["close"].shift(1)
    df["price_ret"] = df["close"].pct_change().fillna(0.0)

    daily = df.groupby("day")
    rows = []
    for day, g in daily:
        a = window_slope(g, *ASIA, cols=["slope"], price="close")
        e = window_slope(g, *LONDON, cols=["slope"], price="close")
        rows.append({
            "date": a["_date"] or e["_date"],
            "asia_slope": a["slope"], "asias_ret": a["ret"], "asia_vol": a["vol"],
            "lon_slope": e["slope"], "lon_ret": e["ret"], "lon_vol": e["vol"],
        })
    R = pd.DataFrame(rows).dropna()

    # same-sign rate + correlation
    asia_sign = np.sign(R["asia_slope"])
    lon_sign = np.sign(R["lon_slope"])
    same_sign = (asia_sign == lon_sign)
    # skip near-flat
    active = (R["asia_slope"].abs() > 1e-6) & (R["lon_slope"].abs() > 1e-6)
    same_active = same_sign[active]

    corr_slope = R["asia_slope"].corr(R["lon_slope"])
    corr_ret = R["asias_ret"].corr(R["lon_ret"])

    print("=" * 66)
    print("亚洲盘(UTC 0-7) vs 欧洲盘/伦敦(UTC 7-13) 线性回归趋势对比")
    print("=" * 66)
    print(f"样本天数   : {len(R)}")
    print(f"亚洲盘斜率同号率 : {same_sign.mean()*100:.1f}%   (active {same_active.mean()*100:.1f}%)")
    print(f"斜率相关性   : r={corr_slope:.3f}")
    print(f"区间收益相关性: r={corr_ret:.3f}")
    print("-" * 66)
    print("亚洲盘波动 (日内std, 美元):")
    print(f"  均值 {R['asia_vol'].mean():.2f}  中位 {R['asia_vol'].median():.2f}  "
          f"最高 {R['asia_vol'].max():.2f}  最低 {R['asia_vol'].min():.2f}")
    print("欧洲盘波动 (日内std, 美元):")
    print(f"  均值 {R['lon_vol'].mean():.2f}  中位 {R['lon_vol'].median():.2f}  "
          f"最高 {R['lon_vol'].max():.2f}  最低 {R['lon_vol'].min():.2f}")
    print("-" * 66)
    print("亚洲盘 vs 欧洲盘 同向时段的平均区间收益:")
    up_both = R[(R["asia_slope"] > 0) & (R["lon_slope"] > 0)]
    dn_both = R[(R["asia_slope"] < 0) & (R["lon_slope"] < 0)]
    opp = R[(R["asia_slope"] * R["lon_slope"] < 0)]
    if len(up_both):
        print(f"  双涨 : {len(up_both)} 天, 欧盘平均 {up_both['lon_ret'].mean()*100:+.2f}%")
    if len(dn_both):
        print(f"  双跌 : {len(dn_both)} 天, 欧盘平均 {dn_both['lon_ret'].mean()*100:+.2f}%")
    if len(opp):
        print(f"  背离 : {len(opp)} 天, 欧盘平均 {opp['lon_ret'].mean()*100:+.2f}% (无延续性)")
    print(f"  合计 : 双涨{len(up_both)} 双跌{len(dn_both)} 背离{len(opp)}")


if __name__ == "__main__":
    main()
