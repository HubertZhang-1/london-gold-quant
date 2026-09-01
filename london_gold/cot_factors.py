# -*- coding: utf-8 -*-
"""CFTC COT 5-factor voting model (medium-period timing layer), from Guosen report.

Five independent factors from COMEX gold COT disaggregated positions:
  1. MgrMny long / total         (managed-money speculative sentiment)
  2. ProdMerc short / total      (hedger lock breadth)
  3. top-4 short concentration    (short dispersion/concentration) - approximated
     by ProdMerc+Swap short share
  4. prod net position (long-short)  (hedger stress / bottom signal)
  5. OtherRepo long change        (semi-institutional marginal momentum)

Vote rule: each factor above its threshold votes +1 (bullish), below its
threshold votes -1 (bearish); net votes >= min_votes -> long, <= -min_votes -> short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cot_factors(cot: pd.DataFrame, window: int = 104) -> pd.DataFrame:
    """Build the 5 factor values (each column) on a weekly COT frame.

    Input columns (long/short by category) — CFTC disaggregated naming:
      Prod_Merc_*_Long/Short_All, Swap_*_Long/Short_All, M_Money_*_Long/Short_All,
      Other_Rept_*_Long/Short_All, NonRept_*_Long/Short_All, plus
      Conc_Gross_LE_4_TDR_Short_All (top-4 short concentration).
    Returns a frame with 'date' plus one factor column per model factor.
    """
    out = cot.copy().sort_values("date").reset_index(drop=True)

    def ratio(num, den):
        return num / (den + 1e-9)

    def get_col(*candidates):
        for cand in candidates:
            if cand in out.columns:
                return cand
        # fallback: fuzzy match by prefix
        for c in out.columns:
            for cand in candidates:
                prefix = cand.split("_Long")[0].split("_Short")[0]
                if prefix and prefix in c and ("_All" in c or c.endswith("_All")):
                    return c
        return None

    prod_l = get_col("Prod_Merc_Positions_Long_All")
    prod_s = get_col("Prod_Merc_Positions_Short_All")
    swap_l = get_col("Swap_Positions_Long_All")
    swap_s = get_col("Swap_Positions_Short_All", "Swap__Positions_Short_All")
    mgr_l = get_col("M_Money_Positions_Long_All")
    mgr_s = get_col("M_Money_Positions_Short_All")
    oth_l = get_col("Other_Rept_Positions_Long_All")
    oth_s = get_col("Other_Rept_Positions_Short_All")
    non_l = get_col("NonRept_Positions_Long_All")
    tot_l = get_col("Tot_Rept_Positions_Long_All")
    tot_s = get_col("Tot_Rept_Positions_Short_All")
    conc4 = next((c for c in out.columns if "Conc_Gross_LE_4_TDR_Short" in c), None)

    def get(name):
        return out[name] if name else pd.Series(0.0, index=out.index)

    prod_l, prod_s = get(prod_l), get(prod_s)
    swap_l, swap_s = get(swap_l), get(swap_s)
    mgr_l, mgr_s = get(mgr_l), get(mgr_s)
    oth_l = get(oth_l)
    tot_l = get(tot_l) if tot_l else (prod_l + swap_l + mgr_l + oth_l)
    tot_s = get(tot_s) if tot_s else (prod_s + swap_s + mgr_s)

    total_long = tot_l + get(non_l)
    total_short = tot_s

    f1 = ratio(mgr_l, total_long)                     # speculative long share
    f2 = ratio(prod_s, total_short)                   # hedger short share
    # top-4 short concentration: use hedger+swap short share (robust proxy for
    # concentrated shorts). Avoid the raw absolute contract count.
    f3 = ratio(prod_s + swap_s, total_short)
    f4 = prod_l - prod_s                              # producer net (contracts)
    f5 = oth_l.diff().fillna(0.0)                     # other-reportable long change

    out = out.assign(f_spec_long_share=f1, f_hedger_short=f2,
                     f_short_concentration=f3, f_prod_net=f4,
                     f_other_long_chg=f5)
    return out


def cot_5factor_vote_factors(factors: pd.DataFrame, window: int = 104) -> pd.DataFrame:
    """Turn raw factor columns into [-1,1] direction contributions.

    Each contribution is +1 bullish (gold), -1 bearish, 0 neutral, based on
    whether the factor is above/below its rolling median (or threshold).
    """
    out = factors.copy()
    med = out[["f_spec_long_share", "f_hedger_short", "f_short_concentration",
               "f_prod_net", "f_other_long_chg"]].rolling(window, min_periods=int(window * 0.7)).median()

    # Managed-money long share: high = sentiment extended (tend to fade -> bearish)
    out["v_spec_long_share"] = np.sign(med["f_spec_long_share"] - out["f_spec_long_share"])
    # Hedger short share: high hedger short = heavy hedging, often top -> bearish
    out["v_hedger_short"] = -np.sign(out["f_hedger_short"] - med["f_hedger_short"])
    # Short concentration: high concentration -> squeeze potential (bullish)
    out["v_short_concentration"] = np.sign(out["f_short_concentration"] - med["f_short_concentration"])
    # Producer net: very negative = hedger panic -> bottom (bullish); very positive -> top (bearish)
    out["v_prod_net"] = np.sign(med["f_prod_net"] - out["f_prod_net"])
    # Other-reportable long change: rising semi-institutional longs -> bullish
    out["v_other_long_chg"] = np.sign(out["f_other_long_chg"])  # rising longs = +1

    for c in ("v_spec_long_share", "v_hedger_short", "v_short_concentration",
              "v_prod_net", "v_other_long_chg"):
        out[c] = np.nan_to_num(out[c], nan=0.0)
    return out


def cot_vote_signal(votes: pd.DataFrame, min_votes: int = 2) -> pd.Series:
    """Combine the 5 votes into a composite timing signal (-1/0/+1)."""
    vcols = ["v_spec_long_share", "v_hedger_short", "v_short_concentration",
             "v_prod_net", "v_other_long_chg"]
    net = votes[vcols].sum(axis=1)
    sig = np.where(net >= min_votes, 1, np.where(net <= -min_votes, -1, 0))
    return pd.Series(sig, index=votes.index)


def cot_timing_score(votes: pd.DataFrame) -> pd.Series:
    """Continuous [-1,1] timing score = net votes / total votes."""
    vcols = ["v_spec_long_share", "v_hedger_short", "v_short_concentration",
             "v_prod_net", "v_other_long_chg"]
    return (votes[vcols].sum(axis=1) / len(vcols)).clip(-1, 1)
