# -*- coding: utf-8 -*-
"""Out-of-sample backtest of the reverse-engineered hedged martingale-grid EA."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.martingale_grid import MartingaleConfig, run_martingale_backtest  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest reverse-engineered martingale grid EA")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "XAUUSD_5m_2026.csv"))
    parser.add_argument("--atr-mult", type=float, default=1.0)
    parser.add_argument("--tp-atr", type=float, default=1.0)
    parser.add_argument("--hedge-atr", type=float, default=1.6)
    parser.add_argument("--max-layers", type=int, default=6)
    parser.add_argument("--balance", type=float, default=100_000.0)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "martingale_grid_5m"))
    args = parser.parse_args()

    df = load(Path(args.csv))
    config = MartingaleConfig(
        initial_balance_usc=args.balance,
        grid_atr_mult=args.atr_mult,
        take_profit_atr=args.tp_atr,
        hedge_atr=args.hedge_atr,
        max_layers=args.max_layers,
    )
    result = run_martingale_backtest(df, config)
    s = result.stats

    print("=" * 62)
    print("HEDGED MARTINGALE GRID  (reverse-engineered EA)")
    print("=" * 62)
    print(f"balance / grid step / tp / hedge / layers : "
          f"${config.initial_balance_usc:,.0f} / ATRx{args.atr_mult:.2f} / "
          f"{args.tp_atr:.2f} / {args.hedge_atr:.2f} / {args.max_layers}")
    print("-" * 62)
    print(f"final equity   : ${s['final_equity']:,.2f}  ({s['total_return_pct']:+.1f}%)")
    print(f"trades / win%  : {s['trades']} / {s['winrate']*100:.1f}%")
    print(f"net pnl        : ${s['net_pnl']:+,.2f}")
    print(f"avg win / loss : ${s['avg_win']:+,.2f} / ${s['avg_loss']:+,.2f}")
    print(f"profit factor  : {s['profit_factor']:.2f}")
    print(f"max drawdown   : {s['max_drawdown_pct']:.1f}%")
    print(f"scalps vs hedge: {int(s['trades']-s['hedge_trades'])} / {s['hedge_trades']}")
    print(f"terminal       : {s['terminal_reason'] or 'none'}")
    print(f"[benchmark] buy&hold: {(float(df.iloc[-1]['close'])/float(df.iloc[0]['close'])-1)*100:+.1f}%")

    # save artifacts
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out.with_name(out.stem + "_trades.csv"), index=False, encoding="utf-8-sig")
    result.equity.to_csv(out.with_name(out.stem + "_equity.csv"), index=False, encoding="utf-8-sig")
    print(f"wrote {out}_trades.csv / {out}_equity.csv")


if __name__ == "__main__":
    main()
