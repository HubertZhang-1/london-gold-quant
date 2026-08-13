# -*- coding: utf-8 -*-
"""Intraday gold open-range breakout backtest CLI (Yahoo GC=F 1h data)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.backtest import CostConfig, run_backtest
from london_gold.intraday_data import fetch_intraday
from london_gold.intraday_strategy import open_range_breakout_signals
from london_gold.report import format_table, stats_row, write_equity_svg


def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Intraday gold open-range breakout backtest")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "london_gold_intraday_config.json")
    parser.add_argument("--update", action="store_true", help="force refresh Yahoo data")
    parser.add_argument("--quick", action="store_true", help="only run default params")
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--risk-pct", dest="risk_pct", type=float, default=None)
    parser.add_argument("--margin-call-pct", dest="margin_call_pct", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    cost_cfg = config["costs"]
    cost = CostConfig(
        capital=float(cost_cfg["capital"]),
        position_oz=float(cost_cfg["position_oz"]),
        spread=float(cost_cfg["spread"]),
        slippage=float(cost_cfg["slippage"]),
        commission_per_oz=float(cost_cfg["commission_per_oz"]),
        leverage=float(cost_cfg.get("leverage", 0.0)),
        max_oz=float(cost_cfg.get("max_oz", 0.0)),
        risk_per_trade_pct=float(cost_cfg.get("risk_per_trade_pct", 0.0)),
        margin_call_pct=float(cost_cfg.get("margin_call_pct", 0.0)),
    )
    if args.leverage is not None:
        cost.leverage = args.leverage
    if args.risk_pct is not None:
        cost.risk_per_trade_pct = args.risk_pct
    if args.margin_call_pct is not None:
        cost.margin_call_pct = args.margin_call_pct

    df = fetch_intraday(force=args.update, cache_path=data_cfg["cache"])
    start = pd.Timestamp(data_cfg.get("start", "2024-08-01"))
    df = df[df["date"] >= start].reset_index(drop=True)
    if len(df) < 500:
        raise SystemExit(f"not enough bars after {start.date()}: {len(df)}")

    print(f"Intraday gold data: {len(df)} bars  {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    results = []
    cfg = config["strategies"]["open_range_breakout"]
    params_list = [cfg["default"]] if args.quick else expand_grid(cfg.get("grid", {}))
    for params in params_list:
        signaled = open_range_breakout_signals(df, **params)
        result = run_backtest(signaled, cost=cost, name="open_range_breakout", params=params, reentry_after_stop=False)
        result["stats"]["label"] = f"ORB({params['range_hours']},MA{params.get('ma_filter', 0)},S{params.get('stop_mult', 1.0):g})"
        results.append(result)

    results.sort(key=lambda r: (r["stats"]["sharpe"], r["stats"]["total_return"]), reverse=True)
    headers = ["策略", "总收益%", "年化%", "夏普", "最大回撤%", "交易数", "胜率%", "盈亏比"]
    print()
    print(format_table(headers, [stats_row(r["stats"], r["stats"]["label"]) for r in results]))

    prefix = args.out or f"gc_h1_orb_{datetime.now().strftime('%Y%m%d')}"
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = report_dir / prefix

    grid_rows = []
    for r in results:
        s = r["stats"]
        grid_rows.append(
            {
                "label": s["label"],
                "params": json.dumps(s["params"], ensure_ascii=False),
                "total_return": s["total_return"],
                "annual_return": s["annual_return"],
                "sharpe": s["sharpe"],
                "max_drawdown": s["max_drawdown"],
                "trade_count": s["trade_count"],
                "win_rate": s["win_rate"],
                "profit_factor": s["profit_factor"],
                "final_equity": s["final_equity"],
            }
        )
    pd.DataFrame(grid_rows).to_csv(f"{prefix_path}_grid.csv", index=False, encoding="utf-8-sig")

    best = results[0]
    best["trades"].to_csv(f"{prefix_path}_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": best["dates"], "equity": best["equity"]}).to_csv(
        f"{prefix_path}_equity.csv", index=False, encoding="utf-8-sig"
    )
    top5 = results[:5]
    if top5:
        write_equity_svg(
            [(r["stats"]["label"], r["dates"], r["equity"]) for r in top5],
            f"{prefix_path}_equity.svg",
        )

    lines = [
        f"# 黄金日内开盘区间突破回测 {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- 数据: GC=F 1小时，{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}，共 {len(df)} 根",
        f"- 成本: 点差 ${cost.spread}，滑点 ${cost.slippage}，手续费 ${cost.commission_per_oz}/oz",
        f"- 资金: ${cost.capital:,.0f}，杠杆 {cost.leverage:g}x，单笔风险 {cost.risk_per_trade_pct * 100:.0f}%，强平线 {cost.margin_call_pct * 100:.0f}%",
        "",
        "## 结果",
        "",
        format_table(headers, [stats_row(r["stats"], r["stats"]["label"]) for r in results]),
        "",
        "> GC=F 是 COMEX 黄金期货，作为伦敦金代理；实盘应以经纪商 XAUUSD 点差/隔夜利息复核。",
        "",
    ]
    Path(str(prefix_path) + ".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreports written to {prefix_path}.md / _grid.csv / _trades.csv / _equity.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
