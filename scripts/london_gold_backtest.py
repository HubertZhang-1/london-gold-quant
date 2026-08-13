# -*- coding: utf-8 -*-
"""London gold strategy backtest CLI.

Usage:
    py311 scripts/london_gold_backtest.py [--update] [--quick] [--start 2015-01-01]
"""
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
from london_gold.data import fetch_daily
from london_gold.report import format_table, stats_row, write_equity_svg
from london_gold.strategies import (
    donchian_breakout_signals,
    ema_cross_signals,
    rsi_reversal_signals,
)

STRATEGY_FUNCS = {
    "donchian_breakout": donchian_breakout_signals,
    "ema_cross": ema_cross_signals,
    "rsi_reversal": rsi_reversal_signals,
}


def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def label_for(name: str, params: dict) -> str:
    stop = params.get("stop_mult", 0)
    if name == "donchian_breakout":
        return f"DON({params['entry_n']},{params['exit_n']},MA{params.get('ma_filter', 0)},S{stop:g})"
    if name == "ema_cross":
        return f"EMA({params['fast_n']},{params['slow_n']},S{stop:g})"
    if name == "rsi_reversal":
        return f"RSI({params['rsi_n']},{params['oversold']}/{params['overbought']},MA{params.get('ma_filter', 0)},S{stop:g})"
    return f"{name}({params})"


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="London gold strategy backtest")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "london_gold_config.json")
    parser.add_argument("--update", action="store_true", help="force refresh daily data")
    parser.add_argument("--quick", action="store_true", help="only run default params")
    parser.add_argument("--start", default=None, help="backtest start date, e.g. 2015-01-01")
    parser.add_argument("--out", default=None, help="report prefix under reports/")
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
    )

    df = fetch_daily(force=args.update, cache_path=data_cfg["cache"])
    start = pd.Timestamp(args.start or data_cfg.get("start", "2010-01-01"))
    df = df[df["date"] >= start].reset_index(drop=True)
    if len(df) < 260:
        raise SystemExit(f"not enough bars after {start.date()}: {len(df)}")

    print(f"London gold data: {len(df)} bars  {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    results = []
    for name, cfg in config["strategies"].items():
        func = STRATEGY_FUNCS[name]
        params_list = [cfg["default"]] if args.quick else expand_grid(cfg.get("grid", {}))
        for params in params_list:
            signaled = func(df, **params)
            result = run_backtest(signaled, cost=cost, name=name, params=params)
            result["stats"]["label"] = label_for(name, params)
            results.append(result)

    results.sort(key=lambda r: (r["stats"]["sharpe"], r["stats"]["total_return"]), reverse=True)

    headers = ["策略", "总收益%", "年化%", "夏普", "最大回撤%", "交易数", "胜率%", "盈亏比"]
    table = [stats_row(r["stats"], r["stats"]["label"]) for r in results]
    print()
    print(format_table(headers, table))

    today = datetime.now().strftime("%Y%m%d")
    prefix = args.out or f"london_gold_{today}"
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = report_dir / prefix

    grid_rows = []
    for r in results:
        s = r["stats"]
        grid_rows.append(
            {
                "strategy": s["name"],
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
    best_trades = best["trades"]
    best_trades.to_csv(f"{prefix_path}_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": best["dates"], "equity": best["equity"]}).to_csv(
        f"{prefix_path}_equity.csv", index=False, encoding="utf-8-sig"
    )

    top5 = results[:5]
    if top5:
        write_equity_svg(
            [(r["stats"]["label"], r["dates"], r["equity"]) for r in top5],
            f"{prefix_path}_equity.svg",
        )

    md = build_markdown(df, cost, results, best, prefix)
    Path(str(prefix_path) + ".md").write_text(md, encoding="utf-8")
    print(f"\nreports written to {prefix_path}.md / _grid.csv / _trades.csv / _equity.svg")
    return 0


def build_markdown(df: pd.DataFrame, cost: CostConfig, results: list[dict], best: dict, prefix: str) -> str:
    headers = ["策略", "总收益%", "年化%", "夏普", "最大回撤%", "交易数", "胜率%", "盈亏比"]
    top = [stats_row(r["stats"], r["stats"]["label"]) for r in results[:10]]
    worst = [stats_row(r["stats"], r["stats"]["label"]) for r in results[-5:]]
    best_s = best["stats"]

    lines = [
        f"# 伦敦金回测报告 {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- 数据区间: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}，共 {len(df)} 根日线",
        f"- 合约: XAU（伦敦金），计价 USD/oz",
        f"- 成本: 点差 ${cost.spread}，滑点 ${cost.slippage}，手续费 ${cost.commission_per_oz}/oz",
        f"- 资金: ${cost.capital:,.0f}，每次 ${cost.position_oz:g} oz",
        "",
        "## Top 10",
        "",
        format_table(headers, top),
        "",
        "## Worst 5",
        "",
        format_table(headers, worst),
        "",
        "## 最优策略",
        "",
        f"**{best_s['label']}**",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 总收益 | {best_s['total_return']:.2f}% |",
        f"| 年化收益 | {best_s['annual_return']:.2f}% |",
        f"| 夏普比率 | {best_s['sharpe']:.2f} |",
        f"| 最大回撤 | {best_s['max_drawdown']:.2f}% |",
        f"| 交易次数 | {best_s['trade_count']} |",
        f"| 胜率 | {best_s['win_rate']:.1f}% |",
        f"| 盈亏比 | {best_s['profit_factor']:.2f} |",
        f"| 平均单笔 | ${best_s['avg_trade']:.2f} |",
        f"| 期末权益 | ${best_s['final_equity']:,.2f} |",
        "",
        "## 输出文件",
        "",
        f"- 参数网格: `reports/{prefix}_grid.csv`",
        f"- 交易明细: `reports/{prefix}_trades.csv`",
        f"- 净值曲线: `reports/{prefix}_equity.csv` / `reports/{prefix}_equity.svg`",
        "",
        "> 免责声明: 回测基于历史数据，不代表未来收益；参数网格存在过拟合风险，实盘前请用样本外数据复核。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
