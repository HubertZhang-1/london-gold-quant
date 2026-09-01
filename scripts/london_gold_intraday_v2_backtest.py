# -*- coding: utf-8 -*-
"""Second-generation intraday gold strategy backtest CLI (1h GC=F)."""
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
from london_gold.intraday_strategies_v2 import (
    combine_ensemble,
    momentum_trend_signals,
    session_breakout_signals,
    zscore_reversion_signals,
)
from london_gold.report import format_table, stats_row, write_equity_svg

FUNCS = {
    "session_breakout": session_breakout_signals,
    "momentum_trend": momentum_trend_signals,
    "zscore_reversion": zscore_reversion_signals,
}


def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def label_for(name: str, params: dict) -> str:
    if name == "session_breakout":
        return f"SES({params['session']},{params['range_bars']},S{params['stop_mult']:g},MA{params.get('ma_filter', 0)},{'ATR' if params.get('atr_filter', True) else 'RAW'})"
    if name == "momentum_trend":
        return f"MOM({params['fast_bars']},{params['slow_bars']},MA{params.get('ma_filter', 0)},S{params['stop_mult']:g})"
    if name == "zscore_reversion":
        return f"Z({params['window']},{params['entry_z']},MA{params.get('ma_filter', 0)},S{params['stop_mult']:g})"
    return f"{name}({params})"


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_cost(config: dict, args) -> CostConfig:
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
    return cost


def main() -> int:
    parser = argparse.ArgumentParser(description="Second-generation intraday gold backtest")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "london_gold_intraday_v2_config.json")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--risk-pct", dest="risk_pct", type=float, default=None)
    parser.add_argument("--margin-call-pct", dest="margin_call_pct", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    cost = build_cost(config, args)

    df = fetch_intraday(
        force=args.update,
        interval=data_cfg.get("interval", "1h"),
        period=data_cfg.get("period", "2y"),
        cache_path=data_cfg["cache"],
    )
    start = pd.Timestamp(args.start or data_cfg.get("start", "2024-08-01"))
    df = df[df["date"] >= start].reset_index(drop=True)
    if args.end:
        df = df[df["date"] <= pd.Timestamp(args.end)].reset_index(drop=True)
    if len(df) < 500:
        raise SystemExit(f"not enough bars after {start.date()}: {len(df)}")
    print(f"Intraday gold data: {len(df)} bars  {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    results = []
    for name, cfg in config["strategies"].items():
        func = FUNCS[name]
        params_list = [cfg["default"]] if args.quick else expand_grid(cfg.get("grid", {}))
        for params in params_list:
            signaled = func(df, **params)
            result = run_backtest(signaled, cost=cost, name=name, params=params, reentry_after_stop=False)
            result["stats"]["label"] = label_for(name, params)
            results.append(result)

    ensemble_cfg = config.get("ensemble") or {}
    if ensemble_cfg.get("use_defaults", True):
        frames = [FUNCS[name](df, **config["strategies"][name]["default"]) for name in ensemble_cfg["members"]]
        combined = combine_ensemble(frames, min_votes=int(ensemble_cfg.get("min_votes", 2)))
        result = run_backtest(combined, cost=cost, name="ensemble", params={"min_votes": 2}, reentry_after_stop=False)
        result["stats"]["label"] = "ENS(2/3)"
        results.append(result)

    results.sort(key=lambda r: (r["stats"]["sharpe"], r["stats"]["total_return"]), reverse=True)
    headers = ["策略", "总收益%", "年化%", "夏普", "最大回撤%", "交易数", "胜率%", "盈亏比"]
    print()
    print(format_table(headers, [stats_row(r["stats"], r["stats"]["label"]) for r in results]))

    prefix = args.out or f"gc_h1_v2_{datetime.now().strftime('%Y%m%d')}"
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = report_dir / prefix

    rows = []
    for r in results:
        s = r["stats"]
        rows.append(
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
    pd.DataFrame(rows).to_csv(f"{prefix_path}_grid.csv", index=False, encoding="utf-8-sig")

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

    md = [
        f"# 日内策略 V2 回测 {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- 数据: GC=F {data_cfg.get('interval', '1h')}，{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}，共 {len(df)} 根",
        f"- 成本: 点差 ${cost.spread}，滑点 ${cost.slippage}，手续费 ${cost.commission_per_oz}/oz",
        f"- 资金: ${cost.capital:,.0f}，杠杆 {cost.leverage:g}x，单笔风险 {cost.risk_per_trade_pct * 100:.0f}%，强平线 {cost.margin_call_pct * 100:.0f}%",
        "",
        "## 结果",
        "",
        format_table(headers, [stats_row(r["stats"], r["stats"]["label"]) for r in results]),
        "",
        "## 策略说明",
        "",
        "- SES: 伦敦/纽约时段开盘区间突破，加 ATR 波动率过滤",
        "- MOM: 双动量（快/慢 ROC）+ 均线趋势过滤",
        "- Z: 波动率标准化均值回归（Z-score），只沿趋势方向逆势入场",
        "- ENS: 三策略 2/3 投票合成",
        "",
        "> GC=F 是 COMEX 黄金期货代理；实盘应以经纪商 XAUUSD 真实点差和隔夜利息复核。",
        "",
    ]
    Path(str(prefix_path) + ".md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nreports written to {prefix_path}.md / _grid.csv / _trades.csv / _equity.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
