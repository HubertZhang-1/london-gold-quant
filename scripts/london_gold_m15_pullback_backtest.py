# -*- coding: utf-8 -*-
"""Backtest the M15 momentum-pullback strategy on a selected UTC period."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from london_gold.intraday_data import fetch_intraday
from london_gold.m15_execution import M15ExecutionConfig, run_m15_backtest
from london_gold.m15_pullback import M15StrategyConfig, momentum_pullback_signals


def prepare_period(
    df: pd.DataFrame,
    start: str,
    end: str,
    warmup_bars: int = 64,
) -> dict[str, pd.DataFrame]:
    """Return the evaluation bars plus the warmup input required by indicators."""
    start_at = pd.Timestamp(start, tz="UTC")
    end_at = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    if end_at <= start_at:
        raise ValueError("end must not be before start")

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=True)
    normalized = normalized.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    evaluation_mask = (normalized["date"] >= start_at) & (normalized["date"] < end_at)
    positions = normalized.index[evaluation_mask]
    if len(positions) == 0:
        raise ValueError(f"no bars in requested period {start} through {end}")
    input_start = max(0, int(positions[0]) - int(warmup_bars))
    input_end = int(positions[-1]) + 1
    input_frame = normalized.iloc[input_start:input_end].reset_index(drop=True)
    evaluation = normalized.loc[evaluation_mask].reset_index(drop=True)
    return {"input": input_frame, "evaluation": evaluation}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _format_metric(value) -> str:
    if value == float("inf"):
        return "∞"
    return str(value)


def write_report(
    prefix: Path,
    source: pd.DataFrame,
    signaled: pd.DataFrame,
    result: dict,
    start: str,
    end: str,
    strategy_config: M15StrategyConfig,
    execution_config: M15ExecutionConfig,
) -> None:
    trades_path = Path(str(prefix) + "_trades.csv")
    equity_path = Path(str(prefix) + "_equity.csv")
    skipped_path = Path(str(prefix) + "_skipped.csv")
    report_path = Path(str(prefix) + ".md")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    result["trades"].to_csv(trades_path, index=False, encoding="utf-8-sig")
    result["skipped"].to_csv(skipped_path, index=False, encoding="utf-8-sig")
    equity = pd.DataFrame({"date": result["dates"], "equity": result["equity"]})
    start_at = pd.Timestamp(start, tz="UTC")
    end_at = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    equity = equity[(equity["date"] >= start_at) & (equity["date"] < end_at)]
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")

    stats = result["stats"]
    confirmed = int((signaled["signal"] != 0).sum())
    lines = [
        f"# GC=F 15分钟动量回踩代理回测：{start} 至 {end}",
        "",
        "> **数据身份：GC=F COMEX 黄金期货代理，不是经纪商 XAUUSD 报价。本文结果只能用于策略工具初检。**",
        "",
        "## 数据与规则",
        "",
        f"- 实际数据覆盖：{source['date'].iloc[0]} 至 {source['date'].iloc[-1]}，评估区间 {len(source)} 根 K 线",
        "- 开仓窗口：伦敦与纽约 08:00–17:00 当地交易时段的动态重叠区间",
        f"- 确认信号：{confirmed} 个；成交：{stats['trade_count']} 笔",
        f"- 初始资金：${execution_config.capital:,.2f}；单笔风险：{execution_config.risk_per_trade * 100:.2f}%",
        f"- 成本：点差 ${execution_config.spread:.2f}，单边滑点 ${execution_config.slippage:.2f}，单边佣金 ${execution_config.commission_per_oz:.2f}/oz",
        "- 信号在收盘确认，下一根 K 线开盘成交；日内平仓，不隔夜",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 总收益率 | {_format_metric(stats['total_return'])}% |",
        f"| 期末净值 | ${stats['final_equity']:,.2f} |",
        f"| 最大回撤 | {_format_metric(stats['max_drawdown'])}% |",
        f"| 交易数 | {stats['trade_count']} |",
        f"| 胜率 | {_format_metric(stats['win_rate'])}% |",
        f"| 盈利因子 | {_format_metric(stats['profit_factor'])} |",
        f"| 平均每笔 | ${stats['avg_trade']:,.2f} |",
        f"| 每笔期望 | {stats['expectancy_r']:.3f}R |",
        "",
        "## 配置快照",
        "",
        "```json",
        json.dumps(
            {"strategy": asdict(strategy_config), "execution": asdict(execution_config)},
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## 限制",
        "",
        "- 单月样本不能证明策略长期有效，也不能满足设计规格要求的滚动样本外门槛。",
        "- GC=F 的交易时段、基差和成本结构不同于场外 XAUUSD。",
        "- 当前没有目标经纪商 Bid/Ask、真实滑点和重大新闻日历，因此不能据此进入实盘。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="M15 gold momentum-pullback proxy backtest")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "london_gold_m15_pullback_config.json",
    )
    parser.add_argument("--start", default="2026-08-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", default="gc_m15_pullback_202608")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    cache_path = args.cache or PROJECT_ROOT / data_config["cache"]
    raw = fetch_intraday(
        symbol=config["symbol"],
        interval=data_config["interval"],
        period=data_config["period"],
        force=args.update,
        cache_path=cache_path,
    )
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    end_day = pd.Timestamp(args.end, tz="UTC")
    if raw["date"].max().normalize() < end_day:
        raise SystemExit(
            f"incomplete data: latest bar {raw['date'].max()} is before requested end date {args.end}"
        )

    prepared = prepare_period(raw, args.start, args.end, warmup_bars=80)
    strategy_config = M15StrategyConfig(**config["strategy"])
    execution_config = M15ExecutionConfig(**config["execution"])
    signaled = momentum_pullback_signals(prepared["input"], strategy_config)
    start_at = pd.Timestamp(args.start, tz="UTC")
    end_at = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    outside_evaluation = (signaled["date"] < start_at) | (signaled["date"] >= end_at)
    signaled.loc[outside_evaluation, ["signal", "stop_dist"]] = 0
    result = run_m15_backtest(signaled, execution_config)

    report_prefix = PROJECT_ROOT / "reports" / args.out
    write_report(
        report_prefix,
        prepared["evaluation"],
        signaled.loc[~outside_evaluation],
        result,
        args.start,
        args.end,
        strategy_config,
        execution_config,
    )
    print(
        f"Data: {prepared['evaluation']['date'].iloc[0]} ~ {prepared['evaluation']['date'].iloc[-1]} "
        f"({len(prepared['evaluation'])} bars)"
    )
    print(json.dumps(result["stats"], ensure_ascii=False))
    print(f"Reports: {report_prefix}.md and CSV artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
