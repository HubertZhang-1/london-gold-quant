# -*- coding: utf-8 -*-
"""Reporting for the bidirectional ATR-grid proxy backtester."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .grid_replica_data import DataAudit
from .grid_replica_engine import BacktestResult


def compute_grid_stats(
    result: BacktestResult,
    initial_balance: float = 100_000.0,
) -> dict:
    """Compute summary statistics for one grid backtest result."""
    trades = result.trades
    events = result.events
    equity = result.equity

    if len(trades):
        gross_profit = float(trades.loc[trades["pnl_usc"] > 0, "pnl_usc"].sum())
        gross_loss = abs(float(trades.loc[trades["pnl_usc"] < 0, "pnl_usc"].sum()))
    else:
        gross_profit = 0.0
        gross_loss = 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (0.0 if gross_profit == 0 else float("inf"))

    max_gross_lots = float(events["gross_lots"].max()) if len(events) else 0.0
    max_abs_net_lots = float(events["net_lots"].abs().max()) if len(events) else 0.0
    hedge_count = int((events["event"] == "hedge_open").sum()) if len(events) else 0

    return {
        "initial_balance": initial_balance,
        "final_balance": float(result.stats["final_balance"]),
        "final_equity": float(result.stats["final_equity"]),
        "total_return_pct": (
            (float(result.stats["final_equity"]) / initial_balance - 1.0) * 100.0
            if initial_balance > 0
            else 0.0
        ),
        "max_drawdown_pct": float(equity["drawdown_pct"].max()) * 100.0 if len(equity) else 0.0,
        "profit_factor": profit_factor,
        "closed_trades": int(len(trades)),
        "max_gross_lots": max_gross_lots,
        "max_abs_net_lots": max_abs_net_lots,
        "hedge_count": hedge_count,
        "terminal_reason": str(result.stats.get("terminal_reason") or ""),
    }


def reconcile_result(
    result: BacktestResult,
    initial_balance: float = 100_000.0,
) -> list[str]:
    """Verify that closed PnL plus open PnL reconciles to final equity."""
    errors: list[str] = []
    closed_pnl = float(result.trades["pnl_usc"].sum()) if len(result.trades) else 0.0
    open_pnl = (
        float(result.final_positions["unrealized_pnl_usc"].sum())
        if len(result.final_positions)
        else 0.0
    )
    final_equity = float(result.equity.iloc[-1]["equity"]) if len(result.equity) else initial_balance
    expected = initial_balance + closed_pnl + open_pnl
    if abs(expected - final_equity) > 1e-6:
        errors.append(
            f"equity mismatch: expected {expected:.6f} but final equity is {final_equity:.6f}"
        )
    if abs(float(result.stats["final_equity"]) - final_equity) > 1e-6:
        errors.append(
            f"stats final_equity {result.stats['final_equity']:.6f} differs from equity frame {final_equity:.6f}"
        )
    return errors


def write_grid_reports(
    results: list[BacktestResult],
    audit: DataAudit,
    config: dict,
    prefix: Path,
) -> list[Path]:
    """Write the summary markdown, scenario CSV and per-scenario event CSVs."""
    prefix = Path(prefix)
    written: list[Path] = []
    symbol = str(config.get("symbol", "GC=F"))
    proxy_for = str(config.get("proxy_for", "XAUUSD.c"))

    rows = []
    for result in results:
        stats = compute_grid_stats(result)
        rows.append(
            {
                "scenario": result.scenario,
                **{key: stats[key] for key in (
                    "initial_balance",
                    "final_balance",
                    "final_equity",
                    "total_return_pct",
                    "max_drawdown_pct",
                    "profit_factor",
                    "closed_trades",
                    "max_gross_lots",
                    "max_abs_net_lots",
                    "hedge_count",
                    "terminal_reason",
                )},
            }
        )
    summary = pd.DataFrame(rows)

    md_path = prefix.with_suffix(".md")
    interval = str(config.get("interval") or config.get("data", {}).get("interval", "5m"))
    md_lines = [
        f"# {symbol}：双向 ATR 网格回测",
        "",
        f"> **数据身份：{symbol}，{interval} 粒度，非经纪商 {proxy_for} 实时报价。本文结果只能用于策略工具初检。**",
        "",
        "## 数据审计",
        "",
        f"- 数据粒度：{interval}；输入行数：{audit.rows_in}；输出行数：{audit.rows_out}",
        f"- 重复时间戳：{audit.duplicate_rows}；缺失间隔（审计固定按 5 分钟基准估算，仅对 5m 数据有意义）：{audit.missing_intervals}",
        f"- 覆盖范围：{audit.first_timestamp} 至 {audit.last_timestamp}",
        "",
        "## 场景结果",
        "",
    ]
    if len(summary):
        md_lines.append(summary.to_markdown(index=False))
    else:
        md_lines.append("（无场景结果）")
    md_lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 事件驱动逐笔模拟：单根 K 线内按 O-H-L-C 或 O-L-H-C 路径连续穿越网格价位。",
            "- 网格步长由前一根已收盘 K 线的 Wilder ATR × 系数计算，并在 min/max 之间截断。",
            "- 双向网格：涨时向下加空单网格，跌时向上加多单网格；方向/全局止盈、对冲、日亏损熔断、最大回撤熔断均按配置生效。",
            "- GC=F 的交易时段、基差和成本结构不同于场外 XAUUSD，实盘前需用目标经纪商 Bid/Ask 与真实滑点复核。",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    written.append(md_path)

    scenarios_path = prefix.with_name(f"{prefix.stem}_scenarios.csv")
    summary.to_csv(scenarios_path, index=False, encoding="utf-8-sig")
    written.append(scenarios_path)

    for result in results:
        scenario = result.scenario.replace(" ", "_").replace("/", "_")
        events_path = prefix.with_name(f"{prefix.stem}_{scenario}_events.csv")
        result.events.to_csv(events_path, index=False, encoding="utf-8-sig")
        written.append(events_path)

    return written
