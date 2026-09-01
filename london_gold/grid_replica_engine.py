# -*- coding: utf-8 -*-
"""Event-driven bidirectional ATR-grid proxy backtester."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

import pandas as pd


@dataclass(frozen=True)
class GridConfig:
    initial_balance_usc: float = 100_000.0
    base_lot: float = 0.02
    lot_increment: float = 0.01
    max_layers_per_side: int = 9
    spread: float = 0.35
    usc_per_price_lot: float = 100.0
    direction_target_fixed: float = 100.0
    direction_target_balance_pct: float = 0.0005
    global_target_fixed: float = 250.0
    global_target_balance_pct: float = 0.0012
    cooldown_seconds: int = 30
    hedge_loss_pct: float = 0.015
    hedge_exposure_ratio: float = 0.30
    hedge_fraction: float = 0.60
    hedge_unlock_loss_pct: float = 0.0075
    lot_step: float = 0.01
    daily_loss_limit: float = 0.08
    max_drawdown: float = 0.20


@dataclass
class Position:
    ticket: int
    cycle_id: int
    kind: str
    side: int
    lots: float
    entry_time: pd.Timestamp
    entry_fill: float
    entry_mid: float
    layer: int
    source_side: int = 0


@dataclass(frozen=True)
class BacktestResult:
    scenario: str
    events: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    final_positions: pd.DataFrame
    stats: dict


def path_nodes(row: pd.Series, path_mode: str) -> tuple[float, ...]:
    """Return deterministic O-H-L-C or O-L-H-C path nodes."""
    if path_mode == "OHLC":
        return float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    if path_mode == "OLHC":
        return float(row["open"]), float(row["low"]), float(row["high"]), float(row["close"])
    raise ValueError(f"unsupported path mode: {path_mode}")


def fill_price(mid: float, order_side: int, spread: float) -> float:
    """Apply half-spread against a market order."""
    return mid + order_side * spread / 2.0


def layer_lots(layer: int, config: GridConfig) -> float:
    return round(config.base_lot + (layer - 1) * config.lot_increment, 2)


def position_unrealized(
    position: Position,
    mid: float,
    spread: float,
    usc_per_price_lot: float,
) -> float:
    exit_fill = fill_price(mid, -position.side, spread)
    return position.side * (exit_fill - position.entry_fill) * position.lots * usc_per_price_lot


def account_equity(
    balance: float,
    positions: list[Position],
    mid: float,
    config: GridConfig,
) -> float:
    return balance + sum(
        position_unrealized(position, mid, config.spread, config.usc_per_price_lot)
        for position in positions
    )


def exposure(positions: list[Position]) -> tuple[float, float]:
    """Return signed net lots and absolute gross lots."""
    net = round(sum(position.side * position.lots for position in positions), 8)
    gross = round(sum(abs(position.lots) for position in positions), 8)
    return net, gross


def floored_lots(raw: float, step: float) -> float:
    """Floor a raw lot size to the broker lot step without float drift."""
    if raw <= 0 or step <= 0:
        return 0.0
    raw_decimal = Decimal(str(raw))
    step_decimal = Decimal(str(step))
    units = (raw_decimal / step_decimal).to_integral_value(rounding=ROUND_FLOOR)
    return float(units * step_decimal)


def _validate_inputs(df: pd.DataFrame, config: GridConfig, path_mode: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "grid_step"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if path_mode not in {"OHLC", "OLHC"}:
        raise ValueError(f"unsupported path mode: {path_mode}")
    if (
        config.initial_balance_usc <= 0
        or config.base_lot <= 0
        or config.lot_increment < 0
        or config.max_layers_per_side <= 0
        or config.spread < 0
        or config.usc_per_price_lot <= 0
        or config.hedge_loss_pct <= 0
        or config.hedge_unlock_loss_pct < 0
        or config.hedge_unlock_loss_pct >= config.hedge_loss_pct
    ):
        raise ValueError("invalid grid configuration")
    data = df.copy().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], utc=True)
    if data.empty or data["grid_step"].isna().any() or (data["grid_step"] <= 0).any():
        raise ValueError("grid_step must be finite and positive")
    return data


def run_grid_backtest(
    df: pd.DataFrame,
    config: GridConfig | None = None,
    path_mode: str = "OHLC",
) -> BacktestResult:
    """Run one deterministic intrabar scenario with continuous event crossings."""
    config = config or GridConfig()
    data = _validate_inputs(df, config, path_mode)
    balance = float(config.initial_balance_usc)
    positions: list[Position] = []
    events: list[dict] = []
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []
    ticket = 0
    cycle_id = 0
    close_group_id = 0
    sequence = 0
    peak_equity = balance
    daily_realized_pnl = 0.0
    day_start_balance = balance
    current_day = None
    cooldown_until: pd.Timestamp | None = None
    paused_sides: set[int] = set()
    terminal_reason = ""
    terminal_time: pd.Timestamp | None = None
    first_cycle = True
    blocked_keys: set[tuple] = set()
    tolerance = 1e-9

    event_columns = [
        "sequence", "time", "event", "reason", "ticket", "cycle_id", "kind", "side",
        "layer", "lots", "mid", "fill", "balance", "equity", "grid_step", "close_group_id",
        "net_lots", "gross_lots", "source_side",
    ]
    trade_columns = [
        "ticket", "cycle_id", "kind", "side", "lots", "entry_time", "exit_time",
        "entry_fill", "exit_fill", "pnl_usc", "holding_seconds", "close_group_id", "reason",
    ]
    final_position_columns = [
        "ticket", "cycle_id", "kind", "side", "layer", "lots", "entry_time", "entry_fill",
        "unrealized_pnl_usc",
    ]

    def unrealized(mid: float, selected: list[Position] | None = None) -> float:
        owned = positions if selected is None else selected
        return sum(
            position_unrealized(position, mid, config.spread, config.usc_per_price_lot)
            for position in owned
        )

    def update_peak(mid: float) -> float:
        nonlocal peak_equity
        equity = account_equity(balance, positions, mid, config)
        peak_equity = max(peak_equity, equity)
        return equity

    def record_event(
        event: str,
        reason: str,
        time: pd.Timestamp,
        mid: float,
        step: float,
        position: Position | None = None,
        fill: float | None = None,
        group_id: int | None = None,
    ) -> None:
        nonlocal sequence
        sequence += 1
        net_lots, gross_lots = exposure(positions)
        events.append(
            {
                "sequence": sequence,
                "time": time,
                "event": event,
                "reason": reason,
                "ticket": position.ticket if position else 0,
                "cycle_id": position.cycle_id if position else cycle_id,
                "kind": position.kind if position else "account",
                "side": ("buy" if position.side > 0 else "sell") if position else "",
                "layer": position.layer if position else 0,
                "lots": position.lots if position else 0.0,
                "mid": mid,
                "fill": fill if fill is not None else mid,
                "balance": balance,
                "equity": account_equity(balance, positions, mid, config),
                "grid_step": step,
                "close_group_id": group_id or 0,
                "net_lots": net_lots,
                "gross_lots": gross_lots,
                "source_side": position.source_side if position else 0,
            }
        )

    def additions_blocked() -> bool:
        daily_loss = max(0.0, -daily_realized_pnl)
        return daily_loss >= day_start_balance * config.daily_loss_limit - tolerance

    def side_positions(side: int) -> list[Position]:
        return [position for position in positions if position.kind == "grid" and position.side == side]

    def open_position(
        side: int,
        layer: int,
        time: pd.Timestamp,
        mid: float,
        step: float,
        reason: str,
        kind: str = "grid",
        lots: float | None = None,
        source_side: int = 0,
    ) -> Position:
        nonlocal ticket
        ticket += 1
        position = Position(
            ticket=ticket,
            cycle_id=cycle_id,
            kind=kind,
            side=side,
            lots=layer_lots(layer, config) if lots is None else lots,
            entry_time=time,
            entry_fill=fill_price(mid, side, config.spread),
            entry_mid=mid,
            layer=layer,
            source_side=source_side,
        )
        positions.append(position)
        event_name = "hedge_open" if kind == "hedge" else "open"
        record_event(event_name, reason, time, mid, step, position, position.entry_fill)
        update_peak(mid)
        return position

    def open_cycle(time: pd.Timestamp, mid: float, step: float) -> bool:
        nonlocal cycle_id, first_cycle
        if terminal_reason or additions_blocked() or (cooldown_until is not None and time < cooldown_until):
            return False
        cycle_id += 1
        reason = "cycle_start" if first_cycle else "cycle_restart"
        first_cycle = False
        open_position(1, 1, time, mid, step, reason)
        open_position(-1, 1, time, mid, step, reason)
        return True

    def close_positions(
        selected: list[Position],
        time: pd.Timestamp,
        mid: float,
        step: float,
        reason: str,
        existing_group_id: int | None = None,
    ) -> int:
        nonlocal balance, daily_realized_pnl, close_group_id
        if not selected:
            return existing_group_id or 0
        if existing_group_id is None:
            close_group_id += 1
            group_id = close_group_id
        else:
            group_id = existing_group_id
        closed: list[tuple[Position, float, float]] = []
        total_pnl = 0.0
        for position in selected:
            exit_fill = fill_price(mid, -position.side, config.spread)
            pnl = (
                position.side
                * (exit_fill - position.entry_fill)
                * position.lots
                * config.usc_per_price_lot
            )
            total_pnl += pnl
            closed.append((position, exit_fill, pnl))
        balance += total_pnl
        daily_realized_pnl += total_pnl
        selected_tickets = {position.ticket for position in selected}
        positions[:] = [position for position in positions if position.ticket not in selected_tickets]
        for position, exit_fill, pnl in closed:
            trade_rows.append(
                {
                    "ticket": position.ticket,
                    "cycle_id": position.cycle_id,
                    "kind": position.kind,
                    "side": "buy" if position.side > 0 else "sell",
                    "lots": position.lots,
                    "entry_time": position.entry_time,
                    "exit_time": time,
                    "entry_fill": position.entry_fill,
                    "exit_fill": exit_fill,
                    "pnl_usc": pnl,
                    "holding_seconds": max(0.0, (time - position.entry_time).total_seconds()),
                    "close_group_id": group_id,
                    "reason": reason,
                }
            )
            event_name = "hedge_close" if position.kind == "hedge" and reason in {
                "hedge_unlock", "orphan_hedge_cleanup"
            } else "close"
            record_event(event_name, reason, time, mid, step, position, exit_fill, group_id)
        update_peak(mid)
        return group_id

    def clean_orphan_hedges(time: pd.Timestamp, mid: float, step: float, group_id: int) -> None:
        while True:
            net_lots, _ = exposure(positions)
            orphan = None
            for hedge in [position for position in positions if position.kind == "hedge"]:
                without = net_lots - hedge.side * hedge.lots
                if abs(net_lots) > abs(without) + tolerance:
                    orphan = hedge
                    break
            if orphan is None:
                break
            close_positions([orphan], time, mid, step, "orphan_hedge_cleanup", group_id)
        active_sources = {position.source_side for position in positions if position.kind == "hedge"}
        paused_sides.intersection_update(active_sources)

    def trigger_terminal(reason: str, time: pd.Timestamp, mid: float, step: float) -> None:
        nonlocal terminal_reason, terminal_time
        if terminal_reason:
            return
        if positions:
            close_positions(list(positions), time, mid, step, reason)
        terminal_reason = reason
        terminal_time = time
        paused_sides.clear()
        record_event("circuit_breaker", reason, time, mid, step)

    def process_immediate(time: pd.Timestamp, mid: float, step: float) -> bool:
        nonlocal cooldown_until
        if terminal_reason:
            return False
        equity = update_peak(mid)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 1.0
        if balance <= 0 or equity <= 0:
            trigger_terminal("bankruptcy", time, mid, step)
            return True
        if drawdown >= config.max_drawdown - tolerance:
            trigger_terminal("max_drawdown", time, mid, step)
            return True

        if positions:
            global_target = max(config.global_target_fixed, balance * config.global_target_balance_pct)
            if unrealized(mid) >= global_target - tolerance:
                close_positions(list(positions), time, mid, step, "global_target")
                paused_sides.clear()
                cooldown_until = time + pd.Timedelta(seconds=config.cooldown_seconds)
                return True

        direction_target = max(config.direction_target_fixed, balance * config.direction_target_balance_pct)
        for side in (1, -1):
            owned = side_positions(side)
            if owned and unrealized(mid, owned) >= direction_target - tolerance:
                group_id = close_positions(owned, time, mid, step, "direction_target")
                clean_orphan_hedges(time, mid, step, group_id)
                return True

        hedges = [position for position in positions if position.kind == "hedge"]
        equity = account_equity(balance, positions, mid, config)
        floating_loss = max(0.0, balance - equity)
        if hedges and floating_loss <= equity * config.hedge_unlock_loss_pct + tolerance:
            close_positions(hedges, time, mid, step, "hedge_unlock")
            paused_sides.clear()
            return True

        net_lots, gross_lots = exposure(positions)
        if gross_lots > 0 and abs(net_lots) > tolerance:
            source_side = 1 if net_lots > 0 else -1
            hedge_side = -source_side
            has_same_hedge = any(
                position.kind == "hedge" and position.side == hedge_side for position in positions
            )
            ratio = abs(net_lots) / gross_lots
            if (
                not has_same_hedge
                and floating_loss >= equity * config.hedge_loss_pct - tolerance
                and ratio >= config.hedge_exposure_ratio - tolerance
            ):
                lots = min(abs(net_lots), floored_lots(abs(net_lots) * config.hedge_fraction, config.lot_step))
                if lots >= config.lot_step - tolerance:
                    open_position(
                        hedge_side,
                        0,
                        time,
                        mid,
                        step,
                        "hedge_trigger",
                        kind="hedge",
                        lots=lots,
                        source_side=source_side,
                    )
                    paused_sides.add(source_side)
                    return True

        if not positions and open_cycle(time, mid, step):
            return True
        return False

    def metric_root(
        current_mid: float,
        end_mid: float,
        current_value: float,
        end_value: float,
        target: float,
    ) -> float | None:
        delta = end_value - current_value
        if abs(delta) <= tolerance:
            return None
        ratio = (target - current_value) / delta
        if ratio <= tolerance or ratio > 1.0 + tolerance:
            return None
        return current_mid + (end_mid - current_mid) * min(1.0, ratio)

    def interpolate_time(
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        start_mid: float,
        end_mid: float,
        event_mid: float,
    ) -> pd.Timestamp:
        if abs(end_mid - start_mid) <= tolerance:
            return end_time
        ratio = abs((event_mid - start_mid) / (end_mid - start_mid))
        return start_time + (end_time - start_time) * ratio

    def add_block_event(side: int, time: pd.Timestamp, mid: float, step: float) -> None:
        key = (time.date(), side)
        if key in blocked_keys:
            return
        blocked_keys.add(key)
        dummy = Position(0, cycle_id, "grid", side, 0.0, time, mid, mid, 0)
        record_event("addition_blocked", "daily_loss_limit", time, mid, step, dummy)

    def traverse_segment(
        start_mid: float,
        end_mid: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        step: float,
    ) -> None:
        current_mid = start_mid
        current_time = start_time
        guard = 0
        while guard < 500 and not terminal_reason:
            guard += 1
            immediate_guard = 0
            while process_immediate(current_time, current_mid, step):
                immediate_guard += 1
                if terminal_reason or immediate_guard >= 100:
                    break
            if terminal_reason:
                break
            if immediate_guard >= 100:
                raise RuntimeError("immediate event loop did not converge")
            if abs(end_mid - current_mid) <= tolerance:
                update_peak(end_mid)
                break

            candidates: list[tuple[float, int, str, float, pd.Timestamp]] = []

            def add_candidate(price: float | None, priority: int, label: str) -> None:
                if price is None:
                    return
                progress = abs((price - current_mid) / (end_mid - current_mid))
                if progress <= tolerance or progress > 1.0 + tolerance:
                    return
                time = interpolate_time(current_time, end_time, current_mid, end_mid, price)
                candidates.append((progress, priority, label, price, time))

            equity_now = account_equity(balance, positions, current_mid, config)
            equity_end = account_equity(balance, positions, end_mid, config)
            drawdown_equity = peak_equity * (1.0 - config.max_drawdown)
            if equity_now > drawdown_equity + tolerance and equity_end <= drawdown_equity + tolerance:
                add_candidate(metric_root(current_mid, end_mid, equity_now, equity_end, drawdown_equity), 0, "risk")
            if equity_now > tolerance and equity_end <= tolerance:
                add_candidate(metric_root(current_mid, end_mid, equity_now, equity_end, 0.0), 0, "risk")

            if positions:
                pnl_now = equity_now - balance
                pnl_end = equity_end - balance
                target = max(config.global_target_fixed, balance * config.global_target_balance_pct)
                if pnl_now < target - tolerance and pnl_end >= target - tolerance:
                    add_candidate(metric_root(current_mid, end_mid, pnl_now, pnl_end, target), 1, "global")

            direction_target = max(config.direction_target_fixed, balance * config.direction_target_balance_pct)
            for side in (1, -1):
                owned = side_positions(side)
                if not owned:
                    continue
                pnl_now = unrealized(current_mid, owned)
                pnl_end = unrealized(end_mid, owned)
                if pnl_now < direction_target - tolerance and pnl_end >= direction_target - tolerance:
                    add_candidate(
                        metric_root(current_mid, end_mid, pnl_now, pnl_end, direction_target),
                        2,
                        f"direction_{side}",
                    )

            hedges = [position for position in positions if position.kind == "hedge"]
            total_now = equity_now - balance
            total_end = equity_end - balance
            if hedges:
                unlock_level = -config.hedge_unlock_loss_pct * balance / (1.0 + config.hedge_unlock_loss_pct)
                if total_now < unlock_level - tolerance and total_end >= unlock_level - tolerance:
                    add_candidate(
                        metric_root(current_mid, end_mid, total_now, total_end, unlock_level),
                        3,
                        "unlock",
                    )
            else:
                net_lots, gross_lots = exposure(positions)
                ratio = abs(net_lots) / gross_lots if gross_lots > 0 else 0.0
                if abs(net_lots) > tolerance and ratio >= config.hedge_exposure_ratio - tolerance:
                    hedge_level = -config.hedge_loss_pct * balance / (1.0 + config.hedge_loss_pct)
                    if total_now > hedge_level + tolerance and total_end <= hedge_level + tolerance:
                        add_candidate(
                            metric_root(current_mid, end_mid, total_now, total_end, hedge_level),
                            3,
                            "hedge",
                        )

            direction = 1 if end_mid > current_mid else -1
            grid_side = -1 if direction > 0 else 1
            owned = side_positions(grid_side)
            if owned and len(owned) < config.max_layers_per_side and grid_side not in paused_sides:
                grid_price = owned[-1].entry_mid + direction * step
                crossed = grid_price <= end_mid + tolerance if direction > 0 else grid_price >= end_mid - tolerance
                if crossed:
                    if additions_blocked():
                        add_block_event(grid_side, current_time, current_mid, step)
                    else:
                        add_candidate(grid_price, 4, f"grid_{grid_side}")

            if not positions and cooldown_until is not None and current_time < cooldown_until <= end_time:
                duration = (end_time - current_time).total_seconds()
                ratio = (cooldown_until - current_time).total_seconds() / duration if duration > 0 else 1.0
                restart_mid = current_mid + (end_mid - current_mid) * ratio
                candidates.append((ratio, 5, "restart", restart_mid, cooldown_until))

            if not candidates:
                current_mid, current_time = end_mid, end_time
                update_peak(current_mid)
                process_immediate(current_time, current_mid, step)
                break

            _, _, label, event_mid, event_time = min(candidates, key=lambda item: (item[0], item[1]))
            current_mid, current_time = event_mid, event_time
            update_peak(current_mid)
            if label.startswith("grid_"):
                side = int(label.split("_")[1])
                owned = side_positions(side)
                open_position(side, len(owned) + 1, current_time, current_mid, step, "grid_add")
            elif label == "restart":
                open_cycle(current_time, current_mid, step)
            else:
                process_immediate(current_time, current_mid, step)
        if guard >= 500:
            raise RuntimeError("segment event loop did not converge")

    for bar_index, row in data.iterrows():
        bar_time = row["date"]
        trading_day = bar_time.date()
        if current_day != trading_day:
            current_day = trading_day
            day_start_balance = balance
            daily_realized_pnl = 0.0
            blocked_keys = {key for key in blocked_keys if key[0] == trading_day}
        step = float(row["grid_step"])
        nodes = path_nodes(row, path_mode)
        node_times = tuple(bar_time + pd.Timedelta(seconds=100 * index) for index in range(4))
        if not positions and not terminal_reason:
            open_cycle(node_times[0], nodes[0], step)
        for index in range(3):
            traverse_segment(nodes[index], nodes[index + 1], node_times[index], node_times[index + 1], step)
        close_mid = nodes[-1]
        close_equity = account_equity(balance, positions, close_mid, config)
        net_lots, gross_lots = exposure(positions)
        equity_rows.append(
            {
                "date": node_times[-1],
                "bar_index": bar_index,
                "mid": close_mid,
                "balance": balance,
                "equity": close_equity,
                "gross_lots": gross_lots,
                "net_lots": net_lots,
                "peak_equity": peak_equity,
                "drawdown_pct": (peak_equity - close_equity) / peak_equity if peak_equity > 0 else 1.0,
            }
        )

    final_mid = float(data.iloc[-1]["close"])
    final_positions = pd.DataFrame(
        [
            {
                "ticket": position.ticket,
                "cycle_id": position.cycle_id,
                "kind": position.kind,
                "side": "buy" if position.side > 0 else "sell",
                "layer": position.layer,
                "lots": position.lots,
                "entry_time": position.entry_time,
                "entry_fill": position.entry_fill,
                "unrealized_pnl_usc": position_unrealized(
                    position, final_mid, config.spread, config.usc_per_price_lot
                ),
            }
            for position in positions
        ],
        columns=final_position_columns,
    )
    event_frame = pd.DataFrame(events, columns=event_columns)
    trade_frame = pd.DataFrame(trade_rows, columns=trade_columns)
    equity_frame = pd.DataFrame(equity_rows)
    stats = {
        "final_balance": balance,
        "final_equity": float(equity_frame.iloc[-1]["equity"]),
        "closed_tickets": len(trade_frame),
        "terminal_reason": terminal_reason,
        "terminal_time": terminal_time,
    }
    return BacktestResult(
        scenario=f"{path_mode}_spread_{config.spread}",
        events=event_frame,
        trades=trade_frame,
        equity=equity_frame,
        final_positions=final_positions,
        stats=stats,
    )
