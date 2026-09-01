# XAUUSD M15 August Proxy Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved 15-minute momentum-pullback strategy and run a clearly labeled August 2026 backtest using GC=F as a temporary public proxy for XAUUSD.

**Architecture:** Add a pure signal state machine and a dedicated intraday execution engine rather than changing the existing hourly strategy behavior. Reuse the current Yahoo intraday loader for complete August 2026 GC=F bars, execute signals on the next bar open, and generate auditable CSV/Markdown reports.

**Tech Stack:** Python 3.11, pandas, numpy, requests, unittest, standard-library `zoneinfo`.

**Spec:** `docs/superpowers/specs/2026-09-01-xauusd-m15-intraday-design.md`

## Global Constraints

- This plan implements only data refresh, strategy, historical execution, tests, and the August 2026 report; paper trading remains a later plan.
- Open new positions only during the dynamic overlap of `Europe/London` and `America/New_York`; close all positions before the overlap ends.
- Risk at most 0.5% of current equity per trade, stop opening trades after 1.5% realized daily loss, and allow at most three entries per UTC trading date.
- Signals use completed bars and fill at the next bar open.
- Do not add third-party dependencies or modify unrelated dirty files.
- Label GC=F output as a COMEX futures proxy, never as broker XAUUSD evidence.

---

### Task 1: Momentum-Pullback Signal State Machine

**Files:**
- Create: `london_gold/m15_pullback.py`
- Test: `test_london_gold_m15.py`

**Interfaces:**
- Consumes: a pandas DataFrame with `date`, `open`, `high`, `low`, and `close` columns.
- Produces: `M15StrategyConfig` and `momentum_pullback_signals(df, config) -> pd.DataFrame` with `signal`, `stop_dist`, `session_open`, `setup_state`, `ema`, `atr`, `fast_roc`, and `slow_roc`.

- [ ] **Step 1: Write failing tests for overlap detection, breakout lookback, pullback confirmation, and timeout**

```python
def test_overlap_handles_august_dst():
    bars = synthetic_m15("2026-08-03 12:00", periods=24)
    result = momentum_pullback_signals(bars, relaxed_config())
    assert result.loc[result["date"] == pd.Timestamp("2026-08-03 13:00"), "session_open"].item()
    assert not result.loc[result["date"] == pd.Timestamp("2026-08-03 16:15"), "session_open"].item()

def test_confirmation_uses_next_bar_signal_without_lookahead():
    bars = pullback_fixture()
    result = momentum_pullback_signals(bars, relaxed_config())
    assert result.loc[BREAKOUT_INDEX, "signal"] == 0
    assert result.loc[CONFIRM_INDEX, "signal"] == 1
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run: `py -3.11 -m unittest test_london_gold_m15.M15SignalTests -v`

Expected: FAIL because `london_gold.m15_pullback` does not exist.

- [ ] **Step 3: Implement configuration, indicators, dynamic overlap, and the setup state machine**

```python
@dataclass(frozen=True)
class M15StrategyConfig:
    ema_bars: int = 48
    fast_bars: int = 4
    slow_bars: int = 16
    breakout_bars: int = 8
    pullback_bars: int = 3
    atr_bars: int = 14
    min_stop_atr: float = 0.8
    max_stop_atr: float = 2.0

def momentum_pullback_signals(df: pd.DataFrame, config: M15StrategyConfig) -> pd.DataFrame:
    """Return close-confirmed signals; execution is handled by the next-open engine."""
```

Use `ZoneInfo("Europe/London")` and `ZoneInfo("America/New_York")` to derive the 08:00-17:00 London and 08:00-17:00 New York local-session intersection for every UTC bar. Compute breakout levels with `.shift(1).rolling(config.breakout_bars)` so the current bar never enters its own channel. A setup expires after `pullback_bars`, on trend invalidation, or when the overlap closes.

- [ ] **Step 4: Run signal tests**

Run: `py -3.11 -m unittest test_london_gold_m15.M15SignalTests -v`

Expected: all `M15SignalTests` pass.

- [ ] **Step 5: Do not commit**

Leave the tested changes uncommitted because project instructions require an explicit user request before `git commit`.

### Task 2: Intraday Risk and Execution Engine

**Files:**
- Create: `london_gold/m15_execution.py`
- Modify: `test_london_gold_m15.py`

**Interfaces:**
- Consumes: the signal DataFrame from Task 1 and `M15ExecutionConfig`.
- Produces: `run_m15_backtest(df, config) -> dict` containing `trades`, `equity`, `dates`, `stats`, and `skipped`.

- [ ] **Step 1: Write failing tests for next-open fills and account limits**

```python
def test_signal_fills_on_next_open():
    result = run_m15_backtest(signaled_fixture(), execution_config())
    assert result["trades"].iloc[0]["entry_time"] == signaled_fixture().iloc[SIGNAL_INDEX + 1]["date"]

def test_position_risks_half_percent():
    result = run_m15_backtest(signaled_fixture(), execution_config(capital=100_000))
    trade = result["trades"].iloc[0]
    assert trade["initial_risk"] <= 500.01

def test_daily_limits_block_new_entries():
    result = run_m15_backtest(three_loss_fixture(), execution_config())
    assert (result["skipped"]["reason"] == "daily_loss_limit").any()
```

- [ ] **Step 2: Run execution tests and verify failure**

Run: `py -3.11 -m unittest test_london_gold_m15.M15ExecutionTests -v`

Expected: FAIL because `london_gold.m15_execution` does not exist.

- [ ] **Step 3: Implement costs, position sizing, stops, and exits**

```python
@dataclass(frozen=True)
class M15ExecutionConfig:
    capital: float = 100_000.0
    risk_per_trade: float = 0.005
    daily_loss_limit: float = 0.015
    max_daily_trades: int = 3
    spread: float = 0.35
    slippage: float = 0.10
    commission_per_oz: float = 0.10
    min_position_oz: float = 0.01
    max_hold_bars: int = 12

def run_m15_backtest(df: pd.DataFrame, config: M15ExecutionConfig) -> dict:
    """Execute completed-bar signals at the next open with daily risk controls."""
```

Size ounces from `(current_equity * 0.005) / (stop_dist + round_trip_cost_per_oz)`, round down to `min_position_oz`, and never increase risk by rounding. Check the initial stop against intrabar high/low, move to cost-covering breakeven after 1R, trail by ATR afterward, exit after 12 bars or on the final session bar, and record every blocked signal reason.

- [ ] **Step 4: Run execution and signal tests**

Run: `py -3.11 -m unittest test_london_gold_m15 -v`

Expected: all tests pass.

- [ ] **Step 5: Do not commit**

Leave the tested changes uncommitted.

### Task 3: August Backtest CLI and Report

**Files:**
- Create: `config/london_gold_m15_pullback_config.json`
- Create: `scripts/london_gold_m15_pullback_backtest.py`
- Modify: `test_london_gold_m15.py`
- Generate: `reports/gc_m15_pullback_202608.md`
- Generate: `reports/gc_m15_pullback_202608_trades.csv`
- Generate: `reports/gc_m15_pullback_202608_equity.csv`
- Generate: `reports/gc_m15_pullback_202608_skipped.csv`

**Interfaces:**
- Consumes: `fetch_intraday(interval="15m", period="60d")`, Task 1 signals, Task 2 execution, and JSON configuration.
- Produces: a command-line exit code plus auditable August 2026 artifacts.

- [ ] **Step 1: Write a failing CLI configuration test**

```python
def test_august_filter_is_inclusive_and_has_warmup():
    prepared = prepare_period(sample_bars(), "2026-08-01", "2026-08-31", warmup_bars=64)
    assert prepared["evaluation"].min() == pd.Timestamp("2026-08-01", tz="UTC")
    assert prepared["evaluation"].max() < pd.Timestamp("2026-09-01", tz="UTC")
    assert len(prepared["input"]) > len(prepared["evaluation"])
```

- [ ] **Step 2: Run the CLI tests and verify failure**

Run: `py -3.11 -m unittest test_london_gold_m15.M15CliTests -v`

Expected: FAIL because the CLI module and configuration do not exist.

- [ ] **Step 3: Implement the config and CLI**

The configuration fixes the approved parameters and costs:

```json
{
  "symbol": "GC=F",
  "proxy_for": "XAUUSD",
  "interval": "15m",
  "risk_per_trade": 0.005,
  "daily_loss_limit": 0.015,
  "max_daily_trades": 3,
  "strategy": {"ema_bars": 48, "fast_bars": 4, "slow_bars": 16, "breakout_bars": 8, "pullback_bars": 3}
}
```

The CLI accepts `--start`, `--end`, `--update`, `--cache`, and `--out`. It loads at least 64 warmup bars before the requested start, but statistics and exported trades include only entries from `[start, end + 1 day)`. Its Markdown heading and limitations section explicitly state `GC=F COMEX proxy; not broker XAUUSD`.

- [ ] **Step 4: Run all new tests**

Run: `py -3.11 -m unittest test_london_gold_m15 -v`

Expected: all tests pass.

- [ ] **Step 5: Refresh the proxy data and run August 2026**

Run: `py -3.11 scripts/london_gold_m15_pullback_backtest.py --update --start 2026-08-01 --end 2026-08-31 --out gc_m15_pullback_202608`

Expected: the loader reports bars covering August 2026 and writes the four report artifacts. If the refreshed source lacks full August coverage, stop and report the exact last timestamp instead of presenting partial-month metrics as complete.

- [ ] **Step 6: Inspect report consistency**

Verify that trade entry times fall within August, no entry is outside the overlap window, risk per trade is at most 0.5% of entry equity, daily entries are at most three, and the equity file ends on or before August 31 UTC.

- [ ] **Step 7: Do not commit**

Leave implementation and generated reports uncommitted.

### Task 4: Regression Verification

**Files:**
- Test: `test_london_gold.py`
- Test: `test_london_gold_m15.py`

**Interfaces:**
- Consumes: all implementation from Tasks 1-3.
- Produces: verification evidence that the new slice works without breaking the existing London-gold toolkit.

- [ ] **Step 1: Run the new focused suite**

Run: `py -3.11 -m unittest test_london_gold_m15 -v`

Expected: PASS.

- [ ] **Step 2: Run the existing self-check**

Run: `py -3.11 test_london_gold.py`

Expected: exit code 0 and `all London gold self checks passed`.

- [ ] **Step 3: Compile changed Python files**

Run: `py -3.11 -m py_compile london_gold/m15_pullback.py london_gold/m15_execution.py scripts/london_gold_m15_pullback_backtest.py test_london_gold_m15.py`

Expected: exit code 0 with no output.

- [ ] **Step 4: Check repository diff without touching unrelated files**

Run: `git diff --check -- london_gold/m15_pullback.py london_gold/m15_execution.py scripts/london_gold_m15_pullback_backtest.py test_london_gold_m15.py config/london_gold_m15_pullback_config.json docs/superpowers/plans/2026-09-01-xauusd-m15-august-backtest.md`

Expected: exit code 0 with no whitespace errors.

- [ ] **Step 5: Do not commit**

Report changed files, tests, exact data coverage, August metrics, and the GC=F proxy limitation to the user.
