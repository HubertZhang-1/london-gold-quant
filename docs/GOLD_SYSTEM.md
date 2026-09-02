# 伦敦金量化策略说明书（GOLD SYSTEM）

> 本说明书把整套伦敦金（XAUUSD）量化操作逻辑整理成一份可执行、可复现的规范：
> 决策流程图（mermaid + 纯文本兜底）、分层规则、参数、API、以及当前回测表现。
> 数据：黄金日线 `data/XAUUSD_1d.csv`（2004-2026）；宏观 `data/macro_daily.csv`（DXY/VIX/10Y，2017-09 起）。
> 回测窗口：2024-01-01 → 2026-08-28（牛市窗口），并用完整 2019-2026 校验不爆仓。

---

## 0. 一句话逻辑

**牛市才做、熊市不做、震荡只作壁上观；顺着明确上升趋势做多，靠「方向判断 + 节奏过滤 + 仓位风险预算 + 硬止损」活下来，而不是靠高杠杆赌方向。**

---

## 1. 决策流程图

### 1.1 mermaid 版本（GitHub / 支持 mermaid 的工具可直接渲染）

```mermaid
flowchart TD
    START([每日收盘后]) --> LOAD[加载数据: 黄金日线 + 宏观 DXY/VIX/10Y]
    LOAD --> CALC[计算指标: ATR / 效率比 / ADX / CHOP / EMA / bull分 / 宏观分]

    CALC --> G1{宏观门: 宏观分?}
    G1 -- "偏空(宏观分<0)" --> DAMP[宏观降档: 杠杆缓存 0.5x]
    G1 -- "中性/偏多" --> DAMP
    G1 -- "未知(2017前)" --> NODAMP[不降档 ×1.0]

    DAMP --> G2{牛市门: bull分 ≥ 0.55?}
    NODAMP --> G2
    G2 -- "否<br>熊市/不明朗" --> FLAT_END([空仓, 不限时间])
    G2 -- "是" --> G3{节奏门: 行情节奏=确认上升趋势?}
    G3 -- "否<br>震荡/区间/下跌" --> FLAT_END
    G3 -- "是" --> G4{入场: 收阳 + 微观多头对齐?}
    G4 -- "否" --> WAIT([持有/观望])
    G4 -- "是" --> SIZE[风险预算定仓: 手数= 账户2% / (止盈差×100oz)]
    SIZE --> POS((持有做多))
    POS --> EXIT{离场: 止损/止盈/时间止损?}
    EXIT -- "止损 ATR×2.5" --> CLOSE_LOSE([止损离场])
    EXIT -- "止盈 2×止损" --> CLOSE_WIN([止盈离场])
    EXIT -- "持仓>30根K线" --> CLOSE_TIME([时间止损离场])
    POS --> CB{熔断: 峰值回撤≥20%?}
    CB -- "是" --> HALT([强制平仓停机])
    CB -- "否" --> LOOP([进入下一根K线])
```

### 1.2 纯文本版本（任何环境都能看）

```
每日收盘
   └─ 加载 黄金日线 + 宏观(DXY/VIX/10Y)
        └─ 计算 ATR / 效率比 / ADX / Choppiness / EMA / bull分 / 宏观分
             ├─ 宏观门: 宏观分偏空? ──是──▶ 宏观降档(杠杆×0.5) ；中性/偏多→×1.0
             │         │
             │         └─〔未知(2017前)〕→ 不降档 ×1.0
             ├─ 牛市门: bull分 ≥ 0.55? ──否──▶ 空仓(不限时间)
             │         │
             │         └─是▶ 节奏门: 行情节奏=确认上升趋势? ──否──▶ 空仓
             │                  │
             │                  └─是▶ 入场: 收阳 + 微观多头对齐? ──否──▶ 观望
             │                           │
             │                           └─是▶ 风险预算定仓(单笔止损2%)
             │                                    │
             │                                    └─▶ 持有多单
             │                                         ├─ 止损 ATR×2.5 → 止损离场
             │                                         ├─ 止盈 2×止损 → 止盈离场
             │                                         ├─ 持仓>30根 → 时间止损离场
             │                                         └─ 峰值回撤≥20% → 熔断停机
             └─ 下一根K线
```

---

## 2. 四层信号架构

| 层 | 数据/指标 | 作用 | 输出 |
|---|---|---|---|
| **宏观层** | DXY / 10Y / VIX | 方向风标 + 风险开关 | 宏观分 `[-1,1]` → 降杠杆倍数 |
| **状态层** | bull分(EMA斜率+价格vsEMA+trend) | 牛市 vs 非牛市 | `bull` `[0,1]` → 是否允许做多 |
| **节奏层** | Choppiness + 效率比 + ADX + EMA方向 | 趋势 vs 震荡 | `state` `dir` `signal` → 是否进 |
| **微观层** | macd/aroon/trend_adx/ema_spread/bulls_bears/momentum/bb | 进场点 | 信号分(置信度) |

---

## 3. 分层规则（详细）

### 3.1 宏观门（风险降档，不空仓）
- **宏观分** `macro_direction_score`：由 DXY/10Y/VIX 的变化合成 `[-1,1]`。
  - 实际利率↑ → 利空黄金；美元↑ → 利空黄金；VIX 仅危机时利多。
- **映射**：`杠杆倍数 = macro_lev_lo + (macro_lev_hi - macro_lev_lo) × (宏观分+1)/2`
  - 生产配置 `macro_lev_lo=0.5, macro_lev_hi=1.0`：宏观越偏空，杠杆越低。
  - **未知（2017 年前无宏观数据）→ 倍数=1.0 不降档**，避免误伤。

### 3.2 牛市门（只做牛市的根基）
- **bull 分** `_bull_score`：`0.4×(EMA斜率为正) + 0.3×(价格>EMA50) + 0.3×trend_regime`，clip 到 `[0,1]`。
- **规则**：`bull < bull_thr(0.55)` → 完全空仓，不限时间。（熊市不参与）

### 3.3 节奏门（震荡识别器）
- `market_state` 合成三个正交量：
  - **Choppiness Index**：`>chop_hi(68)` → 区间/震荡；`<38` → 趋势。
  - **效率比**：`< er_thr(0.10)` → 噪声/震荡。
  - **ADX**：`< adx_thr(16)` → 无趋势。
  - **方向**：EMA 斜率确定 `dir`（+1 上 / -1 下）。
- **规则**：仅当 `state=="trend"` 且 `dir>0`（确认上升趋势）时 `signal=+1`（允许做多），否则 `signal=0`（空仓/观望）。

### 3.4 微观入场
- 入场需同时满足：收阳 `close>open` + 微观多头对齐（信号分与趋势同向）。
- 单笔收益目标 = 止盈距离（`2.5×ATR × 2`），超过即止盈。

### 3.5 仓位（风险预算定仓）
- **手数** = `账户余额 × risk_per_trade_pct / (止损距离 × 100盎司/lot)`
- 生产 `risk_per_trade_pct=0.02`：**单笔止损只冒账户 2%**。这是把回撤压到 9% 的关键，替代"马丁翻倍"。

### 3.6 离场
- **硬止损**：`entry - stop_mult×ATR`（`stop_mult=2.5`）。
- **止盈**：`entry + rr×stop`（`rr=2.0`）。
- **时间止损**：持仓 > `max_bars_in_trade(30)` 根 K 线未达目标 → 平仓。

### 3.7 熔断
- 峰值回撤 ≥ `margin_call_pct(20%)` → 强制平仓停机。
- 这是唯一保命阀；风控设计下正常不会触发。

---

## 4. 参数表

| 参数 | 默认 | 含义 |
|---|---|---|
| `bull_thr` | 0.55 | 牛市门阈值 |
| `chop_hi` | 68.0 | Choppiness 震荡上限 |
| `rhythm_er_thr` | 0.10 | 效率比震荡阈值 |
| `rhythm_adx_thr` | 16.0 | 趋势强度阈值 |
| `stop_mult` | 2.5 | 止损=ATR×2.5 |
| `rr` | 2.0 | 止盈=2×止损 |
| `risk_per_trade_pct` | 0.02 | 单笔止损占账户 2% |
| `macro_lev_lo/hi` | 0.5 / 1.0 | 宏观降档倍数 |
| `max_bars_in_trade` | 30 | 时间止损 |
| `margin_call_pct` | 0.20 | 回撤熔断线 |

---

## 5. API 参考

```python
from london_gold.bull_grid import BullGridConfig, run_bull_grid_backtest
from london_gold.macro_factors import macro_direction_score, forward_fill_macro
from london_gold.indicators import market_state, choppiness_index

# 行情节奏识别器
ms = market_state(close, high, low, er_thr=0.10, adx_thr=16.0, chop_hi=68.0)
#  -> DataFrame[state, dir, signal, chop, er, adx]

# 宏观分 -> 降杠杆倍数
macro = forward_fill_macro(macro_direction_score(macro_df)["macro_score"], gold_dates)

# 牛市单向版回测
cfg = BullGridConfig(bull_thr=0.55, chop_hi=68.0, rhythm_er_thr=0.10,
                     rhythm_adx_thr=16.0, stop_mult=2.5, rr=2.0,
                     risk_per_trade_pct=0.02, macro_lev_lo=0.5, macro_lev_hi=1.0)
res = run_bull_grid_backtest(daily_df, cfg, macro_series=macro)
#  -> {"stats": {...}, "equity": DataFrame, "trades": DataFrame}
```

---

## 6. 当前回测表现（牛市单向版 + 节奏门，生产配置）

### 2024-2026 牛市窗口
| 指标 | 数值 |
|---|---|
| 总收益 | **+88.5%** |
| 盈亏因子 | **3.32** |
| 最大回撤 | **9.1%** |
| 胜率 | 65% |
| 交易数 | 34 |

### 完整区间 2019-2026
**+138.2%、PF 2.36、回撤 9.1%、胜率 55%、75 笔、不爆仓。**

### 逐年
| 年 | 收益 | PF | 回撤 | 判断 |
|---|---|---|---|---|
| 2019 | +8.3% | 2.37 | 6.1% | 微利 |
| 2020 | +4.9% | 1.29 | 10.8% | 微利 |
| 2021 | +0.6% | 1.13 | 5.1% | 持平 |
| 2022 | +7.8% | 2.22 | 7.2% | 熊市不亏 |
| 2023 | +1.1% | 0.98 | 6.2% | 持平 |
| 2024 | +13.6% | 1.99 | 9.1% | ✅ 牛市 |
| 2025 | +39.3% | 3.88 | 8.3% | ✅ 牛市强 |
| 2026 | -1.2% | 0.83 | 9.8% | 小亏(收窄) |

---

## 7. 诚实的边界与代价

1. **不是预测工具**：核心是"过滤"，把熊市/震荡/宏观逆风提前挡在外，而非精确预测方向。
2. **无法救回所有亏损年**：2026 这种"上半年假趋势下跌"，节奏门只能挡真震荡，挡不了伪装成趋势的下跌，仍小亏 -1.2%。这是黄金固有结构，非参数能完全消除。
3. **收益-回撤权衡**：`risk_per_trade_pct` 是收益/回撤的旋钮。2% → 收益 88.5%/回撤 9.1%；降到 1% → 收益约 33%/回撤 3%；升到 3% → 收益更高但回撤更大。
4. **数据覆盖**：宏观层仅 2017-09 起有效；2004-2017 无宏观数据（按中性处理不降档）。
5. **高杠杆已被证伪**：10x/15x 在完整区间会爆仓（见 conf_x15_aggressive.md）。本系统用低杠杆+风险预算，放弃暴利换长久。

---

## 8. 可复现文件

- 说明书：`docs/GOLD_SYSTEM.md`（本文）
- 策略引擎：`london_gold/bull_grid.py`（牛市单向版）
- 节奏识别器：`london_gold/indicators.py`（`choppiness_index` / `market_state`）
- 宏观：`london_gold/macro_factors.py`、`london_gold/bull_adaptive.py`
- 回测：`scripts/backtest_bull_grid.py`
- 数据：`data/XAUUSD_1d.csv`、`data/macro_daily.csv`
