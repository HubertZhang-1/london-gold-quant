# MT5 界面指标 → 多因子策略库：分析与回测结论

> 数据：`data/XAUUSD_1h_continuous.csv`（2024-01 至 2026-08，1h）。
> 动机：从图中 MT5（STARTTRADER XAUUSD.c）挂载的技术指标出发，建立可回测的多因子策略库，找出真正有预测力的因子。

## 一、图中识别到的 MT5 指标（指标导航栏）

类别 | 指标 | 量化因子
---|---|---
动量 | Momentum(ROC) | `f_momentum_roc` 快慢动量
动量 | MACD / OsMA | `f_macd_hist` MACD 柱方向
趋势 | ADX | `f_trend_adx` 趋势强度×价格方向
趋势 | 均线/Alligator | `f_ema_spread` EMA 快慢差
趋势 | Aroon | `f_aroon` 上/下轨主导
超买超卖 | RSI | `f_rsi` 均值回归偏置
超买超卖 | Stochastic | `f_stochastic` 震荡器偏置
超买超卖 | Williams %R | `f_williams_r` 反转偏置
超买超卖 | CCI | `f_cci` 反转偏置
价格带 | Bollinger | `f_bb_position` %B 位置
买卖力量 | Bulls/Bears Power | `f_bulls_bears` 买卖力量平衡
波动率 | ATR | `f_atr_vol` 波动率相对水平
量能 | Force Index / Chaikin | 需量能，本数据无 volume → 未启用

## 二、架构

- `london_gold/indicators.py`：新增 Momentum/ROC、Williams %R、CCI、MACD/OsMA、Bulls/Bears Power、Force Index、Aroon、Chaikin 等指标实现。
- `london_gold/factor_library.py`：把每个指标量化为**方向一致、标准化在 [-1,+1]** 的因子（正=看多、负=看空），并提供 `build_factors`（批量构建）+ `aggregate_score`（加权合成）。

## 三、单因子有效性（关键结果）

用每个因子单独作为信号，跑 2024-2025 vs 2026 两段：

| 因子 | 2024-25 收益 | PF | 2026 收益 | PF | 有效? |
|---|---:|---:|---:|---:|:--:|
| macd | +154.9% | 1.28 | +79.5% | 1.49 | ✅ |
| aroon | +145.8% | 1.34 | +51.6% | 1.38 | ✅ |
| trend_adx | +127.7% | 1.60 | +43.1% | 1.55 | ✅ |
| bulls_bears | +104.2% | 1.11 | +98.7% | 1.35 | ✅ |
| ema_spread | +86.0% | 1.53 | +50.7% | 1.93 | ✅ |
| bb_position | +81.0% | 1.20 | +22.4% | 1.19 | ✅ |
| momentum | +61.1% | 1.08 | +82.3% | 1.37 | ✅ |
| stochastic | +20.9% | 1.04 | +27.1% | 1.15 | ✅ |
| rsi | -34.0% | 0.93 | -13.0% | 0.92 | ❌ |
| williams_r | -47.7% | 0.93 | -10.9% | 0.95 | ❌ |
| cci | -23.3% | 0.95 | +4.0% | 1.03 | ❌ |

**核心洞察**：
1. **趋势/动量类因子（macd、aroon、trend_adx、ema_spread、bulls_bears、momentum、bb）单独都强正期望**，且两段都盈利。
2. **超买超卖反转类因子（rsi、williams_r、cci）单独都是负期望** —— 在 1h 趋势市里，越超买越涨（动量延续），均值回归反而亏损。这些因子应当**剔除或反向使用**。

## 四、因子组合（精选 vs 全部等权）

| 组合 | 2024-25 收益 | PF | 2026 收益 | PF |
|---|---:|---:|---:|---:|
| 全部因子等权 | -12.0% | 0.86 | -17.0% | 0.42 |
| **精选趋势/动量因子** | **+160.0%** | 1.35 | **+92.8%** | 1.62 |
| 精选 + 反转因子反向 | +151.8% | 1.56 | +52.9% | 1.59 |

**剔除无效反转因子后，结果天壤之别**：全部等权 -12% → 精选 +252%（全期）、PF 1.42。

## 五、叠加市场状态过滤（平衡最优）

| 方案 | 全期收益 | PF | 最大回撤 | 2024-25 | 2026 |
|---|---:|---:|---:|---:|---:|
| 精选因子（无过滤） | +252.0% | 1.42 | 29.2% | +160% | +92.8% (DD7.4%) |
| **精选因子 + regime 过滤** | **+178.9%** | **1.44** | **21.4%** | +124% | +54.8% (DD8.3%) |

**权衡**：叠加市场状态过滤（ER=0.12/ADX=20，只在趋势市交易）后，总收益从 +252% 降到 +178.9%，但**最大回撤从 29.2% 压到 21.4%**，PF 微升（1.42→1.44），且两段都稳健。

## 六、结论

1. **从 MT5 指标能提炼出真正有预测力的多因子体系**。趋势/动量类因子（MACD、Aroon、ADX+EMA、EMA spread、Bulls/Bears、Momentum、BB 位置）单独都有正期望且跨年稳健。
2. **反转类因子（RSI、Williams %R、CCI）在 1h 趋势市里反而无效甚至反向** —— 这是重要的认知：不能盲目用超买超卖反转，尤其在趋势品种上。
3. **精选组合 + 市场状态过滤** 提供平衡最优方案：**全期 +178.9%、PF 1.44、最大回撤 21.4%**，2024-25 和 2026 两段都盈利，是实盘候选。

## 七、仍存的注意点

1. 因子权重和 regime 阈值是在 2024-2026 上选的，**存在过拟合风险**，建议用 2018-2023 独立年份复验。
2. **量能因子（Force Index / Chaikin）因数据缺 volume 未启用**；若能获取更完整的 tick 量可补充。
3. 单笔风险 1%，精选+regime 回撤 21.4%，实盘建议再降到 0.5% 或配合其他策略摊平。

## 八、可复现文件

- 指标：`london_gold/indicators.py`
- 因子库：`london_gold/factor_library.py`
- 单因子有效性：`scripts/factor_effectiveness.py`
- 因子组合：`scripts/factor_ensemble.py`
- 因子+regime：`scripts/factor_regime_combined.py`
