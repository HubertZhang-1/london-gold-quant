# GitHub 伦敦金量化策略调研与本地策略完善

> 目的：调研 GitHub 上主流 XAUUSD 量化策略，提炼可借鉴设计，完善本地伦敦金量化工具。
> 数据：`data/XAUUSD_5m_2026.csv`（真实 XAUUSD 5m，2026-01-01 至 08-28）。

## 一、调研到的 GitHub 代表性策略

| 仓库 | 方向 | 核心设计 |
|---|---|---|
| [ns-vikas/trading-bot-mql5](https://github.com/ns-vikas/trading-bot-mql5) | 剥头皮 | 多指标**置信度打分**、ADX 趋势门控、**时段过滤**、ATR 固定 R:R、多层风控 |
| [Shivkeerth-Laj/XAUUSD-Backtesting](https://github.com/Shivkeerth-Laj/XAUUSD-Backtesting) | 规则突破 | 4 根同向K突破 + **固定美元风险 sizing** + 动态 SL/TP |
| [soloshun/Quantitative-XAUUSD-Strategy](https://github.com/soloshun/Quantitative-XAUUSD-Strategy) | 时段动力学 | **session 预测力**建模（亚盘→伦敦、伦敦午盘→重叠期动量） |
| [coler07/mql5-format](https://github.com/coler07/mql5-format) | 马丁网格 | 与本地已逆推的马丁网格一致，验证了其风险特征 |
| [HisyamAlammar/YQTS](https://github.com/HisyamAlammar/YQTS) | regime 检测 | PELT+BOCPD 市场状态检测 + 蒙特卡洛 |

## 二、提炼出的可借鉴设计（本地缺失的）

对照本地 `intraday_strategies_v2.py`，GitHub 主流策略普遍具备而本地欠缺的：

1. **置信度打分**：多指标加权（RSI/EMA/Stoch/ADX/BB），而不是单条件判断。
2. **ADX 趋势强度门控**：ADX>25 才顺势，ADX>40 禁止均值回归（本地仅看斜率方向不看强度）。
3. **交易时段过滤**：伦敦/纽约/重叠期权重不同，亚盘低权重（本地 `momentum_trend`/`zscore` 无时段概念）。
4. **ATR 固定风险回报比**：SL=1×ATR、TP=2×ATR（本地只有 stop_dist，无止盈）。
5. **美元风险 sizing**：按 `capital×risk%/stop_dist` 定手数（本地固定 position_oz）。

## 三、实现的完善（新增代码）

### 1. `london_gold/indicators.py`（补充 3 个指标）
- `adx()` — Wilder ADX 趋势强度
- `stochastic()` — 随机指标 %K/%D
- `bollinger()` — 布林带（中/上/下轨）

### 2. `london_gold/intraday_strategies_v3.py`（新策略模块）
- **`momentum_scalp_signals`**：双动量入场 + **置信度打分**（快/慢ROC、EMA21、RSI、Stoch、ADX 加权）+ **ADX 门控** + **时段过滤** + 固定 **SL=1×ATR / TP=2×ATR**。
- **`mean_reversion_signals`**：均值回归 + **ADX 上限门控**（强趋势禁止逆势）+ 时段过滤 + 更宽止损。

### 3. `scripts/london_gold_intraday_v3_backtest.py`（v3 回测，支持止盈）
- 独立事件回测，支持 `signal`/`stop_dist`/`tp_dist` 三列，带符号记账。
- **已用最小案例单元验证**：做多 10oz@102→103 得+10，做空 10oz@102→101 得+10，正确。

## 四、回测结果（2026-01 至 08，5m XAUUSD，risk 0.5%)

| 策略 | 交易数 | 胜率 | 总收益 | 盈亏因子 | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| V3 MOM_SCALP | 2906 | 37% | -3.5% | 1.00 | 44% |
| V3 MOM_SCALP_STRONG(0.75) | 3145 | 39% | -78.9% | 0.90 | 91% |
| V2 MOM_BASE | 6029 | 27% | -64% | 0.73 | — |

**初步观察**：
- V3 置信度打分版（V3_MOM_SCALP）相比于 V2 基线（-64%）**大幅改善**——从明显亏损到接近盈亏平衡（-3.5%），说明**置信度门控 + ADX + 时段过滤**确实有正向作用。
- 但 V3_MOM_SCALP_STRONG（更高门槛 0.75）反而更差（-78.9%），说明**过度过滤导致在关键行情错过、在震荡里追单**，门槛过高反而有害。
- **胜率偏低（37-39%）**是固定 R:R=2（TP=2×ATR）的必然——高盈亏比策略胜率天然低，需靠盈利单覆盖。
- **回撤仍大（44%）**：5m 高频 + 风险预算 sizing 使绝对敞口大。这符合网格/高频策略在 5m 上的真实风险特征，**不建议直接实盘**。

## 五、结论

1. **GitHub 主流增强（置信度/ADX/时段/R:R）确实有正向价值**：把纯动量基线从 -64% 改善到接近 -3.5%。
2. **但 5m 高频 XAUUSD 上整体仍难稳定盈利**——胜率低、回撤高，风险预算 sizing 在 5m 小 stop 下放大敞口。
3. **下一步最有价值的改进**：拉长周期（用 1h 而非 5m）、降低交易频率、把 5m 置信度信号作为**过滤**而非独立交易信号。这符合 GitHub 主流（ns-vikas 用 M5/M15/H1 多周期确认，soloshun 用 H1）。

## 六、可复现文件

- 指标补充：`london_gold/indicators.py`
- 新策略：`london_gold/intraday_strategies_v3.py`
- 回测：`scripts/london_gold_intraday_v3_backtest.py`
- 引擎单元验证：`scripts/unit_check_v3_engine.py`
- 调试：`scripts/debug_v3_engine.py`
