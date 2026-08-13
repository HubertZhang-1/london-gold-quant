# 伦敦金（XAU）量化工具

一套基于日线的伦敦金量化模块，覆盖数据、策略、回测、实时信号四个环节，全部复用项目里已有的 `pandas` / `numpy` / `akshare`。

## 数据

- 日线历史: `ak.futures_foreign_hist(symbol="XAU")`，免费、无需 Key，缓存到 `data/london_gold_daily.csv`
- 实时报价: `ak.futures_foreign_commodity_realtime(symbol=["XAU"])`，返回伦敦金最新价、买价、卖价、涨跌幅
- 当前接口不提供稳定免费的 XAUUSD 分钟历史；需要日内回测时，可把自有分钟数据按同样 `date/open/high/low/close` 结构放入 CSV 后复用现有引擎

## 策略

`london_gold/strategies.py` 内置三种日线策略，全部带 ATR 止损和可选的均线趋势过滤：

1. **唐奇安通道突破** `donchian_breakout`
   - 收盘价突破 N 日最高/最低入场，跌破 M 日通道反向离场
   - 默认 `entry_n=40`、`exit_n=20`、`ma_filter=100`、`stop_mult=3.0`
2. **EMA 金叉死叉** `ema_cross`
   - 快线在上持多、在下持空，始终持仓
   - 默认 `fast_n=10`、`slow_n=40`、`stop_mult=3.0`
3. **RSI 回踩反转** `rsi_reversal`
   - 只沿趋势方向逆势入场：上升趋势中 RSI 超卖做多，下降趋势中 RSI 超买做空
   - 默认 `rsi_n=14`、`oversold=30`、`overbought=70`、`ma_filter=50`、`stop_mult=2.5`

## 回测

`scripts/london_gold_backtest.py` 会展开 `config/london_gold_config.json` 里的参数网格，按以下规则成交：

- 信号在收盘后产生，次根 K 线开盘成交
- 每次成交计半点点差 + 滑点 + 每盎司手续费
- 持仓期间按日线最低/最高价触发 ATR 止损
- 输出收益、年化、夏普、最大回撤、胜率、盈亏比

```bash
py311 scripts/london_gold_backtest.py --update
py311 scripts/london_gold_backtest.py --quick
py311 scripts/london_gold_backtest.py --from 2026-08-01 --to 2026-08-13
```

`--from/--to` 指定评估区间（仍用完整历史计算指标），输出该区间的区间收益、平仓盈亏和胜率。报告写入 `reports/london_gold_YYYYMMDD.md`、`_grid.csv`、`_trades.csv`、`_equity.csv`、`_equity.svg`。

## 实时信号

```bash
py311 scripts/london_gold_scan.py --update
```

输出三套策略的当前方向、触发价位、止损位和 RSI 读数，同时保存 JSON 快照到 `data/london_gold_scan.json`，方便接到邮件、企业微信或监控面板。

## 50 倍杠杆短线

短线配置在 `config/london_gold_short_config.json`：快参数（如 5/15 EMA、10/5 唐奇安、RSI2），回测引擎支持按杠杆动态算手数、按单笔风险限额控仓、以及净值跌破阈值自动强平。

```bash
# 2% 单笔风险 + 50 倍上限
py311 scripts/london_gold_backtest.py --config config/london_gold_short_config.json

# 完全不控仓、直接满 50 倍
py311 scripts/london_gold_backtest.py --config config/london_gold_short_config.json --risk-pct 0
```

需要说明：50 倍杠杆下价格反向 2% 就会亏光本金；伦敦金单日经常波动 1%-2%，隔夜跳空可能直接击穿强平价。当前免费数据源只有日线，所以这里的“短线”指持仓几天的快参数策略；分钟级回测需要接入分钟行情（自有 CSV 或带 API Key 的数据源）。

## 日内版本（1 小时 GC=F）

免费日内数据用 Yahoo 的 COMEX 黄金期货 `GC=F` 作为伦敦金代理，1 小时 K 线缓存在 `data/gc_h1.csv`。日内策略采用 UTC 日开盘区间突破：用每天前 2-4 根小时线形成区间，突破后做多/做空，止损放在区间外侧，日终强制平仓。

```bash
# 2% 单笔风险 + 50 倍杠杆上限
py311 scripts/london_gold_intraday_backtest.py

# 满 50 倍不控仓（用于对比爆仓风险）
py311 scripts/london_gold_intraday_backtest.py --risk-pct 0
```

2024-08 至 2026-08 的 1 小时回测显示：2% 风控下各参数组合多为正收益，但单笔风险若只靠 2% 限额、实际杠杆通常只有 1-2 倍；满 50 倍会在 1-5 笔交易内触发强平或穿仓。GC=F 是期货代理，实盘仍应以经纪商 XAUUSD 的真实点差、隔夜利息和滑点复核。

## 参数与风控

默认 `capital=100,000 USD`、每次 `10 oz`、点差 `$0.35`、滑点 `$0.10`、手续费 `$0.10/oz`。伦敦金杠杆高、波动大，实盘前应：

- 用小仓位和更严格的止损先跑样本外验证
- 核对经纪商实际点差、佣金、隔夜利息和最小手数
- 不要把参数网格里的最优结果直接当成期望收益
