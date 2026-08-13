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
```

报告写入 `reports/london_gold_YYYYMMDD.md`、`_grid.csv`、`_trades.csv`、`_equity.csv`、`_equity.svg`。

## 实时信号

```bash
py311 scripts/london_gold_scan.py --update
```

输出三套策略的当前方向、触发价位、止损位和 RSI 读数，同时保存 JSON 快照到 `data/london_gold_scan.json`，方便接到邮件、企业微信或监控面板。

## 参数与风控

默认 `capital=100,000 USD`、每次 `10 oz`、点差 `$0.35`、滑点 `$0.10`、手续费 `$0.10/oz`。伦敦金杠杆高、波动大，实盘前应：

- 用小仓位和更严格的止损先跑样本外验证
- 核对经纪商实际点差、佣金、隔夜利息和最小手数
- 不要把参数网格里的最优结果直接当成期望收益
