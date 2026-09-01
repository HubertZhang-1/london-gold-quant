# 多因子选股策略（标准版）

触发方式：用户说"多因子选股"时，运行 `python scripts/multi_factor_screener.py`。

## 筛选条件

| 条件 | 说明 |
|------|------|
| 剔除 ST | 排除名称含 `ST` / `*ST` 的股票 |
| 盈利要求 | PE > 0，排除亏损股 |
| 市值区间 | 50 - 200 亿 |
| 涨停动量 | 近 20 个交易日内出现过涨停（主板 10%，创业板/科创板 20%，北交所 30%） |

## 四维因子评分

总分 0-100，权重如下：

| 因子 | 权重 | 计算方式 |
|------|------|----------|
| 低估值 PE | 35% | `50 / PE * 30`，PE 越低分越高 |
| 合理 PB | 20% | `10 / PB * 25`，PB 越低分越高 |
| 市值规模 | 25% | 百亿以上接近满分，兼顾流动性 |
| 换手活跃 | 20% | `换手率 * 5`，越活跃越高 |

## 输出

- 终端显示 Top 30
- 完整结果保存到 `data/screen_limitup_YYYYMMDD.csv`
- CSV 字段：`code, name, total, pe, pb, mcap_yi, turnover, price, last_limitup`

## 数据源

- 全市场快照：新浪财经 `Market_Center.getHQNodeData`（约 5500 只）
- 20 日 K 线：新浪财经 `CN_MarketData.getKLineData`（并发检查涨停）
