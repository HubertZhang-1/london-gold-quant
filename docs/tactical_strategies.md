# 可执行的因子战术策略体系

> 提炼自《因子投资》《寻找Alpha》《量化交易》《价值投资》等经典著作
> 所有策略均可直接用 Python/vnpy/xtquant 实现

---

## 策略一：双动量趋势跟踪
**来源**: 《寻找Alpha》Carhart四因子, 《量化交易》Chan

### 核心理念
```
相对动量 + 绝对动量双重确认:
  相对动量: 股票/ETF在过去N个月跑赢基准
  绝对动量: 股票/ETF价格高于其X月均线
```

### 战术实现

```
信号生成:
  1. 计算过去12个月收益(剔除近1个月)
     momentum_12_1 = (close[-12m] / close[-1m]) - 1
  2. 计算价格与200日均线关系
     trend_ma200 = close / MA(close, 200) > 1
  3. 买入条件
     momentum_12_1 > 0 AND trend_ma200 == True

退出条件:
  止损: 价格跌破MA(50) 或 单笔亏损>-8%
  止盈: momentum_12_1 < -10% 或 价格跌破MA(200)
  时间: 持有超过6个月自动评估

仓位:
  单个品种 <= 20% 总资金
  总仓位 <= 80%
```

### Python骨架
```python
def momentum_signal(closes, lookback=252, skip=21, ma_period=200):
    """双动量信号"""
    if len(closes) < lookback + ma_period:
        return 0
    rel_mom = closes[-skip] / closes[-lookback] - 1  # 相对动量
    abs_mom = closes[-1] / np.mean(closes[-ma_period:])  # 绝对动量
    if rel_mom > 0 and abs_mom > 1:
        return 1  # 买入
    elif abs_mom < 0.95:
        return -1  # 卖出
    return 0
```

---

## 策略二：均值回归组合
**来源**: 《量化交易》Chan, 《投资最重要的事》Marks

### 核心理念
```
价格围绕价值波动, 过度偏离后回归:
  买入: 价格低于合理估值2个标准差
  卖出: 价格回到均值或过度高估
```

### 战术实现

```
信号生成:
  1. 计算Z-score
     z = (price - MA(price, 60)) / std(price, 60)
  2. 买入条件
     z < -2.0  (极度低估)
  3. 卖出条件
     z > 0     (回到均值) 或 z > 2.0 (做空信号)

退出条件:
  止损: z-score进一步恶化到 < -3.0
  止盈: z-score回到 -0.5 ~ 0 区间

仓位:
  凯利公式: f = (p * b - q) / b
  简化: 固定仓位, z-score越低仓位越大
  z<-2.0: 100%目标仓位
  -2.0<z<-1.5: 50%目标仓位
  -1.5<z<-1.0: 25%目标仓位
```

### Python骨架
```python
def mean_reversion_signal(closes, window=60, entry_z=-2.0, exit_z=0):
    """均值回归信号"""
    if len(closes) < window + 1:
        return 0
    ma = np.mean(closes[-window:])
    std = np.std(closes[-window:])
    z = (closes[-1] - ma) / std

    if z < entry_z:      # 超卖 -> 买入
        return 1
    elif z > -entry_z:    # 超买 -> 卖出
        return -1
    elif z > exit_z and pos > 0:  # 回到均值 -> 平多
        return -1
    elif z < -exit_z and pos < 0: # 回到均值 -> 平空
        return 1
    return 0

def kelly_position(win_rate, avg_win, avg_loss):
    """凯利仓位"""
    b = avg_win / abs(avg_loss)
    p = win_rate
    return (p * b - (1-p)) / b  # 返回资金比例
```

---

## 策略三：价值+质量多因子打分
**来源**: 《因子投资》《Quality Investing》, Asness QMJ

### 核心理念
```
多维度综合打分, 选取高质量低估值的公司:
  价值分: EP + BP + CP 三因子等权
  质量分: ROE + 毛利率 + 低杠杆 三因子等权
  总分: 价值分 * 0.5 + 质量分 * 0.5
```

### 战术实现

```
因子计算:
  价值因子:
    EP = 每股收益 / 股价 (高EP = 低估)
    BP = 每股净资产 / 股价 (高BP = 低估)
    CP = 每股经营现金流 / 股价

  质量因子:
    ROE = 净利润 / 净资产
    Gross_Margin = (营收-成本) / 营收
    Debt_Equity = 总负债 / 净资产 (越低越好)

打分标准化:
  对全市场股票分行业, 每个因子计算Z-score
  价值分 = Z(EP) + Z(BP) + Z(CP)
  质量分 = Z(ROE) + Z(Gross_Margin) - Z(Debt_Equity)

买入条件:
  总分 > 行业平均 + 1个标准差
  且 质量分 > 0 (避免价值陷阱)

退出条件:
  总分降至行业平均以下
  或 质量分转为负值
  或 持有满12个月重新评估
```

### Python骨架
```python
def score_stock(ep, bp, cp, roe, gm, de):
    """价值+质量打分"""
    value = zscore(ep) + zscore(bp) + zscore(cp)
    quality = zscore(roe) + zscore(gm) - zscore(de)
    total = value * 0.5 + quality * 0.5
    if quality > 0 and total > 1:  # 质量过关且总分>1σ
        return "BUY", total
    elif total < 0:
        return "SELL", total
    return "HOLD", total
```

---

## 策略四：低波动防御
**来源**: 《低波动投资》, Ang et al. (2006)

### 核心理念
```
低波动异象: 低Beta/低波动股票长期跑赢高波动股票
在市场下跌时提供保护, 在上涨时适度参与
```

### 战术实现

```
信号生成:
  1. 计算过去1年日收益率的标准差
     volatility = std(daily_return, 252)
  2. 选择波动率最低的20%股票
  3. 在低波动池中按股息率排序

买入条件:
  入选低波动池 + 股息率 > 市场平均

退出条件:
  波动率超过全市场中位数
  或 股息率降至市场平均以下

仓位:
  等权配置低波动组合
  单个品种 <= 10%
  满仓运行
```

### Python骨架
```python
def low_vol_portfolio(stocks, vol_window=252, vol_percentile=0.2):
    """低波动组合构建"""
    vols = []
    for s in stocks:
        vol = np.std(s["returns"][-vol_window:]) * np.sqrt(252)
        vols.append(vol)
    threshold = np.percentile(vols, vol_percentile * 100)
    selected = [s for s, v in zip(stocks, vols) if v <= threshold]
    return selected  # 等权配置
```

---

## 策略五：ETF行业轮动
**来源**: 《因子投资》《量化交易》

### 核心理念
```
不同行业在不同经济周期中表现差异显著
通过动量+基本面判断行业强弱, 轮动配置
```

### 战术实现

```
信号生成:
  1. 计算各行业ETF过去N个月收益排名
     rank = argsort(returns[-N:])
  2. 取排名前3的行业
  3. 排除绝对动量为负的行业

买入条件:
  排名前3 + 绝对动量 > 0

轮动频率:
  每月第一个交易日重新评估
  卖出不再符合条件的行业
  买入新符合条件的行业

仓位:
  符合条件的行业等权配置
  最多持有5个行业
  最少持有3个行业
```

### Python骨架
```python
def sector_rotation(etf_returns, top_n=3, lookback=63):
    """行业轮动"""
    mom = {}
    for name, ret in etf_returns.items():
        r = ret[-1] / ret[-lookback] - 1
        abs_mom = ret[-1] / np.mean(ret[-lookback:])
        if abs_mom > 1:
            mom[name] = r
    ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_n]]  # 选出的行业
```

---

## 风险管理系统

### 核心风控原则 (来自《投资最重要的事》)
```
1. 保护本金优先于追求收益
2. 理解所处周期位置
3. 风险控制在前, 收益在后
4. 分散化是唯一的免费午餐
```

### 风控参数
```yaml
单品种止损: -8% (硬止损)
单品种仓位: <= 20%
总仓位限制: <= 80% (非极端行情)
最大回撤阈值: -15% (降低仓位到50%)
极端回撤阈值: -25% (清仓停止交易)
相关性控制: 持仓品种平均相关性 < 0.6
```

### 资金管理公式
```
f* = (p * b - q) / b    # 凯利公式
  p = 胜率
  q = 1-p
  b = 平均盈亏比

实际仓位 = f* * 0.25  (保守型: 使用25%凯利)
实际仓位 = f* * 0.5   (平衡型: 使用50%凯利)
```

---

## 策略选择指南

| 市场环境 | 推荐策略 | 依据 |
|---------|---------|------|
| 强趋势上涨 | 双动量趋势跟踪 | 动量策略在趋势市中表现最佳 |
| 震荡市中 | 均值回归组合 | 价格围绕价值波动时回归策略有效 |
| 熊市/下跌 | 低波动防御 | 低波动股在下跌市中跌幅更小 |
| 不确定/切换 | 价值+质量多因子 | 基本面因子长期稳健 |
| 任何市场 | ETF行业轮动 | 捕捉结构性机会 |

---

## 从理论到代码的落地路径

```
策略指标计算 (Pandas/NumPy)
    ↓
信号生成逻辑 (条件判断)
    ↓
回测验证 (vnpy BacktestingEngine)
    ↓
参数优化 (网格搜索)
    ↓
实盘运行 (xtquant / vnpy CTA)
    ↓
绩效跟踪 (Dashboard监控)
    ↓
策略调整 (基于IC/回撤反馈)
```

> 本文档基于经典文献精华提炼, 可直接作为策略开发的技术规格书使用
> 更新: 2026-07-24
