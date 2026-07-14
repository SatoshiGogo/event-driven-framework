# 事件研究框架

这个目录提供一套用于研究规则型入场/出场事件的框架。底层标准格式统一为：

- 行：交易日
- 列：股票代码
- 值：数值型宽表

市场数据、因子数据、风格控制数据都通过 `massim.load_mongo()` 读取，默认返回 pivot-table。

## 默认运行

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research
```

默认样本池为 Mongo 中可读取到的全市场股票，即不向 `ms.load_mongo()` 传入股票列表过滤。默认会读取：

- `MKT_Ashare/adj_open`
- `MKT_Ashare/adj_high`
- `MKT_Ashare/adj_low`
- `MKT_Ashare/adj_close`
- `MKT_Ashare/volume`
- `MKT_Ashare/amount`
- `DFJG/Lncap`，用于市值风格控制

默认测试：

- `ma20_break_after_trend`：强趋势后跌破 MA20
- `volume_stall_near_high`：高位放量滞涨
- `volume_breakout`：放量突破

输出：

- `outputs/event_study_report.html`：交互式 HTML 报告
- `outputs/event_summary.csv`：事件效应统计
- `outputs/event_observations.csv`：事件逐笔观测长表
- `outputs/recent_events.csv`：最近事件
- `outputs/path_*.csv`：事件前后平均路径
- `outputs/strategy_backtest_*.csv`：各持有期/手续费组合的逐日策略路径
- `outputs/strategy_metrics_*.csv`：年化收益、夏普、最大回撤、卡玛等策略指标

如果只想快速调试两只股票，可以显式传入 `--codes`：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research `
  --codes 300308 300274
```

全市场逐笔事件可能很多，HTML 报告默认只嵌入最多 100000 行逐笔观测用于浏览器端交互；完整逐笔结果仍写入 `outputs/event_observations.csv`。可以调整或关闭这个限制：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research --report-max-rows 200000
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research --report-max-rows 0
```

全市场计算耗时较长时，可以打开事件级多进程并行。并行粒度是“事件”，这样每个子进程只处理一个事件矩阵的全部标签和对照组，减少全市场宽表的重复复制：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research `
  --n-jobs 3
```

默认会显示 `EventStudyRunner.run` 的进度条；如需关闭：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research --no-progress
```

显著性使用“事件日期聚类 + 零效应中心化”的 bootstrap，默认重复 2000 次。同一天触发的股票会作为一个簇共同重采样，避免把同日横截面样本误当成完全独立。也可以请求 GPU 辅助；未安装 `cupy` 时会自动回落到 CPU：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research --use-gpu
```

## 使用 YAML 选择事件类

通过 `--event-config` 可以只运行配置文件中启用的事件。仓库内的
`event_config.yaml` 保留了原来的三个默认事件，并给出了一个默认关闭的因子事件示例：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research `
  --event-config event_study_framework/event_config.yaml
```

未传 `--event-config` 时默认读取包内的 `event_config.yaml`；传入其他 YAML 路径时，
只加载该文件中 `enabled: true` 的事件，不会再自动追加其他事件。配置格式如下：

```yaml
version: 1
run:
  codes: null
  start: "20130101"
  end: null
  output_dir: event_study_framework/outputs
  horizons: [1, 3, 5, 10, 20]
  label_types: [ret]
  bootstrap: 2000
  significance_level: 0.05
  control_modes: [same_date, same_date_style]
  style_name: lncap
  style_bins: 5
  purge_days: 10
  residualize_styles: false
  n_jobs: 1
  show_progress: true
  use_gpu: false
  report_max_rows: 100000
  cache_dir: event_study_framework/cache
  use_cache: true
  refresh_cache: false
  backtest_fees: [0.0, 0.002]
  annualization: 252
  risk_free_rate: 0.0

events:
  - class: ma_break
    enabled: true
    event_name: my_ma_break
    params:
      ma_window: 20
      trend_window: 60
      trend_return_threshold: 0.25
```

`run` 中的值会成为 `parse_args()` 的默认值；命令行显式参数优先。例如 YAML 中设置
`bootstrap: 2000` 后，临时传入 `--bootstrap 5000` 只覆盖本次运行。`--event-config`
本身默认指向包内的 `event_config.yaml`，因此正常运行不再需要重复填写整组参数。

`events.py` 中直接定义的所有具体 `Event` 子类都会被自动发现，不需要再写工厂函数或
手工注册表。YAML 既可以写完整类名（如 `BreakoutVolumeEvent`），也可以写自动生成的
snake_case 短名（如 `breakout_volume`）。`event_name` 是可选的显式报告标识；它只能
包含 ASCII 字母、数字、下划线、短横线和点，且所有启用事件的名称必须唯一。

也可以从可信 Python 模块加载自定义事件类。类必须继承独立的 `Event` 基类，
并实现 `compute()`：

```python
from typing import Dict, Optional

import pandas as pd

from event_study_framework import Event
from event_study_framework.data import MarketPanel


class TurnoverSpikeEvent(Event):
    """Example event supplied by an external trusted module."""

    required_fields = ("close", "volume")
    required_factors = {
        "turnover_rate": "FactorDB/turnover_rate",
    }

    def __init__(self, multiple: float = 2.0) -> None:
        """Initialize the turnover-spike threshold."""

        super().__init__(event_name="turnover_spike", direction="exit")
        self.multiple = multiple

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Return dates whose volume exceeds its rolling average."""

        if factors is None:
            raise ValueError("turnover_rate is required")
        turnover = factors["turnover_rate"]
        return turnover > turnover.rolling(20, min_periods=20).mean() * self.multiple
```

这就是新增事件所需的全部 Python 代码：`required_fields` 声明标准行情字段，
`required_factors` 直接声明“事件内使用的名称 -> Mongo 数据库/集合”。主程序会先解析
当前 YAML 启用的全部事件，再对行情字段和因子依赖取并集、去重后统一读取。多个事件可
共享同一因子；若同一个因子名被声明成两个不同数据源，程序会在读库前报错。

若该类位于 `my_events/custom.py`，配置中的类名写为
`my_events.custom:TurnoverSpikeEvent`。动态导入会执行目标模块代码，因此只应
加载可信配置和可信模块。

默认只构建 `ret` 收益标签，以减少约三分之二的标签计算和长表体积。如确实需要路径内最大有利/不利变动，可显式启用：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research `
  --label-types ret mfe mae
```

## 增量事件缓存

默认开启事件级持久缓存，目录由 YAML 的 `run.cache_dir` 决定。缓存键包含：

- 事件类、事件名称及构造参数
- 持有期、手续费、年化口径、控制组、Bootstrap、风格控制等研究参数
- 股票池、日期范围、数据截面摘要及因子/风格数据摘要
- 事件类源码与核心评估算法源码摘要

第一次运行会显示 `Event cache misses` 并写入结果；之后相同事件会显示在
`Event cache hits` 中，不再执行该事件的检测和统计。新增事件或修改某一个事件参数时，
只计算对应 miss。YAML 中移除或禁用的事件不会进入当前 CSV 和 HTML 报告，但原缓存不会
删除；恢复完全相同的配置时可以再次直接命中。

需要强制重算当前事件时使用：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research --refresh-cache
```

需要完全绕过缓存时使用 `--no-cache`。缓存采用本机 Python pickle 保存，请只使用本框架
自己生成且位于可信工作区中的缓存文件。若 Mongo 历史数据发生回溯修订而日期和抽样摘要
未变化，应使用 `--refresh-cache` 主动刷新。

## 核心设计

事件生成和事件评价解耦。事件生成程序只负责生成 0/1 宽表：

```python
event_matrix = pd.DataFrame(0, index=close.index, columns=close.columns)
event_matrix[condition] = 1
```

事件评价器接受：

```python
event_matrices = {
    "my_event": event_matrix,
}
```

## 控制时间漂移和风格影响

支持以下控制组：

- `same_date`：同一交易日的非事件股票作为对照，控制市场时间漂移。
- `same_date_style`：同一交易日、同一风格分桶的非事件股票作为对照，默认风格为 `lncap`。
- `event_stock_history`：同一股票历史非事件样本作为对照。
- `unconditional`：全样本非事件作为对照，主要用于粗略基准。

参数示例：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research `
  --control-modes same_date same_date_style `
  --style-name lncap `
  --style-bins 5 `
  --purge-days 10
```

也可以对未来收益标签先做风格残差化：

```powershell
& "D:\Anaconda3\envs\py385\python.exe" -m event_study_framework.run_event_research --residualize-styles
```

## 加入 Mongo 因子

因子数据源只在事件类中声明，不再使用 `--factor`，也不再在 YAML 的 `run` 中重复配置：

```python
class MyFactorEvent(Event):
    """Event that consumes two automatically loaded factors."""

    required_fields = ("close",)
    required_factors = {
        "quality": "FactorDB/quality_score",
        "momentum": "FactorDB/momentum_score",
    }
```

`compute()` 中直接使用 `factors["quality"]` 和 `factors["momentum"]`。启用该事件后，
主程序会自动加载这两个因子；禁用后则不会读取它们。

## 自定义事件

```python
from event_study_framework.study import EventMeta, EventStudyConfig, EventStudyRunner

event_matrices = {
    "my_event": my_event_matrix,  # 0/1 date-by-code DataFrame
}
event_meta = {
    "my_event": EventMeta(name="my_event", direction="entry", cooldown_days=10),
}

runner = EventStudyRunner(
    panel=panel,
    event_matrices=event_matrices,
    event_meta=event_meta,
    style_controls={"lncap": lncap},
    config=EventStudyConfig(control_modes=("same_date", "same_date_style")),
)
result = runner.run()
```

## 交互报告

HTML 报告支持：

- 选择事件
- 选择控制组
- 默认查看 `ret`；仅在命令行显式启用后显示 `MFE`、`MAE`
- 下拉选择持有期
- 下拉选择 0 或 0.2% 的单边手续费
- 拖动观察日期区间
- 查看事件后收益分布和相对控制组超额分布
- 对比事件路径与同日未触发事件的平均收益路径（横轴显示 T-/T/T+ 刻度）
- 查看事件策略、市场等权基准及相对基准超额的累计收益曲线
- 查看年化收益、夏普比率、最大回撤、卡玛比率、平均持仓和开平仓笔数
- 查看面向开仓/平仓方向的优势、命中率、聚类置信区间、p 值、FDR q 值和证据结论

策略回测在事件日之后的下一交易日按复权收盘价成交。持有期为 H 时，持有 H 个
收盘到收盘收益区间；同一股票持仓期间再次触发会把退出日延长至最新信号之后 H 天。
组合日收益是当日持仓股票收益的等权平均，股票离开有效交易池时目标仓位归零；手续费
按目标权重的单边换手收取，因此一次完整开平仓会支付两次所选费率。
