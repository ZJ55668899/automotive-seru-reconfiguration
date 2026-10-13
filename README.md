# automotive-seru-reconfiguration

This repository is the official implementation of "基于分阶段约束处理的汽车总装seru生产系统动态重构优化方法" (submitted to *Journal of Manufacturing Systems*).

本仓库为论文《基于分阶段约束处理的汽车总装seru生产系统动态重构优化方法》的核心代码，包含候选方案生成（合并/拆分/班组重分配算子）、C1—C4约束检查、离散事件仿真（DES）评价与代理辅助NSGA-III求解的实现。

## Requirements

- Python >= 3.10
- numpy >= 1.26
- simpy >= 4.0
- deap >= 1.4

Tested with Python 3.14, numpy 2.4, SimPy 4.1, DEAP 1.4.

## Data

算例由内置生成器按论文表4的参数现场生成（U形主线、四类产品族、六维班组技能、AGV路径网格），无需外部数据。

## Usage

- 性质检查（算子可行性闭包、交叉恢复、DES统计口径、代理模型）：

  `python tests/sanity_checks.py`

- 端到端演示（小规模算例，秒级完成，输出Pareto候选集）：

  `python examples/run_demo.py`

## Citation

论文录用后更新引用信息。
