# automotive-seru-reconfiguration

This repository is the official implementation of "基于分阶段约束处理的汽车总装seru生产系统动态重构优化方法" (submitted to *Journal of Manufacturing Systems*).

本仓库为论文《基于分阶段约束处理的汽车总装seru生产系统动态重构优化方法》的核心代码，包含候选方案生成（合并/拆分/班组重分配算子）、C1—C4约束检查、离散事件仿真（DES）评价与代理辅助NSGA-III求解的实现。

## Requirements

- Python >= 3.10
- numpy >= 1.26
- simpy >= 4.0
- deap >= 1.4
- matplotlib >= 3.8

Tested with Python 3.14, numpy 2.4, SimPy 4.1, DEAP 1.4.

## Data

算例由内置生成器按论文表4的参数现场生成（U形主线、四类产品族、六维班组技能、AGV路径网格），无需外部数据。固定工装比例可用 `make_instance(..., fixture_share=...)` 覆盖（表4默认0.30）。

## Usage

- 性质检查（算子可行性闭包、交叉恢复、DES统计口径、代理模型）：

  `python tests/sanity_checks.py`

- 端到端演示（小规模算例，秒级完成，输出Pareto候选集）：

  `python examples/run_demo.py`

- 演示并输出Pareto前沿图：

  `python examples/plot_demo_front.py`（生成 `examples/demo_front.png`）

- 固定工装比例灵敏度的配对核验（5.10节作用路径）：

  `python examples/fixture_share_sensitivity.py`

演示输出示例：

```text
instance S-demo: N=12, teams=9, serus in X_prev=2, lambda=0.098/min
X_prev: f1=9.28 h, f2=0.0 kCNY, f3=0.123, rho=4.402%, orders=59
search: 15 generations, 50 full DES calls, wall clock 0.0 min
final front (DES-verified, R=3 replications; 0/7 executable at tau=5%):
  K=2: f1=8.29 h, f2=7.8 kCNY, f3=0.034, rho=8.832%±4.41%  [NOT executable (best-effort)]
  K=4: f1=8.10 h, f2=18.0 kCNY, f3=0.200, rho=9.002%±4.42%  [NOT executable (best-effort)]
  ...
total wall clock: 1.5 s
```

注：演示算例规模极小，最终复核中所有方案的 ρ 均高于 5% 阈值，因此按算法2的最后一步筛选后无可执行方案，返回带 `executable=False` 标记的复核前沿作为兜底输出。主实验规模下返回集合经 ρ≤τ 筛选且重新进行非支配排序。

## Citation

论文录用后更新引用信息。
