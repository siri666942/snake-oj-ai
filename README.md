# 贪吃蛇 OJ 自动策略

这是一个用于课程作业 OJ 版贪吃蛇的自动策略项目。当前可提交版本是 C 语言交互程序，使用 BFS、DFS 候选路径搜索、完整蛇移动模拟和安全评分函数来自动选择移动方向。

## 目录结构

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── 大作业题目-贪吃蛇.docx
├── tools/
│   └── judge.py
└── strategies/
    ├── bfs_dfs_survival/
    │   ├── snake_oj.c
    │   └── snake_oj_algorithm_plan.md
    └── dl_rl/
        ├── README.md
        ├── snake_ai/
        ├── tests/
        └── snake_learning_experiments_plan.md
```

## 文件说明

- `strategies/bfs_dfs_survival/snake_oj.c`
  - OJ 提交用源码。
  - 读取 20 行地图和第 21 行增长频率 `N`。
  - 每步输出方向和移动前分数，并调用 `fflush(stdout)`。
  - 收到 `100 100` 后输出碰撞前地图和分数。

- `strategies/bfs_dfs_survival/snake_oj_algorithm_plan.md`
  - 当前 C 语言策略的算法设计文档。
  - 说明状态维护、BFS、DFS 候选路径、评分函数和保命策略。

- `tools/judge.py`
  - 本地模拟 OJ 交互的估分工具。
  - 会自动编译指定策略目录下的 `snake_oj.c`。
  - 默认跑 10 个不同 `N` 的随机样例，并输出原始分、加权总分和基础分档位。

- `strategies/dl_rl/snake_learning_experiments_plan.md`
  - 后续神经网络 / 强化学习实验方案。
  - 目前是研究计划，不是 OJ 提交版本。

- `strategies/dl_rl/snake_ai/`
  - Python 版实验环境和学习型策略代码。
  - 包含 OJ 规则环境、教师策略、特征提取、监督学习、DQN、评估和 MLP 导出工具。

- `strategies/dl_rl/tests/`
  - Python 环境规则测试，覆盖吃食物增长、反向死亡、尾巴格移动等关键规则。

## 环境要求

- C 编译器：`gcc`
- Python：建议 Python 3.10 或更高
- 学习实验可选依赖：`numpy`、`torch`、`pytest`

在 Windows PowerShell 中确认：

```powershell
gcc --version
python --version
```

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

## 编译 OJ 程序

进入策略目录：

```powershell
cd C:\Users\lenovo\Desktop\贪吃蛇\strategies\bfs_dfs_survival
gcc .\snake_oj.c -O2 -Wall -Wextra -o .\snake_oj.exe
```

OJ 提交时通常只需要提交：

```text
strategies/bfs_dfs_survival/snake_oj.c
```

不要提交本地生成的 `.exe` 文件。

## 本地估分

在项目根目录运行：

```powershell
cd C:\Users\lenovo\Desktop\贪吃蛇
python .\tools\judge.py
```

默认测试：

- 使用策略目录：`strategies/bfs_dfs_survival`
- 固定随机种子：`20260516`
- 测试 `N = 1,2,4,8,16,32,64,128,256,512`

输出示例：

```text
strategy=C:\Users\lenovo\Desktop\贪吃蛇\strategies\bfs_dfs_survival
base_seed=20260516
case,N,raw_score,reason
1,1,80,dead
...
weighted_total=753.08
base_grade=50
note=本地随机模拟分数只用于估计，OJ隐藏数据分数以正式提交为准。
```

其中：

- `raw_score`：单个本地样例吃食物得到的原始分。
- `reason=dead`：该局最终死亡结束，属于正常结束方式。
- `weighted_total`：按作业题目中的 `1 / (log2(N) + 1)` 权重计算的本地加权总分。
- `base_grade`：按题目基础分规则换算出的基础分档位。

换一个随机种子：

```powershell
python .\tools\judge.py --seed 12345
```

每次使用新随机种子：

```powershell
python .\tools\judge.py --random
```

测试其他策略目录：

```powershell
python .\tools\judge.py .\strategies\bfs_dfs_survival
```

## Python 学习实验

这些内容不是 OJ 必交版本，主要用于后续探索 CNN / MLP / 监督学习 / 强化学习策略。

快速烟测：

```powershell
cd C:\Users\lenovo\Desktop\贪吃蛇
python -m strategies.dl_rl.snake_ai.run_smoke
```

运行规则测试：

```powershell
pytest .\strategies\dl_rl\tests
```

生成教师数据：

```powershell
python -m strategies.dl_rl.snake_ai.data --samples 300000 --output strategies/dl_rl/data/teacher_300k.npz
```

训练 MLP 监督学习模型：

```powershell
python -m strategies.dl_rl.snake_ai.train_supervised --data strategies/dl_rl/data/teacher_300k.npz --kind mlp --output strategies/dl_rl/checkpoints/mlp_supervised.pt --epochs 10
```

评估教师策略或模型策略：

```powershell
python -m strategies.dl_rl.snake_ai.evaluate --policy teacher --cases 300
python -m strategies.dl_rl.snake_ai.evaluate --policy mlp --checkpoint strategies/dl_rl/checkpoints/mlp_supervised.pt --cases 300
```

更多命令见 `strategies/dl_rl/README.md`。

## 当前算法摘要

当前 OJ 策略不是只走到食物的最短路，而是：

1. 用 BFS 计算蛇头到当前食物的最短距离。
2. 用 DFS 枚举不超过 `最短距离 + DELTA` 的安全候选路径。
3. 对每条候选路径完整模拟蛇移动，包括吃食物、自然增长、尾巴移动和反向移动限制。
4. 对吃完食物后的局面评分，重点看可达空间、能否追尾、路径长度和困死风险。
5. 如果没有安全吃食物路径，则优先追尾，再选择可达空间最大的安全方向。

## 注意事项

- 本地估分不等于 OJ 隐藏数据最终分数，只能用于比较策略好坏。
- `.gitignore` 已忽略 `.exe`、Python 缓存、训练数据和模型 checkpoint，后续上传仓库时建议只上传源码、文档和工具脚本。
- 本项目使用 MIT License 开源。

