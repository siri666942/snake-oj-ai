# 贪吃蛇 DL/RL 实验框架说明

这个目录用于执行 `snake_learning_experiments_plan.md` 里的四类学习方案：

1. `MLP + 监督学习`
2. `CNN + 监督学习`
3. `MLP + DQN 强化学习`
4. `CNN + DQN 强化学习`

实验目标不是替代现有 OJ 版 C 程序，而是在同一套本地规则环境里训练和评估学习型策略。现有 `strategies/bfs_dfs_survival/snake_oj.c` 仍然是当前最稳的 OJ 提交兜底方案；学习模型只有在本地评估稳定接近或超过它时，才值得继续移植到 C。

## 目录结构

```text
strategies/dl_rl/
├── snake_learning_experiments_plan.md   原始实验方案
├── README.md                            本说明文档
├── snake_ai/                            Python 实验代码
│   ├── env.py                           OJ 规则环境 SnakeEnv
│   ├── teacher.py                       BFS/DFS teacher 和 fast teacher
│   ├── features.py                      CNN/MLP 输入特征
│   ├── models.py                        MLPPolicy / CNNPolicy
│   ├── data.py                          teacher 数据生成
│   ├── train_supervised.py              监督学习训练
│   ├── train_dqn.py                     DQN 强化学习训练
│   ├── train_all.py                     一行命令训练完整流水线
│   ├── evaluate.py                      统一评估
│   ├── export_mlp_c.py                  MLP 权重导出为 C 头文件
│   └── run_smoke.py                     最小端到端自检
├── tests/                               环境规则单测
├── data/                                生成的训练数据
├── checkpoints/                         训练得到的模型
├── results/                             评估 CSV
└── export/                              导出的 C 权重
```

## 核心设计

### 环境

`SnakeEnv` 复刻作业 OJ 规则：

- 地图固定 `20x20`
- `#` 墙、`O` 障碍、`H` 蛇头、`B` 蛇身、`F` 食物
- 初始蛇长 3
- 吃食物加 10 分并刷新食物
- 每 `N` 步自然增长 1 节
- 吃食物和自然增长同一步只增长 1 节
- 普通移动时可以进入旧尾巴格，增长时不能进入旧尾巴格
- 反向移动、撞墙、撞障碍、撞自身都会死亡

### Teacher

有两个 teacher：

- `teacher`：强 teacher，移植了现有 C baseline 的 BFS/DFS 候选路径思想，更适合正式 imitation learning 数据。
- `fast_teacher`：轻量 teacher，优先 BFS 找食物，失败后追尾或走最大空间；速度更快，适合 smoke test 和快速造数据。

正式训练建议优先用 `teacher strong`，但如果数据生成太慢，可以先用 `fast_teacher` 跑通训练流程。

### 输入特征

CNN 输入是 `8 x 20 x 20`：

- 墙
- 障碍物
- 蛇头
- 蛇身
- 食物
- 蛇尾
- 当前方向
- `step % N / N`

MLP 输入是手工特征：

- 四个方向分别计算安全性、到食物距离、可达空间、能否追尾、是否被困、是否反向、是否吃食物
- 全局加入蛇长、分数、步数、`N`、当前方向 one-hot、食物相对位置等

## 推荐执行顺序

所有命令都在项目根目录运行：

```powershell
cd C:\Users\lenovo\Desktop\贪吃蛇
```

### 1. 一行命令跑完整训练

如果你不想手动分阶段执行，直接用封装好的流水线：

```powershell
python -m strategies.dl_rl.snake_ai.train_all
```

默认 `full` 预设会自动执行：

1. 跑环境规则单测
2. 生成 `300000` 条 strong teacher 数据
3. 训练 MLP supervised
4. 训练 CNN supervised
5. 用 MLP supervised 初始化 MLP DQN 并训练
6. 训练 CNN DQN
7. 统一评估 teacher、MLP、CNN、DQN
8. 导出 MLP 权重到 `strategies/dl_rl/export/mlp_weights.h`

如果中途断了，或者已经生成过数据/模型，可以加 `--resume` 跳过已有产物：

```powershell
python -m strategies.dl_rl.snake_ai.train_all --resume
```

如果只想快速检查流水线：

```powershell
python -m strategies.dl_rl.snake_ai.train_all --preset smoke
```

如果想比 smoke 更认真，但不想等完整长训：

```powershell
python -m strategies.dl_rl.snake_ai.train_all --preset fast
```

如果只关心最可能落地到 OJ 的 MLP，可以跳过 CNN：

```powershell
python -m strategies.dl_rl.snake_ai.train_all --skip-cnn
```

如果只想做监督学习，不跑 DQN：

```powershell
python -m strategies.dl_rl.snake_ai.train_all --skip-dqn
```

如果想把某次训练产物放到单独目录，避免覆盖默认输出：

```powershell
python -m strategies.dl_rl.snake_ai.train_all --preset smoke --root strategies/dl_rl/runs/smoke_001
```

流水线会写出：

```text
strategies/dl_rl/pipeline_config.json
strategies/dl_rl/data/teacher_<teacher>_<samples>.npz
strategies/dl_rl/checkpoints/*.pt
strategies/dl_rl/results/pipeline_<preset>_metrics.json
strategies/dl_rl/export/mlp_weights.h
```

### 2. 手动分阶段执行

下面这些命令保留给需要单独调某一阶段的时候使用。正常训练优先用 `train_all.py`。

#### 2.1 先跑最小自检

```powershell
python -m strategies.dl_rl.snake_ai.run_smoke
```

这个命令会做一遍完整链路：

- 生成 256 条 teacher 样本
- 训练 1 个 smoke MLP 监督模型
- 训练 1 个 smoke CNN 监督模型
- 跑极小规模 MLP DQN
- 跑极小规模 CNN DQN
- 评估 `fast_teacher`

这一步只验证代码能跑通，不代表模型效果。

#### 2.2 跑环境规则测试

```powershell
python -m pytest strategies/dl_rl/tests -q
```

目前测试覆盖：

- 吃食物和自然增长同一步只增长 1 节
- 反向移动死亡
- 普通移动可进旧尾巴格，增长时不可进旧尾巴格

#### 2.3 生成监督学习数据

正式 teacher 数据：

```powershell
python -m strategies.dl_rl.snake_ai.data --samples 300000 --output strategies/dl_rl/data/teacher_300k.npz --teacher strong
```

快速 teacher 数据：

```powershell
python -m strategies.dl_rl.snake_ai.data --samples 300000 --output strategies/dl_rl/data/teacher_fast_300k.npz --teacher fast
```

如果要追求更高效果，可以把 `--samples` 提到 `1000000` 或更高。`strong` 数据更接近现有 C baseline，但生成更慢。

#### 2.4 训练监督模型

训练 MLP：

```powershell
python -m strategies.dl_rl.snake_ai.train_supervised --data strategies/dl_rl/data/teacher_300k.npz --kind mlp --output strategies/dl_rl/checkpoints/mlp_supervised.pt --epochs 10
```

训练 CNN：

```powershell
python -m strategies.dl_rl.snake_ai.train_supervised --data strategies/dl_rl/data/teacher_300k.npz --kind cnn --output strategies/dl_rl/checkpoints/cnn_supervised.pt --epochs 10
```

MLP 是更现实的 OJ 移植候选。CNN 用于实验对比，不优先移植到 C。

#### 2.5 训练 DQN 强化学习模型

MLP DQN 可以从监督模型初始化：

```powershell
python -m strategies.dl_rl.snake_ai.train_dqn --kind mlp --output strategies/dl_rl/checkpoints/mlp_dqn.pt --episodes 50000 --init-checkpoint strategies/dl_rl/checkpoints/mlp_supervised.pt
```

CNN DQN：

```powershell
python -m strategies.dl_rl.snake_ai.train_dqn --kind cnn --output strategies/dl_rl/checkpoints/cnn_dqn.pt --episodes 50000
```

DQN 默认奖励：

- 吃食物：`+10`
- 死亡：`-50`
- 每步：`-0.01`

50,000 episodes 在 CPU 上会比较久。建议先用较小 episodes 确认曲线，再拉长训练。

#### 2.6 统一评估

评估强 teacher：

```powershell
python -m strategies.dl_rl.snake_ai.evaluate --policy teacher --cases 300
```

评估 fast teacher：

```powershell
python -m strategies.dl_rl.snake_ai.evaluate --policy fast_teacher --cases 300
```

评估 MLP：

```powershell
python -m strategies.dl_rl.snake_ai.evaluate --policy mlp --checkpoint strategies/dl_rl/checkpoints/mlp_supervised.pt --cases 300
```

评估 CNN：

```powershell
python -m strategies.dl_rl.snake_ai.evaluate --policy cnn --checkpoint strategies/dl_rl/checkpoints/cnn_supervised.pt --cases 300
```

评估结果会追加写入：

```text
strategies/dl_rl/results/eval.csv
```

主要指标：

- `avg_score`：平均分
- `min_score`：最低分
- `score_std`：分数波动
- `avg_steps`：平均存活步数
- `death_rate`：死亡率
- `food_count`：平均吃食物数量
- `final_metric`：综合指标，公式为 `avg_score - 0.3 * score_std + 0.2 * min_score`

## MLP 导出到 C

如果 MLP 在评估中接近或超过现有 BFS/DFS baseline，可以先导出权重：

```powershell
python -m strategies.dl_rl.snake_ai.export_mlp_c --checkpoint strategies/dl_rl/checkpoints/mlp_supervised.pt --output strategies/dl_rl/export/mlp_weights.h
```

这个命令只生成权重头文件，不会自动生成完整 OJ C 程序。真正提交前还需要把：

- MLP 特征提取
- 前向传播
- 安全动作过滤
- OJ 交互输入输出

整合进 C 代码，并用 `tools/judge.py` 回归测试。

## 当前已验证结果

当前 smoke 链路已跑通，但 smoke 模型只用了 256 条样本和 1 个 epoch，不能代表正式效果。

已验证命令：

```powershell
python -m compileall strategies/dl_rl
python -m pytest strategies/dl_rl/tests -q
python -m strategies.dl_rl.snake_ai.run_smoke
python tools/judge.py strategies/bfs_dfs_survival
```

现有 C baseline 本地固定种子结果：

```text
weighted_total=753.08
base_grade=50
```

所以最终 OJ 提交建议仍以现有 BFS/DFS C 版本为兜底。学习模型要先通过 300 个以上固定测试图评估，再决定是否移植。

## 常见判断

如果只是为了完成作业并拿稳定分：

```text
优先使用 strategies/bfs_dfs_survival/snake_oj.c
```

如果要做学习实验报告：

```text
python -m strategies.dl_rl.snake_ai.train_all --preset full
```

如果要冲更高分：

```text
先训练 MLP supervised，再用它初始化 MLP DQN；只有当评估结果稳定接近 baseline，才考虑导出到 C。
```
