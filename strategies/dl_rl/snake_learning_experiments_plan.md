# 贪吃蛇神经网络 / 强化学习实验方案文档

目标：在传统搜索算法已经作为 baseline 的基础上，额外尝试四类学习型方案，并用同一套本地模拟环境进行对比。

四个主方案：

```text
1. CNN + 监督学习 / 模仿学习
2. CNN + 强化学习
3. MLP + 监督学习 / 模仿学习
4. MLP + 强化学习
```

其中：

```text
CNN / MLP：模型结构
监督学习 / 强化学习：训练方式
```

---

## 0. 统一前提：先写本地环境

所有方案都依赖一个 Python 本地贪吃蛇环境。

环境必须复刻 OJ 规则：

```text
地图：20×20
边缘：#
空地：.
蛇头：H
蛇身：B
食物：F
障碍物：O
障碍物数量：10 个
初始蛇长：3，含蛇头
每吃 1 个食物：+10 分
每 N 步自然增长 1 节
如果吃食物和自然增长发生在同一步：只增长 1 节
增长方式：尾巴原位置不清空
撞墙 / 撞障碍 / 撞自身：死亡
不能直接反向移动，否则死亡
```

环境接口：

```python
state = env.reset()
next_state, reward, done, info = env.step(action)
```

动作定义：

```text
0 = W
1 = A
2 = S
3 = D
```

环境需要支持：

```text
1. 随机生成地图
2. 随机生成食物
3. 随机生成 N
4. 固定随机种子，方便复现实验
5. 记录每局分数、步数、死亡原因
```

---

## 1. 统一评估方式

所有算法都用同一批测试地图评估。

建议：

```text
训练集：随机生成大量地图
验证集：固定 100 张地图
测试集：固定 300 或 1000 张地图
```

记录指标：

```text
avg_score：平均分
min_score：最低分
score_std：分数波动
avg_steps：平均存活步数
death_rate：死亡率
food_count：平均吃食物数量
```

最终排序不要只看最高分，建议看：

```text
final_metric = avg_score - 0.3 * score_std + 0.2 * min_score
```

含义：

```text
平均分高
波动小
最差情况别太差
```

---

# 方案一：CNN + 监督学习 / 模仿学习

## 1.1 核心思想

先用传统搜索算法当老师，跑大量局，记录：

```text
当前局面 state -> 老师选择的动作 action
```

然后训练 CNN 模仿老师。

这不是直接优化得分，而是学习传统算法的决策模式。

## 1.2 数据生成

用传统搜索算法运行很多局。

每一步保存一个样本：

```text
input: 当前地图状态
label: 传统算法选择的方向
```

样本格式：

```python
(state_tensor, action_label)
```

其中 `action_label` 是：

```text
0 = W
1 = A
2 = S
3 = D
```

建议生成：

```text
至少 100,000 条样本
更好：500,000 ~ 2,000,000 条样本
```

## 1.3 CNN 输入设计

输入为多通道 20×20 矩阵。

推荐通道：

```text
channel 0：墙 #
channel 1：障碍物 O
channel 2：蛇头 H
channel 3：蛇身 B
channel 4：食物 F
channel 5：蛇尾
channel 6：当前方向 curDir / 3.0，填满整张图
channel 7：step % N / N，填满整张图
```

输入形状：

```text
8 × 20 × 20
```

## 1.4 网络结构

小 CNN 即可：

```text
Conv2D(8, 32, kernel_size=3, padding=1)
ReLU
Conv2D(32, 64, kernel_size=3, padding=1)
ReLU
Conv2D(64, 64, kernel_size=3, padding=1)
ReLU
Flatten
Linear(64*20*20, 256)
ReLU
Linear(256, 4)
```

输出 4 个动作 logits。

## 1.5 损失函数

使用交叉熵：

```text
loss = CrossEntropyLoss(logits, action_label)
```

训练目标：

```text
让模型预测老师选择的动作
```

## 1.6 推理方式

每一步：

```text
1. 把当前局面转成 8×20×20 tensor
2. CNN 输出 4 个动作分数
3. 按分数从高到低尝试动作
4. 如果动作非法或会死，尝试下一个
5. 选择第一个合法动作
```

注意：不要盲目执行模型最高分动作，要做安全过滤。

## 1.7 优点

```text
训练稳定
实现相对简单
能学习地图空间模式
可以作为后续 RL 的预训练模型
```

## 1.8 缺点

```text
上限受老师限制
老师错误会被模型学走
不直接优化最终得分
C 语言部署 CNN 比较麻烦
```

---

# 方案二：CNN + 强化学习

## 2.1 核心思想

不让模型模仿老师，而是让模型自己玩游戏，通过奖励学习怎么走得分更高。

CNN 负责看地图，RL 负责训练策略。

## 2.2 推荐算法

优先尝试 DQN。

原因：

```text
动作离散：W/A/S/D
输入是网格图
输出是 4 个动作的价值 Q(s, a)
```

DQN 输出：

```text
Q(W), Q(A), Q(S), Q(D)
```

选择 Q 值最高的动作。

## 2.3 输入设计

和 CNN 监督学习一致：

```text
8 × 20 × 20
```

通道：

```text
墙
障碍物
蛇头
蛇身
食物
蛇尾
当前方向
step % N
```

## 2.4 网络结构

可以复用 CNN 监督学习结构，只是输出含义不同。

```text
Conv2D(8, 32, 3, padding=1)
ReLU
Conv2D(32, 64, 3, padding=1)
ReLU
Conv2D(64, 64, 3, padding=1)
ReLU
Flatten
Linear(..., 256)
ReLU
Linear(256, 4)
```

输出是 4 个 Q 值。

## 2.5 奖励函数

推荐先用较稳的 reward：

```text
吃食物：+10
死亡：-50
每走一步：-0.01
普通移动：0
```

更贴近 OJ 的版本：

```text
吃食物：+10
死亡：-20
每走一步：0
```

注意：

```text
只用最终分数作为 reward 最贴近目标，但太稀疏，训练会很慢。
中间 reward 是为了帮助模型更快知道哪些动作有效。
```

## 2.6 DQN 训练流程

```text
1. 初始化 Q 网络
2. 初始化 target Q 网络
3. 初始化 replay buffer
4. 每局 env.reset()
5. 每一步用 epsilon-greedy 选择动作
6. 执行动作，得到 next_state, reward, done
7. 存入 replay buffer
8. 随机采样 batch
9. 用 DQN loss 更新 Q 网络
10. 定期同步 target network
```

DQN 目标：

```text
target = reward + gamma * max_a Q_target(next_state, a)
```

如果 done：

```text
target = reward
```

## 2.7 推荐超参数

```text
episodes：50,000 ~ 200,000
replay buffer：100,000
batch size：64
learning rate：1e-4
gamma：0.99
epsilon：1.0 线性下降到 0.05
target network update：每 1000 或 5000 步
max steps per episode：可设 1000 或 2000，防止无限绕圈
```

## 2.8 可选升级

```text
Double DQN
Dueling DQN
Prioritized Replay
NoisyNet
```

建议顺序：

```text
先普通 DQN 跑通
再 Double DQN
再 Dueling DQN
最后再考虑 prioritized replay
```

## 2.9 优点

```text
直接围绕得分优化
理论上可能超过传统算法
能学习复杂空间模式
```

## 2.10 缺点

```text
训练不稳定
数据需求大
容易学到短视策略
C 语言部署 CNN 麻烦
24 小时内不保证超过传统搜索 baseline
```

---

# 方案三：MLP + 监督学习 / 模仿学习

## 3.1 核心思想

仍然让传统搜索算法当老师，但不把整张地图喂给模型。

而是人工提取特征，再让 MLP 学：

```text
特征向量 -> 老师动作
```

## 3.2 特征设计

推荐按“四个方向”提取特征。

对每个方向 d，模拟走一步，如果不死，计算：

```text
1. 该方向是否安全 safe[d]
2. 走完后到食物的 BFS 距离 distFood[d]
3. 走完后可达空间 reachable[d]
4. 走完后是否能到蛇尾 canReachTail[d]
5. 走完后是否被困 trapped[d]
6. 该方向是否反向 opposite[d]
7. 走完后是否吃到食物 eatFood[d]
```

全局特征：

```text
蛇长度 len / 400
当前分数 score / 1000
当前步数 step / 1000
step % N / N
N / 512
当前方向 curDir one-hot
食物与蛇头相对位置：drFood / 20, dcFood / 20
曼哈顿距离 foodDist / 40
```

最终特征向量示例：

```text
4 个方向 × 7 个方向特征 = 28
全局特征约 10
总维度约 38~50
```

## 3.3 数据生成

和 CNN 监督学习一样：

```text
用传统搜索算法跑大量局
每一步保存：
features -> teacher_action
```

建议数据量：

```text
100,000 ~ 1,000,000 条
```

## 3.4 网络结构

```text
Input(feature_dim)
Linear(feature_dim, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 64)
ReLU
Linear(64, 4)
```

输出 4 个动作 logits。

## 3.5 损失函数

```text
CrossEntropyLoss(logits, teacher_action)
```

## 3.6 推理方式

```text
1. 对当前局面提取特征
2. MLP 输出 4 个动作分数
3. 按分数从高到低尝试
4. 选择第一个合法且不死的动作
```

## 3.7 优点

```text
训练快
部署简单
C 语言实现前向传播容易
参数少，能硬编码进 C
对小作业很现实
```

## 3.8 缺点

```text
看不到完整地图空间形状
效果依赖人工特征设计
上限受老师限制
```

---

# 方案四：MLP + 强化学习

## 4.1 核心思想

用人工特征表示当前局面，然后用强化学习训练 MLP，让它直接优化游戏得分。

也就是：

```text
手工特征 -> MLP -> 四个动作价值
```

训练方式用 DQN。

## 4.2 输入特征

可以复用 MLP 监督学习的特征。

方向特征：

```text
safe[d]
distFood[d]
reachable[d]
canReachTail[d]
trapped[d]
opposite[d]
eatFood[d]
```

全局特征：

```text
len / 400
score / 1000
step / 1000
step % N / N
N / 512
curDir one-hot
food relative row / 20
food relative col / 20
foodManhattan / 40
```

## 4.3 网络结构

```text
Input(feature_dim)
Linear(feature_dim, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 64)
ReLU
Linear(64, 4)
```

输出：

```text
Q(W), Q(A), Q(S), Q(D)
```

## 4.4 奖励函数

推荐：

```text
吃食物：+10
死亡：-50
每走一步：-0.01
```

可测试变体：

```text
Variant A:
吃食物 +10，死亡 -20，每步 0

Variant B:
吃食物 +10，死亡 -50，每步 -0.01

Variant C:
吃食物 +10，死亡 -100，每步 -0.01
```

## 4.5 训练流程

同 DQN：

```text
1. 初始化 Q 网络和 target 网络
2. 初始化 replay buffer
3. epsilon-greedy 探索
4. 与环境交互，存经验
5. 从 buffer 采样 batch
6. 更新 Q 网络
7. 定期同步 target 网络
8. 固定验证集评估
```

## 4.6 推荐超参数

```text
episodes：50,000 ~ 200,000
buffer size：100,000
batch size：64 或 128
learning rate：1e-4 或 3e-4
gamma：0.99
epsilon：1.0 -> 0.05
target update：1000 steps
```

## 4.7 优点

```text
比 CNN + RL 更快
比 CNN 更容易部署到 C
直接围绕得分优化
可作为 OJ 落地方案候选
```

## 4.8 缺点

```text
依赖手工特征
可能学不到复杂空间形状
如果特征设计差，上限会受限
```

---

# 四个方案对比

| 方案 | 输入 | 训练目标 | 优点 | 缺点 | OJ 部署 |
|---|---|---|---|---|---|
| CNN + 监督学习 | 整张地图 | 模仿老师动作 | 稳定，能看空间 | 上限受老师限制 | 麻烦 |
| CNN + 强化学习 | 整张地图 | 最大化得分 | 上限高，目标直接 | 难训，部署麻烦 | 很麻烦 |
| MLP + 监督学习 | 手工特征 | 模仿老师动作 | 快，部署简单 | 依赖特征，上限受老师 | 容易 |
| MLP + 强化学习 | 手工特征 | 最大化得分 | 快，目标直接，部署现实 | 特征限制表达 | 较容易 |

---

# 推荐实验顺序

## 第一阶段：本地环境

```text
先写 Python 环境
用随机动作测试
用传统搜索算法测试
确认分数和规则正确
```

## 第二阶段：MLP + 监督学习

原因：

```text
最容易跑通
可以验证数据生成和特征提取是否正确
也方便未来部署 C
```

## 第三阶段：CNN + 监督学习

原因：

```text
验证整图输入是否能学会传统算法
可以和 MLP 监督学习对比
```

## 第四阶段：MLP + 强化学习

原因：

```text
训练比 CNN RL 快
可以初步验证 RL 是否能提升得分
```

## 第五阶段：CNN + 强化学习

原因：

```text
最重
最难训
但理论上空间表达能力最强
```

---

# 实验记录模板

每次实验记录：

```text
实验编号：
方案：
模型结构：
输入特征：
reward 设计：
训练地图数量：
训练 episode：
学习率：
batch size：
gamma：
epsilon 衰减方式：
验证集 avg_score：
验证集 min_score：
验证集 score_std：
验证集 avg_steps：
死亡主要原因：
备注：
```

示例：

```text
EXP-001
方案：MLP + 监督学习
老师：BFS + DFS候选路径 + 评分函数
样本数：300000
feature_dim：42
模型：42-128-128-64-4
验证 avg_score：xxx
验证 min_score：xxx
验证 std：xxx
结论：xxx
```

---

# 部署判断

如果最终要上 OJ，优先考虑：

```text
1. 传统搜索算法
2. MLP + 强化学习
3. MLP + 监督学习
```

不优先考虑：

```text
CNN + 强化学习
CNN + 监督学习
```

原因：

```text
OJ 版通常只能提交 C 源码；
CNN 参数多，C 推理复杂；
MLP 参数少，更容易硬编码和实现。
```

---

# 总结

四个方案本质区别：

```text
CNN：直接看地图，表达能力强，部署难
MLP：看人工特征，表达能力依赖特征，部署简单

监督学习：模仿传统算法，训练稳定，但上限受老师影响
强化学习：直接围绕得分优化，理论上上限高，但训练难
```

最现实的路线：

```text
传统搜索 baseline 已经在做；
学习型方案先从 MLP + 监督学习开始；
再试 MLP + 强化学习；
CNN 方案作为实验对照；
最终用统一测试集比较平均分、最低分和稳定性。
```
