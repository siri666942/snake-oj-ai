# 贪吃蛇：搜索 + 深度学习方案文档

目标：在传统搜索算法基础上，引入神经网络来学习“局面价值”，用模型替代或增强手写评分函数。

核心思想：

```text
搜索负责往前推演；
神经网络负责评价搜索到的局面好不好。
```

这类方案不是纯强化学习，也不是纯监督模仿。它更像：

```text
有限步搜索 + Value Network
```

也可以扩展成：

```text
有限步搜索 + Policy Network + Value Network
```

---

## 1. 为什么要做搜索 + 深度学习？

传统搜索算法通常会用手写评分函数：

```text
score =
    + reachableSpace
    + canReachTail
    - pathLength
    - trappedPenalty
```

这些特征有用，但它们是人写的，表达能力有限。

深度学习可以尝试学习：

```text
什么样的蛇形状未来更容易高分？
什么样的局面看起来空间大但其实会被困死？
什么样的路径虽然短但后续风险高？
```

所以本方案的目标是：

```text
手写评分函数 → 学出来的评分函数
```

---

## 2. 基本结构

每一步决策时：

```text
1. 枚举当前可行动作 W/A/S/D
2. 对每个动作模拟一步
3. 继续向未来搜索 depth 步
4. 到达搜索叶子节点时，用神经网络评价局面
5. 选总价值最高的路径的第一步
```

伪代码：

```text
choose_action(state):
    best_action = None
    best_value = -INF

    for action in legal_actions:
        next_state = simulate(state, action)
        value = search(next_state, depth - 1)

        if value > best_value:
            best_value = value
            best_action = action

    return best_action
```

搜索函数：

```text
search(state, depth):
    if state is dead:
        return gained_score - death_penalty

    if depth == 0:
        return value_network(state)

    best = -INF

    for action in legal_actions:
        next_state = simulate(state, action)
        value = immediate_gain + search(next_state, depth - 1)
        best = max(best, value)

    return best
```

注意：

```text
死亡不是把历史分数清零；
如果搜索中已经吃到食物，应该保留 gained_score。
```

---

## 3. Value Network 是什么？

Value Network 学的是：

```text
V(state) = 从当前局面继续玩，未来大概还能拿多少分
```

它输入一个局面，输出一个数字。

例如：

```text
V(state) = 60
```

含义：

```text
从这个局面开始，后续大概还能吃 6 个食物，也就是 60 分。
```

---

## 4. Value Network 的训练数据怎么来？

让已有策略跑很多局，保存完整轨迹。

一局轨迹：

```text
S0, S1, S2, ..., ST
```

对应每一步当前分数：

```text
score0, score1, score2, ..., scoreT
```

最终分数：

```text
final_score
```

对每个状态 `St`，训练标签是：

```text
future_score = final_score - score_t
```

例子：

```text
第 0 步：当前分数 0，最终分数 150
label = 150 - 0 = 150

第 30 步：当前分数 40，最终分数 150
label = 150 - 40 = 110

第 80 步：当前分数 100，最终分数 150
label = 150 - 100 = 50
```

所以训练样本是：

```text
state -> future_score
```

注意：

```text
即使最后死亡，也不应该把前面的状态都标成负数。
死亡只是游戏结束，不会清空已经获得的分数。
```

---

## 5. 数据来源

可以混合使用多种策略生成轨迹：

```text
1. 传统搜索算法
2. BFS 贪心算法
3. 随机加安全过滤算法
4. RL 算法
5. 人工调参后的高分算法
```

推荐不要只用一个策略的数据。

原因：

```text
如果只用一个策略，Value Network 只会评价这个策略能到达的局面。
混合数据可以让模型见到更多不同形状和风险局面。
```

推荐数据量：

```text
小实验：50,000 ~ 200,000 个状态
正式实验：500,000 ~ 2,000,000 个状态
```

---

## 6. Value Network 输入形式

有两种路线。

---

# 方案 A：MLP Value Network

## A.1 输入

输入手工特征向量。

推荐特征：

### 全局特征

```text
len / 400
score / 1000
step / 1000
N / 512
step % N / N
当前方向 one-hot
食物相对蛇头行差 / 20
食物相对蛇头列差 / 20
食物曼哈顿距离 / 40
```

### 当前局面安全特征

```text
蛇头到食物 BFS 距离
蛇头到尾巴 BFS 距离
蛇头可达空间 reachableSpace
是否能追尾 canReachTail
reachableSpace / len
```

### 四方向模拟特征

对每个方向 W/A/S/D，模拟走一步后计算：

```text
safe[d]
eatFood[d]
distFoodAfter[d]
reachableAfter[d]
canReachTailAfter[d]
trappedAfter[d]
isReverse[d]
```

总特征维度大约：

```text
40 ~ 80
```

## A.2 网络结构

```text
Input(feature_dim)
Linear(feature_dim, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 64)
ReLU
Linear(64, 1)
```

输出一个数字：

```text
future_score
```

## A.3 损失函数

使用回归损失：

```text
MSELoss(predicted_future_score, true_future_score)
```

也可以试：

```text
HuberLoss
```

HuberLoss 对异常高分 / 极端数据更稳。

## A.4 优点

```text
训练快
部署到 C 简单
可以直接替换传统评分函数
数据需求较少
```

## A.5 缺点

```text
依赖手工特征
看不到完整地图形状
表达能力弱于 CNN
```

---

# 方案 B：CNN Value Network

## B.1 输入

输入整张地图的多通道表示：

```text
8 × 20 × 20
```

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

## B.2 网络结构

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
Linear(256, 1)
```

输出一个数字：

```text
future_score
```

## B.3 损失函数

```text
MSELoss
```

或：

```text
HuberLoss
```

## B.4 优点

```text
能看完整空间结构
可能学到死胡同、包围、狭窄入口等复杂形状
比 MLP 更少依赖人工特征
```

## B.5 缺点

```text
训练更慢
数据需求更大
部署到 C 很麻烦
```

---

## 7. 搜索怎么结合 Value Network？

### 7.1 最简单：一步搜索

每一步枚举四个动作：

```text
for action in W/A/S/D:
    next_state = simulate(state, action)
    value = immediate_gain + V(next_state)
```

选 value 最大的动作。

这个等价于：

```text
用 Value Network 替代四方向评分函数。
```

---

### 7.2 深度搜索

搜索未来 `depth` 步。

推荐：

```text
depth = 3 ~ 8
```

伪代码：

```text
search(state, depth):
    if dead:
        return gained_score - death_penalty

    if depth == 0:
        return V(state)

    best = -INF

    for action in legal_actions:
        next_state = simulate(state, action)
        immediate_gain = next_state.score - state.score
        value = immediate_gain + search(next_state, depth - 1)
        best = max(best, value)

    return best
```

最终选择：

```text
当前四个动作中，search 返回值最高的动作。
```

---

### 7.3 搜索中的死亡处理

死亡不应该永远是 `-INF`。

更合理：

```text
value = gained_score - death_penalty
```

推荐：

```text
death_penalty = 20 ~ 100
```

但如果是当前一步直接死亡，可以给极低分，避免自杀：

```text
immediate death -> -10000
```

区别：

```text
当前直接死：极差
搜索深处吃了很多后死：保留已获得收益，只扣死亡惩罚
```

---

## 8. Policy Network 可选增强

Policy Network 学的是：

```text
P(action | state)
```

输入局面，输出四个动作概率：

```text
W: 0.1
A: 0.2
S: 0.6
D: 0.1
```

它可以用于：

```text
1. 给搜索动作排序
2. 只搜索概率最高的前 k 个动作
3. 减少搜索爆炸
```

训练数据：

```text
state -> action
```

来源可以是：

```text
传统强算法选择的动作
高分轨迹中的动作
RL agent 的动作
```

注意：

```text
Policy Network 不一定直接决定动作；
它主要帮助搜索更快找到好分支。
```

最小可行版本可以先不做 Policy Network，只做 Value Network。

---

## 9. 推荐实验路线

### 第一阶段：MLP Value Network

```text
1. 用传统搜索算法跑很多局
2. 保存 state、score、final_score
3. 计算 future_score = final_score - score
4. 提取 MLP 特征
5. 训练 MLP 回归 future_score
6. 用 depth=1 搜索测试
7. 用 depth=3/5/7 搜索测试
8. 和纯传统搜索 baseline 比较
```

### 第二阶段：CNN Value Network

```text
1. 复用同一批轨迹数据
2. 把 state 转成 8×20×20
3. 训练 CNN 回归 future_score
4. 用 depth=1/3/5 搜索测试
5. 和 MLP Value 比较
```

### 第三阶段：加入 Policy Network

```text
1. 用高分策略动作训练 Policy
2. 搜索时按 Policy 概率排序动作
3. 每层只展开 top-k 动作
4. 比较搜索速度和得分
```

---

## 10. 和其他方案的区别

### 和纯传统搜索

```text
传统搜索 = 搜索 + 手写评分函数
搜索 + DL = 搜索 + 学出来的评分函数
```

### 和纯 RL

```text
纯 RL = 模型直接输出动作价值或动作概率
搜索 + DL = 模型只评价局面，最终动作由搜索决定
```

### 和监督模仿学习

```text
监督模仿 = 学老师怎么走
Value Network = 学一个局面未来还能赚多少分
```

---

## 11. 优缺点

## 优点

```text
比纯 RL 稳
比纯监督学习更接近长期收益
比纯手写评分函数表达能力强
可以和传统搜索算法自然结合
```

## 缺点

```text
需要生成大量轨迹数据
Value Network 会受数据来源影响
CNN 版本部署困难
搜索深度大时仍然会爆炸
训练出来的估值不一定比手写评分更准
```

---

## 12. OJ 部署建议

如果最终要提交 C 语言 OJ：

优先尝试：

```text
MLP Value Network + 有限搜索
```

原因：

```text
MLP 参数少
C 语言前向传播容易写
可以硬编码参数
能自然接入现有传统搜索框架
```

谨慎尝试：

```text
CNN Value Network
```

原因：

```text
参数量大
卷积前向传播复杂
C 文件可能很大
调试成本高
```

---

## 13. 一句话总结

搜索 + 深度学习的本质是：

```text
让搜索负责推演未来，让神经网络负责评价未来局面。
```

最小可行方案：

```text
传统算法生成轨迹数据
MLP Value Network 学 future_score
实战时 depth=3~7 搜索
叶子节点用 MLP 估值
选择估值最高路径的第一步
```
