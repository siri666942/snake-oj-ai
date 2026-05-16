## 第一优先：David Silver 强化学习课

这是最适合你现在状态的入口。它从 MDP、value function、policy、Q-learning、function approximation、planning + learning 一路讲到深度 RL 前的核心骨架。DeepMind 官方 YouTube 有完整 playlist。([YouTube][1])

你重点看：

```text
Lecture 1: Introduction
Lecture 2: Markov Decision Processes
Lecture 3: Planning by Dynamic Programming
Lecture 4: Model-Free Prediction
Lecture 5: Model-Free Control
Lecture 6: Value Function Approximation
Lecture 8: Integrating Learning and Planning
```

这几节正好对应我们今天聊的：

```text
state / action / reward
Value Network
Q / DQN
搜索 + 学习
```

---

## 第二优先：Sutton & Barto《Reinforcement Learning: An Introduction》

这是 RL 经典教材。作者官网提供第二版 PDF。([incompleteideas.net][2])

你不用全读，先读：

```text
Chapter 1: Introduction
Chapter 3: Finite Markov Decision Processes
Chapter 4: Dynamic Programming
Chapter 5: Monte Carlo Methods
Chapter 6: Temporal-Difference Learning
Chapter 9: On-policy Prediction with Approximation
Chapter 11: Off-policy Methods with Approximation
```

这本书适合补“为什么 DQN 的 target 是 reward + gamma * maxQ(next_state)”这种根基。

---

## 第三优先：Stanford CS234

CS234 是系统强化学习课，课程页面写得很清楚：它会讲 RL 的核心挑战和方法，包括 generalization 和 exploration。([Stanford University][3])

它比 David Silver 更课程化，适合你后面真正系统学。官方也有 lecture materials 页面，里面按 topic 给 slides 和补充材料。([Stanford University][4])

你可以把它当第二轮复习资源。

---

## 第四优先：Berkeley CS285 Deep RL

这个是深度强化学习课，更偏现代 deep RL。Berkeley 官方课程页是 CS 285 / Deep Reinforcement Learning。([rail.eecs.berkeley.edu][5])

但我不建议你现在直接冲它。它更适合你已经理解：

```text
MDP
value function
policy
Q-learning
policy gradient
function approximation
```

之后再看。

---

## 你的学习顺序

最小路线：

```text
1. David Silver Lecture 1~6
2. 用 Python 写一个极简 gridworld / snake env
3. 手写 tabular Q-learning
4. 再写 DQN
5. 再看 Sutton & Barto 对应章节补原理
6. 最后看 CS285
```

对应到这次贪吃蛇项目，你应该优先理解这几个词：

```text
MDP：游戏怎么抽象成状态转移
Value：一个局面未来值多少钱
Q：某个局面下某个动作值多少钱
Policy：怎么根据局面选动作
Reward：训练信号
Exploration：为什么要随机试错
Planning：为什么搜索能和神经网络结合
```

一句话：**David Silver 入门，Sutton & Barto 补底层，CS234 系统化，CS285 进 deep RL。**

[1]: https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&utm_source=chatgpt.com "DeepMind x UCL | Introduction to Reinforcement Learning ..."
[2]: https://incompleteideas.net/book/the-book-2nd.html?utm_source=chatgpt.com "Reinforcement Learning: An Introduction"
[3]: https://web.stanford.edu/class/cs234/?utm_source=chatgpt.com "CS234: Reinforcement Learning Winter 2026"
[4]: https://web.stanford.edu/class/cs234/modules.html?utm_source=chatgpt.com "CS234: Reinforcement Learning Winter 2026"
[5]: https://rail.eecs.berkeley.edu/deeprlcourse/?utm_source=chatgpt.com "CS 185/285"
