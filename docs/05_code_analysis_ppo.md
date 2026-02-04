# 05. 源码深度解析：AI 是如何自我进化的？(PPO 算法)

如果说 `model.py` 是 AI 的大脑结构，那么 `algo/ppo.py` 就是**大脑的思维方式**。

**PPO (Proximal Policy Optimization)** 是目前最流行的深度强化学习算法之一（OpenAI 开发，ChatGPT 也在用）。它的核心思想是：**步子迈小点，走得稳一点。**

本文将逐行解析 [ppo.py](file:///d%3A/Project/auto-wzry/main/algo/ppo.py)，带您看懂 AI 是如何通过“反思”来变强的。

---

## 1. 经验笔记本：RolloutBuffer

AI 并不是玩一步学一步，而是先玩一整段（比如几分钟），把这段时间的经历都记下来，晚上统一复盘。这个笔记本就是 `RolloutBuffer`。

```python
class RolloutBuffer:
    def __init__(self):
        # 记录每一步的 6 要素
        self.states = []    # 当时看到了什么？
        self.actions = []   # 当时做了什么？
        self.rewards = []   # 这一步得了多少分？
        self.dones = []     # 游戏结束了吗？
        self.log_probs = [] # 当时有多自信？
        self.values = []    # 当时觉得局势如何？
```

### 关键函数：计算优势 (GAE)
这是 PPO 的核心数学部分。`compute_returns_and_advantages` 计算了两个重要指标：

1.  **Returns (回报)**: 从这一步开始，后面**总共**拿了多少分？
2.  **Advantages (优势)**: 这一步做得**比预期好多少**？
    *   比如 Critic 觉得这一步只能拿 5 分，结果实际拿了 10 分，那么优势就是 +5。说明这一步做对了！
    *   如果优势是正的，我们就要**鼓励**这个动作；如果是负的，就要**抑制**它。

---

## 2. 进化引擎：PPO.update

这是 AI “长脑子”的地方。

```python
    def update(self, rollouts):
        # ... (数据提取与归一化)

        # 核心循环：拿着笔记本反复复盘 (ppo_epochs 次)
        for _ in range(self.ppo_epochs):
            for batch in loader:
                # 1. 回忆当时的情况
                # 这里会再次调用模型 (self.policy.evaluate)，看看现在的模型对当时的情况怎么看
                new_log_probs, values, entropy = self.policy.evaluate(b_states, b_actions)
                
                # 2. 计算 Ratio (概率比)
                # Ratio = 现在做这个动作的概率 / 当时做这个动作的概率
                # 如果 Ratio > 1，说明我们现在更倾向于做这个动作了
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                
                # 3. 计算 Actor Loss (PPO 的精髓：剪裁 Clip)
                # 我们希望 AI 进步，但不要突然改变太大。
                # 如果 Ratio 变化太大（比如超过 1.2 或低于 0.8），我们就把它“剪裁”掉。
                # 这就是 PPO 名字里 "Proximal" (近端/近似) 的由来：只在旧策略附近找新策略。
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 4. 计算 Critic Loss
                # 评论家也要进步，它的预测应该越来越接近真实的 Returns。
                value_loss = (b_returns - values).pow(2).mean()
                
                # 5. 计算 Entropy Loss
                # 鼓励探索，不要过早变得死板。
                entropy_loss = -entropy.mean()
                
                # 6. 总 Loss
                loss = actor_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # 7. 反向传播更新参数
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

---

## 3. 总结

PPO 的工作流程可以用一句话总结：

**“收集数据 -> 计算实际表现比预期好多少 (Advantage) -> 如果好就增加做这个动作的概率，但别增加太多 (Clip) -> 同时修正预期的准确度 (Value Loss)。”**

理解了 [ppo.py](file:///d%3A/Project/auto-wzry/main/algo/ppo.py)，你就理解了 AI 是如何从原本的“随机乱动”一步步收敛到“神操作”的数学原理。
