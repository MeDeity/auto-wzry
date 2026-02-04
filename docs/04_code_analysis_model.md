# 04. 源码深度解析：AI 的大脑是如何构成的？

正如您所说，强化学习的核心在于“模型 (Model)”。如果说算法是灵魂，那么模型就是肉体。理解 `main/model.py` 确实是掌握本项目的关键。

本文将逐行解析 [model.py](file:///d%3A/Project/auto-wzry/main/model.py)，带您看懂 AI 的“视觉中枢”和“决策中枢”是如何用代码实现的。

---

## 1. 补充知识：PyTorch 的拼装艺术

在深入代码之前，我们先回答一个常见疑问：**这些代码里的变量（如 `self.encoder`）是哪里来的？**

```python
class CNNEncoder(nn.Module):
    # ...
```

*   **`nn.Module` 是什么？**
    *   它是一个**基类**（或者说“底座”）。它不包含具体的神经网络层，但它提供了一套“魔法”，让挂载在它上面的积木（如 `nn.Linear`, `nn.Conv2d`）能够被 PyTorch 自动识别、管理和训练。
*   **`self.encoder` 是预定义的吗？**
    *   **不是**。这些名字（`encoder`, `actor_mean`, `critic`）完全是我们自己起的。
    *   你可以把它改成 `self.eye`, `self.hand`, `self.judge`，代码依然能跑。
    *   但为了规范和可读性，我们通常使用通用的命名习惯。
*   **我们是继承了 ResNet18 吗？**
    *   **不是继承，是借用（组合）**。
    *   我们在 `__init__` 里写了 `resnet = models.resnet18(...)`，这相当于我们在自己的工厂里买了一台现成的 ResNet18 发动机。
    *   然后我们用 `list(resnet.children())[:-1]` 把它拆开，只留下了核心的“气缸”（特征提取层），扔掉了原来的“变速箱”（分类层），最后装上我们自己定制的部件。这是 Python 编程中**组合优于继承**的一个典型应用。

---

## 2. 核心理解：Model 就是“数据流水线”

基于上述理解，我们可以把 `model.py` 的作用总结为一句话：

**Model 负责定义如何把各种“数据处理组件”拼接成一条完整的流水线。**

*   **组件 (Components)**: 就像乐高积木。我们常用的组件如下表所示：

| 组件名称 (PyTorch) | 通俗名称 | 核心作用 | 现实类比 | 在本项目中的应用 |
| :--- | :--- | :--- | :--- | :--- |
| `nn.Conv2d` | 卷积层 | **提取特征**。识别图像中的线条、形状、纹理。 | 视网膜/扫描仪 | `CNNEncoder` 中提取画面特征 (兵线、英雄)。 |
| `nn.Linear` | 全连接层 | **变换维度/综合判断**。将特征向量映射到具体的输出（动作/分数）。 | 联想/决策大脑 | `Actor` 输出动作坐标；`Critic` 输出局势评分。 |
| `nn.ReLU` | 激活函数 | **过滤信号**。只保留正向信号，忽略负向信号（增加非线性）。 | 神经元阈值 (小于阈值不放电) | `CNNEncoder` 中每层卷积后都加了它，防止信号衰减。 |
| `nn.Tanh` | 双曲正切 | **归一化输出**。将输出限制在 [-1, 1] 之间。 | 限流阀 | `Actor` 输出动作时，保证坐标不会飞到屏幕外面去。 |
| `nn.Softmax` | 归一化指数 | **计算概率**。将一组数值变成概率分布 (和为1)。 | 投票统计 | (本项目未直接使用，常用于离散动作如贪吃蛇的上下左右)。 |
| `nn.BatchNorm2d` | 批归一化 | **数据整容**。让每一层的数据分布更均匀，训练更稳定。 | 考前心态调整 | `ResNet` 内部大量使用，防止训练中途“心态崩了”(梯度消失/爆炸)。 |
| `nn.Dropout` | 丢弃层 | **防止死记硬背**。随机扔掉一些神经元，强迫模型学通用的规律。 | 随机抽查考试 | (本项目暂未使用，如果发现过拟合可以加上)。 |

*   **拼接 (Assembly)**: `forward` 函数就是说明书。
    *   它规定了数据先流过组件 A，再流过组件 B。
    *   它的核心任务是**保证数据格式 (Shape) 的对齐**。比如组件 A 输出 `(Batch, 512)`，那么组件 B 的输入必须也是 512，否则流水线就卡住了。

---

## 3. 视觉中枢：CNNEncoder

这就是 AI 的眼睛。它的任务是：**把图片变成向量**。

```python
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(CNNEncoder, self).__init__()
        
        # 核心 1: 使用 ResNet18
        # ResNet18 是一个非常经典的深层卷积网络，包含 18 层。
        # weights=None 表示我们不使用预训练权重，而是从零开始学。
        # 就像刚出生的婴儿，一开始什么都看不懂，要慢慢学着认人。
        resnet = models.resnet18(weights=None)
        
        # 核心 2: 去头去尾
        # ResNet 原本是用来做分类的（比如分辨猫和狗），最后输出是 1000 个类别的概率。
        # 我们不需要分类，只需要提取特征。所以把最后的全连接层去掉 ([:-1])。
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 核心 3: 特征映射
        # ResNet 提取出来的特征是 512 维的。
        # 我们接一个全连接层 (Linear)，把它映射到我们想要的维度 (也是 512)。
        # 这就像把看到的复杂画面总结成一句话（特征向量）。
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        # x 是输入的图片张量，形状是 (Batch_Size, 3通道, 224宽, 224高)
        x = self.features(x)      # 图片 -> 特征图 (B, 512, 1, 1)
        x = x.view(x.size(0), -1) # 展平 -> 向量 (B, 512)
        x = F.relu(self.fc(x))    # 激活 -> 输出
        return x
```

### 💡 深度思考：这里面的“智能”在哪里？

您问到了点子上！**是的，`forward` 函数仅仅定义了数据的流向（做数学题的步骤）。**

*   它只规定了：先做卷积，再展平，最后做一次矩阵乘法。
*   它**并没有**规定：怎么识别敌人，怎么躲避技能。

**那么智能到底藏在哪里？**
智能藏在 `self.features` 和 `self.fc` 内部的 **参数 (Weights)** 里。
*   这些层里面包含了**上千万个**数字（权重）。
*   **初始状态**: 这些数字是随机生成的。这时候数据流过去，算出来的结果也是随机的（乱玩）。
*   **训练过程**: `train.py` 会不断修改这上千万个数字。
    *   看到敌人没识别出来？修改一点点权重。
    *   撞墙了？再修改一点点权重。
*   **最终状态**: 这些数字被“雕刻”成了特定的形状。当图片数据流过这些特定的数字时，就会神奇地计算出正确的特征。

**总结**: `model.py` 定义了大脑的**物理结构**（神经元怎么连接），而 **智慧** 是通过训练填入神经元里的数值。

---

## 4. 决策中枢：ActorCritic

这就是 AI 的大脑。它包含两个“人格”：Actor（演员）和 Critic（评论家）。

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(ActorCritic, self).__init__()
        
        # 1. 共享眼睛
        # Actor 和 Critic 共用同一个 CNNEncoder。
        # 这意味着它们看到的画面是一样的。
        self.encoder = CNNEncoder(output_dim=hidden_dim)
        
        # 2. 演员 (Actor) 的脑回路
        # 它的任务是输出动作 (Action)。
        # 动作是连续的坐标 (x, y)，所以我们需要输出一个高斯分布 (正态分布)。
        # 高斯分布由 均值 (Mean) 和 标准差 (Std) 决定。
        
        # -> 输出均值 (Mean)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        
        # -> 输出标准差 (Log Std)
        # 这是一个可学习的参数。一开始标准差很大（动作很随机），
        # 随着训练进行，它会变小（动作越来越精准）。
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # 3. 评论家 (Critic) 的脑回路
        # 它的任务是打分 (Value)。
        # 输入画面特征，输出一个标量数值 (1维)。
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # 第一步：看图
        features = self.encoder(state)
        
        # 第二步：演员思考
        # 使用 Tanh 激活函数，把均值限制在 [-1, 1] 之间。
        # 这样方便后续映射到屏幕坐标 [0, 1]。
        mean = torch.tanh(self.actor_mean(features))
        
        # 计算标准差 (取指数是为了保证它是正数)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        # 第三步：评论家打分
        value = self.critic(features)
        
        return mean, std, value
```

**关键方法解析**:

*   `get_action(state)`: **决策时刻**
    *   这是 AI 在玩游戏时调用的函数。
    *   它根据 Mean 和 Std 构建一个概率分布，然后**从中采样**一个动作。
    *   *为什么要采样？* 为了探索。即使是同样的局面，AI 每次可能也会有微小的不同操作，这样才能发现新战术。

*   `evaluate(state, action)`: **反思时刻**
    *   这是 AI 在训练（晚上复盘）时调用的函数。
    *   它计算：在当时那个局面下，做那个动作的**概率**是多少 (log_prob)？
    *   它还计算：动作的**熵** (entropy)。熵越大，代表动作越随机。我们希望熵适中，既不完全乱动，也不要死板僵化。

---

## 5. 总结

`main/model.py` 其实只做了一件事：**映射**。

*   **输入**: 屏幕截图 (224x224x3)
*   **中间**: ResNet18 提取特征 (512维向量)
*   **输出 A (Actor)**: 动作分布 (均值, 标准差) -> 告诉手怎么动
*   **输出 B (Critic)**: 局面评分 (数值) -> 告诉大脑现在情况好不好

理解了这个文件，你就理解了强化学习最底层的“物理载体”。剩下的 PPO 算法，只是在教这个载体**如何调整参数**，让输出的动作更准确，评分更客观。
