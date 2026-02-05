# 03. 模仿学习 (Imitation Learning) 指南

为了解决强化学习初期探索效率低下的问题（即“随机乱动”阶段），我们引入了 **模仿学习** 模块。通过录制人类高手的操作数据，让模型快速学会基础的游戏理解和操作。

## 1. 核心流程

1.  **录制数据**: 人工操作游戏，记录画面 (State) 和 动作 (Action)。
2.  **行为克隆 (BC)**: 使用监督学习训练一个基础策略网络。
3.  **强化微调**: 将 BC 模型的权重作为 PPO 的初始权重，进行后续训练。

---

## 2. 专家数据录制器 (`main/record_expert.py`)

这是一个用于采集“示教数据”的工具。它会弹出一个窗口显示手机画面，并捕获您的鼠标操作。

### 启动方式
确保手机已连接并开启 USB 调试。

```powershell
python main/record_expert.py
```

### 操作说明
*   **窗口交互**: 
    *   在弹出的 `Expert Recorder` 窗口中，使用鼠标**左键点击**或**拖拽**来模拟手指操作。
    *   脚本会将鼠标在窗口内的相对坐标转换为手机屏幕坐标。
*   **快捷键**:
    *   `r`: **开始/停止 录制**。
        *   按下 `r` 开始：控制台提示 "Recording started..."，画面左上角会出现**红点**。
        *   再次按 `r` 停止：控制台提示 "Recording stopped..."，数据会自动保存。
    *   `q`: **退出程序**。

### 数据存储
录制的数据保存在项目根目录的 `data/expert_data` 下：
*   `ep{id}_{timestamp}.jpg`: 游戏截图（原始画面）。
*   `episode_{id}.json`: 索引文件，记录了每一帧的时间戳以及对应的鼠标动作。

---

## 3. 行为克隆训练器 (`main/train_bc.py`)

在采集了足够的数据（建议至少 3-5 局完整游戏，包含不同场景）后，使用此脚本进行训练。

### 启动方式

```powershell
# 使用 Conda 环境运行 (推荐)
conda run --no-capture-output -n wzry python main/train_bc.py
```

### 训练细节
*   **输入**: 读取 `data/expert_data` 下的所有 JSON 和图片。
*   **模型**: 使用与 PPO 相同的 `CNNEncoder` 结构，但输出层直接拟合鼠标坐标 (MSE Loss)。
*   **输出**: 训练好的模型权重会保存在 `models/` 目录下，例如 `bc_model_epoch_50.pth`。

### 常见问题
*   **Q: 需要录制多少数据？**
    *   A: 越多越好。对于简单的移动和攻击，几千帧（约几分钟）即可看到效果；若要学会连招，可能需要数万帧。
*   **Q: 录制时卡顿怎么办？**
    *   A: 录制脚本默认每帧都保存图片，如果 I/O 瓶颈导致卡顿，可以尝试降低 Scrcpy 分辨率或修改代码减少采样率。

---

## 4. 下一步：接入 PPO

完成 BC 训练后，您将获得一个已经“会玩”的模型权重。在运行 PPO 训练 (`main/train.py`) 时，可以加载这个权重作为起点：

### 4.1 首次启动 (热启动)

使用 `--pretrained-path` 加载 BC 模型，作为 PPO 的初始起点。

```powershell
# 加载 BC 预训练权重启动 PPO 训练
conda run --no-capture-output -n wzry python main/train.py --pretrained-path models/bc_model_epoch_50.pth
```

> **注意**: 请将 `models/bc_model_epoch_50.pth` 替换为您实际生成的模型路径。

### 4.2 断点续训

如果您中断了训练，想从之前的 PPO 存档继续（例如从第 10 轮继续），请使用 `--resume-path`：

```powershell
# 从 PPO 断点继续训练 (脚本会自动识别 Epoch 并从下一轮开始)
conda run --no-capture-output -n wzry python main/train.py --resume-path models/ppo_wzry_epoch_10.pth
```

> **优先级说明**: 如果同时指定了 `resume-path` 和 `pretrained-path`，脚本会**优先使用 resume-path** 进行断点续训。

---

## 5. 多局数据积累与迭代 (Iterative Data Collection)

**Q: 我玩了一局，模型还是很笨，如何让它更聪明？**

这是非常正常的！一局游戏的数据量往往不足以覆盖所有情况（例如：走位、团战、回城等）。正确的做法是 **“增量式教学”**。

### 操作步骤

1.  **不要删除旧数据**：
    *   保留 `data/expert_data` 文件夹中的旧数据。
2.  **再次运行录制脚本**：
    *   直接运行 `python main/record_expert.py`。
    *   脚本会自动检测已有的 `episode_0.json`，并将新的一局保存为 `episode_1.json`、`episode_2.json` 等。
    *   **建议**：专门针对模型薄弱的环节进行录制（例如：如果您发现模型不会走出泉水，就专门录制几分钟“走出泉水”的操作）。
3.  **重新运行 BC 训练**：
    *   运行 `conda run --no-capture-output -n wzry python main/train_bc.py`。
    *   训练脚本会自动加载 `data/expert_data` 下的 **所有** JSON 文件（新旧数据一起训练）。
    *   训练出的新模型（如 `bc_model_epoch_50.pth`）将会融合所有局的经验。

### 常见疑问
**Q: 每次都把旧数据重新喂给模型，会有问题吗？**
**A: 不仅没问题，而且必须这样做！**
目前的训练脚本每次都是**从头开始**训练一个新的模型。如果您只喂给它“新的一局”数据，模型会发生**灾难性遗忘 (Catastrophic Forgetting)**——即学会了新操作（如走出泉水），但把旧操作（如打怪）全忘了。
只有将 **旧数据 + 新数据** 混合在一起训练，模型才能同时掌握所有技能，变得越来越强。

**Q: 数据多了训练时间太长怎么办？**
**A: 使用断点续训！**
从第二局开始，您可以使用 `--resume-path` 参数，加载上一局训练好的模型继续训练。
这样模型已经有了基础，只需要少量 Epoch (如 20 轮) 就能适应新数据，大大节省时间。

```powershell
# 示例：加载上一局的模型 (epoch 50)，再训练 20 轮
conda run --no-capture-output -n wzry python main/train_bc.py --resume-path models/bc_model_epoch_50.pth --epochs 20
```

### 循环迭代
您可以多次重复上述过程：
`录制新数据` -> `重新训练 BC` -> `测试效果` -> `发现不足` -> `针对性录制` -> ...

直到您觉得 BC 模型的基础表现已经可以接受（比如能正常走路、简单平A），再转入 PPO 强化学习阶段。
