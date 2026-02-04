# Auto-WZRY: 王者荣耀强化学习项目

本项目致力于构建一个基于强化学习的王者荣耀 AI Agent。

## 项目结构

本项目包含两个主要目录：

- **`main/`**:  当前项目的核心开发目录。所有的 **新代码实现、改进和重构** 都将在此目录中进行。我们将在参考前人工作的基础上，构建更加模块化、易于维护和扩展的强化学习框架。
- **`WZCQ/`**:  参考项目的源代码（[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT) 改进版）。
    - **注意**: 此文件夹仅作为参考资料，**不应修改**其中的任何内容。
    - 包含原始的训练脚本、模型定义（ResNet + Policy Gradient）以及辅助工具。

## 开发计划 (Roadmap)

我们将在 `main` 目录下逐步实现以下模块：

1.  **环境交互层 (Environment Wrapper)**:
    - 封装 ADB/Scrcpy/Minitouch 的底层交互。
    - 提供统一的 `Observation` (状态) 和 `Action` (动作) 接口，类似 OpenAI Gym/Gymnasium 标准。
2.  **数据处理 (Data Pipeline)**:
    - 优化截图采集与预处理流程。
    - 改进状态标注与奖励函数设计。
3.  **模型架构 (Model Architecture)**:
    - 重构特征提取网络。
    - 探索更先进的 RL 算法 (如 PPO, SAC 等) 替代原始的 Policy Gradient。
4.  **训练循环 (Training Loop)**:
    - 更加稳定的训练流程与日志记录 (TensorBoard/WandB)。

## 环境依赖 (Prerequisites)

开发环境配置建议：

- **Python**: 3.8+
- **PyTorch**: 1.9+ (根据 GPU 版本选择)
- **Android 工具**:
    - ADB (Android Debug Bridge)
    - [Scrcpy](https://github.com/Genymobile/scrcpy) (用于投屏)
    - [Minitouch](https://github.com/openstf/minitouch) (用于高速触控)

## 参考资料

- 原始参考代码位于 `WZCQ/` 目录。
- 更多技术细节请查阅 `WZCQ/readme.md`。
