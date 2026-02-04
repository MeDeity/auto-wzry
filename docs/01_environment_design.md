# 环境交互层设计 (Environment Interaction Layer)

在强化学习任务中，**环境 (Environment)** 是 Agent 进行交互的对象。对于王者荣耀这样的手游，我们需要构建一个高效、低延迟的交互层，让 PC 端的 AI 能够“看”到游戏画面并“摸”到游戏操作。

本项目采用 **“投屏 + 旁路控制”** 的架构，主要包含以下核心组件：

## 1. 架构概览

```mermaid
graph TD
    Device[Android 设备 (手机/模拟器)] 
    PC[PC 主机 (Python Agent)]
    
    Device -- 视频流 (Scrcpy) --> PC
    PC -- 触控指令 (Minitouch) --> Device
    PC -- 管理指令 (ADB) --> Device
```

## 2. 核心组件详解

### 2.1 ADB (Android Debug Bridge)
- **作用**: Android 开发调试的通用工具，是连接 PC 和手机的桥梁。
- **本项目用途**:
    - 检测设备连接状态。
    - 端口转发 (Port Forwarding)，为 Minitouch 提供通信隧道。
    - 启动/停止应用。
- **封装**: `main/env/adb_wrapper.py`

### 2.2 Scrcpy (Screen Copy)
- **作用**: 开源的高性能 Android 投屏工具。
- **特点**: 低延迟 (35~70ms)、高画质。
- **本项目用途**: 将手机画面投射到 PC 屏幕上的一个窗口中。
- **封装**: `main/env/scrcpy_wrapper.py`

### 2.3 Window Capture (窗口截图)
- **作用**: AI 的“眼睛”。
- **原理**: 使用 Windows API (`pywin32`) 直接截取 Scrcpy 投屏窗口的像素数据。
- **为什么不直接用 ADB 截图?**: ADB 截图速度极慢 (0.5~1秒/帧)，无法满足实时游戏需求。窗口截图可以达到 30+ FPS。
- **封装**: `main/env/window_capture.py`

### 2.4 Minitouch
- **作用**: AI 的“手指”。
- **原理**: 直接向 Android 设备的 `/dev/input/event*` 写入触控事件。
- **优势**: 响应速度极快，支持多点触控（移动 + 施法）。相比 `adb shell input tap` (延迟约 300ms)，Minitouch 几乎无延迟。
- **封装**: `main/env/minitouch_wrapper.py` (基于 `pyminitouch` 库)

## 3. 交互循环 (Game Loop)

一个典型的 AI 决策循环如下：

1.  **Observation (观察)**: `WindowCapture` 截取一帧图像。
2.  **Processing (处理)**: 图像缩放、归一化，转换为 Tensor。
3.  **Inference (推理)**: 神经网络根据图像输出动作概率。
4.  **Action (动作)**: 将动作转换为坐标，通过 `Minitouch` 发送到手机。

## 4. 环境搭建与依赖安装

在运行本项目之前，需要准备好以下软件环境。

### 4.1 系统工具依赖

本项目依赖外部工具与 Android 设备进行通信，请确保已安装并配置好以下工具：

1.  **ADB (Android Debug Bridge)**
    -   **下载**: 包含在 Android SDK Platform-Tools 中，可从 [Android 开发者官网](https://developer.android.com/studio/releases/platform-tools) 下载。
    -   **配置**: 解压后，将文件夹路径添加到系统的 `PATH` 环境变量中。
    -   **验证**: 打开终端运行 `adb devices`，应无报错。

2.  **Scrcpy (Screen Copy)**
    -   **下载**: 从 [GitHub Release](https://github.com/Genymobile/scrcpy/releases) 下载 Windows 版本 (例如 `scrcpy-win64-v2.x.zip`)。
    -   **配置**: 解压后，将文件夹路径添加到系统的 `PATH` 环境变量中。
    -   **验证**: 打开终端运行 `scrcpy --version`，应显示版本信息。

3.  **Minitouch (自动安装)**
    -   本项目使用的 Python 库 `pyminitouch` 会在运行时尝试自动推送 `minitouch` 二进制文件到手机。
    -   **注意**: Minitouch 不支持 Android 10+ 的部分设备。如果遇到兼容性问题，后续可能需要切换到 `MaaTouch` 或其他方案。

### 4.2 Python 依赖

本项目基于 Python 3.8+ 开发。请在项目根目录下运行以下命令安装 Python 依赖库：

```bash
pip install -r main/requirements.txt
```

`main/requirements.txt` 包含的核心库如下：
-   `pywin32`: 用于 Windows 窗口截图。
-   `pyminitouch`: 用于 Android 设备触控。
-   `numpy`: 数值计算与图像矩阵处理。
-   `opencv-python`: 图像处理 (Resize, Color Space Convert)。
