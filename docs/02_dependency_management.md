# 依赖管理与自动安装 (Dependency Management)

为了降低环境配置的门槛，本项目实现了一套自动依赖管理机制。在 `main.utils.downloader` 模块中，封装了对 **Scrcpy** 和 **ADB (Platform-Tools)** 的自动下载、解压和路径配置逻辑。

## 1. 工作原理

当 `AdbWrapper` 或 `ScrcpyWrapper` 初始化时，它们会按照以下顺序查找可执行文件：

1.  **系统环境变量 (PATH)**: 首先检查系统中是否已经安装并配置了 `adb` 或 `scrcpy`。如果存在，直接使用系统版本。
2.  **本地工具目录 (tools/)**: 如果 PATH 中未找到，程序会检查项目根目录下的 `tools/` 文件夹。
3.  **自动下载**: 如果本地目录也不存在，程序将自动从官方源下载最新版本的压缩包，并解压到 `tools/` 目录中。

## 2. 目录结构

自动安装完成后，项目的目录结构如下：

```text
d:\Project\auto-wzry\
├── main/
├── tools/                  <-- 自动生成的工具目录
│   ├── scrcpy-win64-v2.4/  <-- Scrcpy 目录
│   │   └── scrcpy.exe
│   └── platform-tools/     <-- ADB 目录
│       └── adb.exe
└── ...
```

## 3. 手动干预

虽然自动化脚本能处理大部分情况，但在以下场景下您可能需要手动干预：

-   **网络问题**: 如果您的网络无法访问 GitHub 或 Google 服务器，自动下载可能会失败。此时您可以手动下载压缩包并解压到 `tools/` 目录。
-   **版本冲突**: 如果您系统中已安装了旧版本的 ADB/Scrcpy 且在 PATH 中，程序会优先使用旧版本。如果您希望使用项目内置的最新版本，请从 PATH 中移除旧版本，或者在代码中显式指定路径。

## 4. 源码解析

核心逻辑位于 `main/utils/downloader.py`：

-   `ensure_scrcpy_installed()`: 检查并安装 Scrcpy。
-   `ensure_adb_installed()`: 检查并安装 Android Platform Tools (包含 ADB)。

这两者都被设计为**幂等**的：如果检测到文件已存在且完整，它们会直接返回路径，不会重复下载。
